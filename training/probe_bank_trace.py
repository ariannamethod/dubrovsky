"""Frozen writer trajectory and fixed-prefix bank-age interventions.

Every bank snapshot is checked against the checkpoint's native recurrence.
Historical generations are replayed, then read with changed banks on the same
actual working/Qwen prefix. Private scenes and checkpoints remain external.
"""
import argparse
import json
from pathlib import Path

import torch

from embedded_experts import ExpertQwen, tensor_hashes
from expert_data import encode_expert_prompt, memory_inputs
from probe_expert_readout import digest, ids_digest, intervene, question_bounds
from probe_role_transfer import prepare_transfer_pairs
from sample_experts import generate
from train_role_gain import with_gain, first_name_margin
from train_role_readout import prefix_state, swapped_state, value_logp


def user_segments(mask):
    if mask.ndim != 2 or mask.shape[0] != 1 or mask.dtype != torch.bool:
        raise ValueError('Trace requires one unpadded boolean user mask')
    positions = torch.where(mask[0])[0].tolist()
    if not positions:
        raise ValueError('No user content')
    spans = []
    for pos in positions:
        if not spans or spans[-1][1] != pos:
            spans.append([pos, pos + 1])
        else:
            spans[-1][1] += 1
    return spans


@torch.no_grad()
def trace_writer(block, source, mask, initial=None):
    """Record exact isolated writes, gates and every user-message endpoint."""
    spans = user_segments(mask)
    if block.config.format_version < 2 or source.shape[:2] != mask.shape:
        raise ValueError('Aligned isolated features are required')
    bank = block.initial_state(1).user if initial is None else initial.clone()
    snapshots = []; tokens = []
    for message, (start, end) in enumerate(spans):
        for pos in range(start, end):
            x = source[:, pos].to(block.source.dtype)
            x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
            fresh = torch.tanh(torch.einsum('bd,edr->ber', x, block.source))
            features = torch.cat((x[:, None].expand(-1, block.config.experts, -1),
                                  bank, fresh, fresh - bank), dim=-1)
            retain = torch.sigmoid((features * block.retain).sum(-1) + block.retain_bias)
            write = torch.sigmoid((features * block.write).sum(-1) + block.write_bias)
            candidate = torch.tanh(retain[..., None] * bank + write[..., None] * fresh)
            tokens.append(dict(position=pos, message=message,
                retain=retain[0].tolist(), write=write[0].tolist(),
                change_rms=(candidate-bank).square().mean().sqrt().item(),
                bank_rms=candidate.square().mean().sqrt().item()))
            bank = candidate
        snapshots.append(bank.clone())
    return dict(spans=spans, banks=snapshots, tokens=tokens)


def distance(a, b):
    difference = (a-b).float()
    return dict(l2=difference.norm().item(), rms=difference.square().mean().sqrt().item(),
                expert_rms=difference.square().mean(-1)[0].sqrt().tolist())


@torch.no_grad()
def checked_trace(block, raw, mask, source):
    trace = trace_writer(block, source, mask)
    for (_, end), bank in zip(trace['spans'], trace['banks']):
        _, state, _ = block(raw[:, :end], mask[:, :end], memory_hidden=source[:, :end])
        if not torch.equal(bank, state.user):
            raise RuntimeError('Writer trace differs from native checkpoint recurrence')
    return trace


def serial_trace(trace):
    return dict(spans=trace['spans'], tokens=trace['tokens'],
                banks=[b.cpu().tolist() for b in trace['banks']], native_parity=True)


@torch.no_grad()
def measure(model, item, state, head32):
    result = dict(first_margin=first_name_margin(model, item, state).item())
    for precision in ('native', 'add32'):
        values = [value_logp(model, p, state, precision=precision, head32=head32)
                  for p in item['alternatives']]
        result[precision] = dict(margin=(values[0].sum()-values[1].sum()).item(),
                                values=[v.tolist() for v in values])
    return result


@torch.no_grad()
def main():
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser()
    for name in ('model', 'states', 'cases', 'revisits', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    a = parser.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (a.output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    model = ExpertQwen.from_pretrained(a.model, dtype='auto', attn_implementation='sdpa').to(a.device).eval()
    model.experts = with_gain(model.experts); model.expert_config = model.experts.config
    model.requires_grad_(False); body = tensor_hashes(model.base)
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    specs = json.loads(a.states.read_text()); cases = json.loads(a.cases.read_text())
    revisits = json.loads(a.revisits.read_text())
    if len(specs) != 3 or len(cases['train']) != 8 or len(cases['validation']) != 4 or len(revisits) != 15:
        raise ValueError('Sealed inventory differs')
    states = {}
    for spec in specs:
        path = a.states.parent/spec['file']
        if digest(path) != spec['sha256']:
            raise ValueError('Expert hash differs')
        states[spec['arm']] = load_file(str(path), device=a.device)
    writer_names = ('source', 'retain', 'write', 'retain_bias', 'write_bias')
    for state in states.values():
        if any(not torch.equal(state[n], states['parent4'][n]) for n in writer_names):
            raise ValueError('Writer differs between checkpoints')
    model.experts.load_state_dict(states['parent4'], strict=True)
    prepared = {}; trajectories = {}; pair_records = []
    for split in ('train', 'validation'):
        items, audit = prepare_transfer_pairs(model, tokenizer, cases[split], a.device)
        prepared[split] = items; write('prefix-'+split+'.json', audit)
        for item in items:
            p = item['alternatives'][0]; cut = p['cut']
            trace = checked_trace(model.experts, p['raw'][:, :cut], p['mask'][:, :cut], p['source'][:, :cut])
            if len(trace['banks']) != 3:
                raise ValueError('Expected two fact messages and the recall question')
            trajectories[item['id']] = trace
        for pair in cases[split]:
            rows = pair['rows']; left, right = [trajectories[r['id']] for r in rows]
            distances = [distance(x, y) for x, y in zip(left['banks'], right['banks'])]
            common = []
            for actual, donor in ((left, right), (right, left)):
                index = 0 if actual is left else 1
                p = next(i for i in items if i['id']==rows[index]['id'])['alternatives'][0]
                start, end = actual['spans'][-1]
                propagated = trace_writer(model.experts, p['source'][:, start:end],
                    p['mask'][:, start:end], donor['banks'][-2])['banks'][-1]
                common.append(distance(actual['banks'][-1], propagated))
            pair_records.append(dict(split=split, pair=pair['id'], role=pair['role'],
                after_messages=distances, same_question_features=common))
    write('writer-traces.json', {k:serial_trace(v) for k,v in trajectories.items()})
    write('source-separation.json', pair_records)
    head32 = model.base.lm_head.weight.detach().float(); measurements = []
    for arm, weights in states.items():
        model.experts.load_state_dict(weights, strict=True)
        frozen = tensor_hashes(model.experts)
        for split, items in prepared.items():
            for item in items:
                other = next(i for i in items if i['pair']==item['pair'] and i['id']!=item['id'])
                own = trajectories[item['id']]['banks']; donor = trajectories[other['id']]['banks']
                state = prefix_state(model, item['alternatives'][0])
                if not torch.equal(state.user, own[-1]):
                    raise RuntimeError('Cached bank differs')
                banks = dict(current_own=own[-1], current_donor=donor[-1],
                             prequestion_own=own[-2], prequestion_donor=donor[-2],
                             zero_final=torch.zeros_like(state.user))
                result = {name:measure(model,item,swapped_state(state,bank),head32)
                          for name,bank in banks.items()}
                measurements.append(dict(arm=arm, split=split, id=item['id'], pair=item['pair'],
                    role=item['role'], cut=item['alternatives'][0]['cut'], results=result))
        if tensor_hashes(model.experts) != frozen:
            raise RuntimeError('Frozen expert checkpoint changed')
        print(json.dumps(dict(measured=arm, rows=len(measurements))), flush=True)
    write('readout.json', measurements)
    model.experts.load_state_dict(states['source4'], strict=True)
    frozen = tensor_hashes(model.experts); count = 0; replay = []
    with (a.output/'replies.jsonl').open('x') as stream:
        for row in revisits:
            prompt = encode_expert_prompt(tokenizer,row,3072)
            ids = torch.tensor([prompt['input_ids']],device=a.device)
            mask = torch.tensor([prompt['user_mask']],device=a.device,dtype=torch.bool)
            source_ids = torch.tensor([prompt['memory_input_ids']],device=a.device)
            source_mask = torch.tensor([prompt['memory_user_mask']],device=a.device,dtype=torch.bool)
            source = model.base.model(input_ids=source_ids,use_cache=False,return_dict=True).last_hidden_state
            raw = model.base.model(input_ids=ids,use_cache=False,return_dict=True).last_hidden_state
            aligned = torch.zeros_like(raw); aligned[mask] = source[source_mask]
            trace = checked_trace(model.experts,raw,mask,aligned)
            cut = question_bounds(mask)[1]
            if row['fact_message_count'] >= len(trace['banks']):
                raise ValueError('Fact-complete snapshot must precede the question')
            fact = trace['banks'][row['fact_message_count']-1]
            before = trace['banks'][-2]
            options = dict(normal=None, prequestion_own=before,
                           zero_final=torch.zeros_like(before))
            if row['group']=='fixed':
                options['prequestion_donor'] = trajectories[row['donor_id']]['banks'][-2]
            else:
                options['fact_own'] = fact
            write('revisit-trace-'+row['id']+'.json',serial_trace(trace))
            for mode, bank in options.items():
                with intervene(model, **({} if bank is None else dict(cut=cut,bank=bank))):
                    answer = generate(model,tokenizer,row,seed=row['seed'],mode='experts',max_new_tokens=192)
                exact = answer['token_ids']==row['reference_token_ids'] if mode=='normal' else None
                if exact is False:
                    raise RuntimeError('Archived normal answer does not replay exactly: '+row['id'])
                record = {k:v for k,v in row.items() if k!='reference_token_ids'}
                record.update(mode=mode, checkpoint='source4-step16', cut=cut,
                    prefix_sha256=ids_digest(prompt['input_ids']), replay_exact=exact, **answer)
                stream.write(json.dumps(record,ensure_ascii=False)+'\n'); stream.flush(); count += 1
                if mode=='normal': replay.append(dict(id=row['id'],exact=True))
                print(json.dumps(dict(reply=count,id=row['id'],mode=mode,eos=answer['eos_reached'])),flush=True)
    if tensor_hashes(model.base)!=body or tensor_hashes(model.experts)!=frozen:
        raise RuntimeError('Frozen weights changed')
    if count != 60:
        raise RuntimeError('Reply inventory differs')
    write('completed.json',dict(stage=35,optimizer_steps=0,body_tensors_preserved=len(body),
        expert_states_frozen=True, writer_identical_across_states=True, paired_rows=len(measurements),
        replies=count, replays=replay, states=specs, primary_promoted=False))


if __name__=='__main__':
    main()
