"""Frozen paired transfer and clarification probes; private inputs stay external.

Names, assistant history, and query reference are crossed independently. Exact
source roles and literary continuation receive separate contextual review.
"""
import argparse
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import re

import torch

from embedded_experts import ExpertQwen, tensor_hashes
from expert_data import encode_expert_prompt
from probe_expert_readout import digest, ids_digest, prepare_hidden, question_bounds
from sample_experts import generate
from train_role_readout import evaluate, prefix_state


@torch.no_grad()
def causal_alternatives(body, sequences, device):
    """Compute a shared prefix once; fork its cache before each answer suffix."""
    common = 0
    while common < min(map(len,sequences)) and len({s[common] for s in sequences}) == 1:
        common += 1
    if common == 0 or common == min(map(len,sequences)):
        raise ValueError('Expected a nonempty shared prefix and differing suffixes')
    prefix = body(input_ids=torch.tensor([sequences[0][:common]],device=device),
                  use_cache=True,return_dict=True)
    result = []
    for sequence in sequences:
        cache = copy.deepcopy(prefix.past_key_values)
        pieces = [prefix.last_hidden_state]
        for token in sequence[common:]:
            out = body(input_ids=torch.tensor([[token]],device=device),
                       past_key_values=cache,use_cache=True,return_dict=True)
            cache = out.past_key_values
            pieces.append(out.last_hidden_state)
        result.append(torch.cat(pieces,dim=1))
    return result, common


@torch.no_grad()
def prepare_transfer_pairs(model, tokenizer, pairs, device):
    result, audit = [], []
    for pair in pairs:
        role = pair['role']
        if role not in ('owner','keeper') or len(pair['rows'])!=2:
            raise ValueError('Expected a two-source role pair')
        prepared = []
        for i,row in enumerate(pair['rows']):
            other = pair['rows'][1-i]
            if row[role]==other[role]:
                raise ValueError('Counterfactual value did not change')
            alternatives = [prepare_hidden(model,tokenizer,row,role.title()+': '+r[role],device)
                            for r in (row,other)]
            for item in alternatives:
                item['cut'] = question_bounds(item['mask'])[1]
                if min(item['positions']) <= item['cut']:
                    raise ValueError('Value overlaps user prefix')
            a,b = alternatives
            if a['cut'] != b['cut']:
                raise ValueError('Alternative token boundaries differ')
            cut = a['cut']
            legacy_difference = (a['raw'][:,:cut].float()-b['raw'][:,:cut].float()).abs().max().item()
            raw,common = causal_alternatives(model.base.model,
                [p['item']['input_ids'] for p in alternatives],device)
            for item,value in zip(alternatives,raw):
                item['raw'] = value
            if not torch.equal(a['raw'][:,:common],b['raw'][:,:common]):
                raise RuntimeError('Forked causal prefix differs')
            audit.append(dict(id=row['id'],lengths=[len(p['item']['input_ids']) for p in alternatives],
                common_tokens=common,legacy_prefix_max_difference=legacy_difference,
                cached_prefix_bitwise_equal=True))
            prepared.append(alternatives)
        for i,row in enumerate(pair['rows']):
            alternatives = prepared[i]
            donor = prepared[1-i][0]
            a = alternatives[0]
            result.append(dict(id=row['id'],pair=pair['id'],role=role,alternatives=alternatives,
                bank=prefix_state(model,donor).user.clone(),
                normal_bank=prefix_state(model,a).user.clone(),
                source_sha256=ids_digest(a['item']['memory_input_ids']),
                donor_sha256=ids_digest(donor['item']['memory_input_ids'])))
    return result,audit


def transfer_pair(pair, *, names=None, assistant_history=None, object_name=None):
    """Cross interventions without changing user facts except an explicit rename."""
    result = copy.deepcopy(pair)
    names = names or {}
    if len(set(names.values())) != len(names):
        raise ValueError('Names must remain distinct')
    pattern = re.compile(r'\b('+'|'.join(map(re.escape, names))+r')\b') if names else None
    def rename(text):
        return pattern.sub(lambda m: names[m.group()], text) if pattern else text
    for row in result['rows']:
        for message in row['history']:
            message['content'] = rename(message['content'])
        row['question'] = rename(row['question'])
        row['owner'], row['keeper'] = rename(row['owner']), rename(row['keeper'])
        if assistant_history is not None:
            messages = [m for m in row['history'] if m['role']=='assistant']
            if len(messages) != len(assistant_history):
                raise ValueError('Assistant history length differs')
            for message, content in zip(messages, assistant_history):
                message['content'] = content
        if object_name is not None:
            if row['question'].count('the object') != 1:
                raise ValueError('Expected one generic object reference')
            row['question'] = row['question'].replace('the object', 'the '+object_name)
    return result


@contextmanager
def base_readout(block, enabled):
    """Disable the expert correction only, preserving prefix state computation."""
    original = block.forward
    existed = 'forward' in block.__dict__
    def forward(*args, **kwargs):
        return original(*args, **{**kwargs, 'enabled': False})
    if enabled:
        block.forward = forward
    try:
        yield
    finally:
        if enabled:
            if existed:
                block.forward = original
            else:
                del block.forward


@torch.no_grad()
def main():
    from transformers import AutoTokenizer
    from safetensors.torch import load_file
    p = argparse.ArgumentParser()
    for name in ('model', 'cases', 'states', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing results')
    data = json.loads(args.cases.read_text())
    state_specs = json.loads(args.states.read_text())
    if len(data['pairs']) != 16 or len(data['revisits']) != 24 or len(state_specs) != 4:
        raise ValueError('Sealed inventory differs')
    args.output.mkdir(parents=True)
    def write(name, value):
        (args.output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    model = ExpertQwen.from_pretrained(args.model, dtype='auto', attn_implementation='sdpa').to(args.device).eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    initial_body = tensor_hashes(model.base)
    head_dtype = model.base.lm_head.weight.dtype
    head32 = model.base.lm_head.weight.detach().float().clone()
    states = {}
    for spec in state_specs:
        path = args.states.parent/spec['path']
        if digest(path) != spec['sha256']:
            raise ValueError('Checkpoint hash mismatch')
        states[spec['arm']] = load_file(str(path), device=args.device)
    model.experts.load_state_dict(states['start24'], strict=True)
    print('Preparing frozen paired features', flush=True)
    prepared, prefix_audit = prepare_transfer_pairs(model, tokenizer, data['pairs'], args.device)
    write('prefix-audit.json',prefix_audit)
    # Assistant substitutions, including their token lengths, preserve the bank.
    bank_groups = {}
    for pair in data['pairs']:
        for row in pair['rows']:
            item = next(x for x in prepared if x['id']==row['id'])
            key = (row['source_id'], row['names'], row['query'])
            prior = bank_groups.setdefault(key, item)
            if prior['source_sha256'] != item['source_sha256'] or not torch.equal(prior['normal_bank'], item['normal_bank']):
                raise RuntimeError('Assistant history changed the isolated user bank')
    write('run.json', dict(stage=31, optimizer_steps=0, cases_sha256=digest(args.cases),
        states=state_specs, body_sha256=digest(args.model/'model.safetensors'),
        native_dtype=str(head_dtype), paired_positions=len(prepared),
        bank_invariance_groups=len(bank_groups), primary_language='en', primary_promoted=False))
    replies = (args.output/'replies.jsonl').open('x')
    count = 0
    replay_checks = []
    def emit(row, arm, group):
        nonlocal count
        prompt = encode_expert_prompt(tokenizer, row, 3072)
        answer = generate(model, tokenizer, row, seed=row['seed'],
            mode='base' if arm=='stage7' else 'experts', max_new_tokens=192)
        record = {**row, 'arm':arm, 'group':group,
            'prefix_sha256':ids_digest(prompt['input_ids']),
            'source_sha256':ids_digest(prompt['memory_input_ids']), **answer}
        replies.write(json.dumps(record, ensure_ascii=False)+'\n'); replies.flush()
        count += 1
        print(json.dumps(dict(reply=count, id=row['id'], arm=arm, eos=answer['eos_reached'])), flush=True)
        if row.get('intervention')=='replay':
            replay_checks.append(dict(id=row['id'], same_tokens=answer['token_ids']==row['previous_token_ids']))
    for spec in state_specs:
        arm = spec['arm']
        model.experts.load_state_dict(states[arm], strict=True)
        initial_experts = tensor_hashes(model.experts)
        with base_readout(model.experts, arm=='stage7'):
            values = evaluate(model, prepared, head32, detailed=True)
        write('values-'+arm+'.json', values)
        for pair in data['pairs']:
            for row in pair['rows']:
                emit(row, arm, 'factorial')
        if arm=='lambda12':
            for row in data['revisits']:
                emit(row, arm, 'revisit')
        if tensor_hashes(model.experts) != initial_experts:
            raise RuntimeError('Frozen expert weights changed')
    replies.close()
    if tensor_hashes(model.base) != initial_body or model.base.lm_head.weight.dtype != head_dtype:
        raise RuntimeError('Frozen body changed')
    if count != 152:
        raise RuntimeError('Reply count differs')
    write('completed.json', dict(optimizer_steps=0, replies=count,
        body_tensors_preserved=len(initial_body), expert_tensors_preserved=True,
        bank_invariance_groups=len(bank_groups), replay_checks=replay_checks,
        primary_promoted=False, production_precision_changed=False))


if __name__=='__main__':
    main()
