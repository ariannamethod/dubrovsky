"""Frozen-prefix expert diagnostics. Private cases and checkpoints stay external.

Interventions replace only the user bank at an explicit token boundary, keeping
the already accumulated working state and the complete Qwen prefix fixed.
Generation uses the production sampler. No optimizer is created.
"""
import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import torch
from torch import nn

from embedded_experts import ExpertQwen, tensor_hashes
from expert_data import (encode_expert_prompt, encode_expert_row,
                         isolated_user_stream, memory_inputs)
from sample_experts import generate
from sft import conversation_messages


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def ids_digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def question_bounds(mask):
    """Last contiguous native user-content segment; prompts are validated first."""
    positions = torch.where(mask[0])[0].tolist()
    if not positions:
        raise ValueError('No user content')
    end = positions[-1] + 1
    start = end - 1
    while start > 0 and bool(mask[0, start-1]):
        start -= 1
    return start, end


class BankIntervention(nn.Module):
    def __init__(self, block, cut, *, bank=None, hold=False):
        super().__init__()
        if cut < 1 or (bank is None) == (not hold):
            raise ValueError('Select exactly one bank replacement or write hold')
        self.block, self.cut, self.bank, self.hold = block, cut, bank, hold
        self.capture = {}

    def forward(self, hidden, user_mask, state=None, **kwargs):
        if state is not None:
            return self.block(hidden, user_mask, state, **kwargs)
        if self.cut >= hidden.shape[1]:
            raise ValueError('Intervention must precede a nonempty suffix')
        if not kwargs.get('memory_enabled', True):
            raise ValueError('Local interventions require the normal memory path')
        source = kwargs.pop('memory_hidden')
        a, before, av = self.block(hidden[:, :self.cut], user_mask[:, :self.cut],
                                  memory_hidden=source[:, :self.cut], **kwargs)
        incoming = before.user.clone() if self.hold else self.bank
        if incoming.shape != before.user.shape or incoming.dtype != before.user.dtype:
            raise ValueError('Replacement bank shape or dtype differs')
        if incoming.device != before.user.device:
            raise ValueError('Replacement bank device differs')
        changed = replace(before, user=incoming.clone())
        suffix_mask = user_mask[:, self.cut:]
        if self.hold:
            suffix_mask = torch.zeros_like(suffix_mask)
        b, after, bv = self.block(hidden[:, self.cut:], suffix_mask, changed,
                                  memory_hidden=source[:, self.cut:], **kwargs)
        self.capture = dict(cut=self.cut, hold=self.hold,
            normal_before=before.user.detach().cpu().tolist(),
            incoming=incoming.detach().cpu().tolist(),
            after=after.user.detach().cpu().tolist(),
            working_before=before.working.detach().cpu().tolist(),
            prefix_output_preserved=True)
        return torch.cat((a, b), 1), after, torch.cat((av, bv), 1)


@contextmanager
def intervene(model, cut=None, bank=None, hold=False):
    original = model.experts
    wrapper = None if cut is None else BankIntervention(original, cut, bank=bank, hold=hold)
    if wrapper is not None:
        model.experts = wrapper
    try:
        yield wrapper
    finally:
        model.experts = original


@torch.no_grad()
def source_bank(model, tokenizer, messages, device):
    stream = isolated_user_stream(tokenizer, messages, 3072)
    ids = torch.tensor([stream['memory_input_ids']], device=device)
    mask = torch.tensor([stream['memory_user_mask']], dtype=torch.bool, device=device)
    raw = model.base.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
    _, state, _ = model.experts(raw, mask, memory_hidden=raw)
    return state.user, ids_digest(stream['memory_input_ids'])


@torch.no_grad()
def prepare_hidden(model, tokenizer, row, answer, device, *, value_offset=None):
    if value_offset is not None and not 0 <= value_offset < len(answer):
        raise ValueError('Value offset must lie inside the answer')
    item = encode_expert_row(tokenizer, dict(row, answer=answer), 3072)
    ids = torch.tensor([item['input_ids']], device=device)
    mask = torch.tensor([item['user_mask']], dtype=torch.bool, device=device)
    source_ids = torch.tensor([item['memory_input_ids']], device=device)
    source_mask = torch.tensor([item['memory_user_mask']], dtype=torch.bool, device=device)
    source = model.base.model(input_ids=source_ids, use_cache=False, return_dict=True).last_hidden_state
    raw = model.base.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
    aligned = torch.zeros_like(raw)
    aligned[mask] = source[source_mask]
    rendered = tokenizer.apply_chat_template(conversation_messages(row) +
        [dict(role='assistant', content=answer, reasoning_content='')], tokenize=False,
        add_generation_prompt=False, enable_thinking=False)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    if encoded['input_ids'] != item['input_ids']:
        raise ValueError('Teacher token offsets disagree with native serialization')
    prefix = tokenizer.apply_chat_template(conversation_messages(row), tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    if not rendered.startswith(prefix + answer):
        raise ValueError('Unexpected assistant serialization')
    # answer is a role cue followed by one value; score the value including its
    # leading whitespace, never the cue, template suffix, or EOS.
    begin = len(prefix) + (answer.index(':') + 1 if value_offset is None else value_offset)
    end = len(prefix) + len(answer)
    positions = [i for i, (a, b) in enumerate(encoded['offset_mapping'])
                 if b > a and begin <= a and b <= end]
    if not positions or any(i < 1 for i in positions):
        raise ValueError('Value has no independently aligned token span')
    return dict(raw=raw, mask=mask, source=aligned, item=item, positions=positions)


@torch.no_grad()
def score_value(model, prepared, mode, *, intervention=None):
    raw, mask, source = [prepared[k] for k in ('raw', 'mask', 'source')]
    options = dict(memory_enabled=mode != 'memory-off', enabled=mode != 'base')
    block = model.experts
    state = None
    cursor = 0
    details = []
    intervention = intervention or {}
    with intervene(model, **intervention) as wrapper:
        for target_pos in prepared['positions']:
            stop = target_pos  # the token before the value predicts this value
            corrected, state, votes = model.experts(raw[:, cursor:stop], mask[:, cursor:stop],
                state, memory_hidden=source[:, cursor:stop], **options)
            h = corrected[:, -1]
            logits = model.base.lm_head(h).float()[0]
            token = prepared['item']['input_ids'][target_pos]
            proposal = torch.einsum('be,ber,erd->bd', votes[:, -1], state.working, block.output)
            rms = torch.sqrt(proposal.square().mean(-1, keepdim=True) + 1e-12)
            delta = proposal * (block.config.max_delta_rms / rms).clamp(max=1.0)
            if mode == 'base':
                delta = torch.zeros_like(delta)
            rounded = h.float() - raw[:, stop-1].float()
            w = model.base.lm_head.weight[token].float()
            details.append(dict(token=token, position=target_pos,
                teacher_prefix_sha256=ids_digest(prepared['item']['input_ids'][:target_pos]),
                log_probability=logits.log_softmax(-1)[token].item(),
                logit=logits[token].item(), votes=votes[0, -1].cpu().tolist(),
                bank=state.user.cpu().tolist(), working=state.working.cpu().tolist(),
                proposal_rms=rms.item(), clipped=bool(rms.item() > block.config.max_delta_rms),
                working_saturated_fraction=((1-state.working.square()) < .01).float().mean().item(),
                float_value_logit_delta=(delta[0]*w).sum().item(),
                rounded_value_logit_delta=(rounded[0]*w).sum().item(),
                rounding_error_rms=(delta-rounded).square().mean().sqrt().item()))
            cursor = stop
        capture = wrapper.capture if wrapper else None
    total = sum(x['log_probability'] for x in details)
    return dict(sum_log_probability=total, mean_log_probability=total/len(details),
                tokens=details, intervention=capture)


def main():
    from transformers import AutoTokenizer
    from safetensors.torch import load_file
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--early-experts', type=Path, required=True)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output directory')
    cases = json.loads(args.cases.read_text())
    core, controls = cases['core'], cases['controls']
    if len(core) != 12 or len(controls) != 8 or sum(c['probe'] for c in core) != 6:
        raise ValueError('Unexpected sealed case inventory')
    args.output.mkdir(parents=True)
    device = args.device
    model = ExpertQwen.from_pretrained(args.model, dtype='auto', attn_implementation='sdpa').to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    initial_body = tensor_hashes(model.base)
    states = {57: load_file(str(args.early_experts), device=device),
              228: load_file(str(args.model/'experts.safetensors'), device=device)}
    hashes = {57:digest(args.early_experts), 228:digest(args.model/'experts.safetensors')}
    expected = cases['expert_sha256']
    if any(hashes[s] != expected[str(s)] for s in states):
        raise ValueError('Expert checkpoint hash mismatch')
    frozen_experts = {s:{k:v.clone() for k,v in state.items()} for s,state in states.items()}
    replies = (args.output/'replies.jsonl').open('x')
    scores = (args.output/'values.jsonl').open('x')
    replay = []
    prepared_cache = {}
    def load(step):
        model.experts.load_state_dict(states[step], strict=True)
    def emit_reply(case, arm, step, mode, row=None, intervention=None, donor_source=None):
        row = case if row is None else row
        prompt = encode_expert_prompt(tokenizer, row, 3072)
        start = time.monotonic()
        with intervene(model, **(intervention or {})) as wrapper:
            answer = generate(model, tokenizer, row, seed=case['seed']+case['turn'],
                              mode=mode, max_new_tokens=192)
            capture = wrapper.capture if wrapper else None
        record = dict(id=case['id'], group=case['group'], case=case['case'], seed=case['seed'],
            turn=case['turn'], language=case['language'], arm=arm, step=step,
            question=row['question'], history=row['history'],
            prefix_sha256=ids_digest(prompt['input_ids']),
            source_sha256=ids_digest(prompt['memory_input_ids']),
            intervention=capture, donor_source_sha256=donor_source,
            seconds=time.monotonic()-start, **answer)
        replies.write(json.dumps(record, ensure_ascii=False)+'\n'); replies.flush()
        if arm == '228/experts':
            match = answer['token_ids'] == case['reference_token_ids']
            replay.append(dict(id=case['id'], exact_tokens=match))
            if not match:
                raise RuntimeError('Archived reply replay differs: '+case['id'])
        print(json.dumps(dict(id=case['id'],arm=arm,eos=answer['eos_reached'])), flush=True)
        return prompt
    def emit_values(case, arm, mode, row=None, intervention=None):
        row = case if row is None else row
        values = {}
        for role in ('owner','keeper'):
            alternatives = {}
            for kind in ('correct','wrong'):
                value = case['values'][role][kind]
                answer = role.title()+': '+value
                key = (case['id'], row['question'], answer)
                if key not in prepared_cache:
                    prepared_cache[key] = prepare_hidden(model, tokenizer, row, answer, device)
                alternatives[kind] = score_value(model, prepared_cache[key], mode,
                                                  intervention=intervention)
            values[role] = dict(alternatives=alternatives,
                mean_margin=alternatives['correct']['mean_log_probability']-alternatives['wrong']['mean_log_probability'],
                sum_margin=alternatives['correct']['sum_log_probability']-alternatives['wrong']['sum_log_probability'])
        scores.write(json.dumps(dict(id=case['id'],arm=arm,values=values),ensure_ascii=False)+'\n');scores.flush()
    for step, mode in [(228,'experts'),(228,'base'),(57,'experts'),(228,'memory-off')]:
        load(step); arm=f'{step}/{mode}'
        for case in core:
            emit_reply(case,arm,step,mode)
            emit_values(case,arm,mode)
        for case in controls:
            if case['language']=='ru' and (step==57 or mode=='memory-off'):
                continue
            emit_reply(case,arm,step,mode)
    load(228)
    for case in core:
        if not case['probe']:
            continue
        prompt = encode_expert_prompt(tokenizer, case, 3072)
        mask = torch.tensor([prompt['user_mask']],dtype=torch.bool,device=device)
        start,end = question_bounds(mask)
        # Verify that isolated donor construction reproduces the actual bank
        # when its source is unchanged, before making counterfactual donors.
        ids = torch.tensor([prompt['input_ids']],device=device)
        before_source=isolated_user_stream(tokenizer,case['history'],3072)
        _,_,before,_,_=model.hidden(ids[:,:start],mask[:,:start],
            **memory_inputs(model,before_source,device))
        _,_,after,_,_=model.hidden(ids,mask,**memory_inputs(model,prompt,device))
        normal_before,_=source_bank(model,tokenizer,case['history'],device)
        normal_after,_=source_bank(model,tokenizer,conversation_messages(case),device)
        if not torch.equal(normal_before,before.user) or not torch.equal(normal_after,after.user):
            raise RuntimeError('Isolated bank replay differs from native prefix')
        for role in ('owner','keeper'):
            donor = case['donors'][role]
            for when,cut in [('before',start),('after',end)]:
                messages = donor['history'] if when=='before' else conversation_messages(donor)
                bank,source_hash = source_bank(model,tokenizer,messages,device)
                intervention=dict(cut=cut,bank=bank)
                arm=f'228/swap-{role}-{when}'
                emit_reply(case,arm,228,'experts',intervention=intervention,donor_source=source_hash)
                emit_values(case,arm,'experts',intervention=intervention)
        intervention=dict(cut=start,hold=True)
        emit_reply(case,'228/hold-question',228,'experts',intervention=intervention)
        emit_values(case,'228/hold-question','experts',intervention=intervention)
        row=dict(case,question=case['paraphrase'])
        emit_reply(case,'228/paraphrase',228,'experts',row=row)
        emit_values(case,'228/paraphrase','experts',row=row)
    replies.close(); scores.close()
    if tensor_hashes(model.base) != initial_body:
        raise RuntimeError('Frozen body changed')
    for s,state in states.items():
        if not all(torch.equal(v,frozen_experts[s][k]) for k,v in state.items()):
            raise RuntimeError('Source expert checkpoint changed')
    if not all(torch.equal(v,states[228][k]) for k,v in model.experts.state_dict().items()):
        raise RuntimeError('Live experts changed')
    report=dict(frozen_body_tensors=len(initial_body),frozen_body_unchanged=True,
        experts_unchanged=True, optimizer_steps=0, primary_promoted=False,
        case_sha256=digest(args.cases),expert_sha256=hashes,replay=replay,
        reply_count=len((args.output/'replies.jsonl').read_text().splitlines()),
        score_count=len((args.output/'values.jsonl').read_text().splitlines()))
    (args.output/'completed.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
