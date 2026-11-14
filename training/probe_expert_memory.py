"""Trace assistant context into later user-bank writes on matched prefixes.

Cases are private JSON: language, user, correct, swapped, question. Both
assistant alternatives preserve user-content IDs. By default they also preserve
masks and positions; --allow-assistant-length-change relaxes only that constraint.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from embedded_experts import ExpertQwen
from expert_data import encode_expert_prompt, isolated_user_stream, memory_inputs


@torch.no_grad()
def probe(model, tokenizer, scene, allow_length_change=False):
    arms = []
    device = next(model.experts.parameters()).device
    for label in ('correct', 'swapped'):
        row = dict(question=scene['question'], history=[
            dict(role='user', content=scene['user']),
            dict(role='assistant', content=scene[label])])
        prompt = encode_expert_prompt(tokenizer, row, 1024)
        ids = torch.tensor([prompt['input_ids']], device=device)
        mask = torch.tensor([prompt['user_mask']], device=device)
        preceding = torch.cat((mask.new_zeros(1), mask[0, :-1]))
        starts = torch.where(mask[0] & ~preceding)[0]
        if len(starts) != 2:
            raise ValueError('Expected two distinct user-content segments')
        cut = int(starts[-1])
        source_before = isolated_user_stream(tokenizer, row['history'][:1], 1024)
        _, _, before, _, _ = model.hidden(ids[:, :cut], mask[:, :cut],
            **memory_inputs(model, source_before, device))
        _, hidden, after, _, _ = model.hidden(ids, mask,
            **memory_inputs(model, prompt, device))
        arms.append(dict(ids=ids, mask=mask, hidden=hidden, cut=cut,
                         before=before.user, after=after.user,
                         source_ids=prompt['memory_input_ids'], source_mask=prompt['memory_user_mask']))
    a, b = arms
    same_positions=(a['ids'].shape==b['ids'].shape and a['cut']==b['cut']
                    and torch.equal(a['mask'],b['mask']))
    if ((not same_positions and not allow_length_change)
            or not torch.equal(a['ids'][a['mask']], b['ids'][b['mask']])):
        raise ValueError('Counterfactual must preserve user-token IDs, masks and positions')
    if a['source_ids'] != b['source_ids'] or a['source_mask'] != b['source_mask']:
        raise ValueError('Isolated source streams differ')
    if not torch.equal(a['before'], b['before']):
        raise ValueError('User banks already differ before the second user message')
    if model.expert_config.format_version==2 and not torch.equal(a['after'],b['after']):
        raise ValueError('Isolated user-bank invariance failed')

    def rms(x):
        return x.float().square().mean().sqrt().item()

    return dict(language=scene['language'],
        **{k: scene[k] for k in ('user', 'correct', 'swapped', 'question')},
        tokens=a['ids'].numel(), last_user_start=a['cut'],
        identical_user_tokens=True, identical_token_positions=same_positions,
        isolated_stream_equal=True,format_version=model.expert_config.format_version,
        bank_before_last_user_equal=True,
        bank_after_last_user_equal=torch.equal(a['after'], b['after']),
        bank_after_rms_difference=rms(a['after'] - b['after']),
        bank_after_max_difference=(a['after'] - b['after']).abs().max().item(),
        last_user_hidden_rms_difference=rms(
            a['hidden'][:, a['cut']:][a['mask'][:, a['cut']:]]
            - b['hidden'][:, b['cut']:][b['mask'][:, b['cut']:]]),
        base_dtype=str(next(model.base.parameters()).dtype),
        expert_dtype=str(next(model.experts.parameters()).dtype))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--allow-assistant-length-change',action='store_true')
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing diagnostic output')
    cases = json.loads(args.cases.read_text())
    fields = ('language', 'user', 'correct', 'swapped', 'question')
    if not cases or any(not isinstance(c, dict) or
            any(not isinstance(c.get(k), str) or not c[k].strip() for k in fields)
            for c in cases):
        raise ValueError('Expected nonempty matched-prefix cases')
    torch.set_num_threads(4)
    model = ExpertQwen.from_pretrained(args.model, dtype='auto',
        attn_implementation='sdpa').to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    reports = []
    for case in cases:
        reports.append(probe(model, tokenizer, case, args.allow_assistant_length_change))
        print(json.dumps(reports[-1], ensure_ascii=False), flush=True)
    result = dict(checkpoint=str(args.model),
        expert_sha256=hashlib.sha256((args.model/'experts.safetensors').read_bytes()).hexdigest(),
        cases_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        purpose='Assistant-context influence with identical user-content IDs; no model mutation',
        allow_assistant_length_change=args.allow_assistant_length_change,
        cases=reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        output.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
