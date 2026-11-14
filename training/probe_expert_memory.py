"""Trace assistant context into later user-bank writes on matched prefixes.

Cases are private JSON: language, user, correct, swapped, question. Both
assistant alternatives must preserve all user-token IDs, masks and positions.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from embedded_experts import ExpertQwen
from expert_data import encode_expert_prompt


@torch.no_grad()
def probe(model, tokenizer, scene):
    arms = []
    device = next(model.experts.parameters()).device
    for label in ('correct', 'swapped'):
        row = dict(question=scene['question'], history=[
            dict(role='user', content=scene['user']),
            dict(role='assistant', content=scene[label])])
        prompt = encode_expert_prompt(tokenizer, row, 1024)
        ids = torch.tensor([prompt['input_ids']], device=device)
        mask = torch.tensor([prompt['user_mask']], device=device)
        hidden = model.base.model(input_ids=ids, use_cache=False,
                                  return_dict=True).last_hidden_state
        preceding = torch.cat((mask.new_zeros(1), mask[0, :-1]))
        starts = torch.where(mask[0] & ~preceding)[0]
        if len(starts) != 2:
            raise ValueError('Expected two distinct user-content segments')
        cut = int(starts[-1])
        _, before, _ = model.experts(hidden[:, :cut], mask[:, :cut])
        _, after, _ = model.experts(hidden[:, cut:], mask[:, cut:], before)
        arms.append(dict(ids=ids, mask=mask, hidden=hidden, cut=cut,
                         before=before.user, after=after.user))
    a, b = arms
    if (a['ids'].shape != b['ids'].shape or a['cut'] != b['cut']
            or not torch.equal(a['mask'], b['mask'])
            or not torch.equal(a['ids'][a['mask']], b['ids'][b['mask']])):
        raise ValueError('Counterfactual must preserve user-token IDs, masks and positions')
    if not torch.equal(a['before'], b['before']):
        raise ValueError('User banks already differ before the second user message')

    def rms(x):
        return x.float().square().mean().sqrt().item()

    return dict(language=scene['language'],
        **{k: scene[k] for k in ('user', 'correct', 'swapped', 'question')},
        tokens=a['ids'].numel(), last_user_start=a['cut'],
        identical_user_tokens=True, identical_token_positions=True,
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
        reports.append(probe(model, tokenizer, case))
        print(json.dumps(reports[-1], ensure_ascii=False), flush=True)
    result = dict(checkpoint=str(args.model),
        expert_sha256=hashlib.sha256((args.model/'experts.safetensors').read_bytes()).hexdigest(),
        cases_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        purpose='Assistant-context influence with identical user-token IDs, masks and positions; no model mutation',
        cases=reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        output.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
