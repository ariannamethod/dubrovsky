"""Score frozen histories under the three predeclared expert interventions."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from embedded_experts import ExpertQwen
from expert_data import encode_expert_row
from dpo import read_preferences


@torch.no_grad()
def score(model, tokenizer, row, answer, mode, max_length):
    item = encode_expert_row(tokenizer, dict(row, answer=answer), max_length)
    device = next(model.experts.parameters()).device
    ids = torch.tensor([item['input_ids']], device=device)
    mask = torch.tensor([item['user_mask']], dtype=torch.bool, device=device)
    labels = torch.tensor(item['labels'], device=device)
    positions = torch.nonzero(labels[1:] != -100).flatten()
    hidden, _, _, _, _ = model.hidden(ids, mask, memory_enabled=mode!='memory-off', enabled=mode!='base')
    nll = 0.0
    for indices in positions.split(32):
        logits = model.base.lm_head(hidden[0, indices]).float()
        nll += F.cross_entropy(logits, labels[indices+1], reduction='sum').item()
    return dict(sum_log_probability=-nll, mean_log_probability=-nll/len(positions), tokens=len(positions))


def main():
    from transformers import AutoTokenizer
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--max-length', type=int, default=3072)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous scores')
    pairs = read_preferences(args.pairs)
    model = ExpertQwen.from_pretrained(args.model, attn_implementation='sdpa').to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        for row in pairs:
            for mode in ('base','experts','memory-off'):
                scores = {key: score(model, tokenizer, row, row[key], mode, args.max_length)
                          for key in ('chosen','rejected')}
                output.write(json.dumps(dict(id=row['id'], family=row.get('family'),
                    language=row['language'], mode=mode, **scores), ensure_ascii=False)+'\n')
                output.flush()


if __name__ == '__main__':
    main()
