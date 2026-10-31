"""DPO on reviewed same-prompt preferences, with the SFT body as frozen reference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

from sft import encode, chat_eos, freeze_language_interface, language_interface_hashes


def read_preferences(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('No reviewed preferences')
    for row in rows:
        for key in ('question', 'chosen', 'rejected', 'reason'):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f'Preference lacks {key}')
        if row['chosen'].strip() == row['rejected'].strip():
            raise ValueError('Chosen and rejected answers are identical')
    return rows


def main():
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoTokenizer, AutoProcessor, Qwen3_5ForConditionalGeneration
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--preferences', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--lr', type=float, default=5e-7)
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--accumulation', type=int, default=8)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh DPO output directory')
    if not (args.model / 'config.json').exists():
        raise ValueError('Expected a local SFT candidate directory')
    rows = read_preferences(args.preferences)
    args.output.mkdir(parents=True)
    torch.manual_seed(20261031); random.seed(20261031)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation='sdpa').cuda()
    model.model.visual.requires_grad_(False)
    tied = freeze_language_interface(model)
    frozen_hashes = language_interface_hashes(model)
    model.config.use_cache = False
    # Deterministic reference/policy likelihoods; sampling remains an evaluation choice.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    encoded = []
    for i, row in enumerate(rows):
        encoded.append([encode(tokenizer, dict(id=str(i), question=row['question'], answer=row[key]), 1024)
                        for key in ('chosen', 'rejected')])
    def logp(row):
        inputs = {k: torch.tensor([v], device='cuda') for k, v in row.items()}
        count = sum(t != -100 for t in row['labels'][1:])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(**inputs).loss
        return -loss * count
    model.eval()
    with torch.no_grad():
        reference = [(logp(c).item(), logp(r).item()) for c, r in encoded]
    if not all(math.isfinite(x) for pair in reference for x in pair):
        raise FloatingPointError('Nonfinite reference log probabilities')
    (args.output / 'reference.json').write_text(json.dumps(reference))
    metadata = {**vars(args), 'preferences_sha256': hashlib.sha256(args.preferences.read_bytes()).hexdigest(),
                'pairs': len(rows), 'torch': torch.__version__, 'transformers': transformers.__version__,
                'reference': 'same SFT weights before updates; sum of supervised answer-token log probabilities',
                'seed': 20261031, 'tied_embeddings': tied, 'language_embeddings': 'frozen',
                'output_head': 'frozen', 'frozen_hashes_before': frozen_hashes}
    (args.output / 'run.json').write_text(json.dumps(metadata, default=str, indent=2))
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, fused=True)
    step, started = 0, time.monotonic()
    for epoch in range(args.epochs):
        indices = list(range(len(rows))); random.shuffle(indices)
        for start in range(0, len(indices), args.accumulation):
            group = indices[start:start + args.accumulation]
            optimizer.zero_grad(set_to_none=True)
            total_loss, margins = 0.0, []
            for i in group:
                chosen, rejected = encoded[i]
                c, r = logp(chosen), logp(rejected)
                margin = args.beta * ((c - r) - (reference[i][0] - reference[i][1]))
                loss = -F.logsigmoid(margin)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite DPO loss')
                (loss / len(group)).backward()
                total_loss += loss.item() / len(group); margins.append(margin.item())
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step(); step += 1
            record = dict(step=step, epoch=epoch+1, loss=total_loss,
                          mean_reward_margin=sum(margins)/len(margins), grad_norm=norm.item(),
                          elapsed_seconds=time.monotonic()-started)
            with (args.output / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            print(json.dumps(record), flush=True)
    after_hashes = language_interface_hashes(model)
    if frozen_hashes != after_hashes:
        raise RuntimeError('Frozen embeddings/head changed during DPO')
    (args.output / 'frozen_integrity.json').write_text(json.dumps(
        {'before': frozen_hashes, 'after': after_hashes, 'identical': True}, indent=2))
    model.config.use_cache = True
    model.generation_config.eos_token_id = chat_eos(model, tokenizer)
    model.gradient_checkpointing_disable()
    candidate = args.output / 'candidate'
    model.to(dtype=torch.bfloat16).save_pretrained(candidate, safe_serialization=True)
    processor.save_pretrained(candidate); tokenizer.save_pretrained(candidate)


if __name__ == '__main__':
    main()
