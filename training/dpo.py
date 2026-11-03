"""DPO on reviewed same-prompt preferences, with the SFT body as frozen reference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

from sft import (encode, freeze_language_interface, language_interface_hashes,
                 checkpoint_steps, save_bf16_candidate)
from checkpoints import CandidateCheckpoints


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


def encode_preference(tokenizer, row, max_length=1024):
    """Both alternatives must see the same complete conversation prefix."""
    return [encode(tokenizer, dict(id=str(row.get('id', 'preference')),
                                   question=row['question'], answer=row[key],
                                   history=row.get('history', [])), max_length)
            for key in ('chosen', 'rejected')]


def validate_holdout(training, validation):
    def context(row):
        return json.dumps([row['question'], row.get('history', [])], sort_keys=True, ensure_ascii=False)
    if {context(row) for row in training} & {context(row) for row in validation}:
        raise ValueError('DPO validation must use held-out conversation prefixes')


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
    p.add_argument('--max-length', type=int, default=1024,
                   help='Maximum complete conversation length; reject rather than truncate')
    p.add_argument('--chosen-nll-weight', type=float, default=0.0,
                   help='Optional preferred-answer NLL anchor (RPO-style; zero is standard DPO)')
    p.add_argument('--validation-preferences', type=Path,
                   help='Reviewed held-out pairs; improvements automatically retain best weights')
    p.add_argument('--eval-every-steps', type=int, default=0)
    p.add_argument('--save-every-steps', type=int, default=5,
                   help='Retain intermediate candidates even without held-out pairs; zero disables periodic saves')
    p.add_argument('--save-steps', type=int, nargs='+', default=[])
    args = p.parse_args()
    if (args.epochs < 1 or args.accumulation < 1 or args.lr <= 0 or args.beta <= 0
            or args.chosen_nll_weight < 0 or args.eval_every_steps < 0 or args.save_every_steps < 0):
        raise ValueError('Invalid DPO optimization or checkpoint settings')
    if args.eval_every_steps and not args.validation_preferences:
        raise ValueError('--eval-every-steps requires --validation-preferences')
    if args.output.exists():
        raise ValueError('Use a fresh DPO output directory')
    if not (args.model / 'config.json').exists():
        raise ValueError('Expected a local SFT candidate directory')
    rows = read_preferences(args.preferences)
    validation_rows = read_preferences(args.validation_preferences) if args.validation_preferences else []
    validate_holdout(rows, validation_rows)
    total_steps = args.epochs * math.ceil(len(rows) / args.accumulation)
    snapshots = checkpoint_steps(args.save_steps, total_steps)
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
    for row in rows:
        encoded.append(encode_preference(tokenizer, row, args.max_length))
    validation = [encode_preference(tokenizer, row, args.max_length) for row in validation_rows]
    def logp(row):
        inputs = {k: torch.tensor([v], device='cuda') for k, v in row.items()}
        count = sum(t != -100 for t in row['labels'][1:])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(**inputs, use_cache=False).loss
        return -loss * count
    model.eval()
    with torch.no_grad():
        reference = [(logp(c).item(), logp(r).item()) for c, r in encoded]
        validation_reference = [(logp(c).item(), logp(r).item()) for c, r in validation]
    if not all(math.isfinite(x) for pair in reference + validation_reference for x in pair):
        raise FloatingPointError('Nonfinite reference log probabilities')
    (args.output / 'reference.json').write_text(json.dumps(reference))
    (args.output / 'validation_reference.json').write_text(json.dumps(validation_reference))
    metadata = {**vars(args), 'preferences_sha256': hashlib.sha256(args.preferences.read_bytes()).hexdigest(),
                'pairs': len(rows), 'history_pairs': sum(bool(row.get('history')) for row in rows),
                'max_encoded_tokens': max(len(x['input_ids']) for pair in encoded for x in pair),
                'torch': torch.__version__, 'transformers': transformers.__version__,
                'reference': 'same SFT weights before updates; sum of supervised answer-token log probabilities',
                'seed': 20261031, 'tied_embeddings': tied, 'language_embeddings': 'frozen',
                'output_head': 'frozen', 'frozen_hashes_before': frozen_hashes}
    metadata.update(total_steps=total_steps, validation_pairs=len(validation),
        validation_preferences_sha256=hashlib.sha256(args.validation_preferences.read_bytes()).hexdigest()
            if args.validation_preferences else None,
        best_selection='held-out mean DPO/RPO objective; generation review still required')
    (args.output / 'run.json').write_text(json.dumps(metadata, default=str, indent=2))
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, fused=True)
    step, started = 0, time.monotonic()
    def log(record):
        record['elapsed_seconds'] = time.monotonic() - started
        with (args.output / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)
    def export(destination, step, epoch):
        save_bf16_candidate(model, tokenizer, processor, destination, step, epoch, frozen_hashes)
        log(dict(step=step, epoch=epoch, checkpoint=str(destination)))
    checkpoints = CandidateCheckpoints(args.output, export)
    @torch.no_grad()
    def evaluate(step, epoch):
        if not validation:
            return
        model.eval()
        losses, nlls = [], []
        for (chosen, rejected), (ref_c, ref_r) in zip(validation, validation_reference):
            c, r = logp(chosen), logp(rejected)
            nll = -c / sum(t != -100 for t in chosen['labels'][1:])
            loss = -F.logsigmoid(args.beta * ((c-r)-(ref_c-ref_r))) + args.chosen_nll_weight*nll
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite DPO validation loss')
            losses.append(loss.item()); nlls.append(nll.item())
        value = sum(losses)/len(losses)
        log(dict(step=step, epoch=epoch, validation_loss=value,
                 validation_chosen_nll=sum(nlls)/len(nlls)))
        checkpoints.consider(value, step, epoch)
        model.train()
    evaluate(0, 0)
    for epoch in range(args.epochs):
        indices = list(range(len(rows))); random.shuffle(indices)
        for start in range(0, len(indices), args.accumulation):
            group = indices[start:start + args.accumulation]
            optimizer.zero_grad(set_to_none=True)
            total_loss, margins, chosen_nlls = 0.0, [], []
            for i in group:
                chosen, rejected = encoded[i]
                c, r = logp(chosen), logp(rejected)
                margin = args.beta * ((c - r) - (reference[i][0] - reference[i][1]))
                chosen_nll = -c / sum(t != -100 for t in chosen['labels'][1:])
                loss = -F.logsigmoid(margin) + args.chosen_nll_weight * chosen_nll
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite DPO loss')
                (loss / len(group)).backward()
                total_loss += loss.item() / len(group); margins.append(margin.item())
                chosen_nlls.append(chosen_nll.item())
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step(); step += 1
            record = dict(step=step, epoch=epoch+1, loss=total_loss,
                          mean_reward_margin=sum(margins)/len(margins),
                          chosen_nll=sum(chosen_nlls)/len(chosen_nlls), grad_norm=norm.item(),
                          elapsed_seconds=time.monotonic()-started)
            log(record)
            if args.eval_every_steps and step % args.eval_every_steps == 0 and start + args.accumulation < len(indices):
                evaluate(step, epoch+1)
            if step in snapshots or (args.save_every_steps and step % args.save_every_steps == 0):
                checkpoints.save(step, epoch+1)
        evaluate(step, epoch+1)
        (args.output / f'epoch-{epoch+1:03d}').symlink_to(
            checkpoints.save(step, epoch+1).name, target_is_directory=True)
    after_hashes = language_interface_hashes(model)
    if frozen_hashes != after_hashes:
        raise RuntimeError('Frozen embeddings/head changed during DPO')
    (args.output / 'frozen_integrity.json').write_text(json.dumps(
        {'before': frozen_hashes, 'after': after_hashes, 'identical': True}, indent=2))
    (args.output / 'candidate').symlink_to(checkpoints.save(step, args.epochs).name, target_is_directory=True)


if __name__ == '__main__':
    main()
