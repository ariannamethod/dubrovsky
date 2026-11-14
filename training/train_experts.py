"""Train checkpoint-owned experts on RU/EN histories with the whole body frozen."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil

import torch
import torch.nn.functional as F

from checkpoints import CandidateCheckpoints
from embedded_experts import ExpertConfig, ExpertQwen, tensor_hashes
from expert_data import encode_expert_row, validate_expert_splits, memory_inputs
from sft import read_rows


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def row_loss(model, row, kl_weight=0.0):
    device = next(model.experts.parameters()).device
    ids = torch.tensor([row['input_ids']], device=device)
    mask = torch.tensor([row['user_mask']], device=device, dtype=torch.bool)
    labels = torch.tensor(row['labels'], device=device)
    positions = torch.nonzero(labels[1:] != -100).flatten()
    if positions.numel() == 0:
        raise ValueError('No supervised next-token positions')
    corrected, original, _, _, _ = model.hidden(ids, mask, **memory_inputs(model, row, device))
    ce = corrected.new_zeros((), dtype=torch.float32)
    kl = ce.clone()
    for indices in positions.split(32):
        logits = model.base.lm_head(corrected[0, indices]).float()
        ce = ce + F.cross_entropy(logits, labels[indices+1], reduction='sum')
        if kl_weight:
            with torch.no_grad():
                reference = model.base.lm_head(original[0, indices]).float().log_softmax(-1)
            kl = kl + F.kl_div(logits.log_softmax(-1), reference,
                               reduction='sum', log_target=True)
    count = positions.numel()
    return (ce + kl_weight * kl) / count, ce.detach().item(), count


def evaluate(model, rows):
    model.eval()
    totals = {language: [0.0, 0] for language in ('en', 'ru')}
    with torch.no_grad():
        for row in rows:
            _, ce, tokens = row_loss(model, row)
            totals[row['language']][0] += ce
            totals[row['language']][1] += tokens
    losses = {k: value / count for k, (value, count) in totals.items()}
    return dict(loss=sum(losses.values()) / len(losses), languages=losses,
                tokens={k: v[1] for k, v in totals.items()})


def main():
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    import transformers
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--base-receipt', type=Path, required=True,
                   help='JSON with repo, revision, subfolder, weights_sha256')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--eval-every', type=int, default=10)
    p.add_argument('--max-steps', type=int, default=0)
    p.add_argument('--max-length', type=int, default=3072)
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--memory-source', choices=['isolated-users', 'legacy-context'], default='isolated-users')
    p.add_argument('--english-kl', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=20261114)
    args = p.parse_args()
    if (min(args.epochs, args.accumulation, args.eval_every, args.max_length, args.rank) < 1
            or args.max_steps < 0 or not math.isfinite(args.lr) or args.lr <= 0
            or not math.isfinite(args.english_kl) or args.english_kl < 0):
        raise ValueError('Invalid training settings')
    if args.output.exists():
        raise ValueError('Use a new output directory')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; no training was started')
    identity = json.loads(args.base_receipt.read_text())
    if any(not identity.get(k) for k in ('repo', 'revision', 'subfolder', 'weights_sha256')):
        raise ValueError('Incomplete frozen-parent identity')
    if digest(args.model/'model.safetensors') != identity['weights_sha256']:
        raise ValueError('Parent weight file does not match the pinned receipt')
    if (args.model/'embedded_experts.json').exists():
        raise ValueError('This pilot starts from the frozen selected base, not a prior expert run')
    train_raw, val_raw = [read_rows(args.data/name) for name in ('train.jsonl', 'validation.jsonl')]
    validate_expert_splits(train_raw, val_raw)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train, validation = [[encode_expert_row(tokenizer, row, args.max_length) for row in rows]
                         for rows in (train_raw, val_raw)]
    random.seed(args.seed); torch.manual_seed(args.seed)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, dtype='auto',
                 attn_implementation='sdpa').to(args.device).eval()
    model = ExpertQwen(base, ExpertConfig(base.config.text_config.hidden_size, rank=args.rank,
        format_version=2 if args.memory_source=='isolated-users' else 1), identity)
    frozen = tensor_hashes(model.base)
    optimizer = torch.optim.AdamW(model.experts.parameters(), lr=args.lr, weight_decay=0)
    args.output.mkdir(parents=True)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings.update(train_records=len(train), validation_records=len(validation),
                    expert_parameters=sum(x.numel() for x in model.experts.parameters()),
                    torch=torch.__version__, transformers=transformers.__version__,
                    base_identity=identity, expert_count=model.expert_config.experts,
                    data_sha256={name: digest(args.data/name)
                                 for name in ('train.jsonl', 'validation.jsonl')})
    (args.output/'run.json').write_text(json.dumps(settings, indent=2)+'\n')

    def export(destination, step, epoch):
        if tensor_hashes(model.base) != frozen:
            raise RuntimeError('Frozen body changed')
        model.save_pretrained(destination, tokenizer=tokenizer,
                              receipt=dict(step=step, epoch=epoch, settings=settings))
        for name in ('preprocessor_config.json', 'processor_config.json', 'video_preprocessor_config.json'):
            if (args.model/name).exists():
                shutil.copyfile(args.model/name, destination/name)
        # Every retained candidate can resume its optimizer; no conversational bank is saved.
        torch.save(dict(optimizer=optimizer.state_dict(), step=step, epoch=epoch,
                        torch_rng=torch.get_rng_state(), random_rng=random.getstate()),
                   destination/'optimizer.pt')

    candidates = CandidateCheckpoints(args.output, export)
    metrics = (args.output/'metrics.jsonl').open('x')
    def assess(step, epoch):
        result = evaluate(model, validation)
        improved = candidates.consider(result['loss'], step, epoch)
        record = dict(event='validation', step=step, epoch=epoch, improved=improved, **result)
        metrics.write(json.dumps(record)+'\n'); metrics.flush()
        print(json.dumps(record), flush=True)
        return result

    assess(0, 0)
    step = 0
    for epoch in range(1, args.epochs+1):
        order = list(train); random.shuffle(order)
        for offset in range(0, len(order), args.accumulation):
            batch = order[offset:offset+args.accumulation]
            model.train(); optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for row in batch:
                loss, _, _ = row_loss(model, row, args.english_kl if row['language'] == 'en' else 0.0)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                (loss / len(batch)).backward()
                batch_loss += loss.detach().item() / len(batch)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.experts.parameters(), 1.0,
                                                       error_if_nonfinite=True).item()
            optimizer.step(); step += 1
            metrics.write(json.dumps(dict(event='train', step=step, loss=batch_loss,
                                          gradient_norm=grad_norm))+'\n'); metrics.flush()
            if step % args.eval_every == 0:
                assess(step, epoch)
            if args.max_steps and step >= args.max_steps:
                break
        if step % args.eval_every:
            assess(step, epoch)
        candidates.save(step, epoch)
        if args.max_steps and step >= args.max_steps:
            break
    metrics.close()
    if tensor_hashes(model.base) != frozen:
        raise RuntimeError('Frozen body changed')
    (args.output/'completed.json').write_text(json.dumps(dict(steps=step,
        frozen_tensors=len(frozen), frozen_unchanged=True, primary_promoted=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
