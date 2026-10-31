"""Full language-body SFT, frozen visual encoder, answer-only loss, empty thinking prefix."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

SYSTEM = 'You are Alexey Dubrovsky.'


def freeze_language_interface(model):
    embeddings = model.get_input_embeddings()
    head = model.get_output_embeddings()
    embeddings.requires_grad_(False)
    head.requires_grad_(False)
    tied = embeddings.weight.data_ptr() == head.weight.data_ptr()
    if model.config.text_config.tie_word_embeddings and not tied:
        raise ValueError('Checkpoint promises tied embeddings but loaded parameters are untied')
    return tied


def language_interface_hashes(model):
    import torch
    hashes = {}
    for name, module in [('input_embeddings', model.get_input_embeddings()),
                         ('output_head', model.get_output_embeddings())]:
        h = hashlib.sha256()
        for chunk in module.weight.detach().split(1024, dim=0):
            h.update(chunk.contiguous().cpu().view(torch.uint8).numpy().tobytes())
        hashes[name] = h.hexdigest()
    return hashes


def chat_eos(model, tokenizer):
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    # The checkpoint config stops at endoftext; the chat template trains im_end.
    # Honor both model termination and the tokenizer's assistant-turn boundary.
    if tokenizer.eos_token_id is not None:
        eos.append(tokenizer.eos_token_id)
    return sorted(set(eos))


def encode(tokenizer, row, max_length):
    messages = [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': row['question']}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=True,
                                          add_generation_prompt=True, enable_thinking=False, return_dict=False)
    complete = tokenizer.apply_chat_template(
        messages + [{'role': 'assistant', 'content': row['answer'], 'reasoning_content': ''}],
        tokenize=True, add_generation_prompt=False, enable_thinking=False, return_dict=False)
    if complete[:len(prefix)] != prefix:
        raise ValueError('Chat template does not preserve the nonthinking prefix')
    if len(complete) > max_length:
        raise ValueError(f"Example {row['id']} needs {len(complete)} tokens; no silent truncation")
    labels = [-100] * len(prefix) + complete[len(prefix):]
    if not any(x != -100 for x in labels):
        raise ValueError('No answer tokens')
    return {'input_ids': complete, 'labels': labels}


def main():
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoProcessor, Qwen3_5ForConditionalGeneration
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='huihui-ai/Huihui-Qwen3.5-0.8B-abliterated')
    p.add_argument('--revision', required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--accumulation', type=int, default=16)
    p.add_argument('--max-length', type=int, default=1024)
    p.add_argument('--max-steps', type=int, default=0)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('This training entry point requires CUDA')
    if args.output.exists():
        raise ValueError('Use a fresh output directory for each run')
    args.output.mkdir(parents=True)
    random.seed(20261031)
    torch.manual_seed(20261031)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    def read(name):
        return [encode(tokenizer, json.loads(line), args.max_length)
                for line in (args.data / name).read_text().splitlines()]
    train, validation = read('train.jsonl'), read('validation.jsonl')
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision, dtype=torch.float32, attn_implementation='sdpa').cuda()
    model.model.visual.requires_grad_(False)
    tied = freeze_language_interface(model)
    frozen_hashes = language_interface_hashes(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01, fused=True)
    total_steps = args.epochs * math.ceil(len(train) / args.accumulation)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = max(1, math.ceil(total_steps * 0.1))
    run = {**vars(args), 'torch': torch.__version__, 'transformers': transformers.__version__,
           'gpu': torch.cuda.get_device_name(), 'trainable_parameters': sum(p.numel() for p in parameters),
           'total_parameters': sum(p.numel() for p in model.parameters()),
           'system': SYSTEM, 'seed': 20261031, 'max_training_tokens': max(len(x['input_ids']) for x in train),
           'total_steps': total_steps, 'parameter_dtype': 'float32', 'autocast': 'bfloat16',
           'visual_encoder': 'frozen', 'language_embeddings': 'frozen', 'output_head': 'frozen',
           'tied_embeddings': tied, 'frozen_hashes_before': frozen_hashes,
           'data_manifest': json.loads((args.data / 'manifest.json').read_text())}
    (args.output / 'run.json').write_text(json.dumps(run, default=str, indent=2))
    print(json.dumps(run, default=str), flush=True)
    def tensors(row):
        return {k: torch.tensor([v], device='cuda') for k, v in row.items()}
    def log(row):
        row['elapsed_seconds'] = round(time.monotonic() - started, 2)
        with (args.output / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    @torch.no_grad()
    def evaluate(step):
        model.eval()
        loss_sum, n = 0.0, 0
        for row in validation:
            count = sum(x != -100 for x in row['labels'][1:])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(**tensors(row)).loss
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite validation loss')
            loss_sum += loss.item() * count
            n += count
        log({'step': step, 'validation_loss': loss_sum / n})
        model.train()
    started = time.monotonic()
    evaluate(0)
    step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        random.shuffle(train)
        for start in range(0, len(train), args.accumulation):
            group = train[start:start + args.accumulation]
            n_tokens = sum(sum(t != -100 for t in x['labels'][1:]) for x in group)
            mean_loss = 0.0
            for row in group:
                count = sum(x != -100 for x in row['labels'][1:])
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(**tensors(row)).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                (loss * count / n_tokens).backward()
                mean_loss += loss.item() * count / n_tokens
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            step += 1
            factor = step / warmup if step <= warmup else 0.1 + 0.9 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup))) / 2
            for group_params in optimizer.param_groups:
                group_params['lr'] = args.lr * factor
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            log({'step': step, 'epoch': epoch + 1, 'train_loss': mean_loss,
                 'grad_norm': norm.item(), 'lr': args.lr * factor,
                 'peak_gpu_gib': torch.cuda.max_memory_allocated() / 2**30})
            if step >= total_steps:
                break
        evaluate(step)
        if step >= total_steps:
            break
    # Candidate stays on the training disk until generation-based selection.
    after_hashes = language_interface_hashes(model)
    if frozen_hashes != after_hashes:
        raise RuntimeError('Frozen language embeddings/head changed during training')
    (args.output / 'frozen_integrity.json').write_text(json.dumps(
        {'before': frozen_hashes, 'after': after_hashes, 'identical': True}, indent=2))
    candidate = args.output / 'candidate'
    model.config.use_cache = True
    model.generation_config.eos_token_id = chat_eos(model, tokenizer)
    model.gradient_checkpointing_disable()
    model.to(dtype=torch.bfloat16).save_pretrained(candidate, safe_serialization=True)
    processor.save_pretrained(candidate)
    tokenizer.save_pretrained(candidate)
    print(json.dumps({'status': 'completed', 'candidate': str(candidate)}), flush=True)


if __name__ == '__main__':
    main()
