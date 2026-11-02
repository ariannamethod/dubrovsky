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
    messages = [{'role': 'system', 'content': SYSTEM}]
    history = row.get('history', [])
    if not isinstance(history, list) or len(history) % 2:
        raise ValueError('History must contain complete user/assistant turns')
    for index, message in enumerate(history):
        role = 'user' if index % 2 == 0 else 'assistant'
        if (not isinstance(message, dict) or message.get('role') != role
                or not isinstance(message.get('content'), str)):
            raise ValueError('History must alternate text user and assistant messages')
        messages.append(dict(role=role, content=message['content'],
                             **({'reasoning_content': ''} if role == 'assistant' else {})))
    messages.append({'role': 'user', 'content': row['question']})
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


def supervision_tail(labels):
    first = next((i for i, token in enumerate(labels) if token != -100), None)
    if first is None:
        raise ValueError('No supervised tokens')
    keep = min(len(labels), len(labels) - first + 1)
    return labels[-keep:], keep


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    seen = set()
    for row in rows:
        if row['id'] in seen:
            raise ValueError(f"Duplicate training record ID in {path}: {row['id']}")
        seen.add(row['id'])
    if not rows:
        raise ValueError('Empty split: '+str(path))
    return rows


def main():
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoProcessor, Qwen3_5ForConditionalGeneration
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='huihui-ai/Huihui-Qwen3.5-0.8B-abliterated')
    p.add_argument('--revision', help='Required for remote models; local candidates use their files')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--accumulation', type=int, default=16)
    p.add_argument('--max-length', type=int, default=1024)
    p.add_argument('--max-steps', type=int, default=0)
    p.add_argument('--save-each-epoch', action='store_true')
    p.add_argument('--answer-logits-only', action='store_true',
                   help='Keep full attention history but project only supervised answer positions plus their predecessor')
    args = p.parse_args()
    if args.epochs < 1 or args.accumulation < 1:
        raise ValueError('Epochs and accumulation must be positive')
    if not Path(args.model).is_dir() and not args.revision:
        raise ValueError('Pin a revision when loading a remote model')
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
        result = []
        for row in read_rows(args.data / name):
            result.append(dict(encode(tokenizer, row, args.max_length),
                               language=row.get('language', 'en')))
        if not result:
            raise ValueError('Empty split: '+name)
        return result
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
        values = {k: torch.tensor([row[k]], device='cuda') for k in ('input_ids','labels')}
        if args.answer_logits_only:
            labels, keep = supervision_tail(row['labels'])
            values['labels'] = torch.tensor([labels], device='cuda')
            values['logits_to_keep'] = keep
        return values
    def log(row):
        row['elapsed_seconds'] = round(time.monotonic() - started, 2)
        with (args.output / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    @torch.no_grad()
    def evaluate(step):
        model.eval()
        loss_sum, n, by_language = 0.0, 0, {}
        for row in validation:
            count = sum(x != -100 for x in row['labels'][1:])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(**tensors(row), use_cache=False).loss
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite validation loss')
            loss_sum += loss.item() * count
            n += count
            values = by_language.setdefault(row['language'], [0.0, 0])
            values[0] += loss.item() * count
            values[1] += count
        log({'step': step, 'validation_loss': loss_sum / n,
             'validation_by_language': {k: {'loss': v[0]/v[1], 'answer_tokens': v[1]}
                                        for k,v in by_language.items()}})
        model.train()
    def save_candidate(destination, step, epoch):
        after_hashes = language_interface_hashes(model)
        if frozen_hashes != after_hashes:
            raise RuntimeError('Frozen language embeddings/head changed during training')
        # Export a CPU BF16 state without changing FP32 training parameters or Adam state.
        state = {k: (v.detach().to(device='cpu', dtype=torch.bfloat16)
                     if v.is_floating_point() else v.detach().cpu())
                 for k,v in model.state_dict().items()}
        previous_cache = model.config.use_cache
        model.config.use_cache = True
        model.generation_config.eos_token_id = chat_eos(model, tokenizer)
        try:
            model.save_pretrained(destination, state_dict=state, safe_serialization=True)
            # save_pretrained infers dtype from the FP32 training model, not supplied state.
            config_path = destination/'config.json'
            config = json.loads(config_path.read_text())
            config['dtype'] = 'bfloat16'
            if 'torch_dtype' in config:
                config['torch_dtype'] = 'bfloat16'
            config_path.write_text(json.dumps(config, indent=2)+'\n')
            processor.save_pretrained(destination)
            tokenizer.save_pretrained(destination)
            (destination/'training_checkpoint.json').write_text(json.dumps({
                'step':step, 'epoch':epoch, 'export_dtype':'bfloat16',
                'purpose':'candidate evaluation; optimizer state not included',
                'frozen_hashes_before_export':after_hashes}, indent=2)+'\n')
        finally:
            model.config.use_cache = previous_cache
        log({'step':step, 'epoch':epoch, 'checkpoint':str(destination)})
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
                    loss = model(**tensors(row), use_cache=False).loss
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
        if args.save_each_epoch:
            last_checkpoint = args.output/f'epoch-{epoch+1:03d}'
            save_candidate(last_checkpoint, step, epoch+1)
        if step >= total_steps:
            break
    # Candidate stays on the training disk until generation-based selection.
    after_hashes = language_interface_hashes(model)
    if frozen_hashes != after_hashes:
        raise RuntimeError('Frozen language embeddings/head changed during training')
    (args.output / 'frozen_integrity.json').write_text(json.dumps(
        {'before': frozen_hashes, 'after': after_hashes, 'identical': True}, indent=2))
    candidate = args.output / 'candidate'
    if args.save_each_epoch:
        candidate.symlink_to(last_checkpoint.name, target_is_directory=True)
    else:
        save_candidate(candidate, step, epoch+1)
    print(json.dumps({'status': 'completed', 'candidate': str(candidate)}), flush=True)


if __name__ == '__main__':
    main()
