"""Paired inference for sparse BF16 weight edits, with restoration and evidence.

The private specification pins checkpoints, two tensor hashes, support, direction,
variants and diagnostic continuations. No gradients, training or checkpoint export.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from compare_sampling import validate_messages
from decoding import sampling_kwargs
from sft import chat_eos
from text_diagnostics import diagnostics, answer_history_diagnostics
from thinking_probe import split_generation
from weight_geometry import project_target, norm_matched_control


def digest(value):
    return hashlib.sha256(value).hexdigest()


def tensor_bits(tensor):
    import torch
    if tensor.dtype != torch.bfloat16:
        raise ValueError('Expected BF16 parameters')
    return tensor.detach().cpu().contiguous().view(torch.uint16).numpy().copy()


def model_digest(model, excluded):
    import torch
    h = hashlib.sha256()
    for name, tensor in model.named_parameters():
        if name in excluded:
            continue
        h.update(name.encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(str(tensor.dtype).encode())
        for chunk in tensor.detach().reshape(-1).split(1024 * 1024):
            h.update(chunk.contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def make_variant(base, layers, variant):
    active = variant.get('layers', [layer['index'] for layer in layers])
    if (not active or len(set(active)) != len(active) or
            not set(active).issubset({layer['index'] for layer in layers})):
        raise ValueError('Expected unique declared layer indices')
    result = {}
    for layer in layers:
        name = layer['name']
        if variant['kind'] == 'baseline' or layer['index'] not in active:
            result[name] = base[name].copy()
        elif variant['kind'] == 'projection':
            result[name] = project_target(base[name], layer['direction'],
                                         layer['columns'], variant['alpha'])
        elif variant['kind'] == 'orthogonal':
            result[name], _ = norm_matched_control(base[name], layer['direction'],
                layer['columns'], variant['alpha'], variant['seed'] + layer['index'])
        else:
            raise ValueError('Unknown variant kind')
        untouched = np.ones(base[name].shape[1], dtype=bool)
        untouched[layer['columns']] = False
        if not np.array_equal(base[name][:, untouched], result[name][:, untouched]):
            raise ValueError('Edit escaped declared columns')
    return result


def dialogue_rows(case, seed, generate):
    """Each variant/seed starts empty and extends only its own actual answers."""
    turns = case.get('user_turns')
    if not isinstance(turns, list) or not turns or any(not isinstance(t, str) or not t.strip() for t in turns):
        raise ValueError('Expected nonempty dialogue turns')
    history = []
    for turn, question in enumerate(turns, 1):
        messages = validate_messages(history + [dict(role='user', content=question)])
        row = generate(dict(case, messages=messages), seed + turn - 1)
        # Keep the exact prompt in each receipt before appending the answer.
        yield dict(row, messages=messages, seed=seed, turn_seed=seed+turn-1, turn=turn)
        history = messages + [dict(role='assistant', content=row['text'])]


def main():
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    p = argparse.ArgumentParser()
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    spec = json.loads(args.spec.read_text())
    target = next(x for x in spec['targets'] if x['id'] == args.target)
    if args.output.exists():
        raise ValueError('Preserve earlier outputs')
    weights = Path(target['path']) / 'model.safetensors'
    if hashlib.file_digest(weights.open('rb'), 'sha256').hexdigest() != target['weight_sha256']:
        raise ValueError('Checkpoint file hash differs')
    tokenizer = AutoTokenizer.from_pretrained(target['path'])
    model = Qwen3_5ForConditionalGeneration.from_pretrained(target['path'],
        dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    model.requires_grad_(False)
    eos = chat_eos(model, tokenizer)
    layers = spec['layers']
    params = dict(model.named_parameters())
    modules = dict(model.named_modules())
    base = {x['name']: tensor_bits(params[x['name']]) for x in layers}
    for name, bits in base.items():
        if digest(bits.tobytes()) != target['tensor_sha256'][name]:
            raise ValueError('Loaded tensor hash differs')
    outside = model_digest(model, base)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    provenance = dict(target=target, spec_sha256=digest(args.spec.read_bytes()),
        torch=torch.__version__, transformers=transformers.__version__,
        device=torch.cuda.get_device_name(), eos_token_ids=eos,
        unchanged_parameter_sha256=outside, gradient_enabled=False)
    started = time.monotonic()

    def install(bits):
        for name, array in bits.items():
            params[name].copy_(torch.from_numpy(array.copy()).view(torch.bfloat16).cuda())
            if digest(tensor_bits(params[name]).tobytes()) != digest(array.tobytes()):
                raise ValueError('Installed tensor bytes differ')

    def prompt_ids(messages, thinking=False):
        return tokenizer.apply_chat_template(validate_messages(messages), tokenize=True,
            add_generation_prompt=True, enable_thinking=thinking, return_dict=False)

    def check_context(length):
        if length > model.config.text_config.max_position_embeddings:
            raise ValueError('Context overflow; no truncation')

    def score(prompt, answer):
        continuation = tokenizer.encode(answer, add_special_tokens=False)
        if not continuation:
            raise ValueError('Empty contrast continuation')
        check_context(len(prompt) + len(continuation))
        start = len(prompt) - 1
        captures = {}
        handles = []
        for layer in layers:
            name = layer['name']; module = modules[name.removesuffix('.weight')]
            u = torch.tensor(layer['direction'], dtype=torch.float32, device='cuda')
            columns = torch.tensor(layer['columns'], device='cuda')
            coefficients = u @ module.weight[:, columns].float()
            def hook(module, inputs, output, name=name, u=u, columns=columns, coefficients=coefficients):
                activations = inputs[0][0, start:].float()
                writes = output[0, start:].float()
                selected_projection = activations[:, columns] @ coefficients
                full_projection = writes @ u
                captures[name] = dict(
                    selected_write_projection=selected_projection.cpu().tolist(),
                    full_write_projection=full_projection.cpu().tolist(),
                    full_write_norm=writes.norm(dim=-1).cpu().tolist())
            handles.append(module.register_forward_hook(hook))
        try:
            ids = torch.tensor([prompt + continuation[:-1]], device='cuda')
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                logits_to_keep=len(continuation), use_cache=False).logits[0].float()
            labels = torch.tensor(continuation, device='cuda')
            logp = logits.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)
            if not torch.isfinite(logp).all():
                raise ValueError('Non-finite likelihood')
            if len(logp) != len(continuation) or any(
                len(x['full_write_norm']) != len(continuation) for x in captures.values()):
                raise ValueError('Prediction/token alignment differs')
            return dict(token_ids=continuation, token_logp=logp.cpu().tolist(),
                mean_logp=logp.mean().item(), prefix16_mean_logp=logp[:16].mean().item(),
                activation_projections=captures)
        finally:
            for handle in handles:
                handle.remove()

    def generate_case(case, seed):
        thinking = case.get('enable_thinking', False)
        messages = validate_messages(case['messages'])
        prompt = prompt_ids(messages, thinking)
        limit = case.get('max_new_tokens', spec['max_new_tokens'])
        check_context(len(prompt) + limit)
        torch.manual_seed(seed)
        ids = torch.tensor([prompt], device='cuda')
        generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
            **sampling_kwargs(spec['sampling'], len(prompt)),
            max_new_tokens=limit, eos_token_id=eos,
            pad_token_id=tokenizer.pad_token_id, use_cache=True)[0, len(prompt):].tolist()
        text = tokenizer.decode(generated, skip_special_tokens=True)
        split = split_generation(tokenizer, generated, thinking, eos)
        return dict(case_id=case['id'], language=case['language'], messages=messages, text=text,
            token_ids=generated, seed=seed, enable_thinking=thinking,
            prompt_tokens=len(prompt), max_new_tokens=limit, sampling=spec['sampling'],
            literal_diagnostics=diagnostics(text, messages[-1]['content']),
            history_diagnostics=answer_history_diagnostics(text, messages), **split)

    with args.output.open('x') as out, torch.inference_mode():
        def emit(row):
            out.write(json.dumps(row, ensure_ascii=False) + '\n'); out.flush()
        emit(dict(kind='provenance', **provenance))
        try:
            for variant in spec['variants']:
                bits = make_variant(base, layers, variant)
                hashes = {n: digest(b.tobytes()) for n, b in bits.items()}
                if hashes != target['variant_sha256'][variant['id']]:
                    raise ValueError('Variant differs from CPU-prepared candidate')
                install(bits)
                emit(dict(kind='installed', variant=variant, tensor_sha256=hashes))
                for case in spec.get('contrasts', []):
                    prompt = prompt_ids(case['messages'])
                    branches = {key: score(prompt, text) for key, text in case['branches'].items()}
                    emit(dict(kind='contrast', variant=variant['id'], case_id=case['id'],
                        language=case['language'], messages=case['messages'],
                        branch_texts=case['branches'], branches=branches,
                        prompt_tokens=len(prompt), enable_thinking=False))
                for case in spec.get('generations', []):
                    for seed in case.get('seeds', spec['seeds']):
                        emit(dict(kind='generation', variant=variant['id'], **generate_case(case, seed)))
                for case in spec.get('dialogues', []):
                    for seed in case.get('seeds', spec['seeds']):
                        for row in dialogue_rows(case, seed, generate_case):
                            emit(dict(kind='dialogue', variant=variant['id'], **row))
                print(json.dumps(dict(target=args.target, variant=variant['id'], complete=True,
                                      elapsed=time.monotonic()-started)), flush=True)
        finally:
            install(base)
            restored = all(np.array_equal(tensor_bits(params[n]), b) for n, b in base.items())
            outside_after = model_digest(model, base)
            emit(dict(kind='restoration', edited_tensors_restored=restored,
                other_parameters_unchanged=outside_after == outside,
                unchanged_parameter_sha256=outside_after))
            if not restored or outside_after != outside:
                raise ValueError('Model restoration/integrity failed')
        emit(dict(kind='complete', elapsed=time.monotonic()-started))


if __name__ == '__main__':
    main()
