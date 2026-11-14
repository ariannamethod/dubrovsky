"""Literal multi-turn expert / memory-off / base comparisons on local checkpoints."""
import argparse
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

from decoding import GeneratedPresencePenalty
from embedded_experts import ExpertQwen
from expert_data import encode_expert_prompt
from sft import chat_eos


@torch.no_grad()
def generate(model, tokenizer, row, *, seed, mode, max_new_tokens=256,
             max_length=3072, temperature=0.8, top_p=1.0, top_k=20, presence_penalty=0.8):
    if mode not in ('experts', 'memory-off', 'base'):
        raise ValueError('Unknown expert ablation')
    if max_new_tokens < 1 or temperature <= 0 or not 0 < top_p <= 1 or top_k < 1:
        raise ValueError('Invalid generation settings')
    prompt = encode_expert_prompt(tokenizer, row, max_length)
    if len(prompt['input_ids']) + max_new_tokens > max_length:
        raise ValueError('Prompt plus generation budget exceeds context limit')
    device = next(model.experts.parameters()).device
    rng = torch.Generator(device=device).manual_seed(seed)
    ids = torch.tensor([prompt['input_ids']], device=device)
    mask = torch.tensor([prompt['user_mask']], device=device, dtype=torch.bool)
    options = dict(memory_enabled=mode != 'memory-off', enabled=mode != 'base')
    model.eval()
    result = model(ids, mask, use_cache=True, logits_to_keep=1, **options)
    eos = chat_eos(model.base, tokenizer)
    penalties = [GeneratedPresencePenalty(ids.shape[1], presence_penalty),
                 TemperatureLogitsWarper(temperature), TopKLogitsWarper(top_k),
                 TopPLogitsWarper(top_p)]
    sequence = ids
    generated = []
    for _ in range(max_new_tokens):
        scores = result.logits[:, -1].float()
        for processor in penalties:
            scores = processor(sequence, scores)
        next_id = torch.multinomial(scores.softmax(-1), num_samples=1, generator=rng)
        token = next_id.item(); generated.append(token)
        sequence = torch.cat((sequence, next_id), dim=1)
        if token in eos:
            break
        # Each accepted generated token updates working memory exactly once.
        result = model(next_id, torch.zeros_like(next_id, dtype=torch.bool),
            state=result.state, past_key_values=result.past_key_values,
            use_cache=True, logits_to_keep=1, **options)
    return dict(text=tokenizer.decode(generated, skip_special_tokens=True),
                raw_text=tokenizer.decode(generated, skip_special_tokens=False),
                token_ids=generated, eos_reached=generated[-1] in eos,
                prompt_tokens=len(prompt['input_ids']),
                sampling=dict(temperature=temperature, top_p=top_p, top_k=top_k,
                              presence_penalty=presence_penalty, repetition_penalty=1.0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--cases', type=Path, required=True, help='JSON array: id, language, turns (user strings)')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--modes', nargs='+', default=['base', 'experts', 'memory-off'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 137])
    p.add_argument('--device', default='cuda')
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--max-length', type=int, default=3072)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing generations')
    cases = json.loads(args.cases.read_text())
    if not cases or any(not c.get('turns') or c.get('language') not in ('en', 'ru')
                        or any(not isinstance(t, str) or not t.strip() for t in c['turns']) for c in cases):
        raise ValueError('Expected nonempty RU/EN conversation cases')
    model = ExpertQwen.from_pretrained(args.model, dtype='auto',
                                      attn_implementation='sdpa').to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        for case in cases:
            for seed in args.seeds:
                for mode in args.modes:
                    history = []
                    for turn, question in enumerate(case['turns']):
                        start = time.monotonic()
                        row = dict(question=question, history=history)
                        answer = generate(model, tokenizer, row, seed=seed+turn, mode=mode,
                                max_new_tokens=args.max_new_tokens, max_length=args.max_length)
                        record = dict(case=case['id'], language=case['language'], seed=seed,
                            mode=mode, turn=turn, question=question, history=list(history),
                            model=str(args.model), seconds=time.monotonic()-start, **answer)
                        output.write(json.dumps(record, ensure_ascii=False)+'\n'); output.flush()
                        history.extend([dict(role='user', content=question),
                                        dict(role='assistant', content=answer['text'])])


if __name__ == '__main__':
    main()
