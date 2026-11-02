"""Controlled decoding comparisons, with one model load and literal branch histories."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time

from chat_probe import conversation_diagnostics
from decoding import sampling_kwargs
from sft import SYSTEM, chat_eos


def validate_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError('Expected nonempty messages')
    if any(not isinstance(message, dict) for message in messages):
        raise ValueError('Expected message objects')
    messages = copy.deepcopy(messages)
    if messages[0].get('role') != 'system':
        messages.insert(0, dict(role='system', content=SYSTEM))
    if len(messages) % 2 or len(messages) < 2:
        raise ValueError('History must end with user')
    for i, message in enumerate(messages):
        role = 'system' if i == 0 else ('user' if i % 2 else 'assistant')
        if (message.get('role') != role or not isinstance(message.get('content'), str)
                or (role != 'assistant' and not message['content'].strip())):
            raise ValueError('Expected alternating text messages ending with user')
    return messages


def validate_inputs(cases, configurations, mode):
    if not isinstance(cases, list) or not cases or not isinstance(configurations, list) or not configurations:
        raise ValueError('Nonempty cases and configurations required')
    for group in (cases, configurations):
        ids = [row.get('id') for row in group]
        if any(not isinstance(x, str) or not x for x in ids) or len(ids) != len(set(ids)):
            raise ValueError('Expected unique nonempty IDs')
    for case in cases:
        if not isinstance(case.get('language'), str) or not case['language']:
            raise ValueError('Case needs language')
        if mode == 'fixed':
            validate_messages(case.get('messages'))
        else:
            turns = case.get('user_turns')
            if not isinstance(turns, list) or not turns or any(not isinstance(t, str) or not t.strip() for t in turns):
                raise ValueError('Dialogue needs nonempty user turns')
    for configuration in configurations:
        s = configuration.get('sampling', {})
        if set(s) != {'temperature', 'top_p', 'top_k', 'presence_penalty'}:
            raise ValueError('Specify all four sampling settings')
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in s.values()):
            raise ValueError('Sampling settings must be finite numbers')
        if s['temperature'] <= 0 or not 0 < s['top_p'] <= 1 or not isinstance(s['top_k'], int) or s['top_k'] < 0 or not 0 <= s['presence_penalty'] <= 2:
            raise ValueError('Invalid sampling range')


def main():
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--revision')
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--configurations', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['fixed', 'dialogue'], default='fixed')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 137])
    p.add_argument('--max-new-tokens', type=int, default=192)
    args = p.parse_args()
    if not Path(args.model).is_dir() and not args.revision:
        raise ValueError('Pin a revision when loading a remote model')
    if args.output.exists():
        raise ValueError('Preserve earlier outputs; choose a new output path')
    if args.max_new_tokens <= 0 or len(args.seeds) != len(set(args.seeds)):
        raise ValueError('Use positive token limit and distinct seeds')
    cases = json.loads(args.cases.read_text())
    configurations = json.loads(args.configurations.read_text())
    validate_inputs(cases, configurations, args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, revision=args.revision,
        dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    eos = chat_eos(model, tokenizer)
    provenance = dict(model=args.model, revision=args.revision, mode=args.mode,
        cases_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        configurations_sha256=hashlib.sha256(args.configurations.read_bytes()).hexdigest(),
        transformers=transformers.__version__, torch=torch.__version__,
        device=torch.cuda.get_device_name(), enable_thinking=False, use_cache=True,
        eos_token_ids=eos, max_new_tokens=args.max_new_tokens,
        partner='Astra-authored fixed evaluation prompts; no external partner API')
    # Each configuration and seed receives a separate dialogue history. Fixed mode
    # receives identical histories; dialogue mode branches on its own actual answers.
    with args.output.open('x') as out, torch.inference_mode():
        for case in cases:
            for configuration in configurations:
                for seed in args.seeds:
                    messages = []
                    turns = case['user_turns'] if args.mode == 'dialogue' else [None]
                    for turn, question in enumerate(turns, 1):
                        if args.mode == 'fixed':
                            messages = validate_messages(case['messages'])
                        else:
                            messages = validate_messages(messages + [dict(role='user', content=question)])
                        prompt = tokenizer.apply_chat_template(messages, tokenize=True,
                            add_generation_prompt=True, enable_thinking=False, return_dict=False)
                        if len(prompt) + args.max_new_tokens > model.config.text_config.max_position_embeddings:
                            raise ValueError('History exceeds context; no silent truncation')
                        turn_seed = seed + turn - 1
                        torch.manual_seed(turn_seed)
                        ids = torch.tensor([prompt], device='cuda')
                        started = time.monotonic()
                        generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                            **sampling_kwargs(configuration['sampling'], len(prompt)),
                            max_new_tokens=args.max_new_tokens, eos_token_id=eos,
                            pad_token_id=tokenizer.pad_token_id, use_cache=True)[0, len(prompt):].tolist()
                        text = tokenizer.decode(generated, skip_special_tokens=True)
                        row = dict(**provenance, case_id=case['id'], language=case['language'],
                            configuration_id=configuration['id'], sampling=dict(**configuration['sampling'],
                                min_p=0.0, repetition_penalty=1.0), seed=seed, turn_seed=turn_seed,
                            turn=turn, prompt_tokens=len(prompt), text=text, token_ids=generated,
                            eos_reached=bool(generated and generated[-1] in eos),
                            empty_answer=not text.strip(),
                            elapsed_seconds=time.monotonic()-started,
                            diagnostics=conversation_diagnostics(text, messages),
                            messages=messages + [dict(role='assistant', content=text)])
                        out.write(json.dumps(row, ensure_ascii=False) + '\n'); out.flush()
                        messages = row['messages']
                        print(json.dumps({k: row[k] for k in ('case_id', 'configuration_id', 'seed', 'turn', 'eos_reached')}), flush=True)


if __name__ == '__main__':
    main()
