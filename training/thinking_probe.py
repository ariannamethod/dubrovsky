"""Compare closed and open native thinking prefixes without hiding generated tags."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from compare_sampling import validate_inputs, validate_messages
from decoding import sampling_kwargs
from sft import chat_eos


def split_generation(tokenizer, ids, enabled, eos):
    body = ids[:-1] if ids and ids[-1] in eos else ids
    close = tokenizer.convert_tokens_to_ids('</think>')
    if close is None or tokenizer.decode([close], skip_special_tokens=False) != '</think>':
        raise ValueError('Tokenizer must expose the native thinking end token')
    reached = close in body
    position = body.index(close) if reached else None
    thinking = body[:position] if reached and enabled else (body if enabled else [])
    answer = body[position+1:] if enabled and reached else ([] if enabled else body)
    content = tokenizer.decode(thinking, skip_special_tokens=False)
    answer_text = tokenizer.decode(answer, skip_special_tokens=False).strip()
    return dict(raw_text=tokenizer.decode(ids, skip_special_tokens=False),
                thinking_end_generated=reached, thinking_end_position=position,
                thinking_tokens=len(thinking), thinking_text=content,
                empty_thinking_closed=bool(enabled and reached and not content.strip()),
                answer=answer_text, empty_final_answer=not answer_text,
                thinking_marker_in_final_answer=any(tag in answer_text for tag in ('<think>','</think>')),
                eos_reached=bool(ids and ids[-1] in eos))


def main():
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--revision')
    p.add_argument('--cases', required=True, type=Path)
    p.add_argument('--configurations', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--seeds', type=int, nargs='+', default=[42,137])
    p.add_argument('--max-new-tokens', type=int, default=192)
    args = p.parse_args()
    if not Path(args.model).is_dir() and not args.revision:
        raise ValueError('Pin remote models')
    if args.output.exists() or args.max_new_tokens <= 0 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Use fresh output, positive token budget and distinct seeds')
    cases = json.loads(args.cases.read_text())
    configurations = json.loads(args.configurations.read_text())
    validate_inputs(cases, configurations, 'fixed')
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, revision=args.revision,
        dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    eos = chat_eos(model, tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output, torch.inference_mode():
        for case in cases:
            messages = validate_messages(case['messages'])
            for configuration in configurations:
                for seed in args.seeds:
                    for enabled in (False, True):
                        prompt = tokenizer.apply_chat_template(messages, tokenize=True,
                            add_generation_prompt=True, enable_thinking=enabled, return_dict=False)
                        if len(prompt)+args.max_new_tokens > model.config.text_config.max_position_embeddings:
                            raise ValueError('Context overflow')
                        torch.manual_seed(seed)
                        ids = torch.tensor([prompt], device='cuda')
                        started = time.monotonic()
                        generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                            **sampling_kwargs(configuration['sampling'],len(prompt)),
                            max_new_tokens=args.max_new_tokens, eos_token_id=eos,
                            pad_token_id=tokenizer.pad_token_id, use_cache=True)[0,len(prompt):].tolist()
                        result = dict(model=args.model, revision=args.revision,
                            case_id=case['id'], language=case['language'], messages=messages,
                            cases_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                            configurations_sha256=hashlib.sha256(args.configurations.read_bytes()).hexdigest(),
                            configuration_id=configuration['id'], sampling=configuration['sampling'],
                            seed=seed, enable_thinking=enabled, max_new_tokens=args.max_new_tokens,
                            prompt_tokens=len(prompt), token_ids=generated, eos_token_ids=eos,
                            elapsed_seconds=time.monotonic()-started,
                            transformers=transformers.__version__, torch=torch.__version__,
                            **split_generation(tokenizer,generated,enabled,eos))
                        output.write(json.dumps(result,ensure_ascii=False)+'\n');output.flush()
                        print(json.dumps({k:result[k]for k in ('case_id','seed','enable_thinking',
                            'empty_thinking_closed','thinking_tokens','eos_reached')}),flush=True)


if __name__ == '__main__':
    main()
