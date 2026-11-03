"""Score native answer continuations without asking the character for a choice.

Each contrast has the same question/history and two reviewed continuations.
Report both summed and mean answer-token log probability; exclude turn-end tokens.
Different token counts and ranking disagreement remain visible in the output.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from sft import conversation_messages


def validate_pairs(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected nonempty grammar pairs')
    seen = set()
    for row in rows:
        for key in ['id', 'language', 'category', 'register', 'question', 'answer', 'rejected']:
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError('Missing nonempty field: ' + key)
        if row['id'] in seen or row['answer'] == row['rejected']:
            raise ValueError('Duplicate ID or identical alternatives')
        seen.add(row['id'])
        conversation_messages(row)


def preference(correct, rejected):
    for item in [correct, rejected]:
        if item['tokens'] <= 0 or not math.isfinite(item['mean_log_probability']):
            raise ValueError('Expected finite likelihood and positive token count')
    summed = correct['mean_log_probability'] * correct['tokens'] - rejected['mean_log_probability'] * rejected['tokens']
    normalized = correct['mean_log_probability'] - rejected['mean_log_probability']
    return dict(sum_margin=summed, mean_margin=normalized,
                preferred_by_sum=summed > 0, preferred_by_mean=normalized > 0,
                equal_token_count=correct['tokens'] == rejected['tokens'],
                ranking_disagrees=(summed > 0) != (normalized > 0))


def main():
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--revision')
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous pair results')
    if not Path(args.model).is_dir() and not args.revision:
        raise ValueError('Pin remote model revision')
    rows = json.loads(args.pairs.read_text())
    validate_pairs(rows)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16,
        attn_implementation='sdpa').cuda().eval().requires_grad_(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out, torch.inference_mode():
        for row in rows:
            messages = conversation_messages(row)
            prefix = tok.apply_chat_template(messages, tokenize=True, return_dict=False,
                                              add_generation_prompt=True, enable_thinking=False)
            values = {}
            for name, answer in [('correct', row['answer']), ('rejected', row['rejected'])]:
                tail = tok.encode(answer, add_special_tokens=False)
                if not tail or tok.decode(prefix + tail) != tok.decode(prefix) + answer:
                    raise ValueError('Answer continuation did not round-trip')
                ids = torch.tensor([prefix + tail], device='cuda')
                labels = torch.tensor([[-100] * len(prefix) + tail], device='cuda')
                loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite pair likelihood')
                values[name] = dict(tokens=len(tail), mean_log_probability=-loss.item(), token_ids=tail)
            result = dict(model=args.model, revision=args.revision, case_id=row['id'],
                language=row['language'], category=row['category'], register=row['register'],
                **values, **preference(values['correct'], values['rejected']),
                pairs_sha256=hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
                prompt_token_ids=prefix, enable_thinking=False, turn_end_scored=False,
                torch=torch.__version__, transformers=transformers.__version__)
            out.write(json.dumps(result) + '\n'); out.flush()


if __name__ == '__main__':
    main()
