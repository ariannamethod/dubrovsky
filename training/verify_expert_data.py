"""Audit native token masks and split boundaries before allocating a training GPU."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from expert_data import encode_expert_row, validate_expert_splits
from sft import read_rows


def verify(tokenizer, data, max_length):
    raw = [read_rows(data/name) for name in ('train.jsonl', 'validation.jsonl')]
    validate_expert_splits(*raw)
    reports = {}
    for name, rows in zip(('train', 'validation'), raw):
        counts, supervised, writes = Counter(), Counter(), Counter()
        max_tokens = 0
        for row in rows:
            item = encode_expert_row(tokenizer, row, max_length)
            main_users = [x for x,m in zip(item['input_ids'],item['user_mask']) if m]
            isolated_users = [x for x,m in zip(item['memory_input_ids'],item['memory_user_mask']) if m]
            if main_users != isolated_users:
                raise ValueError('User-only source changed user-content tokens: '+row['id'])
            if any(write and label != -100 for write, label in zip(item['user_mask'], item['labels'])):
                raise ValueError('Assistant target can write the user bank: '+row['id'])
            language = item['language']
            counts[language] += 1
            supervised[language] += sum(t != -100 for t in item['labels'])
            writes[language] += sum(item['user_mask'])
            max_tokens = max(max_tokens, len(item['input_ids']))
        with (data/(name+'.jsonl')).open('rb') as stream:
            sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        reports[name] = dict(records=dict(counts), supervised_tokens=dict(supervised),
            user_write_positions=dict(writes), max_sequence_tokens=max_tokens, sha256=sha,
            target_writes_user_bank=0, native_template_ids_equal=True, isolated_user_content_ids_equal=True)
    return reports


def main():
    from transformers import AutoTokenizer
    p = argparse.ArgumentParser()
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-length', type=int, default=3072)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing mask audit')
    result = verify(AutoTokenizer.from_pretrained(args.tokenizer), args.data, args.max_length)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
