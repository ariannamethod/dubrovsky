"""Recover the authored Q/A structure without rewriting the character."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import re


def parse(text):
    pattern = re.compile(r'^Q: (.*?)\nA: (.*?)(?=\n\s*\nQ: |\Z)', re.M | re.S)
    matches = list(pattern.finditer(text))
    if pattern.sub('', text).strip() or not matches:
        raise ValueError('Unparsed corpus content; refusing to silently discard text')
    pairs = [(m[1].strip(), m[2].strip()) for m in matches]
    if any(not q or not a for q, a in pairs):
        raise ValueError('Empty question or answer')
    return pairs


def normalize(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))


def split_pairs(pairs, seed=20261031, fraction=0.1):
    # Connected components prevent identical questions OR answers crossing splits.
    unique = sorted(set(pairs))
    parent = list(range(len(unique)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, pair in enumerate(unique):
        for kind, text in enumerate(pair):
            key = (kind, normalize(text))
            if key in seen:
                parent[root(i)] = root(seen[key])
            else:
                seen[key] = i
    groups = collections.defaultdict(list)
    for i, pair in enumerate(unique):
        groups[root(i)].append(pair)
    groups = list(groups.values())
    random.Random(seed).shuffle(groups)
    target = max(1, round(len(unique) * fraction))
    val, train = [], []
    for group in groups:
        (val if len(val) < target else train).extend(group)
    if not train:
        raise ValueError('No training split remains')
    return train, val


def write_jsonl(path, pairs):
    rows = []
    for q, a in pairs:
        ident = hashlib.sha256((q + '\0' + a).encode()).hexdigest()
        rows.append(json.dumps({'id': ident, 'question': q, 'answer': a}, ensure_ascii=False))
    path.write_text('\n'.join(rows) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    raw = args.source.read_bytes()
    pairs = parse(raw.decode('utf-8-sig'))
    train, val = split_pairs(pairs)
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / 'train.jsonl', train)
    write_jsonl(args.output / 'validation.jsonl', val)
    manifest = dict(source_name=args.source.name, sha256=hashlib.sha256(raw).hexdigest(),
                    source_bytes=len(raw), source_pairs=len(pairs), unique_pairs=len(set(pairs)),
                    train_pairs=len(train), validation_pairs=len(val), seed=20261031,
                    split='normalized question/answer connected components; exact pair deduplication',
                    text_edits='strip surrounding whitespace only',
                    note='Near-duplicate passages may remain; validation is not proof of novelty.')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
