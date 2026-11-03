"""Verify literal saved generations before exporting quotes for publication.

The selection manifest names a source file and its SHA-256, model attribution,
quote, and optional case_id/turn/seed selectors. Source datasets remain private;
the resulting receipts contain the selected quote and attribution only.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def verify_quote(root, selection):
    root = Path(root).resolve()
    path = (root / selection['source_file']).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Quote source must be inside the supplied root')
    data = path.read_bytes()
    if sha256(data) != selection['source_sha256']:
        raise ValueError('Quote source hash changed')
    if path.suffix == '.jsonl':
        records = [(n, json.loads(line)) for n, line in enumerate(data.decode().splitlines(), 1)
                   if line.strip()]
    else:
        value = json.loads(data)
        if not isinstance(value, dict):
            raise ValueError('Single JSON quote source must be a generation object')
        records = [(None, value)]
    quote = selection['quote']
    if not isinstance(quote, str) or not quote:
        raise ValueError('Quote must be a nonempty string')
    matches = []
    for line, row in records:
        if any(selection.get(key) is not None and row.get(key) != selection[key]
               for key in ('case_id', 'turn', 'seed')):
            continue
        if row.get('model') != selection['model']:
            continue
        text = row.get('text')
        if not isinstance(text, str) or quote not in text:
            continue
        if not selection.get('excerpt', False) and text != quote:
            continue
        if text.count(quote) != 1:
            raise ValueError('Quote occurs more than once in the selected generation')
        if selection.get('sampling') is not None and selection['sampling'] != row.get('sampling'):
            raise ValueError('Quote sampling attribution differs from the saved generation')
        matches.append((line, row, text))
    if len(matches) != 1:
        raise ValueError('Expected exactly one literal generation matching quote and attribution')
    line, row, text = matches[0]
    start = len(text[:text.index(quote)].encode())
    return {
        'number': selection.get('number'), 'title': selection.get('title'),
        'quote': quote, 'excerpt': bool(selection.get('excerpt', False)),
        'source_file': selection['source_file'], 'source_sha256': sha256(data),
        'source_jsonl_line': line, 'generation_text_sha256': sha256(text.encode()),
        'quote_sha256': sha256(quote.encode()), 'quote_utf8_start': start,
        'quote_utf8_end': start + len(quote.encode()),
        'checkpoint': {key: row.get(key) for key in ('model', 'revision')},
        'generation': {key: row.get(key) for key in
                       ('case_id', 'turn', 'seed', 'turn_seed', 'sampling', 'eos_reached')},
        'verified_literal': True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--selections', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    selections = json.loads(args.selections.read_text())
    receipts = [verify_quote(args.root, selection) for selection in selections]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        json.dump({'format_version': 1, 'quotes': receipts}, out, ensure_ascii=False, indent=2)
        out.write('\n')
    print(json.dumps({'verified_quotes': len(receipts), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
