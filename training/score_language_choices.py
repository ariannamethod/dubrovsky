"""Score explicit held-out word choices; distinguish answers from mere mentions.

This measures grammar-choice task performance, including instruction following.
It does not grade free literary prose. References and generations stay private.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import unicodedata


def normalize(text):
    text = unicodedata.normalize('NFKC', text).strip().casefold()
    while text and unicodedata.category(text[0]).startswith('P'):
        text = text[1:].lstrip()
    while text and unicodedata.category(text[-1]).startswith('P'):
        text = text[:-1].rstrip()
    return ' '.join(text.split())


def classify(text, answer, rejected):
    answer, rejected, normalized = map(normalize, (answer, rejected, text))
    if not answer or not rejected or answer == rejected:
        raise ValueError('Expected two distinct nonempty choices')
    if normalized == answer:
        return 'correct_exact'
    if normalized == rejected:
        return 'rejected_exact'
    words = re.findall(r'\w+', normalized)
    def contains(choice):
        target = re.findall(r'\w+', choice)
        return any(words[i:i + len(target)] == target
                   for i in range(len(words) - len(target) + 1))
    correct, wrong = contains(answer), contains(rejected)
    if correct and wrong:
        return 'both_mentioned'
    if correct:
        return 'correct_mentioned_only'
    if wrong:
        return 'rejected_mentioned_only'
    return 'neither_mentioned'


def score(rows, references):
    refs = {r['id']: r for r in references}
    if len(refs) != len(references):
        raise ValueError('Duplicate reference ID')
    groups, seen, details = {}, set(), []
    for row in rows:
        key = (row['model'], row['case_id'], row['configuration_id'], row['seed'], row['turn'])
        if key in seen:
            raise ValueError('Duplicate generation')
        seen.add(key)
        ref = refs[row['case_id']]
        if row['language'] != ref['language']:
            raise ValueError('Language mismatch')
        verdict = classify(row['text'], ref['answer'], ref['rejected'])
        group = (row['model'], row['configuration_id'], row['language'])
        counts = groups.setdefault(group, Counter())
        counts.update(total=1, **{verdict: 1})
        counts['eos'] += bool(row['eos_reached'])
        details.append(dict(model=row['model'], case_id=row['case_id'],
                            configuration_id=row['configuration_id'], seed=row['seed'],
                            verdict=verdict, text=row['text']))
    return dict(scope='Explicit word-choice task; grammar and instruction following combined',
                groups=[dict(model=k[0], configuration_id=k[1], language=k[2], **v)
                        for k, v in sorted(groups.items())], details=details)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--references', type=Path, required=True)
    p.add_argument('--generations', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rows = [json.loads(line) for f in a.generations for line in f.read_text().splitlines() if line.strip()]
    result = score(rows, json.loads(a.references.read_text()))
    with a.output.open('x') as out:
        out.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
