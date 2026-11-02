"""Apply explicit source-ID exclusions without rewriting authored language or splits."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def curate(splits, translations, exclusions):
    originals = {}
    for split, rows in splits.items():
        for row in rows:
            ident = row['id']
            expected = hashlib.sha256((row['question']+'\0'+row['answer']).encode()).hexdigest()
            if ident != expected or ident in originals:
                raise ValueError('Invalid or duplicate source ID: '+ident)
            originals[ident] = split
    missing = set(exclusions) - set(originals)
    if missing:
        raise ValueError('Exclusion IDs absent from source: '+str(sorted(missing)))
    kept = {split: [row for row in rows if row['id'] not in exclusions]
            for split, rows in splits.items()}
    selected, dropped, duplicate_variants = {}, set(), 0
    for row in translations:
        source_id = row['source_id']
        if source_id not in originals:
            raise ValueError('Unknown translated source: '+source_id)
        if row['split'] != originals[source_id]:
            raise ValueError('Translation changed source split: '+source_id)
        key = (source_id, row['language'])
        if source_id in exclusions:
            dropped.add(key)
        elif key in selected:
            duplicate_variants += 1
        else:
            # Input order defines preference: main run before overlapping pilot.
            selected[key] = dict(row, review_status='pending literary review')
    return kept, list(selected.values()), {
        'excluded_sources': {k: {'split': originals[k], 'reason': v}
                             for k, v in exclusions.items()},
        'excluded_translation_keys': sorted(dropped),
        'overlapping_translation_variants_skipped': duplicate_variants,
        'translation_selection': 'first supplied variant per source ID and language',
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--exclusions', type=Path,
                   default=Path(__file__).with_name('corpus_exclusions.json'))
    p.add_argument('--translations', type=Path, nargs='*', default=[],
                   help='Directories in preference order; raw API responses are ignored')
    p.add_argument('--reference', type=Path, action='append', default=[],
                   help='Unmodified literary reference texts for the next teacher context')
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory; preserve earlier datasets')
    paths = [args.data/(s+'.jsonl') for s in ('train', 'validation')]
    splits = {path.stem: read_jsonl(path) for path in paths}
    translations, translation_paths = [], []
    for directory in args.translations:
        for path in sorted(directory.glob('*.json')):
            if 'response' in path.stem:
                continue
            obj = json.loads(path.read_text())
            translations.extend(obj['records'])
            translation_paths.append(path)
    exclusions = json.loads(args.exclusions.read_text())
    kept, translated, audit = curate(splits, translations, exclusions)
    context = ['# Authored English corpus after explicit source-ID exclusions\n']
    for rows in kept.values():
        context.extend('Q: '+r['question']+'\nA: '+r['answer']+'\n' for r in rows)
    for path in args.reference:
        context.append('\n# Literary source: '+path.name+'\n'+path.read_text())
    args.output.mkdir(parents=True)
    for split, rows in kept.items():
        (args.output/(split+'.jsonl')).write_text(
            ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    (args.output/'translations-pending-review.jsonl').write_text(
        ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in translated))
    (args.output/'context.txt').write_text('\n'.join(context))
    manifest = dict(audit, input_files=[{'path': str(p), 'sha256': sha256(p)}
                      for p in paths+[args.exclusions]+translation_paths+args.reference],
                    source_counts={s:len(r) for s,r in kept.items()},
                    translation_counts=dict(Counter(r['language'] for r in translated)),
                    source_text_edits='none; whole pairs excluded by explicit source ID',
                    split_policy='preserve original split; translations inherit source split',
                    teacher_context='includes validation; use fresh prompts for generative evaluation',
                    training_status='prepared only; translations require literary review')
    (args.output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('source_counts','translation_counts',
                      'excluded_translation_keys','overlapping_translation_variants_skipped')}))


if __name__ == '__main__':
    main()
