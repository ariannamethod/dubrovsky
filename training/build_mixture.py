"""Build a reviewed multilingual continuation set with English replay and provenance."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from prepare_corpus import normalize


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_splits(train, validation):
    if len({r['id'] for r in train+validation}) != len(train)+len(validation):
        raise ValueError('Duplicate record IDs')
    for field in ('question','answer'):
        overlap = {normalize(r[field]) for r in train} & {normalize(r[field]) for r in validation}
        if overlap:
            raise ValueError('Train/development overlap in '+field)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--curated', type=Path, required=True)
    p.add_argument('--new-scenes', type=Path, required=True)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--origin-pairs', type=Path, required=True)
    p.add_argument('--extra-training', type=Path, nargs='*', default=[],
                   help='Reviewed additional JSONL rows; all must inherit a train split')
    p.add_argument('--preference-replay', type=Path,
                   help='Replay reviewed chosen answers while preserving their provenance')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh mixture directory')
    review = json.loads(args.review.read_text())
    english = sorted(jsonl(args.curated/'train.jsonl'), key=lambda r:r['id'])
    train = [dict(r,language='en',provenance='authored English replay')
             for r in english[:review['english_replay_pairs']]]
    validation = [dict(r,language='en',provenance='original English held-out split')
                  for r in jsonl(args.curated/'validation.jsonl')]
    translations = {r['id']:r for r in jsonl(args.curated/'translations-pending-review.jsonl')}
    for ident in review['translation_ids']:
        row = dict(translations[ident], review_status='reviewed for this continuation run')
        if row['split'] != 'train':
            raise ValueError('Do not move a held-out translation to training')
        train.append(row)
    generated = {}
    for language in ('en','ru','he'):
        generated.update({r['id']:r for r in json.loads(
            (args.new_scenes/(language+'.json')).read_text())['records']})
    for ident in review['generated_ids']:
        row = dict(generated[ident], provenance='new GPT-authored scene; reviewed by Astra')
        correction = review.get('corrections',{}).get(ident)
        if correction:
            row['review_edit'] = correction
            for field in ('question','answer'):
                if field in correction:
                    row[field] = correction[field]
        held_out = int(ident.rsplit('-',1)[1]) in review['generated_holdout_indices']
        row['split'] = 'validation' if held_out else 'train'
        (validation if held_out else train).append(row)
    train.extend(jsonl(args.origin_pairs))
    for path in args.extra_training:
        rows = jsonl(path)
        if any(r.get('split') != 'train' for r in rows):
            raise ValueError('Additional translations must inherit a train source split')
        train.extend(rows)
    if args.preference_replay:
        for row in jsonl(args.preference_replay):
            train.append(dict(id='chosen-'+row['id'], source_id=row['id'],
                language=row['language'], split='train', question=row['question'],
                answer=row['chosen'], provenance='reviewed preferred answer replay',
                chosen_provenance=row['chosen_provenance']))
    validate_splits(train, validation)
    args.output.mkdir(parents=True)
    for name,rows in [('train',train),('validation',validation)]:
        (args.output/(name+'.jsonl')).write_text(
            ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    inputs = [args.review,args.origin_pairs,args.curated/'manifest.json']
    inputs += [args.curated/n for n in ('train.jsonl','validation.jsonl','translations-pending-review.jsonl')]
    inputs += [args.new_scenes/(lang+'.json') for lang in ('en','ru','he')]
    inputs += args.extra_training
    if args.preference_replay:
        inputs.append(args.preference_replay)
    manifest = dict(stage='multilingual continuation after English SFT',
        input_files=[{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                     for path in inputs],
        train_by_language=dict(Counter(r['language'] for r in train)),
        validation_by_language=dict(Counter(r['language'] for r in validation)),
        english_replay='first reviewed count after sorting original training IDs; no replacement',
        russian_origins='unchanged historical answers with newly framed prompts; provenance retained',
        evaluation='generated development examples share teacher context; use fresh conversational probes',
        supervised_loss='assistant answers and end marker only; reasoning disabled')
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k.endswith('_by_language')}))


if __name__ == '__main__':
    main()
