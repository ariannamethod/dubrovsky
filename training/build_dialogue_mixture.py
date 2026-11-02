"""Combine reviewed long dialogues with deterministic language replay and held-out data."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from build_mixture import validate_splits
from sft import read_rows


def sample_replay(rows, counts, seed):
    selected=[]
    for language,count in counts.items():
        eligible=sorted((r for r in rows if r.get('language','en')==language
            and not r.get('history') and not r['id'].startswith('dialogue-')),
            key=lambda r:r['id'])
        if len(eligible)<count:
            raise ValueError('Insufficient replay examples for '+language)
        random.Random(seed).shuffle(eligible)
        selected.extend(eligible[:count])
    return selected


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--replay',type=Path,required=True,help='Earlier prepared mixture directory')
    p.add_argument('--dialogues',type=Path,nargs='+',required=True,help='Compiled reviewed dialogue JSONL files')
    p.add_argument('--corrections',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--english-replay',type=int,default=128)
    p.add_argument('--russian-replay',type=int,default=64)
    p.add_argument('--hebrew-replay',type=int,default=64)
    p.add_argument('--seed',type=int,default=20261102)
    args=p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve earlier mixtures')
    counts=dict(en=args.english_replay,ru=args.russian_replay,he=args.hebrew_replay)
    if any(n<0 for n in counts.values()):
        raise ValueError('Replay counts must be nonnegative')
    train=sample_replay(read_rows(args.replay/'train.jsonl'),counts,args.seed)
    inputs=[args.replay/'train.jsonl',args.replay/'validation.jsonl']+args.dialogues
    if args.corrections:
        inputs.append(args.corrections)
    for path in args.dialogues+([args.corrections]if args.corrections else[]):
        rows=read_rows(path)
        if any(r.get('split')!='train'for r in rows):
            raise ValueError('Dialogue additions must belong to training')
        train.extend(rows)
    validation=read_rows(args.replay/'validation.jsonl')
    validate_splits(train,validation)
    args.output.mkdir(parents=True)
    for split,rows in [('train',train),('validation',validation)]:
        (args.output/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in rows))
    manifest=dict(stage='Long dialogue and topic-switch continuation',seed=args.seed,
        train_by_language=dict(Counter(r.get('language','en')for r in train)),
        replay_by_language=counts,history_conditioned_rows=sum(bool(r.get('history'))for r in train),
        input_files=[dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())for path in inputs],
        supervision='Current answer and end marker; complete prior turns remain masked attention context',
        evaluation='Use new partner conversations; all authored demonstrations and rollout corrections are training')
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False))


if __name__=='__main__':
    main()
