"""Turn authored conversations into causal, last-answer-only SFT examples."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def conversation_rows(conversation):
    history, rows = [], []
    for index, turn in enumerate(conversation['turns'], 1):
        if len(turn) != 2 or any(not isinstance(x, str) or not x.strip() for x in turn):
            raise ValueError('Each turn must contain a nonempty question and answer')
        question, answer = turn
        rows.append(dict(id=f"dialogue-{conversation['id']}-{index:02d}",
            language=conversation['language'], split='train', question=question,
            answer=answer, history=list(history), source_turn=index,
            provenance=conversation.get('provenance',
                'new Astra-authored conversational demonstration; not a recovered historical chat')))
        history += [{'role':'user','content':question}, {'role':'assistant','content':answer}]
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve earlier prepared conversations')
    rows, inputs = [], []
    for path in args.input:
        data=path.read_bytes()
        digest=hashlib.sha256(data).hexdigest()
        for conversation in json.loads(data):
            rows.extend(dict(r,source_file=str(path),source_sha256=digest)
                        for r in conversation_rows(conversation))
        inputs.append(dict(path=str(path),sha256=digest))
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty conversations or duplicate turn IDs')
    args.output.mkdir(parents=True)
    (args.output/'train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in rows))
    manifest=dict(inputs=inputs,train_by_language=dict(Counter(r['language']for r in rows)),
        supervision='Only current assistant answer and end marker; all prior turns are context')
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False))


if __name__=='__main__':
    main()
