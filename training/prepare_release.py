"""Package an evaluated candidate with nonthinking chat defaults; never upload here."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from sft import SYSTEM


def main():
    from transformers import AutoTokenizer, AutoProcessor
    p = argparse.ArgumentParser()
    p.add_argument('candidate', type=Path)
    p.add_argument('release', type=Path)
    p.add_argument('--model-card', type=Path, required=True)
    args = p.parse_args()
    if args.release.exists():
        raise ValueError('Release destination already exists')
    if not list(args.candidate.glob('*.safetensors')):
        raise ValueError('No candidate weights')
    tokenizer = AutoTokenizer.from_pretrained(args.candidate)
    processor = AutoProcessor.from_pretrained(args.candidate)
    probes = [[{'role':'system','content':SYSTEM}, {'role':'user','content':q}]
              for q in ['Who are you?', 'Кто ты?', 'מי אתה?']]
    expected = [tokenizer.apply_chat_template(m, tokenize=True, return_dict=False,
                add_generation_prompt=True, enable_thinking=False) for m in probes]
    template = tokenizer.chat_template
    if not isinstance(template, str):
        raise TypeError('Review template variants before packaging')
    # Dubrovsky has no reasoning mode in this release. Keep all native message
    # formatting; choose its existing empty-thinking generation prefix.
    tokenizer.chat_template = '{%- set enable_thinking = false -%}\n' + template
    processor.chat_template = tokenizer.chat_template
    for messages, ids in zip(probes, expected):
        for kwargs in ({}, {'enable_thinking':True}, {'enable_thinking':False}):
            actual = tokenizer.apply_chat_template(messages, tokenize=True,
                return_dict=False, add_generation_prompt=True, **kwargs)
            if actual != ids:
                raise ValueError('Release template changes nonthinking prompt token IDs')
    shutil.copytree(args.candidate, args.release)
    processor.save_pretrained(args.release); tokenizer.save_pretrained(args.release)
    shutil.copyfile(args.model_card, args.release / 'README.md')
    hashes = {}
    for path in sorted(args.release.glob('*.safetensors')):
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024), b''):
                h.update(block)
        hashes[path.name] = h.hexdigest()
    (args.release / 'weights_sha256.json').write_text(json.dumps(hashes, indent=2))
    print(json.dumps({'release': str(args.release), 'template_checks': 9, 'weights_sha256':hashes}))


if __name__ == '__main__':
    main()
