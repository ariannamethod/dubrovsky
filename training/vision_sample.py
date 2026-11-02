"""Photographic grounding and character probes; no image training occurs here."""
import argparse
import hashlib
import json
from pathlib import Path
from decoding import PRESETS, generation_kwargs
from sft import SYSTEM, chat_eos
from sample import diagnostics


DEFAULT_QUESTIONS = [
    dict(id='en-visible', language='en', question='Briefly describe what is visibly present in this photograph.'),
    dict(id='en-character', language='en', question='What trouble has Dubrovsky got into here? Use something visible in the photograph.'),
    dict(id='ru-character', language='ru', question='Во что здесь влип Дубровский? Обыграй предметы на фотографии. Ответь по-русски.'),
    dict(id='he-character', language='he', question='באיזו צרה הסתבך כאן דוברובסקי? שלב פרטים שרואים בתצלום. ענה בעברית.'),
]


def validate_questions(questions):
    if not isinstance(questions, list) or not questions:
        raise ValueError('Expected a nonempty list of image questions')
    ids = set()
    for row in questions:
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key].strip()
            for key in ('id', 'language', 'question')
        ):
            raise ValueError('Each question needs nonempty id, language and question strings')
        if row['id'] in ids:
            raise ValueError('Question IDs must be unique')
        ids.add(row['id'])


def main():
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--revision')
    p.add_argument('--images',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preset',choices=PRESETS,default='qwen-vision')
    p.add_argument('--questions',type=Path,help='JSON list of {id, language, question}; defaults to the original four probes')
    p.add_argument('--seeds',type=int,nargs='+',default=[42])
    args=p.parse_args()
    if not Path(args.model).is_dir() and not args.revision:
        raise ValueError('Pin a revision when loading a remote model')
    if args.output.exists():raise ValueError('Preserve previous outputs')
    questions=json.loads(args.questions.read_text()) if args.questions else DEFAULT_QUESTIONS
    validate_questions(questions)
    if len(args.seeds)!=len(set(args.seeds)):raise ValueError('Use distinct seeds')
    images=sorted(args.images.glob('*.png'))
    if not images:raise ValueError('No PNG photographs found')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    processor=AutoProcessor.from_pretrained(args.model,revision=args.revision)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(args.model,revision=args.revision,
        dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    eos=chat_eos(model,processor.tokenizer)
    questions_hash=hashlib.sha256(json.dumps(questions,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    with args.output.open('x') as out, torch.inference_mode():
        for path in images:
            for item,seed in ((q,s) for q in questions for s in args.seeds):
                language,question=item['language'],item['question']
                torch.manual_seed(seed)
                messages=[{'role':'system','content':SYSTEM},{'role':'user','content':[
                    {'type':'image','url':str(path.resolve())},{'type':'text','text':question}]}]
                inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                    enable_thinking=False,return_dict=True,return_tensors='pt').to('cuda',dtype=torch.bfloat16)
                length=inputs['input_ids'].shape[1]
                ids=model.generate(**inputs,**generation_kwargs(args.preset,length),max_new_tokens=192,
                    eos_token_id=eos,pad_token_id=processor.tokenizer.pad_token_id,use_cache=True)[0,length:].tolist()
                text=processor.tokenizer.decode(ids,skip_special_tokens=True)
                measured=diagnostics(text,set())
                measured['training_8gram_fraction']=None  # No training corpus is loaded here.
                row=dict(model=args.model,revision=args.revision,image=path.name,
                    image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),language=language,
                    question=question,text=text,token_ids=ids,eos_reached=bool(ids and ids[-1] in eos),
                    preset=args.preset,sampling=PRESETS[args.preset],seed=seed,
                    question_id=item['id'],questions_sha256=questions_hash,
                    torch=torch.__version__,transformers=transformers.__version__,
                    enable_thinking=False,use_cache=True,max_new_tokens=192,eos_token_ids=eos,
                    diagnostics=measured)
                out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush()
                print(json.dumps(row,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
