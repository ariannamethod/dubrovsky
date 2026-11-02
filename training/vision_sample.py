"""Photographic grounding and character probes; no image training occurs here."""
import argparse
import hashlib
import json
from pathlib import Path
from decoding import PRESETS, generation_kwargs
from sft import SYSTEM, chat_eos
from sample import diagnostics


def main():
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--revision')
    p.add_argument('--images',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preset',choices=PRESETS,default='qwen-vision')
    args=p.parse_args()
    if args.output.exists():raise ValueError('Preserve previous outputs')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    processor=AutoProcessor.from_pretrained(args.model,revision=args.revision)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(args.model,revision=args.revision,
        dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    eos=chat_eos(model,processor.tokenizer)
    questions=[('en','Briefly describe what is visibly present in this photograph.'),
        ('en','What trouble has Dubrovsky got into here? Use something visible in the photograph.'),
        ('ru','Во что здесь влип Дубровский? Обыграй предметы на фотографии. Ответь по-русски.'),
        ('he','באיזו צרה הסתבך כאן דוברובסקי? שלב פרטים שרואים בתצלום. ענה בעברית.')]
    with args.output.open('w') as out, torch.inference_mode():
        for path in sorted(args.images.glob('*.png')):
            for language,question in questions:
                torch.manual_seed(42)
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
                    preset=args.preset,sampling=PRESETS[args.preset],seed=42,
                    diagnostics=measured)
                out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush()
                print(json.dumps(row,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
