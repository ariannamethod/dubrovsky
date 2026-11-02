"""Compare cached greedy next-token logits with full-context recomputation."""
import argparse
import hashlib
import json
from pathlib import Path

from sft import chat_eos


def main():
    import torch
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--conversation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tokens',type=int,default=8)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Preserve earlier decode comparisons')
    source=args.conversation.read_bytes();messages=json.loads(source)['messages']
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(args.model,dtype=torch.bfloat16,
        attn_implementation='sdpa').cuda().eval()
    prompt=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
        enable_thinking=False,return_dict=False)
    inputs=torch.tensor([prompt],device='cuda');steps=[]
    with torch.inference_mode():
        generated=model.generate(input_ids=inputs,attention_mask=torch.ones_like(inputs),
            max_new_tokens=args.tokens,do_sample=False,use_cache=True,
            eos_token_id=chat_eos(model,tokenizer),pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,output_logits=True)
        tokens=generated.sequences[0,len(prompt):].tolist()
        for index,cached in enumerate(generated.logits):
            full=torch.tensor([prompt+tokens[:index]],device='cuda')
            recomputed=model(input_ids=full,attention_mask=torch.ones_like(full),
                use_cache=False,logits_to_keep=1).logits[0,-1].float()
            cached=cached[0].float()
            if not torch.isfinite(recomputed).all()or not torch.isfinite(cached).all():
                raise FloatingPointError('Nonfinite decode logits')
            delta=(cached-recomputed).abs()
            steps.append(dict(index=index,max_abs_difference=delta.max().item(),
                mean_abs_difference=delta.mean().item(),cached_top1=cached.argmax().item(),
                recomputed_top1=recomputed.argmax().item(),
                top1_equal=bool(cached.argmax()==recomputed.argmax())))
    result=dict(model=args.model,prompt_sha256=hashlib.sha256(source).hexdigest(),
        context_tokens=len(prompt),generated_tokens=tokens,steps=steps,
        top1_matches=sum(s['top1_equal']for s in steps),compared_steps=len(steps),
        torch=torch.__version__,dtype='bfloat16',scope='Recorded greedy continuation from this complete text history')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
