"""Compare full and answer-only vocabulary projection on a native long-chat example."""
import argparse
import json
from pathlib import Path

from sft import encode, read_rows, supervision_tail


def main():
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve earlier parity results')
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    rows=[encode(tokenizer,r,4096)for r in read_rows(args.data)]
    row=max(rows,key=lambda r:len(r['input_ids']))
    model=Qwen3_5ForConditionalGeneration.from_pretrained(args.model,dtype=torch.float32,
        attn_implementation='sdpa').cuda().eval().requires_grad_(False)
    inputs=torch.tensor([row['input_ids']],device='cuda')
    trimmed,keep=supervision_tail(row['labels'])
    results={}
    for name,labels,count in [('full',row['labels'],0),('answer_only',trimmed,keep)]:
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            loss=model(input_ids=inputs,labels=torch.tensor([labels],device='cuda'),
                       logits_to_keep=count,use_cache=False).loss.item()
        results[name]=dict(loss=loss,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
    difference=abs(results['full']['loss']-results['answer_only']['loss'])
    result=dict(model=args.model,context_tokens=len(row['input_ids']),answer_projection_tokens=keep,
        supervised_tokens=sum(x!=-100 for x in row['labels'][1:]),results=results,
        absolute_loss_difference=difference,passed=difference<1e-5,
        torch=torch.__version__,purpose='Keep the complete attention history while omitting unsupervised vocabulary logits')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    if not result['passed']:
        raise AssertionError('Full and answer-only loss differ')


if __name__=='__main__':
    main()
