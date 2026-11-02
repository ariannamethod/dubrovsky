"""One measured conversational turn for training evaluation, preserving full history."""
import argparse
import json
from pathlib import Path
import time

from decoding import PRESETS, generation_kwargs
from sft import SYSTEM, chat_eos
from sample import diagnostics, grams


def conversation_diagnostics(text, messages):
    measured=diagnostics(text,set())
    measured['training_8gram_fraction']=None  # No training corpus is loaded here.
    previous={g for message in messages if message['role']=='assistant'
              for g in grams(message['content'],4)}
    current=grams(text,4)
    measured['prior_answer_4gram_fraction']=(sum(g in previous for g in current)/len(current)
                                             if current else 0)
    return measured


def main():
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--conversation',type=Path,required=True,
                   help='JSON object with messages; last message must be from the partner')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preset',choices=PRESETS,default='baseline')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-new-tokens',type=int,default=192)
    p.add_argument('--no-cache',action='store_true',help='Recompute full context to compare against cached decoding')
    args=p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve earlier conversational outputs')
    source=json.loads(args.conversation.read_text())
    messages=source['messages']
    if not messages or messages[-1]['role']!='user':
        raise ValueError('Conversation must end with a partner user message')
    if any(m.get('role') not in ('system','user','assistant') or
           not isinstance(m.get('content'),str) for m in messages):
        raise ValueError('Expected text chat messages')
    if messages[0]['role']!='system':
        messages=[{'role':'system','content':SYSTEM}]+messages
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(args.model,
        dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    prompt=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
        enable_thinking=False,return_dict=False)
    if len(prompt)+args.max_new_tokens>model.config.text_config.max_position_embeddings:
        raise ValueError('History exceeds context limit; no silent truncation')
    torch.manual_seed(args.seed)
    inputs=torch.tensor([prompt],device='cuda')
    eos=chat_eos(model,tokenizer)
    started=time.monotonic()
    with torch.inference_mode():
        ids=model.generate(input_ids=inputs,attention_mask=torch.ones_like(inputs),
            **generation_kwargs(args.preset,len(prompt)),max_new_tokens=args.max_new_tokens,
            eos_token_id=eos,pad_token_id=tokenizer.pad_token_id,use_cache=not args.no_cache)[0,len(prompt):].tolist()
    text=tokenizer.decode(ids,skip_special_tokens=True)
    measured=conversation_diagnostics(text,messages)
    record=dict(model=args.model,partner=source.get('partner','unspecified'),
        messages=messages+[{'role':'assistant','content':text}],text=text,token_ids=ids,
        prompt_tokens=len(prompt),seed=args.seed,preset=args.preset,sampling=PRESETS[args.preset],
        use_cache=not args.no_cache,
        eos_reached=bool(ids and ids[-1] in eos),elapsed_seconds=time.monotonic()-started,
        diagnostics=measured)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(record,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
