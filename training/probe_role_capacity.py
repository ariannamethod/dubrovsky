"""Audit both-source name decisions and the bounded expert output on train only."""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from embedded_experts import ExpertQwen, tensor_hashes
from probe_expert_readout import digest
from probe_role_transfer import prepare_transfer_pairs
from train_role_readout import evaluate


def sphere_capacity(raw, rows, rms_limit, target=1.):
    """FP32 linear-head bound for an arbitrary residual inside the RMS ball."""
    if rms_limit <= 0 or rows.shape != (2,raw.numel()):
        raise ValueError('Invalid hidden vector, head rows or RMS limit')
    direction=rows[0].float()-rows[1].float()
    norm=direction.norm()
    margin=torch.dot(raw.float(),direction)
    bound=rms_limit*math.sqrt(raw.numel())*norm
    if norm.item()==0:
        delta=torch.zeros_like(raw,dtype=torch.float32)
    else:
        delta=direction*(rms_limit*math.sqrt(raw.numel())/norm)
    return dict(base_margin=margin.item(),bound=bound.item(),
        max_margin=(margin+bound).item(),
        gain_for_zero=max(0.,-margin.item()/bound.item()) if bound.item() else None,
        gain_for_target=max(0.,(target-margin.item())/bound.item()) if bound.item() else None),delta


@torch.no_grad()
def main():
    from transformers import AutoTokenizer
    from safetensors.torch import load_file,save_file
    p=argparse.ArgumentParser()
    for n in ('model','start','cases','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--device',default='cuda');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    def write(n,v):(a.output/n).write_text(json.dumps(v,indent=2)+'\n')
    model=ExpertQwen.from_pretrained(a.model,dtype='auto',attn_implementation='sdpa').to(a.device).eval()
    model.requires_grad_(False);tokenizer=AutoTokenizer.from_pretrained(a.model)
    model.experts.load_state_dict(load_file(str(a.start),device=a.device),strict=True)
    frozen=tensor_hashes(model);data=json.loads(a.cases.read_text())
    if len(data['train'])!=8:raise ValueError('Expected eight original training pairs')
    items,prefix_audit=prepare_transfer_pairs(model,tokenizer,data['train'],a.device)
    head32=model.base.lm_head.weight.detach().float()
    write('prefix-audit.json',prefix_audit)
    write('values.json',evaluate(model,items,head32,detailed=True))
    records=[];tensors={}
    for i,item in enumerate(items):
        left,right=item['alternatives'];x,y=[p['item']['input_ids'] for p in (left,right)]
        t=next(j for j,(u,v) in enumerate(zip(x,y)) if u!=v)
        if t not in left['positions'] or t not in right['positions']:
            raise ValueError('First divergent token lies outside scored name')
        raw=left['raw'][0,t-1];ids=[x[t],y[t]]
        weights=head32[ids];stats,delta=sphere_capacity(raw,weights,model.expert_config.max_delta_rms)
        h,_,_=model.experts(left['raw'][:,:t],left['mask'][:,:t],memory_hidden=left['source'][:,:t])
        native=model.base.lm_head(h[:,-1]).float()[0,ids]
        original=model.base.lm_head(raw[None]).float()[0,ids]
        native_trials=[]
        for gain in (.125,.25,.5,1.,2.,4.,8.,16.):
            # This direction knows the two target rows: capacity probe, never a generated answer.
            candidate=raw+(gain*delta).to(raw.dtype)
            logits=model.base.lm_head(candidate[None]).float()[0,ids]
            native_trials.append(dict(gain=gain,margin=(logits[0]-logits[1]).item()))
        record=dict(id=item['id'],pair=item['pair'],role=item['role'],tokens=ids,
            token_texts=[tokenizer.decode([t]) for t in ids],first_difference=t,
            raw_native_margin=(original[0]-original[1]).item(),
            learned_native_margin=(native[0]-native[1]).item(),
            full_name_token_counts=[len(p['positions']) for p in (left,right)],
            **stats,native_maximizing_direction=native_trials)
        records.append(record)
        tensors[f'{i}.raw']=raw.cpu().contiguous();tensors[f'{i}.rows']=weights.cpu().contiguous()
        tensors[f'{i}.expert_delta_native']=(h[0,-1].float()-raw.float()).cpu().contiguous()
        print(json.dumps(record),flush=True)
    write('capacity.json',records);save_file(tensors,str(a.output/'capacity-features.safetensors'))
    if frozen!=tensor_hashes(model):raise RuntimeError('Frozen tensors changed')
    wrong=[r for r in records if r['learned_native_margin']<=0]
    write('completed.json',dict(optimizer_steps=0,train_positions=len(records),
        wrong_first_tokens=len(wrong),wrong_beyond_fp32_unit_ball=sum(r['max_margin']<0 for r in wrong),
        train_max_gain_for_target=max(r['gain_for_target'] for r in records),
        source_experts_sha256=digest(a.start),cases_sha256=digest(a.cases),
        frozen_tensors_preserved=len(frozen),primary_promoted=False))


if __name__=='__main__':main()
