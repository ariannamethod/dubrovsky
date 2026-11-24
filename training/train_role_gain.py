"""Matched fixed/learned output-strength pilot, with per-source name margins."""
import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from embedded_experts import EmbeddedExperts, ExpertQwen, tensor_hashes
from expert_data import encode_expert_prompt, encode_expert_row
from probe_expert_readout import digest, ids_digest
from probe_expert_memory import probe
from probe_role_transfer import prepare_transfer_pairs
from sample_experts import generate
from train_experts import row_loss
from train_role_readout import (TRAINABLE, configure, prefix_state, value_logp,
    swapped_state, causal_loss, evaluate)


def with_gain(block, maximum=8.):
    if block.config.format_version != 2:
        raise ValueError('Upgrade requires an isolated format-2 parent')
    result = EmbeddedExperts(replace(block.config, format_version=3,
        max_output_gain=maximum)).to(block.output.device)
    state = {**block.state_dict(), 'log_output_gain': block.output.new_zeros(())}
    result.load_state_dict(state, strict=True)
    return result


def first_name_margin(model, item, state):
    """Native full-vocabulary projection at the first differing name token."""
    a,b = item['alternatives']
    x,y = [p['item']['input_ids'] for p in (a,b)]
    t = next(i for i,(u,v) in enumerate(zip(x,y)) if u!=v)
    if t not in a['positions'] or t not in b['positions']:
        raise ValueError('Divergent token is outside the scored names')
    cut = a['cut']
    h,_,_ = model.experts(a['raw'][:,cut:t], a['mask'][:,cut:t], state,
                          memory_hidden=a['source'][:,cut:t])
    logits = model.base.lm_head(h[:,-1]).float()
    return logits[0,x[t]]-logits[0,y[t]]


def per_source_loss(margin, target=1.):
    return F.softplus(target-margin)


def objective(model, item, head32):
    a,b = item['alternatives']; state = prefix_state(model,a)
    if not torch.equal(state.user,item['normal_bank']):
        raise RuntimeError('Frozen writer changed')
    ce = -value_logp(model,a,state).mean()
    margin = first_name_margin(model,item,state)
    normal = (value_logp(model,a,state,precision='add32',head32=head32).sum()
              -value_logp(model,b,state,precision='add32',head32=head32).sum())
    donor = swapped_state(state,item['bank'])
    swapped = (value_logp(model,a,donor,precision='add32',head32=head32).sum()
               -value_logp(model,b,donor,precision='add32',head32=head32).sum())
    role = per_source_loss(margin)
    contrast = causal_loss(normal,swapped)
    return ce+role+12*contrast, dict(ce=ce.item(),first_margin=margin.item(),
        per_source_loss=role.item(),contrast=contrast.item())


def main():
    from transformers import AutoTokenizer
    from safetensors.torch import load_file,save_file
    p=argparse.ArgumentParser()
    for n in ('model','start','cases','replay','acceptance','output'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--device',default='cuda');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    def write(n,v):(a.output/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
    def snapshot(block):return {n:v.detach().clone() for n,v in block.state_dict().items()}
    torch.manual_seed(20261124);random.seed(20261124)
    model=ExpertQwen.from_pretrained(a.model,dtype='auto',attn_implementation='sdpa').to(a.device).eval()
    tokenizer=AutoTokenizer.from_pretrained(a.model)
    model.experts.load_state_dict(load_file(str(a.start),device=a.device),strict=True)
    body=tensor_hashes(model.base); parent=model.experts
    data=json.loads(a.cases.read_text());replay=json.loads(a.replay.read_text())
    acceptance=json.loads(a.acceptance.read_text())
    if len(data['train'])!=8 or len(data['validation'])!=4 or len(replay)!=8:
        raise ValueError('Sealed training inventory differs')
    train,ta=prepare_transfer_pairs(model,tokenizer,data['train'],a.device)
    validation,va=prepare_transfer_pairs(model,tokenizer,data['validation'],a.device)
    write('prefix-audit.json',dict(train=ta,validation=va))
    upgraded=with_gain(parent)
    parity=[]
    with torch.no_grad():
        for item in train:
            q=item['alternatives'][0]
            x=parent(q['raw'],q['mask'],memory_hidden=q['source'])
            y=upgraded(q['raw'],q['mask'],memory_hidden=q['source'])
            ok=all(torch.equal(u,v) for u,v in zip(
                (x[0],x[1].user,x[1].working,x[2]),(y[0],y[1].user,y[1].working,y[2])))
            if not ok:raise RuntimeError('Gain-one conversion changed outputs or state')
            parity.append(dict(id=item['id'],exact=True))
    # Four sampled answers also check the complete decoding path at gain one.
    for row in [v['rows'][0] for v in data['validation']]:
        model.experts=parent
        old=generate(model,tokenizer,row,seed=73,max_new_tokens=192,mode='experts')
        model.experts=upgraded;model.expert_config=upgraded.config
        new=generate(model,tokenizer,row,seed=73,max_new_tokens=192,mode='experts')
        if old['token_ids']!=new['token_ids']:raise RuntimeError('Gain-one decoding differs')
        parity.append(dict(id=row['id'],old=old,new=new,exact=True))
    write('gain-one-parity.json',parity)
    initial=snapshot(upgraded);head32=model.base.lm_head.weight.detach().float()
    encoded=[encode_expert_row(tokenizer,r,3072) for r in replay]
    write('run.json',dict(stage=32,steps_per_arm=16,seed=20261124,
        reader_lr=1e-5,output_lr=1e-3,gain_lr=.15,optimizer_moments='fresh in both arms',
        margin_target=1,contrast_target=.05,contrast_weight=12,replay_weight=.2,
        replay_base_kl=.1,expert_config=asdict(upgraded.config),
        source_experts_sha256=digest(a.start),cases_sha256=digest(a.cases),
        replay_sha256=digest(a.replay),acceptance_sha256=digest(a.acceptance),
        native_dtype=str(model.base.lm_head.weight.dtype),primary_promoted=False))
    states={'parent':initial};metrics=(a.output/'metrics.jsonl').open('x')
    def emit(x):metrics.write(json.dumps(x)+'\n');metrics.flush();print(json.dumps({k:v for k,v in x.items() if k!='rows'}),flush=True)
    for arm in ('fixed1','learned'):
        upgraded.load_state_dict(initial,strict=True)
        optimizer=configure(upgraded)
        allowed=set(TRAINABLE)
        if arm=='learned':
            upgraded.log_output_gain.requires_grad_(True);allowed.add('log_output_gain')
            optimizer.add_param_group(dict(params=[upgraded.log_output_gain],lr=.15))
        random.seed(20261124);torch.manual_seed(20261124)
        order=list(range(8));best=float('inf')
        def assess(step):
            nonlocal best
            path=a.output/arm/f'step-{step:06d}';path.mkdir(parents=True)
            save_file({k:v.cpu().contiguous() for k,v in snapshot(upgraded).items()},str(path/'experts.safetensors'))
            torch.save(dict(optimizer=optimizer.state_dict(),step=step,torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),random_rng=random.getstate()),path/'optimizer.pt')
            write(str(path.relative_to(a.output)/'expert_config.json'),asdict(upgraded.config))
            result=evaluate(model,validation,head32,detailed=True)
            improved=result['loss']<best;best=min(best,result['loss'])
            emit(dict(event='validation',arm=arm,step=step,gain=upgraded.output_gain().item(),best=improved,**result))
        assess(0)
        for step in range(1,17):
            if (step-1)%8==0:random.shuffle(order)
            index=order[(step-1)%8];optimizer.zero_grad(set_to_none=True);stats=[]
            for item in train[2*index:2*index+2]:
                loss,s=objective(model,item,head32);(loss/2).backward();stats.append(s)
            replay_loss,_,_=row_loss(model,encoded[(step-1)%8],kl_weight=.1)
            (.2*replay_loss).backward()
            norm=torch.nn.utils.clip_grad_norm_([v for v in upgraded.parameters() if v.requires_grad],1.,error_if_nonfinite=True)
            gain_gradient=None if upgraded.log_output_gain.grad is None else upgraded.log_output_gain.grad.item()
            optimizer.step()
            with torch.no_grad():upgraded.log_output_gain.clamp_(-math.log(8),math.log(8))
            if any(not torch.isfinite(v).all() for v in upgraded.parameters()):raise RuntimeError('Nonfinite weights')
            if any(not torch.equal(v,initial[n]) for n,v in upgraded.state_dict().items() if n not in allowed):
                raise RuntimeError('Frozen expert tensor changed')
            emit(dict(event='train',arm=arm,step=step,pair=data['train'][index]['id'],
                gain=upgraded.output_gain().item(),gain_gradient=gain_gradient,
                gradient_norm=norm.item(),replay_loss=replay_loss.item(),roles=stats))
            # Every point is retained; validation is a diagnostic, endpoints are preselected.
            assess(step)
        states[arm]=snapshot(upgraded)
        write('train-'+arm+'.json',evaluate(model,train,head32,detailed=True))
    metrics.close();replies=(a.output/'replies.jsonl').open('x');count=0
    def answer(row,arm,group,mode='experts'):
        nonlocal count
        prompt=encode_expert_prompt(tokenizer,row,3072)
        result=generate(model,tokenizer,row,seed=row['seed'],mode=mode,max_new_tokens=192)
        record={**row,'arm':arm,'group':group,'mode':mode,
            'prefix_sha256':ids_digest(prompt['input_ids']),**result}
        replies.write(json.dumps(record,ensure_ascii=False)+'\n');replies.flush();count+=1
        print(json.dumps(dict(reply=count,arm=arm,id=row['id'],eos=result['eos_reached'])),flush=True)
        return result['text']
    for arm in ('stage7','parent','fixed1','learned'):
        upgraded.load_state_dict(states['parent' if arm=='stage7' else arm],strict=True)
        mode='base' if arm=='stage7' else 'experts'
        for pair in data['validation']:
            for row in pair['rows']:answer({**row,'seed':1532},arm,'fixed',mode)
        for scene in acceptance['fresh']:
            history=[]
            for t,question in enumerate(scene['turns']):
                row=dict(id=scene['id']+f'-{t}',scene=scene['id'],turn=t,
                    language=scene['language'],owner=scene['owner'],keeper=scene['keeper'],
                    history=list(history),question=question,seed=scene['seed']+t)
                reply=answer(row,arm,'fresh',mode)
                history.extend([dict(role='user',content=question),dict(role='assistant',content=reply)])
        for row in acceptance['style']:answer(row,arm,'style',mode)
        if arm=='learned':
            for pair in data['validation']:
                for row in pair['rows']:answer({**row,'seed':1532},arm,'fixed-memory-off','memory-off')
    replies.close()
    write('bank-invariance.json',[probe(model,tokenizer,s,True) for s in acceptance['invariance']])
    if tensor_hashes(model.base)!=body:raise RuntimeError('Frozen body changed')
    write('completed.json',dict(optimizer_steps=32,replies=count,body_tensors_preserved=len(body),
        gain_one_exact=True,primary_promoted=False,endpoint_review=['fixed1-step16','learned-step16'],
        learned_gain=upgraded.output_gain().item()))


if __name__=='__main__':main()
