"""Matched gain-four pilot with explicit targets for the counterfactual bank."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random

import torch

from embedded_experts import ExpertQwen, tensor_hashes
from expert_data import encode_expert_prompt, encode_expert_row
from probe_expert_readout import digest, ids_digest
from probe_expert_memory import probe
from probe_role_transfer import prepare_transfer_pairs
from sample_experts import generate
from train_experts import row_loss
from train_role_gain import with_gain, first_name_margin, per_source_loss, objective
from train_role_readout import TRAINABLE, configure, prefix_state, value_logp, swapped_state, evaluate


def source_target_loss(model, item):
    """The counterpart name is correct under its bank, at the same working prefix."""
    own, counterpart = item['alternatives']
    state = prefix_state(model, own)
    if not torch.equal(state.user, item['normal_bank']):
        raise RuntimeError('Frozen writer changed')
    donor = swapped_state(state, item['bank'])
    ce = -value_logp(model, counterpart, donor).mean()
    # first_name_margin is always own minus counterpart; donor reverses the target.
    margin = -first_name_margin(model, item, donor)
    role = per_source_loss(margin)
    return ce + role, dict(donor_ce=ce.item(), donor_target_margin=margin.item(),
                          donor_role_loss=role.item())


def decision_summary(result):
    """Distinguish a favorable source difference from correct source decisions."""
    rows = result['rows']; pairs = sorted({r['pair'] for r in rows})
    good = lambda r: r['native']['margin'] > 0
    switched = lambda r: good(r) and r['native']['swapped_margin'] < 0
    return dict(normal_correct=sum(good(r) for r in rows),
        donor_correct=sum(r['native']['swapped_margin'] < 0 for r in rows),
        source_switch_correct=sum(switched(r) for r in rows),
        normal_complete_pairs=sum(all(good(r) for r in rows if r['pair']==p) for p in pairs),
        four_decision_pairs=sum(all(switched(r) for r in rows if r['pair']==p) for p in pairs))


def main():
    from transformers import AutoTokenizer
    from safetensors.torch import load_file, save_file
    p=argparse.ArgumentParser()
    for name in ('model','start','cases','replay','acceptance','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    def write(name,value):(a.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    def snapshot(block):return {n:v.detach().clone() for n,v in block.state_dict().items()}
    seed=20261125;torch.manual_seed(seed);random.seed(seed)
    model=ExpertQwen.from_pretrained(a.model,dtype='auto',attn_implementation='sdpa').to(a.device).eval()
    tokenizer=AutoTokenizer.from_pretrained(a.model)
    model.experts.load_state_dict(load_file(str(a.start),device=a.device),strict=True)
    model.requires_grad_(False);body=tensor_hashes(model.base)
    data=json.loads(a.cases.read_text());replay=json.loads(a.replay.read_text());acceptance=json.loads(a.acceptance.read_text())
    if len(data['train'])!=8 or len(data['validation'])!=4 or len(replay)!=8:
        raise ValueError('Sealed training inventory differs')
    model.experts=with_gain(model.experts)
    model.expert_config=model.experts.config
    with torch.no_grad():model.experts.log_output_gain.fill_(math.log(4))
    block=model.experts;initial=snapshot(block)
    train,ta=prepare_transfer_pairs(model,tokenizer,data['train'],a.device)
    validation,va=prepare_transfer_pairs(model,tokenizer,data['validation'],a.device)
    write('prefix-audit.json',dict(train=ta,validation=va))
    head32=model.base.lm_head.weight.detach().float()
    encoded=[encode_expert_row(tokenizer,r,3072) for r in replay]
    write('run.json',dict(stage=34,steps_per_arm=16,seed=seed,fixed_gain=4,
        reader_lr=1e-5,output_lr=1e-3,optimizer_moments='fresh in both arms',
        margin_target=1,contrast_target=.05,contrast_weight=12,replay_weight=.2,
        replay_base_kl=.1,source_target_weight=1,expert_config=asdict(block.config),
        source_experts_sha256=digest(a.start),cases_sha256=digest(a.cases),
        replay_sha256=digest(a.replay),acceptance_sha256=digest(a.acceptance),
        native_dtype=str(model.base.lm_head.weight.dtype),primary_promoted=False))
    states={'parent4':initial};metrics=(a.output/'metrics.jsonl').open('x')
    def emit(x):
        metrics.write(json.dumps(x)+'\n');metrics.flush()
        print(json.dumps({k:v for k,v in x.items() if k!='rows'}),flush=True)
    with torch.no_grad():
        baseline=evaluate(model,train,head32,detailed=True)
    write('train-parent4.json',{**baseline,**decision_summary(baseline)})
    for arm in ('control4','source4'):
        block.load_state_dict(initial,strict=True);optimizer=configure(block)
        random.seed(seed);torch.manual_seed(seed);order=list(range(8));best=float('inf')
        def assess(step):
            nonlocal best
            path=a.output/arm/f'step-{step:06d}';path.mkdir(parents=True)
            save_file({k:v.cpu().contiguous() for k,v in snapshot(block).items()},str(path/'experts.safetensors'))
            torch.save(dict(optimizer=optimizer.state_dict(),step=step,torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),random_rng=random.getstate()),path/'optimizer.pt')
            write(str(path.relative_to(a.output)/'expert_config.json'),asdict(block.config))
            result=evaluate(model,validation,head32,detailed=True)
            improved=result['loss']<best;best=min(best,result['loss'])
            emit(dict(event='validation',arm=arm,step=step,gain=block.output_gain().item(),best=improved,
                **result,**decision_summary(result)))
        assess(0)
        for step in range(1,17):
            if (step-1)%8==0:random.shuffle(order)
            index=order[(step-1)%8];optimizer.zero_grad(set_to_none=True);stats=[]
            for item in train[2*index:2*index+2]:
                loss,s=objective(model,item,head32)
                if arm=='source4':
                    extra,detail=source_target_loss(model,item);loss=loss+extra;s.update(detail)
                (loss/2).backward();stats.append(s)
            replay_loss,_,_=row_loss(model,encoded[(step-1)%8],kl_weight=.1)
            (.2*replay_loss).backward()
            gradients={n:v.grad.norm().item() for n,v in block.named_parameters() if v.requires_grad}
            norm=torch.nn.utils.clip_grad_norm_([v for v in block.parameters() if v.requires_grad],1.,error_if_nonfinite=True)
            optimizer.step()
            if any(not torch.isfinite(v).all() for v in block.parameters()):raise RuntimeError('Nonfinite weights')
            if any(not torch.equal(v,initial[n]) for n,v in block.state_dict().items() if n not in TRAINABLE):
                raise RuntimeError('Frozen expert tensor or fixed gain changed')
            emit(dict(event='train',arm=arm,step=step,pair=data['train'][index]['id'],
                gain=block.output_gain().item(),gradient_norm=norm.item(),gradient_norms=gradients,
                replay_loss=replay_loss.item(),roles=stats))
            assess(step)
        states[arm]=snapshot(block)
        result=evaluate(model,train,head32,detailed=True)
        write('train-'+arm+'.json',{**result,**decision_summary(result)})
    metrics.close();replies=(a.output/'replies.jsonl').open('x');count=0
    def answer(row,arm,group,mode='experts'):
        nonlocal count
        prompt=encode_expert_prompt(tokenizer,row,3072)
        result=generate(model,tokenizer,row,seed=row['seed'],mode=mode,max_new_tokens=192)
        record={**row,'arm':arm,'group':group,'mode':mode,'prefix_sha256':ids_digest(prompt['input_ids']),**result}
        replies.write(json.dumps(record,ensure_ascii=False)+'\n');replies.flush();count+=1
        print(json.dumps(dict(reply=count,arm=arm,id=row['id'],eos=result['eos_reached'])),flush=True)
        return result['text']
    for arm in ('stage7','parent4','control4','source4'):
        block.load_state_dict(states['parent4' if arm=='stage7' else arm],strict=True)
        model.requires_grad_(False);mode='base' if arm=='stage7' else 'experts'
        for pair in data['validation']:
            for row in pair['rows']:answer({**row,'seed':1534},arm,'fixed',mode)
        for scene in acceptance['fresh']:
            history=[]
            for t,question in enumerate(scene['turns']):
                row=dict(id=scene['id']+f'-{t}',scene=scene['id'],turn=t,language=scene['language'],
                    owner=scene['owner'],keeper=scene['keeper'],history=list(history),question=question,seed=scene['seed']+t)
                reply=answer(row,arm,'fresh',mode)
                history.extend([dict(role='user',content=question),dict(role='assistant',content=reply)])
        for row in acceptance['style']:answer(row,arm,'style',mode)
        if arm=='source4':
            for pair in data['validation']:
                for row in pair['rows']:answer({**row,'seed':1534},arm,'fixed-memory-off','memory-off')
    replies.close()
    write('bank-invariance.json',[probe(model,tokenizer,s,True) for s in acceptance['invariance']])
    if tensor_hashes(model.base)!=body:raise RuntimeError('Frozen body changed')
    write('completed.json',dict(optimizer_steps=32,replies=count,body_tensors_preserved=len(body),
        primary_promoted=False,endpoint_review=['control4-step16','source4-step16'],fixed_gain=block.output_gain().item()))


if __name__=='__main__':main()
