"""Stage37: matched frozen/trainable isolated writers, native checkpoint output."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random

import torch

from conditional_readout import source_pair_loss, conditional_gate
from embedded_experts import ExpertQwen, tensor_hashes
from expert_data import encode_expert_prompt, encode_expert_row
from output_geometry import correction_bound, output_basis, bf16_rounding_envelope
from probe_expert_readout import digest, ids_digest
from probe_expert_memory import probe
from probe_role_transfer import prepare_transfer_pairs
from sample_experts import generate
from train_experts import row_loss
from train_role_gain import with_gain, first_name_margin
from train_role_readout import TRAINABLE, prefix_state
from isolated_writer import (WRITER, configure_writer, current_states, evaluate_writer, attach_donors, weak_side_changes)
from train_source_targets import decision_summary


def margins(model, item):
    own, donor = current_states(model, item)
    return first_name_margin(model,item,own), first_name_margin(model,item,donor)


@torch.no_grad()
def evaluate(model, items, head32, detailed=True):
    return evaluate_writer(model,items,head32,detailed=detailed)


def gradient_audit(model, items):
    from dataclasses import replace
    configure_writer(model.experts,True)
    parameters=[p for n,p in model.experts.named_parameters() if n in WRITER]
    names=[n for n,p in model.experts.named_parameters() if n in WRITER]
    records=[]
    for item in items[:2]:
        p=item['alternatives'][0];cut=p['cut'];working=prefix_state(model,p)
        own,donor=current_states(model,item)
        m=first_name_margin(model,item,own)
        gradients=torch.autograd.grad(m,parameters)
        # Independent execution path: full native block, then detach only working.
        native=model.experts(p['raw'][:,:cut],p['mask'][:,:cut],memory_hidden=p['source'][:,:cut])[1]
        state=replace(working,user=native.user)
        n=first_name_margin(model,item,state)
        reference=torch.autograd.grad(n,parameters)
        errors={}
        for name,g,r in zip(names,gradients,reference):
            torch.testing.assert_close(g,r,rtol=2e-5,atol=1e-7)
            errors[name]=float((g-r).abs().max())
        assert torch.equal(m,n)
        assert any(g.norm()>0 for g in gradients)
        records.append(dict(id=item['id'],native_margin=float(m.detach()),
            gradient_norms={n:float(g.norm()) for n,g in zip(names,gradients)},
            native_gradient_max_error=errors,bank_exact=torch.equal(own.user,native.user),
            working_shared=torch.equal(own.working,donor.working),working_detached=not own.working.requires_grad))
    model.zero_grad(set_to_none=True)
    return records


@torch.no_grad()
def geometry(model, items):
    block=model.experts; basis=output_basis(block.output); gain=float(block.output_gain())
    result=[]
    for item in items:
        a,b=item['alternatives'];t=a['positions'][0]
        if t!=b['positions'][0] or a['item']['input_ids'][t]==b['item']['input_ids'][t]:
            raise ValueError('Geometry requires first-token name divergence')
        raw=a['raw'][0,t-1]; ids=[p['item']['input_ids'][t] for p in (a,b)]
        rows=model.base.lm_head.weight[ids].float();w=rows[0].double()-rows[1].double()
        bound=correction_bound(block.output,w,block.config.max_delta_rms,gain,basis)
        body=float(raw.double()@w)
        rounding=bf16_rounding_envelope(raw,rows,block.config.max_delta_rms*gain*math.sqrt(raw.numel()))
        result.append(dict(id=item['id'],pair=item['pair'],body_margin=body,**bound,
            rounding_envelope=rounding,own_excluded=body+bound['bound']+rounding<0,
            donor_excluded=body-bound['bound']-rounding>0))
    return result


def main():
    from safetensors.torch import load_file,save_file
    from transformers import AutoTokenizer
    parser=argparse.ArgumentParser()
    for n in ('model','start','cases','replay','acceptance','output'):
        parser.add_argument('--'+n,type=Path,required=True)
    parser.add_argument('--device',default='cuda');a=parser.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    def write(name,value):(a.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    def snapshot():return {k:v.detach().clone() for k,v in model.experts.state_dict().items()}
    seed=2026112637;torch.manual_seed(seed);random.seed(seed)
    model=ExpertQwen.from_pretrained(a.model,dtype='auto',attn_implementation='sdpa').to(a.device).eval()
    model.experts=with_gain(model.experts);model.expert_config=model.experts.config
    model.experts.load_state_dict(load_file(str(a.start),device=a.device),strict=True)
    model.requires_grad_(False);block=model.experts;initial=snapshot();body=tensor_hashes(model.base)
    if abs(float(block.output_gain())-4)>1e-6:raise ValueError('Expected source4 parent')
    tokenizer=AutoTokenizer.from_pretrained(a.model)
    data=json.loads(a.cases.read_text());replay=json.loads(a.replay.read_text());acceptance=json.loads(a.acceptance.read_text())
    if len(data['train'])!=16 or len(data['validation'])!=4 or len(replay)!=8:
        raise ValueError('Sealed inventory differs')
    train,ta=prepare_transfer_pairs(model,tokenizer,data['train'],a.device)
    validation,va=prepare_transfer_pairs(model,tokenizer,data['validation'],a.device)
    attach_donors(train);attach_donors(validation)
    write('prefix-audit.json',dict(train=ta,validation=va))
    write('gradient-audit.json',gradient_audit(model,train))
    model.requires_grad_(False)
    head32=model.base.lm_head.weight.detach().float()
    with torch.no_grad():reference={x['id']:tuple(v.detach() for v in margins(model,x)) for x in train}
    encoded=[encode_expert_row(tokenizer,r,3072) for r in replay]
    baseline=evaluate(model,train,head32,detailed=True)
    write('train-parent.json',{**baseline,**decision_summary(baseline)})
    write('validation-parent.json',evaluate(model,validation,head32,detailed=True))
    write('geometry-parent.json',dict(train=geometry(model,train),validation=geometry(model,validation)))
    gain_controls=[]
    for gain in (.125,.5,1.,2.,4.,8.):
        with torch.no_grad():block.log_output_gain.fill_(math.log(gain))
        result=evaluate(model,train,head32,detailed=True);gain_controls.append(result)
        write('gain-'+str(gain)+'.json',{**result,**decision_summary(result)})
    block.load_state_dict(initial,strict=True)
    write('run.json',dict(stage=37,steps_per_arm=16,seed=seed,fixed_gain=4,
        reader_lr=1e-5,output_lr=1e-3,writer_lr=1e-5,optimizer_moments='fresh in both arms',
        margin_target=1,centre_penalty=.1,replay_weight=.2,replay_base_kl=.1,
        expert_config=asdict(block.config),source_experts_sha256=digest(a.start),
        cases_sha256=digest(a.cases),replay_sha256=digest(a.replay),
        acceptance_sha256=digest(a.acceptance),native_dtype=str(model.base.lm_head.weight.dtype),
        primary_promoted=False,endpoint_review=['frozen-step16','writer-step16']))
    states={'parent':initial};metrics=(a.output/'metrics.jsonl').open('x')
    def emit(x):
        metrics.write(json.dumps(x)+'\n');metrics.flush()
        print(json.dumps({k:v for k,v in x.items() if k!='rows'}),flush=True)
    for arm in ('frozen','writer'):
        block.load_state_dict(initial,strict=True);optimizer=configure_writer(block,arm=='writer')
        allowed=TRAINABLE | WRITER if arm=='writer' else TRAINABLE
        random.seed(seed);torch.manual_seed(seed);order=list(range(16));random.shuffle(order);best=float('inf')
        def assess(step):
            nonlocal best
            path=a.output/arm/f'step-{step:06d}';path.mkdir(parents=True)
            save_file({k:v.cpu().contiguous() for k,v in snapshot().items()},str(path/'experts.safetensors'))
            torch.save(dict(optimizer=optimizer.state_dict(),step=step,torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),random_rng=random.getstate()),path/'optimizer.pt')
            write(str(path.relative_to(a.output)/'expert_config.json'),asdict(block.config))
            result=evaluate(model,validation,head32,detailed=True)
            improved=result['loss']<best;best=min(best,result['loss'])
            emit(dict(event='validation',arm=arm,step=step,best=improved,**result,**decision_summary(result)))
        assess(0)
        for step,index in enumerate(order,1):
            optimizer.zero_grad(set_to_none=True);stats=[]
            for item in train[2*index:2*index+2]:
                own,donor=margins(model,item)
                task,preservation=source_pair_loss(own,donor,*reference[item['id']],centered=True)
                # The separate common-shift penalty is identical in both arms.
                ((task+.1*preservation)/2).backward()
                stats.append(dict(id=item['id'],own=float(own.detach()),donor=float(donor.detach()),
                    task=float(task.detach()),centre_penalty=float(preservation.detach())))
            role_gradients={n:v.grad.norm().item() for n,v in block.named_parameters() if v.requires_grad and v.grad is not None}
            replay_loss,_,_=row_loss(model,encoded[(step-1)%8],kl_weight=.1)
            (.2*replay_loss).backward()
            gradients={n:v.grad.norm().item() for n,v in block.named_parameters() if v.requires_grad}
            norm=torch.nn.utils.clip_grad_norm_([v for v in block.parameters() if v.requires_grad],1.,error_if_nonfinite=True)
            optimizer.step()
            if any(not torch.isfinite(v).all() for v in block.parameters()):raise RuntimeError('Nonfinite weights')
            if any(not torch.equal(v,initial[n]) for n,v in block.state_dict().items() if n not in allowed):
                raise RuntimeError('Frozen expert tensor changed')
            emit(dict(event='train',arm=arm,step=step,pair=data['train'][index]['id'],
                gradient_norm=norm.item(),gradient_norms=gradients,role_gradient_norms=role_gradients,replay_loss=float(replay_loss.detach()),roles=stats))
            assess(step)
        states[arm]=snapshot();result=evaluate(model,train,head32,detailed=True)
        write('train-'+arm+'.json',{**result,**decision_summary(result)})
        write('weak-sides-'+arm+'.json',weak_side_changes(baseline,result))
        write('gate-'+arm+'.json',conditional_gate(baseline,result,gain_controls))
        write('geometry-'+arm+'.json',dict(train=geometry(model,train),validation=geometry(model,validation)))
    metrics.close();model.requires_grad_(False);replies=(a.output/'replies.jsonl').open('x');count=0;baseline_prefixes=[]
    def answer(row,arm,group,mode='experts'):
        nonlocal count
        prompt=encode_expert_prompt(tokenizer,row,3072)
        result=generate(model,tokenizer,row,seed=row['seed'],mode=mode,max_new_tokens=192)
        replies.write(json.dumps({**row,'arm':arm,'group':group,'mode':mode,
            'prefix_sha256':ids_digest(prompt['input_ids']),**result},ensure_ascii=False)+'\n');replies.flush();count+=1
        print(json.dumps(dict(reply=count,arm=arm,id=row['id'],eos=result['eos_reached'])),flush=True)
        return result
    for arm in ('stage7','parent','frozen','writer'):
        block.load_state_dict(states['parent' if arm=='stage7' else arm],strict=True)
        mode='base' if arm=='stage7' else 'experts'
        for pair in data['validation']:
            for row in pair['rows']:answer({**row,'seed':1536},arm,'fixed',mode)
        for scene in acceptance['fresh']:
            history=[]
            for t,question in enumerate(scene['turns']):
                facts=scene['facts'][str(t)] if str(t) in scene['facts'] else {}
                row=dict(id=scene['id']+f'-{t}',scene=scene['id'],turn=t,language=scene['language'],
                    history=list(history),question=question,seed=scene['seed']+t,**facts)
                result=answer(row,arm,'fresh',mode)
                if arm=='stage7' and t in scene['matched_turns']:baseline_prefixes.append(row)
                history.extend([dict(role='user',content=question),dict(role='assistant',content=result['text'])])
        for row in acceptance['style']:answer(row,arm,'style',mode)
        if arm!='stage7':
            for row in baseline_prefixes:answer(row,arm,'matched-stage7-prefix',mode)
        if arm=='writer':
            for row in baseline_prefixes:answer(row,arm,'matched-memory-off','memory-off')
    replies.close()
    invariance={}
    for arm in ('parent','frozen','writer'):
        block.load_state_dict(states[arm],strict=True)
        invariance[arm]=[probe(model,tokenizer,s,True) for s in acceptance['invariance']]
    write('bank-invariance.json',invariance)
    if tensor_hashes(model.base)!=body:raise RuntimeError('Frozen body changed')
    write('completed.json',dict(optimizer_steps=32,replies=count,body_tensors_preserved=len(body),
        writer_trainable_only_in_writer_arm=True,primary_promoted=False,endpoint_review=['frozen-step16','writer-step16']))


if __name__=='__main__':main()
