"""Trace bank-dependent output strength and re-read private literary ambiguities."""
import argparse
import json
from pathlib import Path

import torch

from embedded_experts import ExpertQwen, tensor_hashes
from probe_expert_readout import digest
from probe_role_transfer import prepare_transfer_pairs
from sample_experts import generate
from train_role_gain import with_gain
from train_role_readout import prefix_state, swapped_state


def bounded_delta(block, state, votes):
    """Reconstruct the exact pre-gain FP32 readout from its final working state."""
    delta=torch.einsum('be,ber,erd->bd',votes,state.working,block.output)
    rms=torch.sqrt(delta.square().mean(-1,keepdim=True)+1e-12)
    return delta*(block.config.max_delta_rms/rms).clamp(max=1.)


def mix_delta(normal, reference, gain):
    """Unit gain preserves the original operation; only the bank effect scales."""
    return normal+(gain-1)*(normal-reference)


def positive_gain_for_margin(margin, bank_projection, target=1.):
    if margin>=target:return 1.
    if bank_projection<=0:return None
    return 1.+(target-margin)/bank_projection


@torch.no_grad()
def trace(model, prepared):
    result=[];block=model.experts;gain=block.output_gain()
    for t in prepared['positions']:
        raw=prepared['raw'][:,:t];mask=prepared['mask'][:,:t];source=prepared['source'][:,:t]
        on,state,votes=block(raw,mask,memory_hidden=source)
        off,zero,zvotes=block(raw,mask,memory_hidden=source,memory_enabled=False)
        delta=bounded_delta(block,state,votes[:,-1]);reference=bounded_delta(block,zero,zvotes[:,-1])
        if not torch.equal(on[:,-1],raw[:,-1]+(delta*gain).to(raw.dtype)):
            raise RuntimeError('Normal readout reconstruction differs')
        if not torch.equal(off[:,-1],raw[:,-1]+(reference*gain).to(raw.dtype)):
            raise RuntimeError('Zero-bank reconstruction differs')
        result.append(dict(raw=raw[0,-1],normal=delta[0],zero=reference[0],
            label=prepared['item']['input_ids'][t]))
    return result


@torch.no_grad()
def logp(model, traces, mode, gain):
    total=0.
    for item in traces:
        delta=item['normal']*gain if mode=='uniform' else mix_delta(item['normal'],item['zero'],gain)
        h=item['raw']+delta.to(item['raw'].dtype)
        logits=model.base.lm_head(h[None]).float()
        total+=logits.log_softmax(-1)[0,item['label']].item()
    return total


@torch.no_grad()
def main():
    from transformers import AutoTokenizer
    from safetensors.torch import load_file,save_file
    parser=argparse.ArgumentParser()
    for n in ('model','states','cases','revisits','output'):
        parser.add_argument('--'+n,type=Path,required=True)
    parser.add_argument('--device',default='cuda');a=parser.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    def write(n,x):(a.output/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
    model=ExpertQwen.from_pretrained(a.model,dtype='auto',attn_implementation='sdpa').to(a.device).eval()
    tokenizer=AutoTokenizer.from_pretrained(a.model);body=tensor_hashes(model.base)
    parent_block=model.experts;gain_block=with_gain(parent_block)
    specs=json.loads(a.states.read_text());data=json.loads(a.cases.read_text())
    revisits=json.loads(a.revisits.read_text());all_records=[];features={}
    replies=(a.output/'replies.jsonl').open('x');count=0
    for spec in specs:
        name=spec['name'];path=a.states.parent/spec['file']
        if digest(path)!=spec['sha256']:raise RuntimeError('Source weights hash differs')
        model.experts=parent_block if spec['format']==2 else gain_block
        model.expert_config=model.experts.config
        model.experts.load_state_dict(load_file(str(path),device=a.device),strict=True)
        model.requires_grad_(False);frozen=tensor_hashes(model.experts)
        for split in ('train','validation'):
            items,audit=prepare_transfer_pairs(model,tokenizer,data[split],a.device)
            write('prefix-'+name+'-'+split+'.json',audit)
            for item in items:
                alternatives=[trace(model,p) for p in item['alternatives']]
                left,right=alternatives
                # All paired names diverge at their first scored token here.
                if left[0]['label']==right[0]['label']:
                    raise ValueError('Audit requires names diverging at the first name token')
                x=left[0];ids=[left[0]['label'],right[0]['label']]
                rows=model.base.lm_head.weight[ids].float();direction=rows[0]-rows[1]
                bank=x['normal']-x['zero']
                margin=torch.dot(x['raw'].float()+x['normal'],direction).item()
                projection=torch.dot(bank,direction).item()
                trials=[]
                for mode in ('uniform','bank'):
                    for gain in (1.,2.,4.,8.,16.,32.,64.):
                        scores=[logp(model,t,mode,gain) for t in alternatives]
                        delta=x['normal']*gain if mode=='uniform' else mix_delta(x['normal'],x['zero'],gain)
                        logits=model.base.lm_head((x['raw']+delta.to(x['raw'].dtype))[None]).float()[0,ids]
                        trials.append(dict(mode=mode,gain=gain,correct_logp=scores[0],
                            wrong_logp=scores[1],name_margin=scores[0]-scores[1],
                            first_margin=(logits[0]-logits[1]).item(),
                            delta_rms=delta.square().mean().sqrt().item()))
                # Bank replacement holds the actual working prefix fixed.
                p=item['alternatives'][0];state=prefix_state(model,p)
                donor=swapped_state(state,item['bank']);t=p['positions'][0];cut=p['cut']
                _,dstate,dvotes=model.experts(p['raw'][:,cut:t],p['mask'][:,cut:t],donor,
                    memory_hidden=p['source'][:,cut:t])
                donated=bounded_delta(model.experts,dstate,dvotes[:,-1])[0]
                rec=dict(arm=name,split=split,id=item['id'],pair=item['pair'],
                    actual_gain=model.experts.output_gain().item(),
                    normal_rms=x['normal'].square().mean().sqrt().item(),
                    zero_rms=x['zero'].square().mean().sqrt().item(),
                    bank_component_rms=bank.square().mean().sqrt().item(),
                    source_swap_rms=(x['normal']-donated).square().mean().sqrt().item(),
                    unit_first_margin_fp32=margin,bank_projection_fp32=projection,
                    gain_for_margin_one=positive_gain_for_margin(margin,projection),trials=trials)
                all_records.append(rec)
                key=f'{name}.{split}.{len(all_records)}'
                for label,tensor in [('raw',x['raw']),('normal',x['normal']),('zero',x['zero']),('donor',donated),('rows',rows)]:
                    features[key+'.'+label]=tensor.cpu().contiguous()
                print(json.dumps({k:v for k,v in rec.items() if k!='trials'}),flush=True)
        if name=='learned32':
            for case in revisits:
                base={k:case[k] for k in ('history','question','language','seed','owner','keeper')}
                options=[('replay',base),('named',{**base,'question':case['named_question']}),
                    ('followup',{**base,'history':base['history']+[
                        dict(role='user',content=base['question']),dict(role='assistant',content=case['previous_text'])],
                        'question':case['followup_question'],'seed':base['seed']+1})]
                for intervention,row in options:
                    answer=generate(model,tokenizer,row,seed=row['seed'],mode='experts',max_new_tokens=192)
                    exact=answer['token_ids']==case['previous_token_ids'] if intervention=='replay' else None
                    if exact is False:raise RuntimeError('Historical answer failed exact replay')
                    record=dict(id=case['id'],intervention=intervention,previous_code=case['previous_code'],
                        previous_text=case['previous_text'],replay_exact=exact,**row,**answer)
                    replies.write(json.dumps(record,ensure_ascii=False)+'\n');replies.flush();count+=1
                    print(json.dumps(dict(reply=count,id=case['id'],intervention=intervention)),flush=True)
        if tensor_hashes(model.experts)!=frozen:raise RuntimeError('Frozen experts changed')
    replies.close();write('components.json',all_records)
    save_file(features,str(a.output/'features.safetensors'))
    if tensor_hashes(model.base)!=body:raise RuntimeError('Frozen body changed')
    write('completed.json',dict(optimizer_steps=0,positions=len(all_records),replies=count,
        frozen_body_tensors=len(body),source_states=specs,primary_promoted=False))


if __name__=='__main__':main()
