"""Measure objective gradients and natural role openings without optimizer steps."""
import argparse
import json
from pathlib import Path
import torch
from embedded_experts import ExpertQwen,tensor_hashes
from expert_data import encode_expert_row,encode_expert_prompt
from probe_expert_readout import prepare_hidden,question_bounds,intervene,digest,ids_digest
from train_role_readout import TRAINABLE,prepare_pairs,prefix_state,swapped_state,value_logp,causal_loss
from train_experts import row_loss
from sample_experts import generate


def gradient_vector(loss,parameters):
    gradients=torch.autograd.grad(loss,parameters,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                      for p,g in zip(parameters,gradients)])


def cosine(a,b):
    denominator=a.norm()*b.norm()
    return None if denominator.item()==0 else (torch.dot(a,b)/denominator).item()


def gradient_summary(vectors,sizes):
    weighted=dict(ce=vectors['ce'],contrast=vectors['contrast'],replay=.2*vectors['replay'])
    combined=sum(weighted.values())
    def summarize(v):
        offset=0;groups={}
        for name,size in sizes:
            groups[name]=v[offset:offset+size].norm().item();offset+=size
        if offset!=v.numel():raise ValueError('Gradient layout differs')
        return dict(norm=v.norm().item(),groups=groups)
    return dict(raw={k:summarize(v) for k,v in vectors.items()},
        weighted={k:summarize(v) for k,v in weighted.items()},combined=summarize(combined),
        cosine_ce_contrast=cosine(weighted['ce'],weighted['contrast']),
        cosine_combined_contrast=cosine(combined,weighted['contrast']),
        cosine_replay_contrast=cosine(weighted['replay'],weighted['contrast']))


def pair_gradients(model,items,replay,head32):
    ce=0;contrast=0
    for item in items:
        correct,wrong=item['alternatives'];state=prefix_state(model,correct)
        donor=swapped_state(state,item['bank'])
        ce=ce-value_logp(model,correct,state).mean()/len(items)
        a=value_logp(model,correct,state,precision='add32',head32=head32).sum()-value_logp(model,wrong,state,precision='add32',head32=head32).sum()
        b=value_logp(model,correct,donor,precision='add32',head32=head32).sum()-value_logp(model,wrong,donor,precision='add32',head32=head32).sum()
        contrast=contrast+causal_loss(a,b)/len(items)
    named=[(n,p) for n,p in model.experts.named_parameters() if p.requires_grad]
    parameters=[p for _,p in named]
    vectors=dict(ce=gradient_vector(ce,parameters),contrast=gradient_vector(contrast,parameters))
    style,_,_=row_loss(model,replay,kl_weight=.1)
    vectors['replay']=gradient_vector(style,parameters)
    return dict(loss=dict(ce=ce.item(),contrast=contrast.item(),replay=style.item()),
                **gradient_summary(vectors,[(n,p.numel()) for n,p in named]))


def openings(role):
    return dict(cue=role.title()+':',natural='The object belongs to' if role=='owner' else 'The object is now with',
        character='Dubrovsky checked the receipt. The object belongs to' if role=='owner' else 'Dubrovsky opened the cupboard. The object is now with')


def main():
    from transformers import AutoTokenizer
    from safetensors.torch import load_file
    p=argparse.ArgumentParser()
    for name in ('model','candidate','cases','replay','batches','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda');args=p.parse_args()
    if args.output.exists():raise ValueError('Preserve previous results')
    args.output.mkdir(parents=True)
    data=json.loads(args.cases.read_text());replay=json.loads(args.replay.read_text());batches=json.loads(args.batches.read_text())
    model=ExpertQwen.from_pretrained(args.model,dtype='auto',attn_implementation='sdpa').to(args.device).eval()
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    body=tensor_hashes(model.base);head32=model.base.lm_head.weight.detach().float().clone()
    states={'parent228':{k:v.clone() for k,v in model.experts.state_dict().items()},'readout24':load_file(str(args.candidate),device=args.device)}
    model.experts.requires_grad_(False)
    for n,param in model.experts.named_parameters():param.requires_grad_(n in TRAINABLE)
    encoded={r['id']:encode_expert_row(tokenizer,r,3072) for r in replay}
    cache=prepare_pairs(model,tokenizer,data['train'],args.device)
    gradient_file=(args.output/'gradients.jsonl').open('x');scores=(args.output/'openings.jsonl').open('x');replies=(args.output/'replies.jsonl').open('x')
    count=0;score_count=0
    for arm,state_dict in states.items():
        model.experts.load_state_dict(state_dict)
        for batch in batches:
            items=[i for i in cache if i['pair']==batch['pair']]
            assert len(items)==2
            record=dict(arm=arm,pair=batch['pair'],replay_id=batch['replay_id'],**pair_gradients(model,items,encoded[batch['replay_id']],head32))
            gradient_file.write(json.dumps(record)+'\n');gradient_file.flush()
            print(json.dumps(dict(event='gradient',arm=arm,pair=batch['pair'])),flush=True)
        with torch.no_grad():
            for pair_index,pair in enumerate(data['validation']):
                role=pair['role']
                for direction,row in enumerate(pair['rows']):
                    donor_row=pair['rows'][1-direction]
                    donor=prepare_hidden(model,tokenizer,donor_row,role.title()+': '+donor_row[role],args.device)
                    donor['cut']=question_bounds(donor['mask'])[1];bank=prefix_state(model,donor).user.clone()
                    for name,opening in openings(role).items():
                        alternatives=[]
                        for value in [row[role],donor_row[role]]:
                            prepared=prepare_hidden(model,tokenizer,row,opening+' '+value,args.device,value_offset=len(opening))
                            prepared['cut']=question_bounds(prepared['mask'])[1];alternatives.append(prepared)
                        normal=prefix_state(model,alternatives[0]);changed=swapped_state(normal,bank)
                        record=dict(arm=arm,id=row['id'],role=role,opening=name,text_prefix=opening,
                            values=[row[role],donor_row[role]],positions=[a['positions'] for a in alternatives],
                            token_ids=[[a['item']['input_ids'][j] for j in a['positions']] for a in alternatives])
                        for precision in ['native','head32','add32']:
                            own=[value_logp(model,a,normal,precision=precision,head32=head32) for a in alternatives]
                            swapped=[value_logp(model,a,changed,precision=precision,head32=head32) for a in alternatives]
                            margin=own[0].sum()-own[1].sum();other=swapped[0].sum()-swapped[1].sum()
                            record[precision]=dict(margin=margin.item(),swapped_margin=other.item(),causal_difference=(margin-other).item(),logps=[v.tolist() for v in own],swapped_logps=[v.tolist() for v in swapped])
                        scores.write(json.dumps(record)+'\n');scores.flush();score_count+=1
                    prompt=encode_expert_prompt(tokenizer,row,3072);cut=question_bounds(torch.tensor([prompt['user_mask']],dtype=torch.bool))[1]
                    for intervention in ['normal','swapped-bank']:
                        options={} if intervention=='normal' else dict(cut=cut,bank=bank)
                        with intervene(model,**options):answer=generate(model,tokenizer,row,seed=731+pair_index*2+direction,mode='experts',max_new_tokens=192)
                        record=dict(arm=arm,id=row['id'],intervention=intervention,role=role,owner=row['owner'],keeper=row['keeper'],donor_owner=donor_row['owner'],donor_keeper=donor_row['keeper'],history=row['history'],question=row['question'],prefix_sha256=ids_digest(prompt['input_ids']),**answer)
                        replies.write(json.dumps(record,ensure_ascii=False)+'\n');replies.flush();count+=1
                        print(json.dumps(dict(event='reply',arm=arm,id=row['id'],intervention=intervention)),flush=True)
        assert all(torch.equal(v,state_dict[k]) for k,v in model.experts.state_dict().items())
    assert tensor_hashes(model.base)==body
    for f in (gradient_file,scores,replies):f.close()
    (args.output/'completed.json').write_text(json.dumps(dict(optimizer_steps=0,gradient_records=16,opening_records=score_count,replies=count,body_tensors_preserved=len(body),both_expert_checkpoints_preserved=True,candidate_sha256=digest(args.candidate),primary_promoted=False),indent=2)+'\n')


if __name__=='__main__':main()
