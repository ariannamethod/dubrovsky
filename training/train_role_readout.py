"""Short English role-readout pilot with a frozen body and bank writer.

Private paired sources, authored replay and acceptance scenes are supplied as
files. Production inference remains native BF16. FP32 is used for a separately
recorded causal auxiliary objective and precision diagnostics.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from embedded_experts import ExpertQwen, tensor_hashes
from expert_data import encode_expert_prompt, encode_expert_row
from probe_expert_readout import prepare_hidden, question_bounds, digest, ids_digest
from sample_experts import generate
from train_experts import row_loss

TRAINABLE = frozenset(('memory_read', 'router_memory', 'output'))


def configure(block, reader_lr=1e-5, output_lr=1e-3):
    """Freeze the writer and all other internals; assign explicit parameter groups."""
    block.requires_grad_(False)
    for name, parameter in block.named_parameters():
        parameter.requires_grad_(name in TRAINABLE)
    actual = {n for n, p in block.named_parameters() if p.requires_grad}
    if actual != TRAINABLE:
        raise ValueError('Unexpected expert parameter names')
    return torch.optim.AdamW([
        dict(params=[block.memory_read, block.router_memory], lr=reader_lr),
        dict(params=[block.output], lr=output_lr)], weight_decay=0)


def swapped_state(state, bank):
    """Change the user bank, preserving the same detached working prefix."""
    if bank.shape != state.user.shape or bank.dtype != state.user.dtype:
        raise ValueError('Incompatible bank')
    if bank.device != state.user.device:
        raise ValueError('Incompatible device')
    return replace(state.detached(), user=bank.detach().clone())


def causal_loss(normal_margin, swapped_margin, target=0.05):
    return F.softplus(target - (normal_margin - swapped_margin))


def value_logp(model, prepared, state, *, precision='native', head32=None):
    """Only the post-question suffix receives gradients; score actual values."""
    if precision not in ('native', 'head32', 'add32'):
        raise ValueError('Unknown projection precision')
    cut = prepared['cut']
    end = max(prepared['positions'])
    raw = prepared['raw'][:, cut:end]
    if precision == 'add32':
        raw = raw.float()
    corrected, _, _ = model.experts(raw, prepared['mask'][:, cut:end],
        state, memory_hidden=prepared['source'][:, cut:end])
    positions = torch.tensor(prepared['positions'], device=raw.device)
    h = corrected[0, positions - 1 - cut]
    logits = model.base.lm_head(h).float() if precision == 'native' else F.linear(h.float(), head32)
    labels = torch.tensor(prepared['item']['input_ids'], device=raw.device)[positions]
    return logits.log_softmax(-1).gather(1, labels[:, None]).flatten()


@torch.no_grad()
def prefix_state(model, prepared):
    cut = prepared['cut']
    _, state, _ = model.experts(prepared['raw'][:, :cut], prepared['mask'][:, :cut],
        memory_hidden=prepared['source'][:, :cut])
    return state.detached()


def prepare_pairs(model, tokenizer, pairs, device):
    result = []
    for pair in pairs:
        if pair['role'] not in ('owner', 'keeper') or len(pair['rows']) != 2:
            raise ValueError('Expected two counterfactual sources for one role')
        role = pair['role']
        if any(r['language'] != 'en' for r in pair['rows']):
            raise ValueError('This training pilot is English-only')
        if pair['rows'][0][role] == pair['rows'][1][role]:
            raise ValueError('Counterfactual value did not change')
        for i, row in enumerate(pair['rows']):
            other = pair['rows'][1-i]
            alternatives = []
            for name in (row[role], other[role]):
                p = prepare_hidden(model, tokenizer, row, role.title()+': '+name, device)
                p['cut'] = question_bounds(p['mask'])[1]
                if min(p['positions']) <= p['cut']:
                    raise ValueError('Value overlaps the user prefix')
                alternatives.append(p)
            a, b = alternatives
            if a['cut'] != b['cut'] or not torch.equal(a['raw'][:, :a['cut']], b['raw'][:, :b['cut']]):
                raise ValueError('Alternative answers changed the causal Qwen prefix')
            donor = prepare_hidden(model, tokenizer, other, role.title()+': '+other[role], device)
            donor['cut'] = question_bounds(donor['mask'])[1]
            result.append(dict(id=row['id'], pair=pair['id'], role=role,
                alternatives=alternatives, bank=prefix_state(model, donor).user.clone(),
                normal_bank=prefix_state(model, a).user.clone(),
                source_sha256=ids_digest(a['item']['memory_input_ids']),
                donor_sha256=ids_digest(donor['item']['memory_input_ids'])))
    return result


def row_objective(model, item, head32):
    correct, wrong = item['alternatives']
    state = prefix_state(model, correct)
    if not torch.equal(state.user, item['normal_bank']):
        raise RuntimeError('Frozen writer changed its bank')
    donor = swapped_state(state, item['bank'])
    ce = -value_logp(model, correct, state).mean()
    normal = (value_logp(model, correct, state, precision='add32', head32=head32).sum()
              - value_logp(model, wrong, state, precision='add32', head32=head32).sum())
    swapped = (value_logp(model, correct, donor, precision='add32', head32=head32).sum()
               - value_logp(model, wrong, donor, precision='add32', head32=head32).sum())
    contrast = causal_loss(normal, swapped)
    return ce + contrast, dict(ce=ce.item(), contrast=contrast.item(),
        normal_margin=normal.item(), swapped_margin=swapped.item(),
        causal_difference=(normal-swapped).item())


@torch.no_grad()
def evaluate(model, items, head32, *, detailed=False):
    records = []
    for item in items:
        correct, wrong = item['alternatives']
        state = prefix_state(model, correct)
        if not torch.equal(state.user, item['normal_bank']):
            raise RuntimeError('Frozen writer changed')
        donor = swapped_state(state, item['bank'])
        record = dict(id=item['id'], pair=item['pair'], role=item['role'],
            source_sha256=item['source_sha256'], donor_sha256=item['donor_sha256'])
        for precision in ('native', 'head32', 'add32') if detailed else ('native', 'add32'):
            own = [value_logp(model,p,state,precision=precision,head32=head32) for p in (correct,wrong)]
            other = [value_logp(model,p,donor,precision=precision,head32=head32) for p in (correct,wrong)]
            margin = own[0].sum()-own[1].sum()
            swapped = other[0].sum()-other[1].sum()
            record[precision] = dict(ce=-own[0].mean().item(),margin=margin.item(),
                swapped_margin=swapped.item(),causal_difference=(margin-swapped).item(),
                values=[v.tolist() for v in own],swapped_values=[v.tolist() for v in other])
        records.append(record)
    return dict(loss=sum(r['native']['ce'] for r in records)/len(records),
        native_correct=sum(r['native']['margin']>0 for r in records),
        native_causal_direction=sum(r['native']['causal_difference']>0 for r in records),
        float_causal_direction=sum(r['add32']['causal_difference']>0 for r in records),
        rows=records)


def main():
    from transformers import AutoTokenizer
    from safetensors.torch import save_file, load_file
    parser = argparse.ArgumentParser()
    for name in ('model','cases','replay','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output directory')
    data = json.loads(args.cases.read_text())
    replay = json.loads(args.replay.read_text())
    if len(data['train'])!=8 or len(data['validation'])!=4 or len(replay)!=8:
        raise ValueError('Sealed inventory differs')
    if any(r['language']!='en' for r in replay):
        raise ValueError('Replay must be English')
    train_ids={p['id'] for p in data['train']}
    if train_ids & {p['id'] for p in data['validation']}:
        raise ValueError('Training and validation pair IDs overlap')
    args.output.mkdir(parents=True)
    def write(name,value):
        (args.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    random.seed(20261117); torch.manual_seed(20261117)
    model=ExpertQwen.from_pretrained(args.model,dtype='auto',attn_implementation='sdpa').to(args.device).eval()
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    initial_body=tensor_hashes(model.base)
    initial_experts={n:p.detach().clone() for n,p in model.experts.state_dict().items()}
    optimizer=configure(model.experts)
    head32=model.base.lm_head.weight.detach().float().clone()
    head_dtype=model.base.lm_head.weight.dtype
    train=prepare_pairs(model,tokenizer,data['train'],args.device)
    validation=prepare_pairs(model,tokenizer,data['validation'],args.device)
    replay_encoded=[encode_expert_row(tokenizer,r,3072) for r in replay]
    write('run.json',dict(stage=28,steps=24,seed=20261117,reader_lr=1e-5,output_lr=1e-3,
        replay_weight=.2,replay_base_kl=.1,contrast_weight=1,contrast_target=.05,
        native_dtype=str(head_dtype),trainable=sorted(TRAINABLE),
        trainable_parameters=sum(p.numel() for p in model.experts.parameters() if p.requires_grad),
        frozen_body_tensors=len(initial_body),cases_sha256=digest(args.cases),replay_sha256=digest(args.replay),
        parent_experts_sha256=digest(args.model/'experts.safetensors'),
        body_sha256=digest(args.model/'model.safetensors'),primary_language='en',production_precision_changed=False))
    write('token-audit.json',dict(train=[dict(id=x['id'],lengths=[len(p['item']['input_ids']) for p in x['alternatives']],positions=[p['positions'] for p in x['alternatives']]) for x in train],validation=[dict(id=x['id'],positions=[p['positions'] for p in x['alternatives']]) for x in validation],replay=[dict(id=r['id'],tokens=len(e['input_ids'])) for r,e in zip(replay,replay_encoded)]))
    metrics=(args.output/'metrics.jsonl').open('x')
    def emit(record):
        metrics.write(json.dumps(record)+'\n');metrics.flush()
        print(json.dumps({k:v for k,v in record.items() if k!='rows'}),flush=True)
    def save(step):
        path=args.output/f'step-{step:06d}';path.mkdir()
        save_file({k:v.detach().cpu().contiguous() for k,v in model.experts.state_dict().items()},str(path/'experts.safetensors'))
        torch.save(dict(optimizer=optimizer.state_dict(),step=step,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],random_rng=random.getstate()),path/'optimizer.pt')
        return path
    baseline=evaluate(model,validation,head32,detailed=True)
    emit(dict(event='validation',step=0,**baseline));save(0)
    order=list(range(8));best_loss=float('inf');best_step=None
    for step in range(1,25):
        if (step-1)%8==0: random.shuffle(order)
        index=order[(step-1)%8]
        optimizer.zero_grad(set_to_none=True)
        stats=[]
        for item in train[2*index:2*index+2]:
            loss,record=row_objective(model,item,head32)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite role loss')
            (loss/2).backward();stats.append(record)
        replay_loss,_,_=row_loss(model,replay_encoded[(step-1)%8],kl_weight=.1)
        (.2*replay_loss).backward()
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.experts.parameters() if p.requires_grad],1.0,error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.experts.parameters()):
            raise RuntimeError('Nonfinite expert weights')
        if any(not torch.equal(p,initial_experts[n]) for n,p in model.experts.state_dict().items() if n not in TRAINABLE):
            raise RuntimeError('Frozen expert parameter changed')
        emit(dict(event='train',step=step,pair=data['train'][index]['id'],replay_id=replay[(step-1)%8]['id'],replay_loss=replay_loss.item(),gradient_norm=norm.item(),roles=stats))
        if step%4==0:
            result=evaluate(model,validation,head32,detailed=True)
            improved=result['loss']<best_loss
            if improved: best_loss,best_step=result['loss'],step
            save(step) # every assessed point, including all new best and early points
            emit(dict(event='validation',step=step,best_trained=improved,**result))
    candidate=load_file(str(args.output/f'step-{best_step:06d}'/'experts.safetensors'),device=args.device)
    write('candidate.json',dict(step=best_step,criterion='lowest native role-value validation NLL among trained points',loss=best_loss,parent_loss=baseline['loss'],primary_promoted=False))
    replies=(args.output/'replies.jsonl').open('x')
    count=0
    for arm,state,mode in [('stage7',initial_experts,'base'),('parent228',initial_experts,'experts'),(f'readout{best_step}',candidate,'experts')]:
        model.experts.load_state_dict(state,strict=True)
        for scene in data['fresh']:
            history=[]
            for turn,question in enumerate(scene['turns']):
                row=dict(language=scene['language'],question=question,history=list(history))
                answer=generate(model,tokenizer,row,seed=scene['seed']+turn,mode=mode,max_new_tokens=192)
                record=dict(group='fresh',case=scene['id'],arm=arm,turn=turn,seed=scene['seed'],owner=scene['owner'],keeper=scene['keeper'],**row,**answer)
                replies.write(json.dumps(record,ensure_ascii=False)+'\n');replies.flush();count+=1
                print(json.dumps(dict(event='reply',case=scene['id'],arm=arm,turn=turn,eos=answer['eos_reached'])),flush=True)
                history.extend([dict(role='user',content=question),dict(role='assistant',content=answer['text'])])
        for case in data['matched']:
            row={k:case[k] for k in ('language','history','question')}
            prompt=encode_expert_prompt(tokenizer,row,3072)
            answer=generate(model,tokenizer,row,seed=case['seed']+case['turn'],mode=mode,max_new_tokens=192)
            replies.write(json.dumps(dict(group='matched',case=case['id'],arm=arm,turn=case['turn'],seed=case['seed'],prefix_sha256=ids_digest(prompt['input_ids']),**row,**answer),ensure_ascii=False)+'\n');replies.flush();count+=1
            print(json.dumps(dict(event='matched',case=case['id'],arm=arm)),flush=True)
    replies.close();metrics.close()
    if tensor_hashes(model.base)!=initial_body or model.base.lm_head.weight.dtype!=head_dtype:
        raise RuntimeError('Frozen body or native head dtype changed')
    write('completed.json',dict(optimizer_steps=24,replies=count,best_step=best_step,body_tensors_preserved=len(initial_body),writer_preserved=True,primary_promoted=False,production_precision_changed=False))


if __name__=='__main__':
    main()
