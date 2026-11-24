"""Matched continuation of stage 28: audit actual Adam steps, then vary one loss weight.

Private inputs and optimizer state are supplied by the operator. All transient
diagnostic steps are rolled back. Only the paired bank coefficient differs
between the two training arms; the selected release is never changed here.
"""
import argparse
import copy
import json
from pathlib import Path
import random
import time

import torch

from embedded_experts import ExpertQwen, tensor_hashes
from expert_data import encode_expert_row, encode_expert_prompt
from probe_expert_readout import digest, ids_digest, intervene, question_bounds
from probe_role_gradients import gradient_vector, gradient_summary, cosine
from train_role_readout import (TRAINABLE, configure, prepare_pairs, prefix_state,
    swapped_state, value_logp, causal_loss, evaluate)
from train_experts import row_loss
from sample_experts import generate


def components(model, items, replay, head32):
    ce = 0
    contrast = 0
    for item in items:
        correct, wrong = item['alternatives']
        state = prefix_state(model, correct)
        if not torch.equal(state.user, item['normal_bank']):
            raise RuntimeError('Frozen user bank changed')
        donor = swapped_state(state, item['bank'])
        ce = ce - value_logp(model, correct, state).mean() / len(items)
        normal = sum(sign * value_logp(model, p, state, precision='add32',
                     head32=head32).sum() for sign, p in [(1, correct), (-1, wrong)])
        swapped = sum(sign * value_logp(model, p, donor, precision='add32',
                      head32=head32).sum() for sign, p in [(1, correct), (-1, wrong)])
        contrast = contrast + causal_loss(normal, swapped) / len(items)
    replay_loss, _, _ = row_loss(model, replay, kl_weight=.1)
    return dict(ce=ce, contrast=contrast, replay=replay_loss)


def values(losses):
    return {k: v.item() for k, v in losses.items()}


def flatten(parameters):
    return torch.cat([p.detach().flatten().clone() for p in parameters])


def assign_gradient(parameters, vector):
    if vector.numel() != sum(p.numel() for p in parameters):
        raise ValueError('Gradient layout differs')
    offset = 0
    for p in parameters:
        p.grad = vector[offset:offset+p.numel()].reshape_as(p).clone()
        offset += p.numel()


def actual_step(optimizer, parameters, vector, contrast_gradient):
    """Record the real clipped AdamW displacement, including inherited moments."""
    before = flatten(parameters)
    optimizer.zero_grad(set_to_none=True)
    assign_gradient(parameters, vector)
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    optimizer.step()
    displacement = flatten(parameters) - before
    if not torch.isfinite(displacement).all():
        raise RuntimeError('Nonfinite parameter displacement')
    return dict(preclip_norm=norm.item(), clip_scale=min(1., 1./(norm.item()+1e-6)),
        displacement_norm=displacement.norm().item(),
        cosine_descent_contrast=cosine(-displacement, contrast_gradient),
        first_order_contrast_change=torch.dot(displacement, contrast_gradient).item())


def restore(block, optimizer, state, optimizer_state):
    block.load_state_dict(state, strict=True)
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    optimizer.zero_grad(set_to_none=True)


@torch.no_grad()
def readout_signals(model, items):
    """Pre-cast bounded residual at the first value position; full recurrent prefix."""
    result = []
    for item in items:
        p = item['alternatives'][0]
        end = p['positions'][0]
        raw = p['raw'][:, :end].float()
        corrected, _, _ = model.experts(raw, p['mask'][:, :end],
                                       memory_hidden=p['source'][:, :end])
        result.append((corrected[0, -1] - raw[0, -1]).cpu())
    return torch.stack(result)


def main():
    from transformers import AutoTokenizer
    from safetensors.torch import load_file, save_file
    parser = argparse.ArgumentParser()
    for name in ('model', 'start', 'optimizer', 'cases', 'replay', 'batches',
                 'acceptance', 'output', 'gate'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (args.output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    def emit(handle, record):
        handle.write(json.dumps(record, ensure_ascii=False)+'\n'); handle.flush()
        print(json.dumps({k: v for k, v in record.items() if k in
              ('event', 'arm', 'step', 'pair', 'case', 'turn', 'loss')}), flush=True)
    data = json.loads(args.cases.read_text())
    replay = json.loads(args.replay.read_text())
    batches = json.loads(args.batches.read_text())
    acceptance = json.loads(args.acceptance.read_text())
    assert len(data['train']) == 8 and len(data['validation']) == 4 and len(replay) == 8
    assert all(r['language'] == 'en' for r in replay)
    model = ExpertQwen.from_pretrained(args.model, dtype='auto',
                                     attn_implementation='sdpa').to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    body_hashes = tensor_hashes(model.base)
    native_dtype = model.base.lm_head.weight.dtype
    start = load_file(str(args.start), device=args.device)
    model.experts.load_state_dict(start, strict=True)
    optimizer = configure(model.experts)
    checkpoint = torch.load(args.optimizer, map_location='cpu', weights_only=False)
    assert checkpoint['step'] == 24
    restored_optimizer = copy.deepcopy(checkpoint['optimizer'])
    optimizer.load_state_dict(copy.deepcopy(restored_optimizer))
    assert [g['lr'] for g in optimizer.param_groups] == [1e-5, 1e-3]
    assert all(g['weight_decay'] == 0 for g in optimizer.param_groups)
    head32 = model.base.lm_head.weight.detach().float().clone()
    named = [(n, p) for n, p in model.experts.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named]
    sizes = [(n, p.numel()) for n, p in named]
    train = prepare_pairs(model, tokenizer, data['train'], args.device)
    encoded = {r['id']: encode_expert_row(tokenizer, r, 3072) for r in replay}
    write('run.json', dict(stage=30, start_sha256=digest(args.start),
        optimizer_sha256=digest(args.optimizer), source_step=24, steps_per_arm=24,
        coefficients=[1, 12], replay_weight=.2, replay_kl=.1, contrast_target=.05,
        coefficient_basis='round stage29 train-only median CE/contrast norm 11.6266 to 12',
        trainable=[n for n, _ in named], trainable_parameters=sum(p.numel() for p in parameters),
        cases_sha256=digest(args.cases), replay_sha256=digest(args.replay),
        acceptance_sha256=digest(args.acceptance), native_dtype=str(native_dtype),
        body_sha256=digest(args.model/'model.safetensors'), primary_promoted=False))
    diagnostic = (args.output/'adam-audit.jsonl').open('x')
    for batch in batches:
        restore(model.experts, optimizer, start, restored_optimizer)
        items = [i for i in train if i['pair'] == batch['pair']]
        assert len(items) == 2
        losses = components(model, items, encoded[batch['replay_id']], head32)
        before = values(losses)
        vectors = {k: gradient_vector(v, parameters) for k, v in losses.items()}
        summary = gradient_summary(vectors, sizes)
        for coefficient in (1, 12):
            restore(model.experts, optimizer, start, restored_optimizer)
            vector = vectors['ce'] + coefficient*vectors['contrast'] + .2*vectors['replay']
            update = actual_step(optimizer, parameters, vector, vectors['contrast'])
            with torch.no_grad():
                after = values(components(model, items, encoded[batch['replay_id']], head32))
            emit(diagnostic, dict(event='adam-audit', pair=batch['pair'],
                coefficient=coefficient, before=before, after=after, gradients=summary,
                actual_contrast_change=after['contrast']-before['contrast'], **update))
    diagnostic.close()
    restore(model.experts, optimizer, start, restored_optimizer)
    assert all(torch.equal(v, start[k]) for k, v in model.experts.state_dict().items())
    write('audit-complete.json', dict(transient_steps_rolled_back=16,
        diagnostic_uses_train_only=True, experts_restored=True, optimizer_restored=True))
    print(json.dumps(dict(event='awaiting-audit-review')), flush=True)
    deadline = time.monotonic()+900
    while not args.gate.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Audit review was not received')
        time.sleep(2)
    assert json.loads(args.gate.read_text()) == dict(continue_training=True, coefficients=[1,12], steps=24)
    # Validation and fresh evaluation were not opened to select the coefficient.
    validation = prepare_pairs(model, tokenizer, data['validation'], args.device)
    matched = prepare_pairs(model, tokenizer, acceptance['matched'], args.device)
    base_train_signal = readout_signals(model, train)
    base_val_signal = readout_signals(model, validation)
    random.setstate(checkpoint['random_rng'])
    # Recover the stage-28 permutation before continuing its RNG state.
    original = random.Random(20261117); order = list(range(8))
    for _ in range(3): original.shuffle(order)
    assert original.getstate() == checkpoint['random_rng']
    schedule = []
    for step in range(24):
        if step % 8 == 0: random.shuffle(order)
        schedule.append(dict(pair=data['train'][order[step%8]]['id'], replay_id=replay[step%8]['id']))
    write('schedule.json', schedule)
    metrics = (args.output/'metrics.jsonl').open('x')
    candidates = {}
    def save(arm, step):
        target = args.output/arm/f'step-{step:06d}'; target.mkdir(parents=True)
        save_file({k: v.detach().cpu().contiguous() for k,v in model.experts.state_dict().items()},
                  str(target/'experts.safetensors'))
        torch.save(dict(optimizer=optimizer.state_dict(),step=step,source_step=24,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            random_rng=random.getstate()),target/'optimizer.pt')
        return target
    with torch.no_grad():
        initial_eval = evaluate(model, validation, head32, detailed=True)
    emit(metrics, dict(event='validation', arm='start24', step=0, **initial_eval))
    for coefficient in (1, 12):
        arm = f'lambda{coefficient}'
        restore(model.experts, optimizer, start, restored_optimizer)
        torch.set_rng_state(checkpoint['torch_rng'])
        if checkpoint['cuda_rng']: torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        save(arm, 0)
        for step, batch in enumerate(schedule, 1):
            items = [i for i in train if i['pair'] == batch['pair']]
            losses = components(model, items, encoded[batch['replay_id']], head32)
            loss_values = values(losses)
            vectors = {k: gradient_vector(v, parameters) for k,v in losses.items()}
            update = actual_step(optimizer, parameters,
                vectors['ce']+coefficient*vectors['contrast']+.2*vectors['replay'], vectors['contrast'])
            assert all(torch.equal(v,start[k]) for k,v in model.experts.state_dict().items() if k not in TRAINABLE)
            emit(metrics, dict(event='train',arm=arm,step=step,**batch,objectives=loss_values,**update))
            if step%4 == 0:
                result = evaluate(model,validation,head32,detailed=True)
                save(arm,step)
                emit(metrics,dict(event='validation',arm=arm,step=step,**result))
        candidates[arm] = {k:v.detach().clone() for k,v in model.experts.state_dict().items()}
        train_change = readout_signals(model,train)-base_train_signal
        val_change = readout_signals(model,validation)-base_val_signal
        mean = train_change.mean(0)
        energy = val_change.square().sum(-1).mean()
        write(arm+'-conditioning.json',dict(train_positions=len(train),validation_positions=len(validation),
            train_mean_squared_norm=mean.square().sum().item(),validation_mean_squared_norm=energy.item(),
            validation_energy_after_train_center=(val_change-mean).square().sum(-1).mean().item(),
            train_common_fraction=(mean.square().sum()/train_change.square().sum(-1).mean()).item(),
            definition='new pre-cast bounded correction relative to start24; means fit on train only'))
        save_file(dict(train_change=train_change,validation_change=val_change,train_mean=mean),
                  str(args.output/(arm+'-conditioning.safetensors')))
    metrics.close()
    # Both endpoints receive the same full reading, without choosing by validation NLL.
    replies = (args.output/'replies.jsonl').open('x')
    count = 0
    arms = [('stage7',start,'base'),('start24',start,'experts')]+[(a,s,'experts') for a,s in candidates.items()]
    for arm,state,mode in arms:
        model.experts.load_state_dict(state,strict=True)
        for scene in acceptance['fresh']:
            history = []
            for turn,question in enumerate(scene['turns']):
                row = dict(language=scene['language'],history=list(history),question=question)
                answer = generate(model,tokenizer,row,seed=scene['seed']+turn,mode=mode,max_new_tokens=192)
                emit(replies,dict(event='reply',group='fresh',arm=arm,case=scene['id'],turn=turn,
                    owner=scene['owner'],keeper=scene['keeper'],**row,**answer)); count+=1
                history += [dict(role='user',content=question),dict(role='assistant',content=answer['text'])]
        for index,item in enumerate(matched):
            row = next(r for p in acceptance['matched'] for r in p['rows'] if r['id']==item['id'])
            prompt = encode_expert_prompt(tokenizer,row,3072)
            cut = question_bounds(torch.tensor([prompt['user_mask']],dtype=torch.bool))[1]
            interventions = ['normal','swapped-bank','zero-bank'] if arm=='lambda12' else ['normal']
            for intervention in interventions:
                options = {} if intervention=='normal' else dict(cut=cut,
                    bank=item['bank'] if intervention=='swapped-bank' else torch.zeros_like(item['bank']))
                with intervene(model,**options):
                    answer=generate(model,tokenizer,row,seed=921+index,mode=mode,max_new_tokens=192)
                emit(replies,dict(event='reply',group='matched',arm=arm,case=row['id'],turn=0,
                    intervention=intervention,prefix_sha256=ids_digest(prompt['input_ids']),**row,**answer));count+=1
    replies.close()
    assert tensor_hashes(model.base)==body_hashes
    assert model.base.lm_head.weight.dtype==native_dtype
    assert all(torch.equal(v,start[k]) for k,v in model.experts.state_dict().items() if k not in TRAINABLE)
    write('completed.json',dict(training_steps=48,steps_per_arm=24,diagnostic_steps_rolled_back=16,
        replies=count,body_tensors_preserved=len(body_hashes),writer_preserved=True,
        production_precision_changed=False,primary_promoted=False,endpoint_step=24))


if __name__=='__main__':
    main()
