"""Differentiable isolated user writes with a matched detached working prefix.

Inference still uses EmbeddedExperts. This training replay omits non-writing
positions, whose bank transition is the identity, and keeps all user transitions.
"""
from dataclasses import replace
import math
import torch
from train_role_readout import TRAINABLE, configure, prefix_state, evaluate

WRITER = frozenset(('source', 'retain', 'write', 'retain_bias', 'write_bias'))


def configure_writer(block, train_writer, writer_lr=1e-5):
    optimizer = configure(block)
    if train_writer:
        parameters = []
        for name, p in block.named_parameters():
            if name in WRITER:
                p.requires_grad_(True)
                parameters.append(p)
        optimizer.add_param_group(dict(params=parameters, lr=writer_lr))
    return optimizer


def replay_bank(block, source, mask):
    """Batch-one native writer; source is frozen, the recurrent bank is not."""
    if block.config.format_version < 2:
        raise ValueError('An isolated user writer is required')
    if source.ndim != 3 or source.shape[0] != 1 or mask.shape != source.shape[:2] or mask.dtype != torch.bool:
        raise ValueError('Expected batch-one features and boolean write mask')
    bank = block.initial_state(1).user
    for pos in mask[0].nonzero().flatten().tolist():
        x = source[:, pos].detach().to(block.source.dtype)
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        fresh = torch.tanh(torch.einsum('bd,edr->ber', x, block.source))
        features = torch.cat((x[:, None].expand(-1, block.config.experts, -1),
                              bank, fresh, fresh-bank), dim=-1)
        retain = torch.sigmoid((features*block.retain).sum(-1)+block.retain_bias)
        write = torch.sigmoid((features*block.write).sum(-1)+block.write_bias)
        bank = torch.tanh(retain[..., None]*bank+write[..., None]*fresh)
    return bank


def attach_donors(items):
    for item in items:
        peers = [p for p in items if p['pair'] == item['pair'] and p['id'] != item['id']]
        if len(peers) != 1:
            raise ValueError('Expected exactly one counterfactual source')
        item['donor_prepared'] = peers[0]['alternatives'][0]


def current_states(model, item):
    prepared = item['alternatives'][0]
    state = prefix_state(model, prepared)
    banks = []
    for p in (prepared, item['donor_prepared']):
        banks.append(replay_bank(model.experts, p['source'][:, :p['cut']], p['mask'][:, :p['cut']]))
    if not torch.equal(state.user, banks[0]):
        raise RuntimeError('Differentiable replay changed the native user bank')
    # replace preserves the bank's autograd graph; working is common and detached.
    return tuple(replace(state, user=bank) for bank in banks)


@torch.no_grad()
def evaluate_writer(model, items, head32, detailed=True):
    refreshed = []
    for item in items:
        own, donor = current_states(model, item)
        refreshed.append({**item, 'normal_bank': own.user, 'bank': donor.user})
    return evaluate(model, refreshed, head32, detailed=detailed)


def weak_side_changes(parent, candidate):
    """Full-name wrong-side progress, separately from opposite-source contrast."""
    p = {r['id']: r for r in parent['rows']}
    c = {r['id']: r for r in candidate['rows']}
    if p.keys() != c.keys():
        raise ValueError('Different case inventories')
    rows = []
    for key in p:
        a, b = p[key]['native'], c[key]['native']
        old = (a['margin'], a['swapped_margin']); new = (b['margin'], b['swapped_margin'])
        if not all(math.isfinite(v) for v in (*old, *new)):
            raise ValueError('Nonfinite source margin')
        c0, d0 = (old[0]+old[1])/2, (old[0]-old[1])/2
        c1, d1 = (new[0]+new[1])/2, (new[0]-new[1])/2
        rows.append(dict(id=key, pair=p[key]['pair'], old_weak=d0-abs(c0),
            new_weak=d1-abs(c1), weak_change=d1-abs(c1)-d0+abs(c0),
            contrast_change=d1-d0, center_change=c1-c0,
            own_repaired=old[0]<=0<new[0], own_lost=old[0]>0>=new[0],
            donor_repaired=old[1]>=0>new[1], donor_lost=old[1]<0<=new[1]))
    failed = [r for r in rows if r['old_weak'] <= 0]
    return dict(rows=rows, previously_failed=len(failed),
        weak_improved=sum(r['weak_change']>0 for r in failed),
        contrast_improved=sum(r['contrast_change']>0 for r in failed),
        **{name:sum(r[name] for r in rows) for name in ('own_repaired','own_lost','donor_repaired','donor_lost')})
