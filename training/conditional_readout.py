"""Paired source objectives and a gate requiring the same conditional repair."""
import torch
import torch.nn.functional as F


def source_pair_loss(own, donor, parent_own, parent_donor, *, centered):
    """Own-minus-other name margins at ONE actual working/Qwen prefix.

    The centered arm only trains the opposite source component. Its task loss
    has zero gradient along a shared shift of both margins. Both arms receive
    the same separate penalty for moving the source-pair centre.
    """
    parent_center = ((parent_own+parent_donor)/2).detach()
    center, contrast = (own+donor)/2, (own-donor)/2
    left, right = (parent_center+contrast, parent_center-contrast) if centered else (own, donor)
    task = (F.softplus(1-left)+F.softplus(1+right))/2
    preservation = (center-parent_center).square()
    return task, preservation


def conditional_gate(parent, candidate, gain_controls):
    """Native full-name decisions; same-pair actual and centred repair required."""
    def indexed(result): return {r['id']: r for r in result['rows']}
    p, c = indexed(parent), indexed(candidate)
    if set(p) != set(c): raise ValueError('Different source inventories')
    controls = [indexed(g) for g in gain_controls]
    if any(set(g)!=set(p) for g in controls): raise ValueError('Different gain inventory')
    def success(r):
        return r['native']['margin'] > 0 and r['native']['swapped_margin'] < 0
    pairs = {r['pair'] for r in p.values()}
    def complete(rows):
        return {pair for pair in pairs if all(success(r) for r in rows.values() if r['pair']==pair)}
    actual = complete(c)-complete(p)
    centered_rows = {}
    for key, r in c.items():
        old = p[key]['native']; new = r['native']
        center = (old['margin']+old['swapped_margin'])/2
        contrast = (new['margin']-new['swapped_margin'])/2
        centered_rows[key] = dict(pair=r['pair'],native=dict(margin=center+contrast,swapped_margin=center-contrast))
    conditional = complete(centered_rows)-complete(p)
    gain_repaired = set().union(*(complete(g)-complete(p) for g in controls))
    retained = all(c[k]['native']['margin']>0 for k in p if p[k]['native']['margin']>0)
    repaired = actual & conditional - gain_repaired
    return dict(pass_gate=bool(repaired) and retained, retained_parent_names=retained,
        actual_new_pairs=sorted(actual), conditional_new_pairs=sorted(conditional),
        gain_only_new_pairs=sorted(gain_repaired), same_pair_repairs=sorted(repaired))
