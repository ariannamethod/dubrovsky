"""Upper bounds for Dubrovsky's mixture-before-clipping output, in FP64.

The four output row spaces must be joined before projecting: cancellation between
expert proposals can expose a direction absent from each separately clipped one.
These are bounds on the real-valued correction; native rounding is separate.
"""
import math
import torch


def output_basis(output):
    matrix = output.detach().double().flatten(0, 1)
    _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    threshold = torch.finfo(matrix.dtype).eps * max(matrix.shape) * singular.max()
    return vh[singular > threshold]


def correction_bound(output, direction, max_rms, gain, basis=None):
    if output.ndim != 3 or direction.shape != output.shape[-1:]:
        raise ValueError('Expected [expert, rank, width] and a head difference')
    if not (math.isfinite(gain) and gain > 0 and math.isfinite(max_rms) and max_rms > 0):
        raise ValueError('Positive finite gain and RMS bound required')
    o, w = output.detach().double(), direction.detach().double()
    basis = output_basis(o) if basis is None else basis
    # For z_e in [-1,1]^r and softmax a_e, |w . sum a_e z_e O_e|
    # <= max_e ||O_e w||_1. Radial clipping can only reduce this bound.
    cube = torch.einsum('erd,d->er', o, w).abs().sum(-1).max()
    radius = max_rms * math.sqrt(o.shape[-1])
    subspace = radius * (basis @ w).norm()
    return dict(cube=float(gain*cube), subspace=float(gain*subspace),
                bound=float(gain*torch.minimum(cube, subspace)), rank=basis.shape[0])


def bf16_rounding_envelope(raw, rows, radius):
    """Conservative round-to-nearest envelope for two casts plus head rounding.

    Assumes normal BF16 values, FP32 dot accumulation and no overflow. An extra
    FP32 summation allowance covers either ordering of the head dot products.
    This envelope uses the full radius, so it also covers the subspace bound.
    """
    raw, rows = raw.double(), rows.double()
    w = rows[0]-rows[1]; u = 2.**-8
    # bf16(raw + bf16(delta)) versus raw+delta, ||delta||_2 <= radius.
    coordinate = u*raw.abs()
    norm_error = (2*u+u*u)*radius
    addition = coordinate @ w.abs() + norm_error*w.norm()
    row_error = (rows.abs() @ coordinate) + norm_error*rows.norm(dim=-1)
    maximum_logits = (rows @ raw).abs() + radius*rows.norm(dim=-1) + row_error
    head_cast = u*maximum_logits.sum()
    gamma = raw.numel()*2.**-24/(1-raw.numel()*2.**-24)
    accumulation = gamma*((rows.abs() @ raw.abs()).sum()
                          +(1+2*u+u*u)*radius*rows.norm(dim=-1).sum()
                          +(rows.abs() @ coordinate).sum())
    return float(addition+head_cast+accumulation+1e-5)
