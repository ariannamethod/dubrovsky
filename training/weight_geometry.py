"""CPU-only geometry of a published sparse weight edit and a target checkpoint.

Inputs are raw BF16 bit arrays. No model inference or behavioral scoring occurs.
The leading left singular vector estimates the direction of a rounded rank-one
edit. Projection uses the target's own column coefficients.
"""
import numpy as np


def bf16_float(bits):
    bits = np.asarray(bits)
    if bits.dtype.kind != 'u' or bits.dtype.itemsize != 2:
        raise ValueError('Expected uint16 BF16 storage bits')
    value = (bits.astype(np.uint32) << 16).view(np.float32)
    if not np.isfinite(value).all():
        raise ValueError('Non-finite tensor')
    return value


def bf16_bits(value):
    value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError('Non-finite tensor')
    bits = value.view(np.uint32)
    rounded = ((bits.astype(np.uint64) + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    if not np.isfinite((rounded.astype(np.uint32) << 16).view(np.float32)).all():
        raise ValueError('BF16 overflow')
    return rounded


def norm(value):
    return float(np.linalg.norm(np.asarray(value, dtype=np.float64)))


def cosine(a, b):
    denominator = norm(a) * norm(b)
    return float(np.sum(a * b, dtype=np.float64) / denominator) if denominator else None


def infer_direction(base_bits, edited_bits, columns):
    base = bf16_float(base_bits).astype(np.float64)
    edited = bf16_float(edited_bits).astype(np.float64)
    if base.shape != edited.shape or base.ndim != 2:
        raise ValueError('Expected equal matrix shapes')
    columns = np.asarray(columns, dtype=int)
    if columns.ndim != 1 or not len(columns) or len(set(columns)) != len(columns):
        raise ValueError('Unique nonempty column indices required')
    if np.any(columns < 0) or np.any(columns >= base.shape[1]):
        raise ValueError('Column index outside tensor')
    changed_columns = np.flatnonzero(np.any(base_bits != edited_bits, axis=0))
    if set(changed_columns) != set(columns):
        raise ValueError('Declared support differs from the actual edit')
    delta = edited[:, columns] - base[:, columns]
    left, singular, _ = np.linalg.svd(delta, full_matrices=False)
    direction = left[:, 0]
    residual = delta - np.outer(direction, direction @ delta)
    return direction, dict(
        rank_one_energy_fraction=float(singular[0]**2 / np.sum(singular**2)),
        rank_one_relative_residual=norm(residual) / norm(delta),
        changed_entries=int(np.count_nonzero(base_bits != edited_bits)),
        changed_columns=len(changed_columns),
        relative_edit_norm=norm(delta) / norm(base),
    )


def project_target(target_bits, direction, columns, alpha):
    """Subtract alpha*u*(u.T@target) on the declared columns, preserving others."""
    target = bf16_float(target_bits).astype(np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    columns = np.asarray(columns, dtype=int)
    if (direction.shape != (target.shape[0],) or
            not np.isclose(norm(direction), 1, rtol=1e-8, atol=1e-8)):
        raise ValueError('Expected a unit direction with matching rows')
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError('Expected a finite nonnegative alpha')
    if columns.ndim != 1 or not len(columns) or len(set(columns)) != len(columns):
        raise ValueError('Unique nonempty column indices required')
    if np.any(columns < 0) or np.any(columns >= target.shape[1]):
        raise ValueError('Column index outside tensor')
    result = target_bits.copy()
    selected = target[:, columns]
    result[:, columns] = bf16_bits(selected - alpha*np.outer(direction, direction@selected))
    return result


def compare_target(base_bits, target_bits, direction, columns):
    base, target = [bf16_float(x).astype(np.float64) for x in (base_bits, target_bits)]
    if base.shape != target.shape:
        raise ValueError('Expected equal tensor shapes')
    delta = target-base
    coefficients_base = direction@base[:, columns]
    coefficients_target = direction@target[:, columns]
    numerator = norm(direction@delta)
    return dict(
        relative_target_change=norm(delta)/norm(base),
        target_delta_norm=norm(delta),
        direction_overlap_norm_fraction=numerator/norm(delta) if norm(delta) else 0.,
        selected_coefficient_cosine=cosine(coefficients_base, coefficients_target),
        selected_coefficient_relative_change=norm(coefficients_target-coefficients_base)/norm(coefficients_base)
            if norm(coefficients_base) else None,
        selected_direction_norm=norm(coefficients_target),
        all_columns_direction_norm=norm(direction@target),
    )


def norm_matched_control(target_bits, direction, columns, alpha, seed):
    """Orthogonal random-direction control with matched pre-rounding edit norm."""
    target = bf16_float(target_bits).astype(np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    if direction.shape != (target.shape[0],) or not np.isclose(norm(direction),1,rtol=1e-8,atol=1e-8):
        raise ValueError('Expected a matching unit direction')
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=len(direction))
    vector -= direction*(direction@vector)
    if norm(vector) < 1e-12:
        raise ValueError('Cannot construct an orthogonal control')
    vector /= norm(vector)
    selected = target[:, columns]
    denominator = norm(vector@selected)
    if denominator == 0:
        raise ValueError('Control direction writes zero on selected columns')
    matched_alpha = alpha*norm(direction@selected)/denominator
    result = project_target(target_bits,vector,columns,matched_alpha)
    return result, dict(seed=seed,alpha=float(matched_alpha),
        direction_cosine=float(vector@direction),
        requested_pre_rounding_norm=alpha*norm(direction@selected))
