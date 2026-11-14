from __future__ import annotations
import math
from collections.abc import Mapping
import torch
from torch import Tensor
from .ordinal_readout import accumulation_dtype

def _joint_map(joint_critical: Tensor) -> Tensor:
    if joint_critical.ndim == 3:
        joint_critical = joint_critical.unsqueeze(1)
    if joint_critical.ndim != 4 or joint_critical.shape[1] != 1:
        raise ValueError('joint_critical must be [N,1,H,W] or [N,H,W]')
    if not joint_critical.is_floating_point() or min(joint_critical.shape[-2:]) < 1:
        raise ValueError('joint_critical requires a nonempty floating spatial map')
    if not torch.isfinite(joint_critical).all() or ((joint_critical < 0) | (joint_critical > 1)).any():
        raise ValueError('joint_critical must be finite in [0,1]')
    return joint_critical

def _check_threshold(threshold: float) -> None:
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError('threshold must be finite in (0,1]')

def canonical_capture(capture: Tensor | None, reference: Tensor) -> Tensor:
    reference = _joint_map(reference)
    if capture is None:
        return torch.ones_like(reference, dtype=accumulation_dtype(reference))
    if capture.ndim == 3:
        capture = capture.unsqueeze(1)
    if capture.ndim != 4 or capture.shape[0] != reference.shape[0] or capture.shape[-2:] != reference.shape[-2:] or (capture.shape[1] < 1):
        raise ValueError('capture must be [N,Q,H,W] on the context grid, with Q>=1')
    if not capture.is_floating_point() or capture.device != reference.device:
        raise ValueError('capture must be floating point on the context device')
    if not torch.isfinite(capture).all() or ((capture < 0) | (capture > 1)).any():
        raise ValueError('capture must be finite in [0,1]')
    return capture.to(accumulation_dtype(reference))

def build_component_labels(joint_critical: Tensor, threshold: float=0.5) -> Tensor:
    h = _joint_map(joint_critical)
    _check_threshold(threshold)
    n, _, height, width = h.shape
    if n == 0:
        return torch.zeros_like(h, dtype=torch.long)
    masks = (h.detach()[:, 0] >= threshold).to('cpu').tolist()
    output = []
    for mask in masks:
        labels = [[0] * width for _ in range(height)]
        identity = 0
        for y in range(height):
            for x in range(width):
                if not mask[y][x] or labels[y][x]:
                    continue
                identity += 1
                labels[y][x] = identity
                queue = [(y, x)]
                cursor = 0
                while cursor < len(queue):
                    cy, cx = queue[cursor]
                    cursor += 1
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < height and 0 <= nx < width and mask[ny][nx] and (labels[ny][nx] == 0):
                            labels[ny][nx] = identity
                            queue.append((ny, nx))
        output.append(labels)
    return torch.tensor(output, dtype=torch.long, device=h.device).unsqueeze(1)

def _checked_labels(component_labels: Tensor, h: Tensor) -> Tensor:
    labels = component_labels
    if labels.ndim == 3:
        labels = labels.unsqueeze(1)
    integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if labels.shape != h.shape or labels.dtype not in integer_dtypes:
        raise ValueError('component_labels must be integer [N,1,H,W] context identities')
    if ((labels < 0) | (labels > h.shape[-2] * h.shape[-1])).any():
        raise ValueError('component labels must be between zero and the context pixel count')
    return labels.detach().to(device=h.device, dtype=torch.long)

def prepare_component_context(joint_critical: Tensor, threshold: float=0.5, *, component_labels: Tensor | None=None, eps: float=1e-08) -> dict[str, Tensor]:
    h = _joint_map(joint_critical)
    _check_threshold(threshold)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be positive and finite')
    labels = build_component_labels(h, threshold) if component_labels is None else _checked_labels(component_labels, h)
    work = h.to(accumulation_dtype(h)).flatten(1)
    flat_labels = labels.flatten(1)
    n = h.shape[0]
    k = int(labels.max().item()) if n else 0
    masses = work.new_zeros(n, k + 1).scatter_add(1, flat_labels, work)[:, 1:]
    peaks = work.new_zeros(n, k + 1).scatter_reduce(1, flat_labels, work, reduce='amax', include_self=True)[:, 1:]
    valid = masses > eps
    return {'labels': labels, 'peaks': peaks, 'masses': masses, 'component_valid': valid, 'num_components': valid.sum(-1)}

def critical_support_readout(joint_critical: Tensor, capture: Tensor | None=None, threshold: float=0.5, power: float=2.0, *, component_labels: Tensor | None=None, eps: float=1e-08) -> dict[str, Tensor]:
    if not math.isfinite(power) or power < 1:
        raise ValueError('power must be finite and at least one')
    h = _joint_map(joint_critical)
    t = canonical_capture(capture, h)
    context = prepare_component_context(h, threshold, component_labels=component_labels, eps=eps)
    n, q, height, width = t.shape
    k = context['masses'].shape[1]
    work = h.to(t.dtype).flatten(2)
    if k == 0:
        c = work.sum(-1) * 0 + t.flatten(2).sum(-1) * 0
        empty = t.new_zeros(n, q, 0)
        return {**context, 'c': c, 'critical': c, 'critical_presence': c, 'coverage': empty, 'captured_masses': empty, 'scores': empty, 'capture': t}
    indices = context['labels'].flatten(2).expand(n, q, height * width).reshape(n * q, height * width)
    weighted = (work * t.flatten(2)).reshape(n * q, height * width)
    captured = t.new_zeros(n * q, k + 1).scatter_add(1, indices, weighted)[:, 1:].reshape(n, q, k)
    masses = context['masses'][:, None]
    valid = context['component_valid'][:, None]
    coverage = torch.where(valid, (captured / masses.clamp_min(eps)).clamp(0, 1), torch.zeros_like(captured))
    scores = context['peaks'][:, None] * coverage.pow(power)
    scores = torch.where(valid, scores, torch.zeros_like(scores))
    c = scores.amax(-1)
    return {**context, 'c': c, 'critical': c, 'critical_presence': c, 'coverage': coverage, 'captured_masses': captured, 'scores': scores, 'capture': t}

def component_critical(joint_critical: Tensor, capture: Tensor | None=None, context: Mapping[str, Tensor] | None=None, *, threshold: float=0.5, power: float=2.0, eps: float=1e-08) -> dict[str, Tensor]:
    labels = None if context is None else context['labels']
    return critical_support_readout(joint_critical, capture, threshold, power, component_labels=labels, eps=eps)
