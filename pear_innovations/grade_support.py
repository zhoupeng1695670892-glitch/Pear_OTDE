from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import torch
from torch import Tensor, nn
import torch.nn.functional as F
EVIDENCE_NAMES = ('visible', 'damage', 'critical')

def _accumulation_dtype(x: Tensor) -> torch.dtype:
    return torch.float64 if x.dtype == torch.float64 else torch.float32

def _spatial_map(x: Tensor, reference: Tensor, name: str) -> Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(1)
    if x.shape != reference.shape:
        raise ValueError(f'{name} must have shape {tuple(reference.shape)}, got {tuple(x.shape)}')
    return x.to(device=reference.device, dtype=_accumulation_dtype(reference))

def nested_region_weights(visible: Tensor, damage: Tensor, critical: Tensor) -> Tensor:
    if visible.ndim != 4 or visible.shape[1] != 1:
        raise ValueError('evidence maps must be [N,1,H,W]')
    if damage.shape != visible.shape or critical.shape != visible.shape:
        raise ValueError('all evidence maps must have identical shapes')
    return torch.cat((visible * (1 - damage), visible * damage * (1 - critical), visible * damage * critical), dim=1)

def weighted_upper_tail(scores: Tensor, support: Tensor, values: Tensor | None=None, *, fraction: float=0.1, eps: float=1e-06) -> dict[str, Tensor]:
    if not 0 < fraction <= 1 or eps <= 0:
        raise ValueError('fraction must be in (0,1], and eps must be positive')
    if scores.ndim != 2 or scores.shape != support.shape or scores.shape[1] == 0:
        raise ValueError('scores and support must share shape [N,L] with L>0')
    if values is not None and (values.ndim != 3 or values.shape[:2] != scores.shape):
        raise ValueError('values must have shape [N,L,D]')
    dtype = _accumulation_dtype(scores)
    scores, support = (scores.to(dtype), support.to(dtype))
    ordered_scores, indices = torch.sort(scores, dim=-1, descending=True, stable=True)
    ordered_support = support.gather(1, indices)
    support_mass = support.sum(-1, keepdim=True)
    budget = fraction * support_mass
    before = ordered_support.cumsum(-1) - ordered_support
    selected = torch.minimum((budget - before).clamp_min(0), ordered_support)
    valid = support_mass > eps
    selected = selected * valid.to(dtype)
    selected_mass = selected.sum(-1, keepdim=True)
    denominator = selected_mass.clamp_min(fraction * eps)
    tail_mean = (selected * ordered_scores).sum(-1, keepdim=True) / denominator
    weights = torch.zeros_like(selected).scatter(1, indices, selected)
    result = {'mean': tail_mean, 'weights': weights, 'valid': valid, 'support_mass': support_mass, 'selected_mass': selected_mass}
    if values is not None:
        result['token'] = (weights.unsqueeze(-1) * values.to(dtype)).sum(1) / denominator
    return result

class EvidenceHeads(nn.Module):

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.visible = nn.Conv2d(feature_dim, 1, 1)
        self.damage = nn.Conv2d(feature_dim, 1, 1)
        self.critical = nn.Conv2d(feature_dim, 1, 1)

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        return {name: getattr(self, name)(features) for name in EVIDENCE_NAMES}

class _ROIGradeBase(nn.Module):

    def __init__(self, in_channels: int, feature_dim: int, num_grades: int) -> None:
        super().__init__()
        if min(in_channels, feature_dim) < 1 or num_grades < 2:
            raise ValueError('positive channel sizes and at least two grades are required')
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.num_grades = num_grades
        self.stem = nn.Sequential(nn.Conv2d(in_channels, feature_dim, 1, bias=False), nn.SiLU())
        self.evidence_heads = EvidenceHeads(feature_dim)
        self.classifier = nn.Linear(feature_dim, num_grades)

    def _prepare(self, roi_features: Tensor, evidence_override: Mapping[str, Tensor] | None, diagnostic: bool) -> tuple[Tensor, dict[str, Tensor], dict[str, Any]]:
        if roi_features.ndim != 4 or roi_features.shape[1] != self.in_channels:
            raise ValueError(f'ROI features must be [N,{self.in_channels},H,W]')
        if not roi_features.is_floating_point() or min(roi_features.shape[-2:]) < 1:
            raise ValueError('ROI features require floating dtype and positive spatial dimensions')
        if evidence_override is not None and (not diagnostic):
            raise ValueError('evidence_override is oracle/diagnostic-only; set diagnostic=True explicitly')
        features = self.stem(roi_features)
        logits = self.evidence_heads(features)
        probabilities = {name: value.sigmoid() for name, value in logits.items()}
        used = probabilities.copy()
        if evidence_override is not None:
            unknown = set(evidence_override) - set(EVIDENCE_NAMES)
            if unknown:
                raise ValueError(f'unknown evidence names: {sorted(unknown)}')
            for name, value in evidence_override.items():
                value = _spatial_map(value, logits[name], name).detach()
                if not bool(torch.isfinite(value).all()) or bool(((value < 0) | (value > 1)).any()):
                    raise ValueError(f'diagnostic {name} must be finite probabilities in [0,1]')
                used[name] = value
        aux: dict[str, Any] = {'evidence_logits': logits, 'evidence_probs': probabilities, 'used_evidence_probs': used, 'evidence_override_used': evidence_override is not None}
        return (features, used, aux)

class BasicROIGradeHead(_ROIGradeBase):

    def __init__(self, in_channels: int, feature_dim: int=64, num_grades: int=3, *, pooling: str='gap') -> None:
        super().__init__(in_channels, feature_dim, num_grades)
        if pooling not in {'gap', 'meanmax'}:
            raise ValueError("pooling must be 'gap' or 'meanmax'")
        self.pooling = pooling
        width = feature_dim * (2 if pooling == 'meanmax' else 1)
        self.readout = nn.Sequential(nn.Linear(width, feature_dim), nn.SiLU())

    def forward(self, roi_features: Tensor, *, evidence_override: Mapping[str, Tensor] | None=None, diagnostic: bool=False) -> dict[str, Any]:
        features, _, aux = self._prepare(roi_features, evidence_override, diagnostic)
        flat = features.flatten(2)
        pooled = flat.mean(-1)
        if self.pooling == 'meanmax':
            pooled = torch.cat((pooled, flat.amax(-1)), dim=-1)
        embedding = self.readout(pooled)
        return {'grade_logits': self.classifier(embedding), 'embedding': embedding, 'aux': aux}

class EvidencePoolingGradeHead(_ROIGradeBase):

    def __init__(self, in_channels: int, feature_dim: int=64, num_grades: int=3, *, tail_fraction: float=0.1, eps: float=1e-06, detach_region_weights: bool=False) -> None:
        super().__init__(in_channels, feature_dim, num_grades)
        if not 0 < tail_fraction <= 1 or eps <= 0:
            raise ValueError('tail_fraction must be in (0,1], and eps must be positive')
        self.tail_fraction = tail_fraction
        self.eps = eps
        self.detach_region_weights = detach_region_weights
        self.readout = nn.Sequential(nn.Linear(5 * feature_dim + 10, feature_dim), nn.SiLU())

    def forward(self, roi_features: Tensor, *, evidence_override: Mapping[str, Tensor] | None=None, diagnostic: bool=False) -> dict[str, Any]:
        features, evidence, aux = self._prepare(roi_features, evidence_override, diagnostic)
        dtype = _accumulation_dtype(features)
        values = features.flatten(2).transpose(1, 2).to(dtype)
        visible = evidence['visible'].detach().to(dtype)
        damage, critical = (evidence['damage'].to(dtype), evidence['critical'].to(dtype))
        if self.detach_region_weights:
            damage, critical = (damage.detach(), critical.detach())
        weights = nested_region_weights(visible, damage, critical)
        flat_weights = weights.flatten(2)
        region_mass = flat_weights.sum(-1)
        visible_mass = visible.flatten(2).sum(-1)
        region_valid = region_mass > self.eps
        visible_valid = visible_mass > self.eps
        relative_mass = region_mass / visible_mass.clamp_min(self.eps)
        tokens = torch.bmm(flat_weights, values) / region_mass.clamp_min(self.eps).unsqueeze(-1)
        tokens = tokens * region_valid.unsqueeze(-1).to(dtype)
        visible_flat = visible.flatten(1)
        visible_token = (visible_flat.unsqueeze(-1) * values).sum(1) / visible_mass.clamp_min(self.eps)
        global_token = torch.where(visible_valid, visible_token, values.mean(1))
        tail = weighted_upper_tail((damage * critical).flatten(1), visible_flat, values, fraction=self.tail_fraction, eps=self.eps)
        tail['support_token'] = tail['token']
        evidence_tail_weights = tail['weights'] * (damage * critical).flatten(1)
        evidence_tail_mass = evidence_tail_weights.sum(-1, keepdim=True)
        tail['evidence_mass'] = evidence_tail_mass
        tail['evidence_valid'] = evidence_tail_mass > self.eps
        tail['token'] = (evidence_tail_weights.unsqueeze(-1) * values).sum(1) / evidence_tail_mass.clamp_min(self.eps) * tail['evidence_valid'].to(dtype)
        extent = relative_mass[:, 1:].sum(-1, keepdim=True)
        visible_fraction = visible_mass / values.shape[1]
        statistics = torch.cat((relative_mass, region_valid.to(dtype), visible_valid.to(dtype), extent, tail['mean'], visible_fraction), dim=-1)
        pooled = torch.cat((global_token, tokens.flatten(1), tail['token'], statistics), dim=-1).to(features.dtype)
        embedding = self.readout(pooled)
        aux.update({'region_weights': weights, 'region_mass': region_mass, 'relative_mass': relative_mass, 'region_tokens': tokens, 'region_valid': region_valid, 'visible_mass': visible_mass, 'visible_valid': visible_valid, 'visible_fraction': visible_fraction, 'extent': extent, 'tail': tail, 'global_token': global_token})
        return {'grade_logits': self.classifier(embedding), 'embedding': embedding, 'aux': aux}

def evidence_supervision_loss(output: Mapping[str, Any], targets: Mapping[str, Tensor] | None=None, *, valid_masks: Mapping[str, Tensor] | None=None, loss_weights: Mapping[str, float] | None=None, eps: float=1e-06) -> dict[str, Any]:
    if eps <= 0:
        raise ValueError('eps must be positive')
    aux = output['aux'] if 'aux' in output else output
    logits: Mapping[str, Tensor] = aux['evidence_logits']
    targets, valid_masks, loss_weights = (targets or {}, valid_masks or {}, loss_weights or {})
    for mapping in (targets, valid_masks, loss_weights):
        if set(mapping) - set(EVIDENCE_NAMES):
            raise ValueError('unknown evidence supervision name')
    if 'damage' in targets and 'visible' not in targets and ('damage' not in valid_masks):
        raise ValueError('damage supervision needs GT visible support or an explicit conditional valid mask')
    if 'critical' in targets and 'damage' not in targets and ('critical' not in valid_masks):
        raise ValueError('critical supervision needs GT damage support or an explicit conditional valid mask')
    clean_targets: dict[str, Tensor] = {}
    known: dict[str, Tensor] = {}
    for name, target in targets.items():
        target = _spatial_map(target, logits[name], name).detach()
        if bool((torch.isfinite(target) & (target > 1)).any()):
            raise ValueError(f'{name} target exceeds 1')
        known[name] = torch.isfinite(target) & (target >= 0)
        clean_targets[name] = torch.where(known[name], target, torch.zeros_like(target))
    losses, counts, instances = ({}, {}, {})
    for name in EVIDENCE_NAMES:
        prediction = logits[name].to(_accumulation_dtype(logits[name]))
        zero = prediction.sum() * 0
        if name not in clean_targets:
            losses[name], counts[name], instances[name] = (zero, prediction.new_zeros(()), prediction.new_zeros(()))
            continue
        weight = known[name].to(prediction.dtype)
        if name in valid_masks:
            confidence = _spatial_map(valid_masks[name], logits[name], f'valid_masks.{name}').detach()
            if not bool(torch.isfinite(confidence).all()) or bool(((confidence < 0) | (confidence > 1)).any()):
                raise ValueError('valid masks must contain finite values in [0,1]')
            weight = weight * confidence
        for support_name in ('visible',) if name == 'damage' else ('visible', 'damage') if name == 'critical' else ():
            if support_name in clean_targets:
                weight = weight * clean_targets[support_name] * known[support_name].to(weight.dtype)
                if support_name in valid_masks:
                    support_confidence = _spatial_map(valid_masks[support_name], logits[name], support_name).detach()
                    weight = weight * support_confidence
        per_pixel = F.binary_cross_entropy_with_logits(prediction, clean_targets[name], reduction='none')
        mass = weight.flatten(1).sum(-1)
        active = mass > eps
        per_roi = (per_pixel * weight).flatten(1).sum(-1) / mass.clamp_min(eps)
        losses[name] = (per_roi * active.to(per_roi.dtype)).sum() / active.sum().clamp_min(1)
        counts[name], instances[name] = (mass.sum().detach(), active.sum().detach())
    total = sum((losses[name] * float(loss_weights.get(name, 1.0)) for name in EVIDENCE_NAMES))
    return {'loss': total, 'losses': losses, 'support_mass': counts, 'annotated_instances': instances}
