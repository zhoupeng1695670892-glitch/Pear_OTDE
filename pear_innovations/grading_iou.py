from __future__ import annotations
import math
import torch
from torch import Tensor, nn
from .critical_support import build_component_labels, critical_support_readout
from .ordinal_readout import accumulation_dtype, extent_log_odds
from .box_geometry import aligned_iou, instance_mean

def _capture_terms(pred: Tensor, reference: Tensor, height: int, width: int, temp: float) -> tuple[Tensor, Tensor, Tensor]:
    aligned_iou(pred, reference, ciou=False)
    if not isinstance(height, int) or not isinstance(width, int) or height < 1 or (width < 1):
        raise ValueError('height and width must be positive integers')
    if not math.isfinite(temp) or temp <= 0:
        raise ValueError('temp must be finite and positive')
    p = pred.to(accumulation_dtype(pred))
    r = reference.detach().to(device=p.device, dtype=p.dtype)
    size = r[:, 2:] - r[:, :2]
    origin = torch.cat((r[:, :2], r[:, :2]), -1)
    pn = (p - origin) / torch.cat((size, size), -1)
    yy, xx = torch.meshgrid((torch.arange(height, device=p.device, dtype=p.dtype) + 0.5) / height, (torch.arange(width, device=p.device, dtype=p.dtype) + 0.5) / width, indexing='ij')

    def membership(box: Tensor) -> Tensor:
        left, top, right, bottom = [box[:, i, None, None] for i in range(4)]
        return torch.sigmoid((xx - left) / temp) * torch.sigmoid((right - xx) / temp) * torch.sigmoid((yy - top) / temp) * torch.sigmoid((bottom - yy) / temp)
    bp = membership(pn)
    br = membership(p.new_tensor([0.0, 0.0, 1.0, 1.0]).expand(p.shape[0], -1))
    capture = torch.where(bp < br, bp / br, torch.ones_like(bp))
    return (capture, bp, br)

def normalized_soft_capture(pred: Tensor, reference: Tensor, height: int, width: int, temp: float=0.04) -> Tensor:
    return _capture_terms(pred, reference, height, width, temp)[0]

class GSFIoULoss(nn.Module):

    def __init__(self, weight: float=0.25, temperature: float=0.04, max_slope: float=8.0, extent_eps: float=0.0001, coverage_weight: float=1.0, reduction: str='mean', ciou: bool=True, min_mass: float=1e-06, eps: float=1e-07, component_threshold: float=0.5, component_power: float=2.0):
        super().__init__()
        values = (weight, temperature, max_slope, extent_eps, coverage_weight, min_mass, eps, component_threshold, component_power)
        if not all((math.isfinite(x) for x in values)):
            raise ValueError('all loss hyperparameters must be finite')
        if min(temperature, max_slope, extent_eps, min_mass, eps, component_power) <= 0 or min(weight, coverage_weight) < 0:
            raise ValueError('invalid positive hyperparameter or negative loss weight')
        if not 0 < component_threshold <= 1:
            raise ValueError('component_threshold must be in (0,1]')
        if component_power < 1:
            raise ValueError('component_power must be at least one')
        if min_mass < eps:
            raise ValueError('min_mass must be at least eps')
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError('invalid reduction')
        self.weight, self.temperature = (weight, temperature)
        self.max_slope, self.extent_eps = (max_slope, extent_eps)
        self.coverage_weight, self.reduction = (coverage_weight, reduction)
        self.ciou, self.min_mass, self.eps = (ciou, min_mass, eps)
        self.component_threshold, self.component_power = (component_threshold, component_power)

    @property
    def per_box_upper_bound(self) -> float:
        span = 2 * math.log((1 + self.extent_eps) / self.extent_eps)
        return self.max_slope / 4 * span + 1 + self.coverage_weight

    def statistics(self, pred: Tensor, target: Tensor, evidence: Tensor | None, group_valid: Tensor | None=None) -> dict[str, Tensor]:
        aligned_iou(pred, target, ciou=False, eps=self.eps)
        p = pred.to(accumulation_dtype(pred))
        n = p.shape[0]
        zero = p.sum(-1) * 0
        false = torch.zeros(n, device=p.device, dtype=torch.bool)
        if evidence is None:
            return {'retention': zero, 'stat_loss': zero, 'coverage_loss': zero, 'extent_loss': zero, 'critical_loss': zero, 'delta_x': zero, 'delta_extent': zero, 'delta_critical': zero, 'extent_reference': zero, 'extent_captured': zero, 'critical_reference': zero, 'critical_captured': zero, 'captured_fraction': zero, 'tv_bound': zero, 'bound_valid': false, 'valid': false, 'valid_extent': false, 'valid_critical': false, 'valid_visible': false, 'per_group': p.new_zeros(n, 3), 'valid_groups': torch.zeros(n, 3, device=p.device, dtype=torch.bool)}
        if evidence.ndim != 4 or evidence.shape[:2] != (n, 3) or min(evidence.shape[-2:]) < 1:
            raise ValueError('evidence must be [N,3,H,W], ordered V,D,S')
        available = torch.ones(n, 3, device=p.device, dtype=torch.bool)
        if group_valid is not None:
            if group_valid.shape != (n, 3) or not torch.isfinite(group_valid).all() or ((group_valid != 0) & (group_valid != 1)).any():
                raise ValueError('group_valid must be [N,3] boolean/0-1 availability')
            available = group_valid.detach().to(device=p.device, dtype=torch.bool)
        e = evidence.detach().to(device=p.device, dtype=p.dtype)
        known = available[:, :, None, None].expand_as(e)
        if (known & (~torch.isfinite(e) | (e < 0) | (e > 1))).any():
            raise ValueError('annotated evidence maps must be finite and in [0,1]')
        e = torch.where(known, e, torch.zeros_like(e))
        bad_d = (e[:, 1] > e[:, 0] + 1e-06).flatten(1).any(-1) & available[:, 0] & available[:, 1]
        bad_s = (e[:, 2] > e[:, 1] + 1e-06).flatten(1).any(-1) & available[:, 1] & available[:, 2]
        bad_sv = (e[:, 2] > e[:, 0] + 1e-06).flatten(1).any(-1) & available[:, 0] & available[:, 2]
        if bad_d.any() or bad_s.any() or bad_sv.any():
            raise ValueError('annotated evidence must satisfy S <= D <= V')
        h, w = e.shape[-2:]
        capture, bp, br = _capture_terms(p, target, h, w, self.temperature)
        v, d, s = e.unbind(1)
        mass = v.flatten(1).sum(-1)
        captured_mass = (capture * v).flatten(1).sum(-1)
        valid_v = available[:, 0] & (mass > self.min_mass)
        valid_e = valid_v & available[:, 1]
        valid_c = available[:, 2]
        bound_valid = valid_e & valid_c & (captured_mass > self.min_mass)
        e_ref = d.flatten(1).sum(-1) / mass.clamp_min(self.eps)
        e_cap = (capture * d).flatten(1).sum(-1) / captured_mass.clamp_min(self.eps)
        e_ref = torch.where(valid_e, e_ref.clamp(0, 1), zero)
        e_cap = torch.where(valid_e, e_cap.clamp(0, 1), zero)
        labels = build_component_labels(s[:, None], threshold=self.component_threshold)
        critical = critical_support_readout(s[:, None], capture=torch.stack((torch.ones_like(capture), capture), 1), threshold=self.component_threshold, power=self.component_power, component_labels=labels)
        c_ref, c_cap = critical['c'].unbind(1)
        c_ref = torch.where(valid_c, c_ref, zero)
        c_cap = torch.where(valid_c, c_cap, zero)
        delta_x = (extent_log_odds(e_cap, self.extent_eps) - extent_log_odds(e_ref, self.extent_eps)).abs()
        delta_c = (c_cap - c_ref).abs()
        noncritical = torch.where(valid_c, 1 - c_ref, torch.ones_like(c_ref))
        extent_loss = torch.where(valid_e, noncritical * (self.max_slope / 4) * delta_x, zero)
        critical_loss = torch.where(valid_c, delta_c, zero)
        fraction = torch.where(valid_v, (captured_mass / mass.clamp_min(self.eps)).clamp(0, 1), zero)
        coverage = torch.where(valid_v, torch.relu(1 - fraction), zero)
        stat = extent_loss + critical_loss
        return {'retention': stat + self.coverage_weight * coverage, 'stat_loss': stat, 'coverage_loss': coverage, 'extent_loss': extent_loss, 'critical_loss': critical_loss, 'delta_x': delta_x, 'delta_extent': (e_cap - e_ref).abs(), 'delta_critical': delta_c, 'extent_reference': e_ref, 'extent_captured': e_cap, 'critical_reference': c_ref, 'critical_captured': c_cap, 'captured_fraction': fraction, 'tv_bound': stat.clamp(0, 1), 'bound_valid': bound_valid, 'valid': valid_e | valid_c | valid_v, 'valid_extent': valid_e, 'valid_critical': valid_c, 'valid_visible': valid_v, 'per_group': torch.stack((extent_loss, critical_loss, coverage), -1), 'valid_groups': torch.stack((valid_e, valid_c, valid_v), -1), 'capture': capture, 'membership_pred': bp, 'membership_target': br, 'support_mass': mass, 'captured_mass': captured_mass, 'component_labels': labels, 'component_coverage': critical['coverage'], 'component_peaks': critical['peaks'], 'component_valid': critical['component_valid']}

    def retention(self, pred: Tensor, target: Tensor, evidence: Tensor | None, group_valid: Tensor | None=None) -> dict[str, Tensor]:
        return self.statistics(pred, target, evidence, group_valid)

    def forward(self, pred: Tensor, target: Tensor, evidence: Tensor | None=None, group_valid: Tensor | None=None, instance_ids: Tensor | None=None, *, return_details: bool=False):
        geometry = 1 - aligned_iou(pred, target, ciou=self.ciou, eps=self.eps)
        info = self.statistics(pred, target, evidence, group_valid)
        per_box = geometry + self.weight * info['retention']
        loss = per_box if self.reduction == 'none' else per_box.sum() if self.reduction == 'sum' else instance_mean(per_box, instance_ids)
        return (loss, dict(info, geometry=geometry, per_box=per_box)) if return_details else loss
