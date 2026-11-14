from __future__ import annotations
import math
from collections.abc import Mapping
from typing import Any
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .grade_support import EvidenceHeads, EVIDENCE_NAMES

def accumulation_dtype(x: Tensor) -> torch.dtype:
    return torch.float64 if x.dtype == torch.float64 else torch.float32

def extent_log_odds(extent: Tensor, eps: float=0.0001) -> Tensor:
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('extent_eps must be positive and finite')
    if not torch.isfinite(extent).all() or ((extent < 0) | (extent > 1)).any():
        raise ValueError('extent must be finite and in [0,1]')
    x = extent.to(accumulation_dtype(extent))
    return torch.log(x + eps) - torch.log1p(eps - x)

def structured_grade_probabilities(extent: Tensor, critical: Tensor, slope: Tensor | float, thresholds: Tensor, *, extent_eps: float=0.0001) -> dict[str, Tensor]:
    if extent.ndim != 2 or extent.shape[1] != 1 or critical.shape != extent.shape:
        raise ValueError('extent and critical must be [N,1]')
    dtype = accumulation_dtype(extent)
    e, c = (extent.to(dtype), critical.to(device=extent.device, dtype=dtype))
    if not torch.isfinite(c).all() or ((c < 0) | (c > 1)).any():
        raise ValueError('critical must be finite and in [0,1]')
    t = torch.as_tensor(thresholds, device=e.device, dtype=dtype)
    a = torch.as_tensor(slope, device=e.device, dtype=dtype)
    if t.shape != (2,) or a.numel() != 1:
        raise ValueError('two scalar thresholds and one scalar slope required')
    if not torch.isfinite(t).all() or not torch.isfinite(a).all() or a <= 0 or (t[1] < t[0]):
        raise ValueError('positive finite slope and finite ordered thresholds required')
    x = extent_log_odds(e, extent_eps)
    z = a * (x - t[None])
    q = z.sigmoid()
    cumulative = q + c * (1 - q)
    gap = a * (t[1] - t[0])
    middle_extent = torch.sigmoid(z[:, :1]) * torch.sigmoid(-z[:, 1:]) * -torch.expm1(-gap)
    raw = torch.cat(((1 - c) * torch.sigmoid(-z[:, :1]), (1 - c) * middle_extent, c + (1 - c) * q[:, 1:]), dim=1)
    log_probs = torch.log(raw.clamp_min(torch.finfo(dtype).tiny))
    log_probs = log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True)
    return {'grade_probs': log_probs.exp(), 'grade_logits': log_probs, 'raw_grade_probs': raw, 'extent_cumulative_probs': q, 'cumulative_probs': cumulative, 'extent_coordinate': x}

class OrdinalEvidenceReadout(nn.Module):

    def __init__(self, in_channels: int, feature_dim: int=64, num_grades: int=3, *, fine_channels: int | None=None, use_fine: bool=False, extent_eps: float=0.0001, eps: float=1e-06, min_slope: float=0.25, max_slope: float=8.0, init_slope: float=2.0, initial_thresholds: tuple[float, float]=(0.01, 0.2)):
        super().__init__()
        if min(in_channels, feature_dim) < 1 or num_grades != 3:
            raise ValueError('positive channels and exactly three grades required')
        if use_fine and (fine_channels is None or fine_channels < 1):
            raise ValueError('fine_channels is required when use_fine=True')
        scalars = (extent_eps, eps, min_slope, max_slope, init_slope, *initial_thresholds)
        if not all((math.isfinite(v) for v in scalars)):
            raise ValueError('hyperparameters must be finite')
        if extent_eps <= 0 or eps <= 0 or (not 0 < min_slope < init_slope < max_slope):
            raise ValueError('invalid epsilon or slope bounds')
        if len(initial_thresholds) != 2 or not 0 < initial_thresholds[0] < initial_thresholds[1] < 1:
            raise ValueError('initial thresholds must be ordered interior proportions')
        self.in_channels, self.feature_dim, self.num_grades = (in_channels, feature_dim, num_grades)
        self.fine_channels, self.use_fine = (fine_channels, bool(use_fine))
        self.extent_eps, self.eps = (extent_eps, eps)
        self.min_slope, self.max_slope = (min_slope, max_slope)
        self.stem = nn.Sequential(nn.Conv2d(in_channels, feature_dim, 1, bias=False), nn.SiLU())
        self.fine_stem = nn.Sequential(nn.Conv2d(fine_channels, feature_dim, 1, bias=False), nn.SiLU()) if use_fine else None
        self.local_context = nn.Sequential(nn.Conv2d(feature_dim, feature_dim, 3, padding=1, groups=feature_dim, bias=False), nn.SiLU())
        self.evidence_heads = EvidenceHeads(feature_dim)
        a0 = (init_slope - min_slope) / (max_slope - min_slope)
        self.raw_slope = nn.Parameter(torch.tensor(math.log(a0 / (1 - a0))))
        b0, b1 = [math.log((v + extent_eps) / (1 - v + extent_eps)) for v in initial_thresholds]
        self.threshold_base = nn.Parameter(torch.tensor(b0))
        self.threshold_gap = nn.Parameter(torch.tensor(math.log(math.expm1(b1 - b0))))
        self.empty_support_logits = nn.Parameter(torch.zeros(3))

    def calibration(self) -> tuple[Tensor, Tensor]:
        a = self.min_slope + (self.max_slope - self.min_slope) * self.raw_slope.sigmoid()
        t = torch.stack((self.threshold_base, self.threshold_base + F.softplus(self.threshold_gap)))
        return (a, t)

    def forward(self, coarse_roi: Tensor, fine_roi: Tensor | None=None, *, evidence_override: Mapping[str, Tensor] | None=None, diagnostic: bool=False) -> dict[str, Any]:
        if coarse_roi.ndim != 4 or coarse_roi.shape[1] != self.in_channels:
            raise ValueError(f'coarse_roi must be [N,{self.in_channels},H,W]')
        if not coarse_roi.is_floating_point() or min(coarse_roi.shape[-2:]) < 1:
            raise ValueError('nonempty floating spatial ROI required')
        if evidence_override is not None and (not diagnostic):
            raise ValueError('evidence_override is diagnostic-only')
        features = self.stem(coarse_roi)
        if self.use_fine:
            if fine_roi is None or fine_roi.ndim != 4:
                raise ValueError('fine_roi required in fine mode')
            if fine_roi.shape[:2] != (coarse_roi.shape[0], self.fine_channels):
                raise ValueError('fine_roi batch/channel mismatch')
            if not fine_roi.is_floating_point() or min(fine_roi.shape[-2:]) < 1:
                raise ValueError('nonempty floating fine ROI required')
            if fine_roi.device != coarse_roi.device:
                raise ValueError('coarse and fine ROI must share a device')
            features = self.fine_stem(fine_roi) + F.interpolate(features, size=fine_roi.shape[-2:], mode='bilinear', align_corners=False)
        features = features + self.local_context(features)
        logits = self.evidence_heads(features)
        probabilities = {name: value.to(accumulation_dtype(value)).sigmoid() for name, value in logits.items()}
        used = probabilities.copy()
        if evidence_override is not None:
            if set(evidence_override) - set(EVIDENCE_NAMES):
                raise ValueError('unknown evidence override')
            for name, value in evidence_override.items():
                if value.ndim == 3:
                    value = value.unsqueeze(1)
                if value.shape != logits[name].shape:
                    raise ValueError('override must match evidence output grid')
                value = value.detach().to(device=features.device, dtype=accumulation_dtype(features))
                if not torch.isfinite(value).all() or ((value < 0) | (value > 1)).any():
                    raise ValueError('override probabilities must be finite in [0,1]')
                used[name] = value
        v = used['visible'].detach()
        d, s = (used['damage'], used['critical'])
        mass = v.flatten(1).sum(-1, keepdim=True)
        valid = mass > self.eps
        extent = (v * d).flatten(1).sum(-1, keepdim=True) / mass.clamp_min(self.eps)
        extent = torch.where(valid, extent.clamp(0, 1), torch.zeros_like(extent))
        witness = v * d * s
        critical = torch.where(valid, witness.flatten(1).amax(-1, keepdim=True), torch.zeros_like(extent))
        slope, thresholds = self.calibration()
        result = structured_grade_probabilities(extent, critical, slope, thresholds, extent_eps=self.extent_eps)
        prior = F.log_softmax(self.empty_support_logits.to(result['grade_logits'].dtype), dim=-1)
        log_probs = torch.where(valid, result['grade_logits'], prior[None])
        prior_cumulative = torch.stack((prior.exp()[1:].sum(), prior.exp()[2]))
        cumulative = torch.where(valid, result['cumulative_probs'], prior_cumulative[None])
        aux = {'evidence_logits': logits, 'evidence_probs': probabilities, 'used_evidence_probs': used, 'evidence_override_used': evidence_override is not None, 'extent': extent, 'critical_presence': critical, 'critical_witness': witness, 'visible_mass': mass, 'visible_valid': valid, 'support_valid': valid, 'visible_fraction': mass / features.shape[-2] / features.shape[-1], 'extent_cumulative_probs': result['extent_cumulative_probs'], 'cumulative_probs': cumulative, 'extent_coordinate': result['extent_coordinate'], 'slope': slope, 'thresholds': thresholds, 'fine_readout_used': self.use_fine}
        return {'grade_logits': log_probs, 'grade_probs': log_probs.exp(), 'embedding': features.flatten(2).mean(-1), 'aux': aux}
