from __future__ import annotations
import math
from collections.abc import Mapping
from typing import Any
import torch
from torch import Tensor
import torch.nn.functional as F
from .grade_support import EVIDENCE_NAMES
from .ordinal_readout import OrdinalEvidenceReadout, accumulation_dtype, structured_grade_probabilities
from .critical_support import build_component_labels, canonical_capture, critical_support_readout

class CSEGGradeHead(OrdinalEvidenceReadout):

    def __init__(self, *args, critical_threshold: float=0.5, critical_power: float=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        if not math.isfinite(critical_threshold) or not 0 < critical_threshold <= 1:
            raise ValueError('critical_threshold must be finite in (0,1]')
        if not math.isfinite(critical_power) or critical_power < 1:
            raise ValueError('critical_power must be finite and at least one')
        self.critical_threshold, self.critical_power = (critical_threshold, critical_power)

    def encode_context(self, coarse_roi: Tensor, fine_roi: Tensor | None=None, *, evidence_override: Mapping[str, Tensor] | None=None, diagnostic: bool=False) -> dict[str, Any]:
        if coarse_roi.ndim != 4 or coarse_roi.shape[1] != self.in_channels:
            raise ValueError(f'coarse_roi must be [N,{self.in_channels},H,W]')
        if not coarse_roi.is_floating_point() or min(coarse_roi.shape[-2:]) < 1:
            raise ValueError('nonempty floating spatial context required')
        if evidence_override is not None and (not diagnostic):
            raise ValueError('evidence_override is diagnostic-only')
        features = self.stem(coarse_roi)
        if self.use_fine:
            if fine_roi is None or fine_roi.ndim != 4:
                raise ValueError('fine_roi required in fine mode')
            if fine_roi.shape[:2] != (coarse_roi.shape[0], self.fine_channels):
                raise ValueError('fine_roi batch/channel mismatch')
            if not fine_roi.is_floating_point() or min(fine_roi.shape[-2:]) < 1:
                raise ValueError('nonempty floating fine context required')
            if fine_roi.device != coarse_roi.device:
                raise ValueError('context scales must share a device')
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
                    raise ValueError('override must match context evidence grid')
                value = value.detach().to(device=features.device, dtype=accumulation_dtype(features))
                if not torch.isfinite(value).all() or ((value < 0) | (value > 1)).any():
                    raise ValueError('override probabilities must be finite in [0,1]')
                used[name] = value
        joint = used['visible'].detach() * used['damage'] * used['critical']
        labels = build_component_labels(joint, self.critical_threshold)
        return {'embedding': features.flatten(2).mean(-1), 'evidence_logits': logits, 'evidence_probs': probabilities, 'used_evidence_probs': used, 'component_labels': labels, 'evidence_override_used': evidence_override is not None, 'fine_readout_used': self.use_fine}

    def readout_from_probabilities(self, used: Mapping[str, Tensor], capture: Tensor | None=None, *, calibration_detach: bool=False, component_labels: Tensor | None=None) -> dict[str, Any]:
        if set(used) != set(EVIDENCE_NAMES):
            raise ValueError('used must contain exactly visible, damage and critical')
        v0, d0, s0 = (used[name] for name in EVIDENCE_NAMES)
        if v0.ndim != 4 or v0.shape[1] != 1 or min(v0.shape[-2:]) < 1:
            raise ValueError('probability fields must be [N,1,H,W]')
        for value in (v0, d0, s0):
            if value.shape != v0.shape or value.device != v0.device or (not value.is_floating_point()):
                raise ValueError('probability fields must have identical floating shapes and devices')
            if not torch.isfinite(value).all() or ((value < 0) | (value > 1)).any():
                raise ValueError('probability fields must be finite in [0,1]')
        dtype = accumulation_dtype(v0)
        v, d, s = (v0.detach().to(dtype), d0.to(dtype), s0.to(dtype))
        t = canonical_capture(capture, v)
        n, q = t.shape[:2]
        visible = t * v
        mass = visible.flatten(2).sum(-1)
        valid = mass > self.eps
        extent = (visible * d).flatten(2).sum(-1) / mass.clamp_min(self.eps)
        extent = torch.where(valid, extent.clamp(0, 1), torch.zeros_like(extent))
        joint = v * d * s
        support = critical_support_readout(joint, t, self.critical_threshold, self.critical_power, component_labels=component_labels)
        critical = torch.where(valid, support['c'], torch.zeros_like(support['c']))
        slope, thresholds = self.calibration()
        prior_logits = self.empty_support_logits
        if calibration_detach:
            slope, thresholds, prior_logits = (slope.detach(), thresholds.detach(), prior_logits.detach())
        probabilities = structured_grade_probabilities(extent.reshape(n * q, 1), critical.reshape(n * q, 1), slope, thresholds, extent_eps=self.extent_eps)
        event_log = probabilities['grade_logits'].reshape(n, q, 3)
        prior = F.log_softmax(prior_logits.to(event_log.dtype), dim=-1)
        log_probs = torch.where(valid[..., None], event_log, prior[None, None])
        cumulative = probabilities['cumulative_probs'].reshape(n, q, 2)
        extent_q = probabilities['extent_cumulative_probs'].reshape(n, q, 2)
        single = q == 1
        aux = {'evidence_probs': dict(used), 'used_evidence_probs': dict(used), 'extent': extent, 'critical_presence': critical, 'critical_witness': joint, 'visible_mass': mass, 'visible_fraction': mass / (v.shape[-2] * v.shape[-1]), 'support_valid': valid, 'visible_valid': valid, 'cumulative_probs': cumulative[:, 0] if single else cumulative, 'extent_cumulative_probs': extent_q[:, 0] if single else extent_q, 'extent_coordinate': probabilities['extent_coordinate'].reshape(n, q), 'slope': slope, 'thresholds': thresholds, 'capture': t, 'critical_support': support, 'component_labels': support['labels'], 'component_peaks': support['peaks'], 'component_masses': support['masses'], 'component_coverage': support['coverage'], 'component_valid': support['component_valid'], 'query_count': q, 'calibration_detached': calibration_detach, 'cumulative_includes_empty_prior': False}
        return {'grade_logits': log_probs[:, 0] if single else log_probs, 'grade_probs': log_probs[:, 0].exp() if single else log_probs.exp(), 'aux': aux}

    def readout_context(self, context: Mapping[str, Any], capture: Tensor | None=None, *, calibration_detach: bool=False) -> dict[str, Any]:
        out = self.readout_from_probabilities(context['used_evidence_probs'], capture, calibration_detach=calibration_detach, component_labels=context['component_labels'])
        out['embedding'] = context['embedding']
        out['aux'].update({name: context[name] for name in ('evidence_logits', 'evidence_probs', 'evidence_override_used', 'fine_readout_used')})
        return out

    def forward(self, coarse_roi: Tensor, fine_roi: Tensor | None=None, *, capture: Tensor | None=None, evidence_override: Mapping[str, Tensor] | None=None, diagnostic: bool=False) -> dict[str, Any]:
        context = self.encode_context(coarse_roi, fine_roi, evidence_override=evidence_override, diagnostic=diagnostic)
        return self.readout_context(context, capture)
