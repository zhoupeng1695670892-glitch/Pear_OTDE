from __future__ import annotations
import math
from collections.abc import Sequence
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .fusion_base import ScaleComparableFusion, _comparison_dtype
from .phase_base import aggregate_evidence_reference, polyphase_gather

def polyphase_restore(phases: Tensor, height: int, width: int) -> Tensor:
    if phases.ndim != 5 or phases.shape[-1] != 4:
        raise ValueError('phases must be B,C,Hc,Wc,4')
    if height < 1 or width < 1 or phases.shape[2:4] != ((height + 1) // 2, (width + 1) // 2):
        raise ValueError('restored dimensions must match the phase grid')
    ordered = phases[..., [0, 2, 1, 3]]
    b, c, h, w, _ = ordered.shape
    full = ordered.reshape(b, c, h, w, 2, 2).permute(0, 1, 2, 4, 3, 5)
    return full.reshape(b, c, 2 * h, 2 * w)[..., :height, :width]

def constrain_reference_energy(base: Tensor, update: Tensor, evidence_weights: Tensor, reference_weights: Tensor, *, active: Tensor | None=None, variance_floor: float=0.1, detach_constraint: bool=False, enable_projection: bool=True):
    if base.ndim != 5 or base.shape[-1] != 4 or update.shape != base.shape:
        raise ValueError('base/update must have equal B,C,Hc,Wc,4 shapes')
    expected = (base.shape[0], 1, *base.shape[2:])
    if evidence_weights.shape != expected or reference_weights.shape != expected:
        raise ValueError('weights must be B,1,Hc,Wc,4')
    if not math.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError('variance_floor must be finite and positive')
    if base.device != update.device or base.dtype != update.dtype:
        raise ValueError('base/update must share device and dtype')
    if any((w.device != base.device for w in (evidence_weights, reference_weights))):
        raise ValueError('base and weights must share a device')
    if active is None:
        active = torch.ones(expected[:-1], device=base.device, dtype=torch.bool)
    if active.shape != expected[:-1] or active.device != base.device or active.dtype != torch.bool:
        raise ValueError('active must be Boolean B,1,Hc,Wc on the input device')
    dtype = _comparison_dtype(base)
    f, u = (base.to(dtype), update.to(dtype))
    we, wr = (evidence_weights.to(dtype), reference_weights.to(dtype))
    anchor = f
    if detach_constraint:
        anchor, we, wr = (anchor.detach(), we.detach(), wr.detach())
    center = (wr * anchor).sum(-1, keepdim=True)
    residual = anchor - center
    reference_variance = (wr * residual.square()).sum(-1, keepdim=True)
    inverse_variance = (reference_variance + variance_floor).reciprocal()
    weighted_residual = we * residual
    normal = (weighted_residual - wr * weighted_residual.sum(-1, keepdim=True)) * inverse_variance
    normal = normal * active.unsqueeze(-1)
    magnitude = normal.abs().amax(dim=(1, 4), keepdim=True)
    nonzero = magnitude > 0
    scaled_normal = normal / torch.where(nonzero, magnitude, torch.ones_like(magnitude))
    norm_sq = scaled_normal.square().sum(dim=(1, 4), keepdim=True)
    dot = (scaled_normal * u).sum(dim=(1, 4), keepdim=True)
    safe_norm_sq = torch.where(nonzero, norm_sq, torch.ones_like(norm_sq))
    correction_scale = (-dot).relu() / safe_norm_sq
    correction = correction_scale * scaled_normal * float(enable_projection)
    corrected = u + correction
    output = (f + corrected).to(base.dtype)

    def energy(tensor):
        values = tensor.to(dtype)
        reference = (wr * values).sum(-1, keepdim=True)
        return (we * (values - reference).square() * inverse_variance).sum((1, 4), keepdim=True).squeeze(-1)
    actual_update = output.to(dtype) - f
    return (output, {'constraint_normal': scaled_normal, 'constraint_active': (active.unsqueeze(-1) & nonzero).squeeze(-1), 'projection_applied': ((dot < 0) & nonzero & bool(enable_projection)).squeeze(-1), 'constraint_dot_before': dot.squeeze(-1), 'constraint_dot_after': (scaled_normal * actual_update).sum((1, 4), keepdim=True).squeeze(-1), 'energy_before': energy(base), 'energy_unconstrained': energy((f + u).to(base.dtype)), 'energy_after': energy(output), 'projection_correction': correction, 'reference_variance': reference_variance.squeeze(-1), 'projected_update': corrected})

class LECF(ScaleComparableFusion):

    def __init__(self, fine_channels: int, coarse_channels: int, out_channels: int, *, enabled: bool=True, enable_projection: bool=True, enable_common: bool=True, enable_unique: bool=True, variance_floor: float=0.1, mass_threshold: float=0.0001, detach_constraint: bool=False, eps: float=1e-06, tau: float=0.25, residual_init: float=0.1, residual_bound: float=1.0, normalize_eps: float=1e-06) -> None:
        if out_channels != fine_channels + coarse_channels:
            raise ValueError('LECF Cout must equal Cf+Cc for native concat fallback')
        for name, value in (('variance_floor', variance_floor), ('eps', eps), ('tau', tau)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not math.isfinite(mass_threshold) or mass_threshold < 0:
            raise ValueError('mass_threshold must be finite and nonnegative')
        super().__init__(fine_channels, coarse_channels, out_channels, enable_common=enable_common, enable_unique=enable_unique, common_scale_comparison=True, use_fine_support=True, gate_unique_with_common=False, residual_init=residual_init, residual_bound=residual_bound, normalize_eps=normalize_eps)
        self.enabled = enabled
        self.enable_projection = enable_projection
        self.variance_floor = variance_floor
        self.mass_threshold = mass_threshold
        self.detach_constraint = detach_constraint
        self.eps, self.tau = (eps, tau)
        self.fine_support_head = nn.Conv2d(out_channels, 2, 1)

    def forward(self, fine: Tensor | Sequence[Tensor], coarse: Tensor | None=None, *, return_aux: bool=False):
        if coarse is None:
            if not isinstance(fine, (tuple, list)) or len(fine) != 2:
                raise ValueError('pass [fine,coarse] or two native-scale tensors')
            fine, coarse = fine
        if not isinstance(fine, Tensor) or not isinstance(coarse, Tensor):
            raise TypeError('fine/coarse must be tensors')
        self._validate(fine, coarse)
        f, c = (self.fine_projection(fine), self.coarse_projection(coarse))
        size = tuple(f.shape[-2:])
        fine_coarse = self.downsample(f)
        fine_low, coarse_up = (self._upsample(fine_coarse, size), self._upsample(c, size))
        exclusive = f - fine_low
        left = F.normalize(fine_coarse.to(_comparison_dtype(f)), dim=1, eps=self.normalize_eps)
        right = F.normalize(c.to(_comparison_dtype(f)), dim=1, eps=self.normalize_eps)
        cosine = (left * right).sum(1, keepdim=True)
        descriptor = torch.cat((left, right, left - right, cosine), 1).to(f.dtype)
        common_gate = self._upsample(self.common_gate(descriptor), size)
        common_update = self.common_value(torch.cat((fine_low, coarse_up), 1))
        logits = self.fine_support_head(f)
        support = logits.to(_comparison_dtype(f)).sigmoid()
        visible, damage = (support[:, :1], support[:, 1:])
        unique_gate = self.unique_gate(torch.cat((f, exclusive, torch.ones_like(f[:, :1])), 1))
        unique_update = self.unique_value(exclusive)
        alpha = self.residual_bound * self.common_scale_raw.tanh()
        beta = self.residual_bound * self.unique_scale_raw.tanh()
        common_residual = alpha.to(f.dtype) * common_gate * common_update * float(self.enable_common)
        unique_residual = beta.to(f.dtype) * unique_gate * unique_update * float(self.enable_unique)
        proposed_update = common_residual + unique_residual
        phases, valid = polyphase_gather(f)
        update_phases, _ = polyphase_gather(proposed_update)
        support_phases, _ = polyphase_gather(support)
        slots = aggregate_evidence_reference(phases, support_phases[:, :1], support_phases[:, 1:], valid, eps=self.eps, tau=self.tau)
        active = (slots['evidence_mass'] > self.mass_threshold) & (slots['reference_mass'] > self.mass_threshold)
        result, constraint = constrain_reference_energy(phases, update_phases, slots['evidence_weights'], slots['reference_weights'], active=active, variance_floor=self.variance_floor, detach_constraint=self.detach_constraint, enable_projection=self.enable_projection)
        method_output = polyphase_restore(result, *size)
        fallback = torch.cat((F.interpolate(coarse, size=size, mode='nearest'), fine), 1)
        output = method_output * float(self.enabled) + fallback * float(not self.enabled)
        if not return_aux:
            return output
        return (output, {**slots, **constraint, 'base': f, 'fine_low': fine_low, 'fine_exclusive': exclusive, 'comparison_fine': left, 'comparison_coarse': right, 'comparison_cosine': cosine, 'common_gate': common_gate, 'unique_gate': unique_gate, 'common_update': common_update, 'unique_update': unique_update, 'common_residual': common_residual, 'unique_residual': unique_residual, 'common_coefficient': alpha, 'unique_coefficient': beta, 'support_logits': logits, 'fine_support_logits': logits, 'fine_support': support, 'predicted_visible': visible, 'predicted_damage': damage, 'phase_values': phases, 'valid_phases': valid, 'proposed_update': proposed_update, 'method_output': method_output, 'fallback': fallback, 'enabled': f.new_tensor(float(self.enabled))})
