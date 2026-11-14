from __future__ import annotations
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F

def polyphase_gather(x: Tensor) -> tuple[Tensor, Tensor]:
    if x.ndim != 4 or min(x.shape[-2:]) < 1:
        raise ValueError('polyphase_gather requires nonempty NCHW features')
    if not x.is_floating_point():
        raise TypeError('polyphase_gather requires floating point input')
    h, w = x.shape[-2:]
    padded = F.pad(x, (0, w % 2, 0, h % 2))
    mask = F.pad(x.new_ones((1, 1, h, w)), (0, w % 2, 0, h % 2))

    def gather(tensor: Tensor) -> Tensor:
        return torch.stack((tensor[..., 0::2, 0::2], tensor[..., 1::2, 0::2], tensor[..., 0::2, 1::2], tensor[..., 1::2, 1::2]), dim=-1)
    return (gather(padded), gather(mask))

def aggregate_evidence_reference(values: Tensor, visible: Tensor, damage: Tensor, valid: Tensor, *, eps: float=1e-06, tau: float=0.25) -> dict[str, Tensor]:
    if values.ndim != 5 or values.shape[-1] != 4:
        raise ValueError('values must have shape B x rank x Hc x Wc x 4')
    if not values.is_floating_point():
        raise TypeError('values must be floating point')
    expected = (values.shape[0], 1, *values.shape[2:])
    if visible.shape != expected or damage.shape != expected:
        raise ValueError(f'visible and damage must have shape {expected}')
    if valid.shape[1:] != expected[1:] or valid.shape[0] not in (1, values.shape[0]):
        raise ValueError("valid mask must broadcast over values' batch/channel axes")
    if any((t.device != values.device for t in (visible, damage, valid))):
        raise ValueError('values and supports must share a device')
    if not math.isfinite(eps) or eps <= 0 or (not math.isfinite(tau)) or (tau <= 0):
        raise ValueError('eps and tau must be finite and positive')
    dtype = torch.float32 if values.dtype in (torch.float16, torch.bfloat16) else values.dtype
    z = values.to(dtype)
    v, d, m = (visible.to(dtype), damage.to(dtype), valid.to(dtype))
    q_e = m * v * d
    q_r = m * v * (1.0 - d)
    s_e = q_e.sum(-1, keepdim=True)
    s_r = q_r.sum(-1, keepdim=True)
    n_valid = m.sum(-1, keepdim=True)
    pseudocount = eps * m / n_valid.clamp_min(1.0)
    w_e = (q_e + pseudocount) / (s_e + eps)
    w_r = (q_r + pseudocount) / (s_r + eps)
    evidence = (w_e * z).sum(-1)
    reference = (w_r * z).sum(-1)
    availability = (s_e / (s_e + tau) * (s_r / (s_r + tau))).squeeze(-1)
    return {'evidence_slot': evidence, 'reference_slot': reference, 'evidence_weights': w_e, 'reference_weights': w_r, 'evidence_mass': s_e.squeeze(-1), 'reference_mass': s_r.squeeze(-1), 'valid_count': n_valid.squeeze(-1), 'availability': availability}

class PhaseSupportCompressor(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, rank: int=16, *, enable_residual: bool=True, slot_mode: str='dual', support_mode: str='predicted', use_availability: bool=True, detach_support: bool=False, eps: float=1e-06, tau: float=0.25, residual_init: float=0.1, alpha_max: float=0.5) -> None:
        super().__init__()
        if min(in_channels, out_channels, rank) <= 0:
            raise ValueError('channel counts and rank must be positive')
        if slot_mode not in ('dual', 'evidence'):
            raise ValueError("slot_mode must be 'dual' or 'evidence'")
        if support_mode not in ('predicted', 'uniform'):
            raise ValueError("support_mode must be 'predicted' or 'uniform'")
        if not math.isfinite(eps) or eps <= 0 or (not math.isfinite(tau)) or (tau <= 0):
            raise ValueError('eps and tau must be finite and positive')
        if not math.isfinite(alpha_max) or alpha_max <= 0:
            raise ValueError('alpha_max must be finite and positive')
        if not math.isfinite(residual_init) or abs(residual_init) >= alpha_max:
            raise ValueError('abs(residual_init) must be strictly below alpha_max')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = rank
        self.enable_residual = enable_residual
        self.slot_mode = slot_mode
        self.support_mode = support_mode
        self.use_availability = use_availability
        self.detach_support = detach_support
        self.eps = eps
        self.tau = tau
        self.alpha_max = alpha_max
        self.main = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)
        self.projection = nn.Conv2d(in_channels, rank, 1, bias=False)
        self.support_head = nn.Sequential(nn.Conv2d(rank, rank, 3, padding=1, groups=rank, bias=False), nn.SiLU(), nn.Conv2d(rank, 2, 1, bias=False))
        slot_channels = 2 * rank if slot_mode == 'dual' else rank
        self.output_projection = nn.Conv2d(slot_channels, out_channels, 1, bias=False)
        self.alpha_raw = nn.Parameter(torch.tensor(math.atanh(residual_init / alpha_max)))

    def forward(self, x: Tensor, *, return_aux: bool=False) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError('PhaseSupportCompressor requires NCHW features with configured channels')
        if not x.is_floating_point():
            raise TypeError('PhaseSupportCompressor requires floating point features')
        if x.shape[0] < 1 or min(x.shape[-2:]) < 1:
            raise ValueError('PhaseSupportCompressor requires nonempty batch and spatial dimensions')
        base = self.main(x)
        z = self.projection(x)
        support_logits = self.support_head(z)
        prob_dtype = torch.float32 if z.dtype in (torch.float16, torch.bfloat16) else z.dtype
        predicted_support = support_logits.to(prob_dtype).sigmoid()
        visible, damage = (predicted_support[:, :1], predicted_support[:, 1:])
        if self.support_mode == 'uniform':
            visible = torch.ones_like(visible)
            damage = torch.full_like(damage, 0.5)
        if self.detach_support:
            visible, damage = (visible.detach(), damage.detach())
        phase_values, valid = polyphase_gather(z)
        phase_support, _ = polyphase_gather(torch.cat((visible, damage), dim=1))
        slots = aggregate_evidence_reference(phase_values, phase_support[:, :1], phase_support[:, 1:], valid, eps=self.eps, tau=self.tau)
        evidence, reference = (slots['evidence_slot'], slots['reference_slot'])
        content = torch.cat((evidence - reference, reference), dim=1) if self.slot_mode == 'dual' else evidence
        projected = self.output_projection(content.to(z.dtype))
        availability = slots['availability'] if self.use_availability else torch.ones_like(slots['availability'])
        coefficient = self.alpha_max * self.alpha_raw.tanh()
        residual = coefficient.to(projected.dtype) * availability.to(projected.dtype) * projected
        residual = residual * float(self.enable_residual)
        output = base + residual
        if not return_aux:
            return output
        return (output, {**slots, 'main': base, 'projection': z, 'support_logits': support_logits, 'predicted_visible': predicted_support[:, :1], 'predicted_damage': predicted_support[:, 1:], 'visible_support': visible, 'damage_support': damage, 'phase_values': phase_values, 'valid_phases': valid, 'used_availability': availability, 'residual_coefficient': coefficient, 'residual': residual})
