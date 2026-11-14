from __future__ import annotations
import math
import torch
from torch import Tensor, nn
from .phase_base import PhaseSupportCompressor, aggregate_evidence_reference, polyphase_gather

def reference_moments(values: Tensor, evidence_weights: Tensor, reference_weights: Tensor, *, variance_floor: float=0.1) -> dict[str, Tensor]:
    if values.ndim != 5 or values.shape[-1] != 4:
        raise ValueError('phase values must be B,C,Hc,Wc,4')
    expected = (values.shape[0], 1, *values.shape[2:])
    if evidence_weights.shape != expected or reference_weights.shape != expected:
        raise ValueError('phase weights must be B,1,Hc,Wc,4')
    if not math.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError('variance_floor must be finite and positive')
    if any((w.device != values.device for w in (evidence_weights, reference_weights))):
        raise ValueError('phase values and weights must share a device')
    dtype = torch.float32 if values.dtype in (torch.float16, torch.bfloat16) else values.dtype
    z, we, wr = (values.to(dtype), evidence_weights.to(dtype), reference_weights.to(dtype))
    mean_e = (we * z).sum(-1)
    mean_r = (wr * z).sum(-1)
    centered = z - mean_r.unsqueeze(-1)
    squared = centered.square()
    variance_r = (wr * squared).sum(-1)
    energy_e = (we * squared).sum(-1)
    denominator = variance_r + variance_floor
    standardized_e = energy_e / denominator
    standardized_r = variance_r / denominator
    return {'evidence_slot': mean_e, 'reference_slot': mean_r, 'mean_difference': mean_e - mean_r, 'reference_variance': variance_r, 'evidence_energy': energy_e, 'standardized_evidence_energy': standardized_e, 'standardized_reference_energy': standardized_r, 'log_energy_contrast': torch.log1p(standardized_e) - torch.log1p(standardized_r)}

class RCPM(PhaseSupportCompressor):

    def __init__(self, in_channels: int, out_channels: int, rank: int=16, *, enable_residual: bool=True, use_second_order: bool=True, support_mode: str='predicted', use_availability: bool=True, detach_support: bool=False, eps: float=1e-06, tau: float=0.25, variance_floor: float=0.1, residual_init: float=0.1, alpha_max: float=0.5) -> None:
        if not math.isfinite(variance_floor) or variance_floor <= 0:
            raise ValueError('variance_floor must be finite and positive')
        super().__init__(in_channels, out_channels, rank, enable_residual=enable_residual, slot_mode='dual', support_mode=support_mode, use_availability=use_availability, detach_support=detach_support, eps=eps, tau=tau, residual_init=residual_init, alpha_max=alpha_max)
        self.variance_floor = variance_floor
        self.use_second_order = use_second_order
        self.output_projection = nn.Conv2d(3 * rank, out_channels, 1, bias=False)

    def forward(self, x: Tensor, *, return_aux: bool=False):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError('RCPM requires configured NCHW features')
        if not x.is_floating_point():
            raise TypeError('RCPM requires floating point features')
        if x.shape[0] < 1 or min(x.shape[-2:]) < 1:
            raise ValueError('RCPM requires a nonempty batch/spatial grid')
        base = self.main(x)
        z = self.projection(x)
        logits = self.support_head(z)
        dtype = torch.float32 if z.dtype in (torch.float16, torch.bfloat16) else z.dtype
        prediction = logits.to(dtype).sigmoid()
        visible, damage = (prediction[:, :1], prediction[:, 1:])
        if self.support_mode == 'uniform':
            visible, damage = (torch.ones_like(visible), torch.full_like(damage, 0.5))
        if self.detach_support:
            visible, damage = (visible.detach(), damage.detach())
        phases, valid = polyphase_gather(z)
        phase_support, _ = polyphase_gather(torch.cat((visible, damage), dim=1))
        slots = aggregate_evidence_reference(phases, phase_support[:, :1], phase_support[:, 1:], valid, eps=self.eps, tau=self.tau)
        moments = reference_moments(phases, slots['evidence_weights'], slots['reference_weights'], variance_floor=self.variance_floor)
        psi = moments['log_energy_contrast'] * float(self.use_second_order)
        descriptor = torch.cat((moments['reference_slot'], moments['mean_difference'], psi), 1)
        projected = self.output_projection(descriptor.to(z.dtype))
        availability = slots['availability'] if self.use_availability else torch.ones_like(slots['availability'])
        coefficient = self.alpha_max * self.alpha_raw.tanh()
        residual = coefficient.to(projected.dtype) * availability.to(projected.dtype) * projected
        residual = residual * float(self.enable_residual)
        output = base + residual
        if not return_aux:
            return output
        return (output, {**slots, **moments, 'main': base, 'projection': z, 'support_logits': logits, 'predicted_visible': prediction[:, :1], 'predicted_damage': prediction[:, 1:], 'visible_support': visible, 'damage_support': damage, 'phase_values': phases, 'valid_phases': valid, 'used_availability': availability, 'descriptor': descriptor, 'residual_coefficient': coefficient, 'residual': residual})
