from __future__ import annotations
import math
from collections.abc import Sequence
import torch
from torch import Tensor, nn
from torch.nn import functional as F

def _bounded_initial_value(value: float, bound: float) -> float:
    if not math.isfinite(bound) or bound <= 0:
        raise ValueError('residual_bound must be finite and positive')
    if not math.isfinite(value) or abs(value) >= bound:
        raise ValueError('abs(residual_init) must be strictly below residual_bound')
    return math.atanh(value / bound)

def _comparison_dtype(x: Tensor) -> torch.dtype:
    return torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype

class FixedBinomialDownsample(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        kernel = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0
        self.register_buffer('kernel', kernel[None, None])

    def forward(self, x: Tensor) -> Tensor:
        kernel = self.kernel.to(device=x.device, dtype=x.dtype)
        kernel = kernel.expand(x.shape[1], 1, 3, 3)
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode='replicate'), kernel, stride=2, groups=x.shape[1])

class ScaleComparableFusion(nn.Module):

    def __init__(self, fine_channels: int, coarse_channels: int, out_channels: int, *, enable_common: bool=True, enable_unique: bool=True, common_scale_comparison: bool=True, use_fine_support: bool=True, gate_unique_with_common: bool=False, residual_init: float=0.1, residual_bound: float=1.0, normalize_eps: float=1e-06) -> None:
        super().__init__()
        if min(fine_channels, coarse_channels, out_channels) <= 0:
            raise ValueError('all channel counts must be positive')
        if not math.isfinite(normalize_eps) or normalize_eps <= 0:
            raise ValueError('normalize_eps must be finite and positive')
        self.fine_channels = fine_channels
        self.coarse_channels = coarse_channels
        self.out_channels = out_channels
        self.enable_common = enable_common
        self.enable_unique = enable_unique
        self.common_scale_comparison = common_scale_comparison
        self.use_fine_support = use_fine_support
        self.gate_unique_with_common = gate_unique_with_common
        self.residual_bound = residual_bound
        self.normalize_eps = normalize_eps
        self.fine_projection = nn.Conv2d(fine_channels, out_channels, 1, bias=False)
        self.coarse_projection = nn.Conv2d(coarse_channels, out_channels, 1, bias=False)
        self.downsample = FixedBinomialDownsample()
        self.fine_support_head = nn.Conv2d(out_channels, 1, 1)
        self.common_gate = nn.Sequential(nn.Conv2d(3 * out_channels + 1, out_channels, 1), nn.SiLU(), nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels), nn.Sigmoid())
        self.common_value = nn.Sequential(nn.Conv2d(2 * out_channels, out_channels, 1, bias=False), nn.SiLU(), nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False))
        self.unique_gate = nn.Sequential(nn.Conv2d(2 * out_channels + 1, out_channels, 1), nn.SiLU(), nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels), nn.Sigmoid())
        self.unique_value = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False), nn.SiLU(), nn.Conv2d(out_channels, out_channels, 1, bias=False))
        initial = _bounded_initial_value(residual_init, residual_bound)
        self.common_scale_raw = nn.Parameter(torch.tensor(initial))
        self.unique_scale_raw = nn.Parameter(torch.tensor(initial))

    @staticmethod
    def _upsample(x: Tensor, size: tuple[int, int]) -> Tensor:
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def _validate(self, fine: Tensor, coarse: Tensor) -> None:
        if fine.ndim != 4 or coarse.ndim != 4:
            raise ValueError('ScaleComparableFusion requires NCHW fine and coarse tensors')
        if not fine.is_floating_point() or not coarse.is_floating_point():
            raise TypeError('ScaleComparableFusion requires floating point features')
        if fine.device != coarse.device or fine.dtype != coarse.dtype:
            raise ValueError('fine and coarse must have the same device and dtype')
        if fine.shape[0] != coarse.shape[0] or fine.shape[0] < 1:
            raise ValueError('fine and coarse need an equal, nonempty batch')
        if fine.shape[1] != self.fine_channels or coarse.shape[1] != self.coarse_channels:
            raise ValueError('input channels differ from the configured channel counts')
        if min(fine.shape[-2:]) < 1:
            raise ValueError('fine spatial dimensions must be nonempty')
        expected = ((fine.shape[-2] + 1) // 2, (fine.shape[-1] + 1) // 2)
        if coarse.shape[-2:] != expected:
            raise ValueError(f'native coarse grid must be {expected}, got {coarse.shape[-2:]}')

    def forward(self, fine: Tensor | Sequence[Tensor], coarse: Tensor | None=None, *, return_aux: bool=False) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if coarse is None:
            if not isinstance(fine, (tuple, list)) or len(fine) != 2:
                raise ValueError('pass [fine, coarse] or two native-scale tensors')
            fine, coarse = fine
        if not isinstance(fine, Tensor) or not isinstance(coarse, Tensor):
            raise TypeError('fine and coarse must be tensors')
        self._validate(fine, coarse)
        f = self.fine_projection(fine)
        c = self.coarse_projection(coarse)
        size = (f.shape[-2], f.shape[-1])
        fine_coarse = self.downsample(f)
        fine_low = self._upsample(fine_coarse, size)
        exclusive = f - fine_low
        coarse_up = self._upsample(c, size)
        left, right = (fine_coarse, c) if self.common_scale_comparison else (f, coarse_up)
        compare_dtype = _comparison_dtype(left)
        left_normalized = F.normalize(left.to(compare_dtype), dim=1, eps=self.normalize_eps)
        right_normalized = F.normalize(right.to(compare_dtype), dim=1, eps=self.normalize_eps)
        cosine = (left_normalized * right_normalized).sum(1, keepdim=True)
        descriptor = torch.cat((left_normalized, right_normalized, left_normalized - right_normalized, cosine), dim=1).to(f.dtype)
        gs_native = self.common_gate(descriptor)
        gs = self._upsample(gs_native, size) if self.common_scale_comparison else gs_native
        common_update = self.common_value(torch.cat((fine_low, coarse_up), dim=1))
        support_logits = self.fine_support_head(f)
        support = support_logits.sigmoid()
        gate_support = support if self.use_fine_support else torch.ones_like(support)
        ge = self.unique_gate(torch.cat((f, exclusive, gate_support), dim=1))
        if self.gate_unique_with_common:
            ge = ge * gs
        unique_update = self.unique_value(exclusive)
        alpha = self.residual_bound * self.common_scale_raw.tanh()
        beta = self.residual_bound * self.unique_scale_raw.tanh()
        common_residual = alpha.to(f.dtype) * gs * common_update
        unique_residual = beta.to(f.dtype) * ge * unique_update
        common_residual = common_residual * float(self.enable_common)
        unique_residual = unique_residual * float(self.enable_unique)
        output = f + common_residual + unique_residual
        if not return_aux:
            return output
        return (output, {'base': f, 'fine_low': fine_low, 'fine_exclusive': exclusive, 'comparison_fine': left_normalized, 'comparison_coarse': right_normalized, 'comparison_cosine': cosine, 'common_gate': gs, 'unique_gate': ge, 'fine_support_logits': support_logits, 'fine_support': support, 'common_update': common_update, 'unique_update': unique_update, 'common_residual': common_residual, 'unique_residual': unique_residual, 'common_coefficient': alpha, 'unique_coefficient': beta})
