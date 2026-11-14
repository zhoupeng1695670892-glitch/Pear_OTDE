from __future__ import annotations
import math
import torch
from torch import Tensor, nn

def _work_dtype(x: Tensor) -> torch.dtype:
    return torch.float64 if x.dtype == torch.float64 else torch.float32

def aligned_iou(pred: Tensor, target: Tensor, *, ciou: bool=True, eps: float=1e-07) -> Tensor:
    if pred.ndim != 2 or pred.shape[-1] != 4 or target.shape != pred.shape:
        raise ValueError('pred and target must both have shape [N,4]')
    p, g = (pred.to(_work_dtype(pred)), target.detach().to(device=pred.device, dtype=_work_dtype(pred)))
    if not torch.isfinite(p).all() or not torch.isfinite(g).all():
        raise ValueError('boxes must be finite')
    pw, gw = (p[:, 2:] - p[:, :2], g[:, 2:] - g[:, :2])
    if (pw <= 0).any() or (gw <= 0).any():
        raise ValueError('xyxy boxes must have strictly positive width and height')
    inter_wh = (torch.minimum(p[:, 2:], g[:, 2:]) - torch.maximum(p[:, :2], g[:, :2])).clamp_min(0)
    inter = inter_wh.prod(-1)
    union = pw.prod(-1) + gw.prod(-1) - inter
    iou = inter / union.clamp_min(eps)
    if not ciou:
        return iou
    enclosure = torch.maximum(p[:, 2:], g[:, 2:]) - torch.minimum(p[:, :2], g[:, :2])
    center_distance = ((p[:, :2] + p[:, 2:] - g[:, :2] - g[:, 2:]) * 0.5).square().sum(-1)
    v = 4 / math.pi ** 2 * (torch.atan(gw[:, 0] / gw[:, 1]) - torch.atan(pw[:, 0] / pw[:, 1])).square()
    alpha = (v / (1 - iou + v).clamp_min(eps)).detach()
    return iou - center_distance / enclosure.square().sum(-1).clamp_min(eps) - alpha * v

def instance_mean(values: Tensor, instance_ids: Tensor | None=None) -> Tensor:
    if values.ndim != 1:
        raise ValueError('values must be [N]')
    if not values.numel():
        return values.sum() * 0
    if instance_ids is None:
        return values.mean()
    if instance_ids.shape != values.shape:
        raise ValueError('instance_ids must match values')
    _, inverse, counts = torch.unique(instance_ids.to(values.device), return_inverse=True, return_counts=True)
    sums = values.new_zeros(counts.numel()).scatter_add(0, inverse, values)
    return (sums / counts.to(values.dtype)).mean()

class SupportCoverageIoULoss(nn.Module):

    def __init__(self, weight: float=0.25, temperature: float=0.04, reduction: str='mean', ciou: bool=True, group_reduction: str='max', min_mass: float=1e-06, eps: float=1e-07):
        super().__init__()
        if not all((math.isfinite(v) for v in (weight, temperature, min_mass, eps))) or weight < 0 or temperature <= 0 or (min_mass <= 0) or (eps <= 0):
            raise ValueError('weight must be nonnegative; temperature/mass/eps positive')
        if reduction not in ('none', 'mean', 'sum') or group_reduction not in ('max', 'mean'):
            raise ValueError('invalid reduction')
        self.weight, self.temperature, self.reduction = (weight, temperature, reduction)
        self.ciou, self.group_reduction, self.min_mass, self.eps = (ciou, group_reduction, min_mass, eps)

    def retention(self, pred: Tensor, target: Tensor, evidence: Tensor | None, group_valid: Tensor | None=None) -> dict[str, Tensor]:
        aligned_iou(pred, target, ciou=False, eps=self.eps)
        p = pred.to(_work_dtype(pred))
        n = p.shape[0]
        if evidence is None:
            zero = p.sum(-1) * 0
            return {'retention': zero, 'per_group': p.new_zeros(n, 0), 'valid_groups': torch.zeros((n, 0), device=p.device, dtype=torch.bool)}
        if evidence.ndim != 4 or evidence.shape[0] != n or min(evidence.shape[1:]) < 1:
            raise ValueError('evidence must be [N,K,H,W] with nonempty K,H,W')
        e = evidence.detach().to(device=p.device, dtype=p.dtype)
        if not torch.isfinite(e).all() or (e < 0).any() or (e > 1).any():
            raise ValueError('evidence values must be finite probabilities in [0,1]')
        k, h, w = e.shape[1:]
        valid = torch.ones((n, k), dtype=torch.bool, device=p.device)
        if group_valid is not None:
            if group_valid.shape != (n, k):
                raise ValueError('group_valid must be [N,K]')
            if not torch.isfinite(group_valid).all() or ((group_valid != 0) & (group_valid != 1)).any():
                raise ValueError('group_valid must contain only finite 0/1 availability flags')
            valid = group_valid.detach().to(device=p.device, dtype=torch.bool)
        g = target.detach().to(device=p.device, dtype=p.dtype)
        wh = g[:, 2:] - g[:, :2]
        pn = (p - torch.cat((g[:, :2], g[:, :2]), -1)) / torch.cat((wh, wh), -1)
        yy, xx = torch.meshgrid((torch.arange(h, device=p.device, dtype=p.dtype) + 0.5) / h, (torch.arange(w, device=p.device, dtype=p.dtype) + 0.5) / w, indexing='ij')

        def membership(box: Tensor) -> Tensor:
            l, t, r, b = [box[:, i, None, None] for i in range(4)]
            tau = self.temperature
            return torch.sigmoid((xx - l) / tau) * torch.sigmoid((r - xx) / tau) * torch.sigmoid((yy - t) / tau) * torch.sigmoid((b - yy) / tau)
        bp = membership(pn)
        reference = p.new_tensor([0.0, 0.0, 1.0, 1.0]).expand(n, -1)
        bg = membership(reference)
        mass = (e * bg[:, None]).sum((-2, -1))
        valid = valid & (mass >= self.min_mass)
        per_group = (e * torch.relu(bg - bp)[:, None]).sum((-2, -1)) / mass.clamp_min(self.eps)
        per_group = torch.where(valid, per_group, torch.zeros_like(per_group))
        if self.group_reduction == 'max':
            retention = per_group.max(-1).values
        else:
            retention = per_group.sum(-1) / valid.sum(-1).clamp_min(1)
        return {'retention': retention, 'per_group': per_group, 'valid_groups': valid, 'membership_pred': bp, 'membership_target': bg, 'support_mass': mass}

    def forward(self, pred: Tensor, target: Tensor, evidence: Tensor | None=None, group_valid: Tensor | None=None, instance_ids: Tensor | None=None, *, return_details: bool=False):
        geometry = 1 - aligned_iou(pred, target, ciou=self.ciou, eps=self.eps)
        info = self.retention(pred, target, evidence, group_valid)
        per_box = geometry + self.weight * info['retention']
        if self.reduction == 'none':
            loss = per_box
        elif self.reduction == 'sum':
            loss = per_box.sum()
        else:
            loss = instance_mean(per_box, instance_ids)
        if return_details:
            return (loss, dict(info, geometry=geometry, per_box=per_box))
        return loss
