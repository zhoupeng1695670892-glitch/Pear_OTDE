from __future__ import annotations
import math
import torch
from torch import Tensor

@torch.no_grad()
def select_sufficient_grade_rois(candidate_gt_ids: Tensor, fruit_scores: Tensor, statistics: dict[str, Tensor] | None, num_instances: int, *, min_visible_coverage: float=0.9, max_extent_delta: float=0.02, max_critical_delta: float=0.1) -> dict[str, Tensor]:
    if not isinstance(num_instances, int) or isinstance(num_instances, bool) or num_instances < 0:
        raise ValueError('num_instances must be a nonnegative integer')
    if candidate_gt_ids.ndim != 1 or fruit_scores.shape != candidate_gt_ids.shape:
        raise ValueError('candidate_gt_ids and fruit_scores must both be [N]')
    if not torch.isfinite(candidate_gt_ids).all() or (candidate_gt_ids != candidate_gt_ids.round()).any() or (candidate_gt_ids < 0).any() or (candidate_gt_ids >= num_instances).any():
        raise ValueError('candidate IDs must be integers in [0,num_instances)')
    if not torch.isfinite(fruit_scores).all() or ((fruit_scores < 0) | (fruit_scores > 1)).any():
        raise ValueError('fruit_scores must be finite probabilities in [0,1]')
    tolerances = (min_visible_coverage, max_extent_delta, max_critical_delta)
    if not all((math.isfinite(x) and 0 <= x <= 1 for x in tolerances)):
        raise ValueError('selection tolerances must be finite and in [0,1]')
    device = fruit_scores.device
    ids = candidate_gt_ids.detach().to(device=device, dtype=torch.long)
    n = ids.numel()
    eligible = torch.zeros(n, device=device, dtype=torch.bool)
    if statistics is not None:
        required = ('bound_valid', 'captured_fraction', 'delta_extent', 'delta_critical')
        for key in required:
            if key not in statistics or statistics[key].shape != (n,):
                raise ValueError(f'statistics[{key!r}] must be [N]')
        complete = statistics['bound_valid'].to(device=device)
        if not torch.isfinite(complete).all() or ((complete != 0) & (complete != 1)).any():
            raise ValueError('bound_valid must be a finite Boolean/0-1 array')
        coverage, extent, critical = (statistics[key].detach().to(device=device) for key in required[1:])
        if not all((torch.isfinite(x).all() and ((x >= 0) & (x <= 1)).all() for x in (coverage, extent, critical))):
            raise ValueError('candidate statistics must be finite and in [0,1]')
        eligible = complete.bool() & (coverage >= min_visible_coverage) & (extent <= max_extent_delta) & (critical <= max_critical_delta)
    chosen = torch.full((num_instances,), -1, device=device, dtype=torch.long)
    for identity in ids[eligible].unique(sorted=True):
        candidates = torch.where(eligible & (ids == identity))[0]
        chosen[identity] = candidates[fruit_scores.detach()[candidates].argmax()]
    return {'candidate_indices': chosen, 'gt_ids': torch.arange(num_instances, device=device, dtype=torch.long), 'fallback': chosen < 0, 'eligible': eligible}
