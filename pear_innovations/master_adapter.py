from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import Tensor
from .feature_adapter import FeatureAdapter, FeatureConfig
from .support_grading import CSEGGradeHead
from .grading_iou import normalized_soft_capture
from .roi import sample_rois

def expanded_context_rois(rois: Tensor, scale: float=1.25) -> Tensor:
    if rois.ndim != 2 or rois.shape[1] != 5 or (not torch.isfinite(rois).all()):
        raise ValueError('finite [N,5] batch-index/xyxy ROIs required')
    if not math.isfinite(scale) or scale < 1:
        raise ValueError('context scale must be finite and at least one')
    if (rois[:, 3:5] - rois[:, 1:3] <= 0).any():
        raise ValueError('ROI area must be positive')
    center = (rois[:, 1:3] + rois[:, 3:5]) / 2
    half = (rois[:, 3:5] - rois[:, 1:3]) * (scale / 2)
    return torch.cat((rois[:, :1], center - half, center + half), -1)

@dataclass(frozen=True)
class PearConfig(FeatureConfig):
    roi_size: int = 18
    fine_roi_size: int = 35
    context_scale: float = 1.25
    component_threshold: float = 0.5
    component_power: float = 2.0
    capture_temperature: float = 0.04
    min_visible_coverage: float = 0.9
    max_extent_delta: float = 0.02
    max_critical_delta: float = 0.1
    response_crop_fraction: float = 0.65

class PearMasterAdapter(FeatureAdapter):

    def __init__(self, master, config: PearConfig | None=None):
        config = config or PearConfig()
        if not math.isfinite(config.context_scale) or config.context_scale < 1:
            raise ValueError('context_scale must be finite and at least one')
        super().__init__(master, config)
        if config.cseg:
            layers = self.master.model
            self.grade_head = CSEGGradeHead(layers[17].cv2.conv.out_channels, feature_dim=config.grade_dim, fine_channels=layers[3].conv.in_channels, use_fine=config.use_fine_roi, extent_eps=config.extent_eps, min_slope=config.min_slope, max_slope=config.max_slope, critical_threshold=config.component_threshold, critical_power=config.component_power)

    def grade_in_context(self, features: dict, reference_rois: Tensor, query_boxes: Tensor, *, calibration_detach: bool=False) -> dict:
        if query_boxes.ndim not in (2, 3) or query_boxes.shape[-1] != 4:
            raise ValueError('query boxes must be [N,4] or [N,Q,4]')
        if query_boxes.shape[0] != reference_rois.shape[0]:
            raise ValueError('one query group per reference context required')
        multi = query_boxes.ndim == 3
        queries = query_boxes if multi else query_boxes[:, None]
        n, q = queries.shape[:2]
        if q < 1:
            raise ValueError('at least one query required')
        coarse = sample_rois(features['roi_feature'], reference_rois, features['image_size'], self.config.roi_size)
        if not self.config.cseg:
            if multi or calibration_detach:
                raise ValueError('structured query calibration requires CSEG enabled')
            result = self.grade_head(coarse)
            result['reference_rois'] = reference_rois
            return result
        fine = sample_rois(features['fine_roi_feature'], reference_rois, features['image_size'], self.config.fine_roi_size) if self.config.use_fine_roi else None
        ctx = self.grade_head.encode_context(coarse, fine)
        h, w = ctx['used_evidence_probs']['visible'].shape[-2:]
        captures = normalized_soft_capture(queries.reshape(-1, 4), reference_rois[:, None, 1:].expand(-1, q, -1).reshape(-1, 4), h, w, temp=self.config.capture_temperature).reshape(n, q, h, w)
        result = self.grade_head.readout_context(ctx, captures, calibration_detach=calibration_detach)
        result['reference_rois'] = reference_rois
        result['query_capture'] = captures
        return result

    def grade(self, features: dict, rois: Tensor) -> dict:
        reference = expanded_context_rois(rois, self.config.context_scale)
        return self.grade_in_context(features, reference, rois[:, 1:])
