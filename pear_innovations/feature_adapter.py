from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from .detector_bridge import DetectorBridgeConfig, DetectorBridge
from .phase_moments import RCPM
from .energy_fusion import LECF
from .ordinal_readout import OrdinalEvidenceReadout
from .grade_support import BasicROIGradeHead
from .roi import sample_rois

@dataclass(frozen=True)
class FeatureConfig(DetectorBridgeConfig):
    fine_roi_size: int = 28
    use_fine_roi: bool = True
    variance_floor: float = 0.1
    projection_enabled: bool = True
    detach_constraint: bool = False
    min_slope: float = 0.25
    max_slope: float = 8.0
    extent_eps: float = 0.0001
    coverage_weight: float = 1.0

class FeatureAdapter(DetectorBridge):

    def __init__(self, master, config: FeatureConfig | None=None):
        config = config or FeatureConfig()
        if config.fine_roi_size < 1 or config.roi_size < 1:
            raise ValueError('ROI grid sizes must be positive')
        super().__init__(master, config)
        layers = self.master.model
        old = layers[3]
        p3_channels = layers[6].conv.in_channels
        p4_channels = layers[14].cv2.conv.out_channels
        roi_channels = layers[17].cv2.conv.out_channels
        self.downsample = RCPM(old.conv.in_channels, old.conv.out_channels, rank=config.phase_rank, enable_residual=config.rcpm, variance_floor=config.variance_floor)
        self.downsample.main = old
        self.fusion = LECF(p3_channels, p4_channels, p3_channels + p4_channels, enabled=config.lecf, enable_projection=config.projection_enabled, variance_floor=config.variance_floor, detach_constraint=config.detach_constraint)
        if config.cseg:
            self.grade_head = OrdinalEvidenceReadout(roi_channels, feature_dim=config.grade_dim, num_grades=3, fine_channels=old.conv.in_channels, use_fine=config.use_fine_roi, extent_eps=config.extent_eps, min_slope=config.min_slope, max_slope=config.max_slope)
        else:
            self.grade_head = BasicROIGradeHead(roi_channels, config.grade_dim, 3)

    def extract(self, images: Tensor) -> dict:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError('RGB BCHW input required')
        if any((s % 32 for s in images.shape[-2:])):
            raise ValueError('pad model input height/width to multiples of 32')
        outputs, x, phase_aux, fusion_aux = ([], images, None, None)
        for i, layer in enumerate(self.master.model):
            if i == 24:
                raw = layer.forward_head([outputs[j] for j in layer.f], **layer.one2many)
                break
            f = layer.f
            inp = x if f == -1 else [x if j == -1 else outputs[j] for j in f] if isinstance(f, list) else outputs[f]
            if i == 3:
                x, phase_aux = self.downsample(inp, return_aux=True)
            elif i == 16:
                x, fusion_aux = self.fusion([outputs[5], outputs[14]], return_aux=True)
            else:
                x = layer(inp)
            outputs.append(x)
        return {'detection': raw, 'roi_feature': outputs[17], 'fine_roi_feature': outputs[2], 'phase_aux': phase_aux, 'fusion_aux': fusion_aux, 'image_size': tuple(images.shape[-2:])}

    def grade(self, features: dict, rois: Tensor) -> dict:
        coarse = sample_rois(features['roi_feature'], rois, features['image_size'], self.config.roi_size)
        if not self.config.cseg:
            return self.grade_head(coarse)
        fine = None
        if self.config.use_fine_roi:
            fine = sample_rois(features['fine_roi_feature'], rois, features['image_size'], self.config.fine_roi_size)
        return self.grade_head(coarse, fine)
