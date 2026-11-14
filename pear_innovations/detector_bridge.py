from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor, nn
from .fusion_base import ScaleComparableFusion
from .phase_base import PhaseSupportCompressor
from .grade_support import EvidencePoolingGradeHead, BasicROIGradeHead
from .roi import sample_rois, nms

@dataclass(frozen=True)
class DetectorBridgeConfig:
    lecf: bool = True
    cseg: bool = True
    rcpm: bool = True
    roi_size: int = 14
    grade_dim: int = 64
    phase_rank: int = 16
    tail_fraction: float = 0.1
    max_detections: int = 100
    pre_nms_topk: int = 1000
    score_threshold: float = 0.05
    nms_iou: float = 0.6

class DetectorBridge(nn.Module):

    def __init__(self, master: nn.Module, config: DetectorBridgeConfig | None=None):
        super().__init__()
        self.config = config or DetectorBridgeConfig()
        self.master = master
        layers = master.model
        if len(layers) != 25 or list(layers[-1].f) != [17, 20, 23]:
            raise ValueError('adapter supports the audited 25-node Master v0.10 graph only')
        if getattr(layers[-1], 'nc', None) != 1 or getattr(layers[-1], 'end2end', False):
            raise ValueError('construct the native DetectionModel with nc=1, end2end=False; retain grades in batch')
        if list(layers[16].f) != [-1, 5] or list(layers[13].f) != [-1, 8]:
            raise ValueError('unexpected native fusion graph')
        for i in (3, 6, 18):
            if not isinstance(getattr(layers[i], 'conv', None), nn.Conv2d):
                raise ValueError(f'node {i} must have the original Conv interface')
        p3_channels = layers[6].conv.in_channels
        p4_channels = layers[14].cv2.conv.out_channels
        roi_channels = layers[17].cv2.conv.out_channels
        if layers[17].cv1.conv.in_channels != p3_channels + p4_channels:
            raise ValueError('unexpected C3k2 input channels at node 17')
        old = layers[3]
        self.downsample = PhaseSupportCompressor(old.conv.in_channels, old.conv.out_channels, rank=self.config.phase_rank, enable_residual=self.config.rcpm)
        self.downsample.main = old
        self.fusion = ScaleComparableFusion(p3_channels, p4_channels, p3_channels + p4_channels, use_fine_support=False) if self.config.lecf else None
        head_cls = EvidencePoolingGradeHead if self.config.cseg else BasicROIGradeHead
        kwargs = dict(in_channels=roi_channels, feature_dim=self.config.grade_dim, num_grades=3)
        if self.config.cseg:
            kwargs['tail_fraction'] = self.config.tail_fraction
        self.grade_head = head_cls(**kwargs)
        self.stride = master.stride

    @property
    def detection_head(self):
        return self.master.model[-1]

    def extract(self, images: Tensor) -> dict:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError('RGB BCHW input required')
        if any((s % 32 for s in images.shape[-2:])):
            raise ValueError('pad model input height/width to multiples of 32')
        outputs, x, phase_aux, fusion_aux = ([], images, None, None)
        for i, layer in enumerate(self.master.model):
            if i == 24:
                feats = [outputs[j] for j in layer.f]
                raw = layer.forward_head(feats, **layer.one2many)
                break
            f = layer.f
            inp = x if f == -1 else [x if j == -1 else outputs[j] for j in f] if isinstance(f, list) else outputs[f]
            if i == 3:
                x, phase_aux = self.downsample(inp, return_aux=True)
            elif i == 16 and self.fusion is not None:
                x, fusion_aux = self.fusion([outputs[5], outputs[14]], return_aux=True)
            else:
                x = layer(inp)
            outputs.append(x)
        return {'detection': raw, 'roi_feature': outputs[17], 'phase_aux': phase_aux, 'fusion_aux': fusion_aux, 'image_size': tuple(images.shape[-2:])}

    def grade(self, features: dict, rois: Tensor) -> dict:
        crops = sample_rois(features['roi_feature'], rois, features['image_size'], self.config.roi_size)
        return self.grade_head(crops)

    def forward(self, images: Tensor, rois: Tensor | None=None) -> dict:
        out = self.extract(images)
        if rois is not None:
            out['grading'] = self.grade(out, rois)
        return out

    @torch.no_grad()
    def predict(self, images: Tensor) -> list[dict[str, Tensor]]:
        if self.training:
            raise RuntimeError('call eval() before predict()')
        features = self.extract(images)
        raw = features['detection']
        head = self.detection_head
        decoded = head._get_decode_boxes(raw).transpose(1, 2)
        if not getattr(head, 'xyxy', False):
            center, wh = (decoded[..., :2], decoded[..., 2:])
            decoded = torch.cat((center - wh / 2, center + wh / 2), -1)
        scores = raw['scores'][:, 0].sigmoid()
        rois, kept_scores, counts = ([], [], [])
        ih, iw = features['image_size']
        for b, (boxes, quality) in enumerate(zip(decoded, scores)):
            boxes = boxes.clone()
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, iw)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, ih)
            valid = (boxes[:, 2:] > boxes[:, :2]).all(-1) & (quality >= self.config.score_threshold) & torch.isfinite(boxes).all(-1) & torch.isfinite(quality)
            idx = torch.where(valid)[0]
            idx = idx[quality[idx].argsort(descending=True)[:self.config.pre_nms_topk]]
            idx = idx[nms(boxes[idx], quality[idx], self.config.nms_iou, self.config.max_detections)]
            rois.append(torch.cat((boxes.new_full((idx.numel(), 1), b), boxes[idx]), -1))
            kept_scores.append(quality[idx])
            counts.append(idx.numel())
        all_rois = torch.cat(rois, 0)
        probs = self.grade(features, all_rois)['grade_logits'].softmax(-1)
        result, offset = ([], 0)
        for roi, quality, count in zip(rois, kept_scores, counts):
            p = probs[offset:offset + count]
            result.append({'boxes': roi[:, 1:], 'fruit_scores': quality, 'grade_probabilities': p, 'grade_scores': quality[:, None] * p, 'grades': p.argmax(-1)})
            offset += count
        return result
