from __future__ import annotations
import torch
from torch import Tensor
from torch.nn import functional as F

def sample_rois(feature: Tensor, rois: Tensor, image_size: tuple[int, int], output_size: int=14) -> Tensor:
    if feature.ndim != 4 or rois.ndim != 2 or rois.shape[1] != 5:
        raise ValueError('expected BCHW features and Nx5 rois')
    if not feature.is_floating_point() or min(feature.shape[1:]) < 1:
        raise ValueError('features must be floating point with nonempty channels and spatial dimensions')
    ih, iw = image_size
    if output_size < 1 or min(ih, iw) < 1:
        raise ValueError('sizes must be positive')
    if not rois.shape[0]:
        return feature[:0, :, :1, :1].expand(0, feature.shape[1], output_size, output_size)
    dtype = torch.float64 if feature.dtype == torch.float64 else torch.float32
    r = rois.to(device=feature.device, dtype=dtype)
    batch_idx = r[:, 0].long()
    if ((batch_idx < 0) | (batch_idx >= feature.shape[0])).any() or not torch.equal(r[:, 0], batch_idx.to(dtype)):
        raise ValueError('invalid ROI batch indices')
    if not torch.isfinite(r).all() or (r[:, 3:] <= r[:, 1:3]).any():
        raise ValueError('ROI boxes must be finite, ordered xyxy')
    u = (torch.arange(output_size, device=r.device, dtype=dtype) + 0.5) / output_size
    y, x = torch.meshgrid(u, u, indexing='ij')
    gx = r[:, 1, None, None] + x * (r[:, 3] - r[:, 1])[:, None, None]
    gy = r[:, 2, None, None] + y * (r[:, 4] - r[:, 2])[:, None, None]
    grid = torch.stack((2 * gx / iw - 1, 2 * gy / ih - 1), -1)
    sampled = F.grid_sample(feature[batch_idx].to(dtype), grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return sampled.to(feature.dtype)

def reframe_gt_masks(masks: Tensor, gt_boxes: Tensor, roi_boxes: Tensor, output_size: int=14) -> Tensor:
    if masks.ndim != 4 or gt_boxes.shape != roi_boxes.shape or gt_boxes.shape != (masks.shape[0], 4):
        raise ValueError('masks [N,K,H,W] and matching Nx4 boxes required')
    n, _, h, w = masks.shape
    if output_size < 1 or min(masks.shape[1:]) < 1:
        raise ValueError('mask channels, spatial size and output_size must be positive')
    dtype = torch.float64 if masks.dtype == torch.float64 else torch.float32
    masks = masks.to(dtype=dtype)
    gt_boxes = gt_boxes.detach().to(device=masks.device, dtype=dtype)
    roi_boxes = roi_boxes.to(device=masks.device, dtype=dtype)
    if not torch.isfinite(gt_boxes).all() or not torch.isfinite(roi_boxes).all():
        raise ValueError('GT and ROI boxes must be finite')
    if (gt_boxes[:, 2:] - gt_boxes[:, :2] <= 0).any() or (roi_boxes[:, 2:] - roi_boxes[:, :2] <= 0).any():
        raise ValueError('GT and ROI boxes must be ordered xyxy')
    if n == 0:
        return masks[:, :, :1, :1].expand(n, masks.shape[1], output_size, output_size)
    wh = gt_boxes[:, 2:] - gt_boxes[:, :2]
    if (wh <= 0).any():
        raise ValueError('invalid GT boxes')
    normalized = (roi_boxes - gt_boxes[:, :2].repeat(1, 2)) / wh.repeat(1, 2)
    coords = normalized * masks.new_tensor([w, h, w, h])
    rois = torch.cat((torch.arange(n, device=masks.device)[:, None], coords), -1)
    return sample_rois(masks, rois, (h, w), output_size)

def box_iou_matrix(a: Tensor, b: Tensor) -> Tensor:
    wh = (torch.minimum(a[:, None, 2:], b[None, :, 2:]) - torch.maximum(a[:, None, :2], b[None, :, :2])).clamp_min(0)
    inter = wh.prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp_min(0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp_min(0).prod(-1)
    return inter / (area_a[:, None] + area_b[None] - inter).clamp_min(1e-07)

@torch.no_grad()
def nms(boxes: Tensor, scores: Tensor, iou_threshold: float=0.6, max_detections: int=100) -> Tensor:
    if scores.ndim != 1 or boxes.shape != (scores.numel(), 4) or (not 0 <= iou_threshold <= 1) or (max_detections < 1):
        raise ValueError('invalid NMS inputs/configuration')
    if boxes.device != scores.device or not torch.isfinite(boxes).all() or (not torch.isfinite(scores).all()):
        raise ValueError('NMS boxes and scores must be finite and share a device')
    if (boxes[:, 2:] - boxes[:, :2] <= 0).any():
        raise ValueError('NMS boxes must be ordered xyxy')
    order = scores.argsort(descending=True, stable=True)
    keep = []
    while order.numel() and len(keep) < max_detections:
        i = order[0]
        keep.append(i)
        remaining = order[1:]
        if not remaining.numel():
            break
        overlap = box_iou_matrix(boxes[i:i + 1], boxes[remaining])[0]
        order = remaining[overlap <= iou_threshold]
    return torch.stack(keep) if keep else torch.empty(0, dtype=torch.long, device=boxes.device)
