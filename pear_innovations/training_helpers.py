from __future__ import annotations
import math
from typing import Callable
import torch
from torch import Tensor
from torch.nn import functional as F
from .grade_support import EVIDENCE_NAMES, evidence_supervision_loss
from .box_geometry import instance_mean
from .statistic_baseline import PointStatisticIoULoss
from .roi import reframe_gt_masks

def global_gt_indices(batch_idx: Tensor, foreground: Tensor, local_ids: Tensor) -> Tensor:
    if foreground.shape != local_ids.shape or foreground.ndim != 2:
        raise ValueError('TAL foreground and local IDs must be [B,A]')
    image_indices, _ = torch.where(foreground)
    if not image_indices.numel():
        return image_indices
    flat = batch_idx.flatten().to(device=foreground.device, dtype=torch.long)
    result = torch.empty_like(image_indices)
    selected_local = local_ids[foreground].long()
    for b in image_indices.unique():
        original = torch.where(flat == b)[0]
        positions = image_indices == b
        ids = selected_local[positions]
        if (ids < 0).any() or (ids >= original.numel()).any():
            raise ValueError('TAL GT index does not match input instance ordering')
        result[positions] = original[ids]
    return result

def one_prediction_per_instance(ids: Tensor, scores: Tensor) -> Tensor:
    if ids.ndim != 1 or ids.shape != scores.shape or (not torch.isfinite(scores).all()):
        raise ValueError('instance IDs and finite quality scores must share shape [N]')
    chosen = []
    for identity in ids.unique(sorted=True):
        idx = torch.where(ids == identity)[0]
        chosen.append(idx[scores[idx].detach().argmax()])
    return torch.stack(chosen) if chosen else ids.new_empty(0)

def _sanitize_joint_maps(joint: Tensor, availability: Tensor) -> tuple[Tensor, Tensor]:
    if joint.ndim != 4 or joint.shape[1] != 3 or min(joint.shape[-2:]) < 1 or (availability.shape != joint.shape[:2]):
        raise ValueError('joint evidence [N,3,H,W] and availability [N,3] required')
    availability = availability.to(device=joint.device)
    if not torch.isfinite(availability).all() or ((availability != 0) & (availability != 1)).any():
        raise ValueError('evidence availability must be Boolean or 0/1')
    availability = availability.bool()
    known = availability[:, :, None, None].expand_as(joint)
    invalid = ~torch.isfinite(joint) | (joint < 0) | (joint > 1)
    if (known & invalid).any():
        raise ValueError('annotated evidence groups must be finite in [0,1]')
    joint = torch.where(known, joint, torch.zeros_like(joint))
    for child in (1, 2):
        both_known = (availability[:, child] & availability[:, child - 1])[:, None, None]
        if (both_known & (joint[:, child] > joint[:, child - 1] + 1e-06)).any():
            raise ValueError('known evidence groups must satisfy critical <= damage <= visible')
    visible_critical_known = (availability[:, 0] & availability[:, 2])[:, None, None]
    if (visible_critical_known & (joint[:, 2] > joint[:, 0] + 1e-06)).any():
        raise ValueError('known critical support must not exceed visible support')
    return (joint, availability)

def _grade_targets_from_joint_maps(joint: Tensor, availability: Tensor) -> tuple[dict, dict]:
    joint, availability = _sanitize_joint_maps(joint, availability)
    v, joint_d, joint_s = joint.split(1, dim=1)
    d = torch.where(v > 1e-06, joint_d / v.clamp_min(1e-06), torch.zeros_like(v))
    s = torch.where(joint_d > 1e-06, joint_s / joint_d.clamp_min(1e-06), torch.zeros_like(v))
    conditional_available = availability.to(torch.int64).cumprod(1).bool()
    d = torch.where(conditional_available[:, 1, None, None, None], d, torch.zeros_like(d))
    s = torch.where(conditional_available[:, 2, None, None, None], s, torch.zeros_like(s))
    targets = dict(zip(EVIDENCE_NAMES, (v, d, s)))
    valid = {name: conditional_available[:, i, None, None, None].to(joint.dtype).expand_as(v) for i, name in enumerate(EVIDENCE_NAMES)}
    return (targets, valid)

def shallow_evidence_loss(phase_aux: dict, targets: Tensor | None=None, valid: Tensor | None=None) -> Tensor:
    logits = phase_aux['support_logits'].float()
    if targets is None:
        return logits.sum() * 0
    if valid is None or targets.shape != valid.shape or targets.ndim != 4 or (targets.shape[:2] != logits.shape[:2]):
        raise ValueError('shallow labels require matching [B,2,H,W] targets and valid masks')
    target = F.interpolate(targets.detach().to(logits), logits.shape[-2:], mode='nearest')
    weight = F.interpolate(valid.detach().to(logits), logits.shape[-2:], mode='nearest')
    if not torch.isfinite(weight).all() or ((weight < 0) | (weight > 1)).any():
        raise ValueError('shallow validity must be in [0,1]')
    known_visible = weight[:, 0:1] > 0
    visible = target[:, 0:1]
    if (known_visible & (~torch.isfinite(visible) | (visible < 0) | (visible > 1))).any():
        raise ValueError('annotated shallow visible targets must be finite in [0,1]')
    visible = torch.where(known_visible, visible, torch.zeros_like(visible))
    weight = weight.clone()
    weight[:, 1:2] *= visible * weight[:, 0:1]
    known = weight > 0
    if (known & (~torch.isfinite(target) | (target < 0) | (target > 1))).any():
        raise ValueError('annotated shallow targets must be finite in [0,1]')
    target = torch.where(known, target, torch.zeros_like(target))
    per_pixel = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    denom = weight.sum((-2, -1))
    active = denom > 0
    value = (per_pixel * weight).sum((-2, -1)) / denom.clamp_min(1e-06)
    return (value * active).sum() / active.sum().clamp_min(1)

def point_readout_training_loss(adapter, batch: dict, criterion, *, retention_weight: float=0.25, grade_weight: float=1.0, evidence_weight: float=0.1, shallow_weight: float=0.1, fusion_support_weight: float=0.1, compose_native: Callable | None=None) -> dict:
    for w in (retention_weight, grade_weight, evidence_weight, shallow_weight, fusion_support_weight):
        if not math.isfinite(w) or w < 0:
            raise ValueError('loss weights must be finite and nonnegative')
    if not adapter.training:
        raise RuntimeError('training loss requires model.train()')
    native = getattr(criterion, 'native_criterion', criterion)
    if not hasattr(native, 'get_assigned_targets_and_loss'):
        raise TypeError('requires audited v8DetectionLoss TAL interface')
    box_criterion = getattr(native, 'bbox_loss', None)
    if str(getattr(box_criterion, 'iou_loss', '')).lower() != 'ciou' or str(getattr(box_criterion, 'iou_aux', 'none')).lower() != 'none' or bool(getattr(box_criterion, 'use_wiseiou', False)):
        raise ValueError('Point-statistic integration requires native CIoU; DFL stays native')
    if getattr(native, 'nc', 1) != 1:
        raise ValueError('native detection criterion must use one fruit class')
    grade_values = batch.get('grades', batch['cls']).flatten()
    if not torch.isfinite(grade_values).all() or ((grade_values != grade_values.round()) | (grade_values < 0) | (grade_values > 2)).any():
        raise ValueError('grade IDs must be 0/1/2 (A/B/C)')
    grades = grade_values.long()
    batch_ids = batch['batch_idx'].flatten()
    if batch_ids.numel() != grades.numel() or batch['cls'].numel() != grades.numel() or batch['bboxes'].shape != (grades.numel(), 4):
        raise ValueError('native boxes, classes, batch indices and grades must agree on M instances')
    if not torch.isfinite(batch_ids).all() or ((batch_ids != batch_ids.round()) | (batch_ids < 0) | (batch_ids >= batch['img'].shape[0])).any():
        raise ValueError('batch_idx must contain integer image indices in [0,B)')
    order = torch.argsort(batch_ids, stable=True)
    fruit_batch = dict(batch)
    fruit_batch['batch_idx'] = batch_ids[order]
    fruit_batch['cls'] = torch.zeros_like(batch['cls'].reshape(-1, 1))
    fruit_batch['bboxes'] = batch['bboxes'].index_select(0, order.to(batch['bboxes'].device))
    features = adapter(batch['img'])
    raw = features['detection']
    assignment, native_vector, native_items = native.get_assigned_targets_and_loss(raw, fruit_batch)
    fg, local_gt, target_boxes, anchors, strides = assignment
    native_loss = native_vector * batch['img'].shape[0]
    if compose_native is not None:
        native_loss, native_items = compose_native(adapter.master, native_loss, native_items)
    elif hasattr(criterion, 'native_criterion'):
        from ultralytics.nn.mixture_loss import compose_native_result
        native_loss, native_items = compose_native_result(adapter.master, native_loss, native_items)
    decoded = native.bbox_decode(anchors, raw['boxes'].permute(0, 2, 1).contiguous()) * strides
    ids = global_gt_indices(batch['batch_idx'], fg, local_gt)
    bidx, _ = torch.where(fg)
    p, g = (decoded[fg], target_boxes[fg])
    evidence = batch.get('evidence')
    available = batch.get('evidence_valid')
    if evidence is not None:
        if available is None:
            raise ValueError('evidence_valid required; missing annotations are not negatives')
        evidence = evidence.detach().to(p)
        available = available.detach().to(device=p.device)
        evidence, available = _sanitize_joint_maps(evidence, available)
        if evidence.shape[0] != grades.numel():
            raise ValueError('evidence must follow original flattened GT instance ordering')
    retain = PointStatisticIoULoss(max_slope=adapter.config.max_slope, extent_eps=adapter.config.extent_eps, coverage_weight=adapter.config.coverage_weight).retention(p, g, None if evidence is None else evidence[ids], None if available is None else available[ids])
    retention = instance_mean(retain['retention'], ids)
    quality = raw['scores'][:, 0].sigmoid()[fg]
    chosen = one_prediction_per_instance(ids, quality)
    rois = torch.cat((bidx[chosen, None].to(p), p[chosen].detach()), -1)
    grade_out = adapter.grade(features, rois)
    grade_loss = F.cross_entropy(grade_out['grade_logits'], grades.to(p.device)[ids[chosen]]) if chosen.numel() else grade_out['grade_logits'].sum() * 0
    if evidence is not None:
        target_size = grade_out['aux']['evidence_logits']['visible'].shape[-1]
        warped = reframe_gt_masks(evidence[ids[chosen]], g[chosen], p[chosen].detach(), target_size)
        targets, valid = _grade_targets_from_joint_maps(warped, available[ids[chosen]])
        evidence_loss = evidence_supervision_loss(grade_out, targets, valid_masks=valid)['loss']
    else:
        evidence_loss = evidence_supervision_loss(grade_out)['loss']
    shallow = shallow_evidence_loss(features['phase_aux'], batch.get('shallow_targets'), batch.get('shallow_valid'))
    fusion_support = shallow_evidence_loss(features['fusion_aux'], batch.get('shallow_targets'), batch.get('shallow_valid'))
    total = native_loss.sum() + grade_weight * grade_loss + retention_weight * retention + evidence_weight * evidence_loss + shallow_weight * shallow + fusion_support_weight * fusion_support
    return {'loss': total, 'native_items': native_items, 'grade_loss': grade_loss, 'retention_loss': retention, 'evidence_loss': evidence_loss, 'shallow_loss': shallow, 'fusion_support_loss': fusion_support, 'retention_details': retain, 'positive_anchors': fg.sum().detach(), 'graded_instances': chosen.numel(), 'grade_output': grade_out}
