from __future__ import annotations
import math
from typing import Callable
import torch
from torch import Tensor
from torch.nn import functional as F
from .training_helpers import global_gt_indices, _sanitize_joint_maps, _grade_targets_from_joint_maps, shallow_evidence_loss
from .grade_support import evidence_supervision_loss
from .ordinal_readout import extent_log_odds
from .box_geometry import instance_mean
from .grading_iou import GSFIoULoss
from .grade_selection import select_sufficient_grade_rois
from .master_adapter import expanded_context_rois
from .roi import reframe_gt_masks

def normalized_gt_xyxy(batch: dict) -> Tensor:
    h, w = batch['img'].shape[-2:]
    boxes = batch['bboxes'].detach().to(batch['img'])
    if boxes.ndim != 2 or boxes.shape[1] != 4 or (not torch.isfinite(boxes).all()):
        raise ValueError('finite normalized xywh boxes required')
    if (boxes[:, 2:] <= 0).any():
        raise ValueError('GT box width and height must be positive')
    scaled = boxes * boxes.new_tensor([w, h, w, h])
    return torch.cat((scaled[:, :2] - scaled[:, 2:] / 2, scaled[:, :2] + scaled[:, 2:] / 2), -1)

def response_query_boxes(gt_boxes: Tensor, fraction: float=0.65) -> Tensor:
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError('response crop fraction must lie in (0,1)')
    q = gt_boxes[:, None].expand(-1, 5, -1).clone()
    wh = gt_boxes[:, 2:] - gt_boxes[:, :2]
    q[:, 1, 2] = gt_boxes[:, 0] + fraction * wh[:, 0]
    q[:, 2, 0] = gt_boxes[:, 2] - fraction * wh[:, 0]
    q[:, 3, 3] = gt_boxes[:, 1] + fraction * wh[:, 1]
    q[:, 4, 1] = gt_boxes[:, 3] - fraction * wh[:, 1]
    return q

def fixed_context_response_loss(adapter, features: dict, gt_boxes: Tensor, batch_ids: Tensor, evidence: Tensor | None, availability: Tensor | None) -> dict:
    zero = features['roi_feature'].sum() * 0
    if evidence is None or not adapter.config.cseg:
        return {'loss': zero, 'context_evidence_loss': zero, 'pairs': 0}
    complete = availability.all(-1) & (evidence[:, 0].flatten(1).sum(-1) > 1e-06)
    idx = torch.where(complete)[0]
    if not idx.numel():
        return {'loss': zero, 'context_evidence_loss': zero, 'pairs': 0}
    g = gt_boxes[idx]
    ref = expanded_context_rois(torch.cat((batch_ids[idx, None].to(g), g), -1), adapter.config.context_scale)
    query = response_query_boxes(g, adapter.config.response_crop_fraction)
    predicted = adapter.grade_in_context(features, ref, query, calibration_detach=True)
    size = predicted['aux']['evidence_logits']['visible'].shape[-1]
    joint = reframe_gt_masks(evidence[idx], g, ref[:, 1:], size)
    targets, valid = _grade_targets_from_joint_maps(joint, availability[idx])
    reference = adapter.grade_head.readout_from_probabilities(targets, predicted['query_capture'].detach(), calibration_detach=True)
    qp = predicted['aux']['cumulative_probs']
    qr = reference['aux']['cumulative_probs'].detach()
    active = predicted['aux']['support_valid'] & reference['aux']['support_valid']
    active = active[:, 1:] & active[:, :1]
    difference = qp[:, 1:] - qp[:, :1] - (qr[:, 1:] - qr[:, :1])
    residual = difference.abs().amax(-1)
    loss = (residual * active).sum() / active.sum().clamp_min(1)
    e = reference['aux']['extent'].detach()
    c = reference['aux']['critical_presence'].detach()
    x = extent_log_odds(e, adapter.config.extent_eps)
    stat = (1 - c[:, :1]) * (adapter.config.max_slope / 4) * (x[:, 1:] - x[:, :1]).abs() + (c[:, 1:] - c[:, :1]).abs()
    pp = predicted['grade_probs']
    tv = 0.5 * (pp[:, 1:] - pp[:, :1]).abs().sum(-1)
    bound = (stat + 2 * residual).clamp(0, 1)
    annotation = evidence_supervision_loss(predicted, targets, valid_masks=valid)['loss']
    return {'loss': loss, 'context_evidence_loss': annotation, 'pairs': int(active.sum().detach()), 'response_residual': residual, 'bound': bound, 'actual_tv': tv, 'valid': active, 'predicted': predicted, 'reference': reference}

def master_training_loss(adapter, batch: dict, criterion, *, retention_weight: float=0.25, grade_weight: float=1.0, evidence_weight: float=0.1, shallow_weight: float=0.1, fusion_support_weight: float=0.1, response_weight: float=0.1, compose_native: Callable | None=None) -> dict:
    for value in (retention_weight, grade_weight, evidence_weight, shallow_weight, fusion_support_weight, response_weight):
        if not math.isfinite(value) or value < 0:
            raise ValueError('loss weights must be finite and nonnegative')
    if not adapter.training:
        raise RuntimeError('model.train() required')
    native = getattr(criterion, 'native_criterion', criterion)
    if not hasattr(native, 'get_assigned_targets_and_loss'):
        raise TypeError('native TAL loss interface required')
    box_loss = getattr(native, 'bbox_loss', None)
    if str(getattr(box_loss, 'iou_loss', '')).lower() != 'ciou' or str(getattr(box_loss, 'iou_aux', 'none')).lower() != 'none' or bool(getattr(box_loss, 'use_wiseiou', False)) or (getattr(native, 'nc', 1) != 1):
        raise ValueError('single-fruit native CIoU configuration required')
    grade_values = batch.get('grades', batch['cls']).flatten()
    if not torch.isfinite(grade_values).all() or ((grade_values != grade_values.round()) | (grade_values < 0) | (grade_values > 2)).any():
        raise ValueError('grade labels must be integers 0/1/2')
    grades = grade_values.long()
    m = grades.numel()
    bids = batch['batch_idx'].flatten()
    if bids.numel() != m or batch['cls'].numel() != m or batch['bboxes'].shape != (m, 4):
        raise ValueError('instance arrays must have the same length')
    if not torch.isfinite(bids).all() or ((bids != bids.round()) | (bids < 0) | (bids >= batch['img'].shape[0])).any():
        raise ValueError('invalid image indices')
    gt = normalized_gt_xyxy(batch)
    order = torch.argsort(bids, stable=True)
    fruit_batch = dict(batch, batch_idx=bids[order], cls=torch.zeros_like(batch['cls'].reshape(-1, 1)), bboxes=batch['bboxes'].index_select(0, order.to(batch['bboxes'].device)))
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
    p, g = (decoded[fg], target_boxes[fg])
    evidence = batch.get('evidence')
    availability = batch.get('evidence_valid')
    if evidence is not None:
        if availability is None:
            raise ValueError('evidence_valid required; missing is not negative')
        evidence, availability = _sanitize_joint_maps(evidence.detach().to(p), availability.detach().to(p.device))
        if evidence.shape[0] != m:
            raise ValueError('evidence must follow original GT ordering')
    cfg = adapter.config
    statistics = GSFIoULoss(temperature=cfg.capture_temperature, max_slope=cfg.max_slope, extent_eps=cfg.extent_eps, coverage_weight=cfg.coverage_weight, component_threshold=cfg.component_threshold, component_power=cfg.component_power).retention(p, g, None if evidence is None else evidence[ids], None if availability is None else availability[ids])
    retention = instance_mean(statistics['retention'], ids)
    quality = raw['scores'][:, 0].sigmoid()[fg]
    selection = select_sufficient_grade_rois(ids, quality, statistics, m, min_visible_coverage=cfg.min_visible_coverage, max_extent_delta=cfg.max_extent_delta, max_critical_delta=cfg.max_critical_delta)
    chosen = selection['candidate_indices']
    grade_boxes = gt.clone()
    selected = chosen >= 0
    if selected.any():
        grade_boxes[selected] = p[chosen[selected]].detach().to(grade_boxes)
    rois = torch.cat((bids[:, None].to(gt), grade_boxes.detach()), -1)
    grade_out = adapter.grade(features, rois)
    grade_loss = F.cross_entropy(grade_out['grade_logits'], grades.to(p.device)) if m else grade_out['grade_logits'].sum() * 0
    if evidence is not None:
        size = grade_out['aux']['evidence_logits']['visible'].shape[-1]
        warped = reframe_gt_masks(evidence, gt, grade_out['reference_rois'][:, 1:].detach(), size)
        targets, valid = _grade_targets_from_joint_maps(warped, availability)
        roi_evidence = evidence_supervision_loss(grade_out, targets, valid_masks=valid)['loss']
    else:
        roi_evidence = evidence_supervision_loss(grade_out)['loss']
    response = fixed_context_response_loss(adapter, features, gt, bids, evidence, availability)
    evidence_loss = roi_evidence + response['context_evidence_loss']
    shallow = shallow_evidence_loss(features['phase_aux'], batch.get('shallow_targets'), batch.get('shallow_valid'))
    fusion = shallow_evidence_loss(features['fusion_aux'], batch.get('shallow_targets'), batch.get('shallow_valid'))
    total = native_loss.sum() + grade_weight * grade_loss + retention_weight * retention + evidence_weight * evidence_loss + shallow_weight * shallow + fusion_support_weight * fusion + response_weight * response['loss']
    return {'loss': total, 'native_items': native_items, 'grade_loss': grade_loss, 'retention_loss': retention, 'evidence_loss': evidence_loss, 'grade_context_evidence_loss': roi_evidence, 'response_context_evidence_loss': response['context_evidence_loss'], 'shallow_loss': shallow, 'fusion_support_loss': fusion, 'response_loss': response['loss'], 'positive_anchors': fg.sum().detach(), 'graded_instances': m, 'grade_output': grade_out, 'selection': selection, 'grade_rois': rois, 'retention_details': statistics, 'response_details': response}
