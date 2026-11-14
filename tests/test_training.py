from types import SimpleNamespace
import unittest
import torch
from torch import nn
from torch.nn import functional as F
from pear_innovations.grade_support import BasicROIGradeHead, evidence_supervision_loss
from pear_innovations.roi import sample_rois
from pear_innovations.detection_training import _grade_targets_from_joint_maps, global_gt_indices, baseline_training_loss, one_prediction_per_instance, shallow_evidence_loss

class TransparentTrainingAdapter(nn.Module):

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(roi_size=4)
        self.master = nn.Module()
        self.master.routing_aux_parameter = nn.Parameter(torch.tensor(0.3))
        self.boxes = nn.Parameter(torch.tensor([[[4.0, 4.0, 11.0, 11.0], [4.0, 4.0, 12.0, 12.0], [10.0, 2.0, 14.0, 8.0], [1.0, 1.0, 2.0, 2.0]], [[2.0, 2.0, 9.0, 9.0], [2.0, 2.0, 10.0, 10.0], [8.0, 8.0, 14.0, 14.0], [1.0, 1.0, 2.0, 2.0]]]).transpose(1, 2).contiguous())
        self.scores = nn.Parameter(torch.tensor([[[2.0, 4.0, 1.0, -1.0]], [[5.0, 1.0, 3.0, -1.0]]]))
        self.feature = nn.Parameter(torch.randn(2, 4, 8, 8))
        self.support_logits = nn.Parameter(torch.randn(2, 2, 8, 8))
        self.grade_head = BasicROIGradeHead(4, feature_dim=5)
        self.observed_rois = None

    def forward(self, images):
        assert images.shape[0] == 2
        return {'detection': {'boxes': self.boxes, 'scores': self.scores}, 'roi_feature': self.feature, 'image_size': images.shape[-2:], 'phase_aux': {'support_logits': self.support_logits}}

    def grade(self, features, rois):
        self.observed_rois = rois.detach().clone()
        crops = sample_rois(features['roi_feature'], rois, features['image_size'], self.config.roi_size)
        return self.grade_head(crops)

class TransparentNativeCriterion:

    def __init__(self, *, no_positive=False):
        self.nc = 1
        self.bbox_loss = SimpleNamespace(iou_loss='ciou', iou_aux='none', use_wiseiou=False)
        self.no_positive = no_positive
        self.calls = 0
        self.last_batch = None
        self.last_vector = None

    def __call__(self, *args, **kwargs):
        raise AssertionError('bridge must not call TAL twice')

    def get_assigned_targets_and_loss(self, raw, batch):
        self.calls += 1
        self.last_batch = batch
        image_ids = batch['batch_idx'].long()
        assert bool((image_ids[1:] >= image_ids[:-1]).all()), 'native preprocess requires grouped GTs'
        assert bool((batch['cls'] == 0).all()), 'single fruit class expected'
        fg = torch.tensor([[True, True, True, False], [True, True, True, False]])
        local_ids = torch.tensor([[0, 0, 1, 0], [0, 0, 1, 0]])
        target_boxes = torch.zeros(2, 4, 4)
        if self.no_positive:
            fg.zero_()
        else:
            for b in range(2):
                xywh = batch['bboxes'][image_ids == b] * 16
                xyxy = torch.cat((xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2), -1)
                target_boxes[b] = xyxy[local_ids[b]]
        vector = torch.stack((raw['boxes'].mean() * 0.01, raw['scores'].square().mean() * 0.02, raw['boxes'].square().mean() * 0.001))
        self.last_vector = vector
        return ((fg, local_ids, target_boxes, torch.zeros(4, 2), torch.ones(4, 1)), vector, vector.detach())

    def bbox_decode(self, anchors, boxes):
        return boxes

def unsorted_batch():
    xyxy = torch.tensor([[2.0, 2.0, 10.0, 10.0], [4.0, 4.0, 12.0, 12.0], [8.0, 8.0, 14.0, 14.0], [10.0, 2.0, 14.0, 8.0]])
    xywh = torch.cat(((xyxy[:, :2] + xyxy[:, 2:]) / 2, xyxy[:, 2:] - xyxy[:, :2]), -1) / 16
    return {'img': torch.randn(2, 3, 16, 16), 'batch_idx': torch.tensor([1.0, 0.0, 1.0, 0.0]), 'cls': torch.tensor([[2.0], [0.0], [1.0], [2.0]]), 'bboxes': xywh}

class IndexAndEvidenceContractTests(unittest.TestCase):

    def test_unsorted_local_to_global(self):
        fg = torch.tensor([[True, False, True], [True, True, False]])
        local = torch.tensor([[1, 99, 0], [1, 0, 99]])
        ids = global_gt_indices(torch.tensor([1, 0, 1, 0]), fg, local)
        torch.testing.assert_close(ids, torch.tensor([3, 1, 2, 0]))
        with self.assertRaises(ValueError):
            global_gt_indices(torch.tensor([0]), fg, local)

    def test_one_per_gt_best_quality_and_stable_ties(self):
        ids = torch.tensor([3, 1, 3, 1, 2])
        scores = torch.tensor([0.1, 0.8, 0.9, 0.8, 0.5], requires_grad=True)
        torch.testing.assert_close(one_prediction_per_instance(ids, scores), torch.tensor([1, 4, 2]))
        self.assertEqual(one_prediction_per_instance(ids[:0], scores[:0]).numel(), 0)
        with self.assertRaises(ValueError):
            one_prediction_per_instance(ids, scores[:, None])

    def test_joint_soft_maps_become_conditional_targets_and_domains(self):
        joint = torch.tensor([[[[0.5, 1.0]], [[0.25, 0.5]], [[0.125, 0.0]]]])
        targets, valid = _grade_targets_from_joint_maps(joint, torch.ones(1, 3, dtype=torch.bool))
        torch.testing.assert_close(targets['visible'], joint[:, :1])
        torch.testing.assert_close(targets['damage'], torch.tensor([[[[0.5, 0.5]]]]))
        torch.testing.assert_close(targets['critical'], torch.tensor([[[[0.5, 0.0]]]]))
        logits = {name: torch.zeros_like(value, requires_grad=True) for name, value in targets.items()}
        loss = evidence_supervision_loss({'evidence_logits': logits}, targets, valid_masks=valid)
        torch.testing.assert_close(loss['support_mass']['visible'], torch.tensor(2.0))
        torch.testing.assert_close(loss['support_mass']['damage'], torch.tensor(1.5))
        torch.testing.assert_close(loss['support_mass']['critical'], torch.tensor(0.75))
        loss['loss'].backward()
        self.assertTrue(all((torch.isfinite(value.grad).all() for value in logits.values())))

    def test_missing_parent_skips_conditional_domains_and_unknown_nan(self):
        joint = torch.tensor([[[[float('nan')]], [[0.8]], [[0.4]]], [[[1.0]], [[float('nan')]], [[0.7]]]])
        available = torch.tensor([[False, True, True], [True, False, True]])
        targets, valid = _grade_targets_from_joint_maps(joint, available)
        self.assertTrue(all((torch.isfinite(target).all() for target in targets.values())))
        self.assertEqual(valid['damage'].sum().item(), 0)
        self.assertEqual(valid['critical'].sum().item(), 0)
        logits = {name: torch.zeros_like(value, requires_grad=True) for name, value in targets.items()}
        out = evidence_supervision_loss({'evidence_logits': logits}, targets, valid_masks=valid)
        out['loss'].backward()
        self.assertEqual(logits['damage'].grad.abs().sum().item(), 0)
        self.assertEqual(logits['critical'].grad.abs().sum().item(), 0)
        self.assertGreater(logits['visible'].grad.abs().sum().item(), 0)

    def test_joint_checks_only_known_groups_but_enforces_known_containment(self):
        joint = torch.tensor([[[[0.0]], [[0.9]], [[100.0]]]])
        _grade_targets_from_joint_maps(joint, torch.tensor([[False, True, False]]))
        with self.assertRaisesRegex(ValueError, 'critical <= damage <= visible'):
            _grade_targets_from_joint_maps(joint, torch.tensor([[True, True, False]]))
        with self.assertRaisesRegex(ValueError, 'finite in'):
            _grade_targets_from_joint_maps(joint, torch.ones(1, 3, dtype=torch.bool))

    def test_shallow_missing_targets_and_conditioning_are_not_negative_labels(self):
        logits = torch.zeros(1, 2, 1, 3, requires_grad=True)
        targets = torch.tensor([[[[1.0, 0.0, float('nan')]], [[1.0, float('nan'), float('nan')]]]])
        valid = torch.tensor([[[[1.0, 1.0, 0.0]], [[1.0, 1.0, 1.0]]]])
        loss = shallow_evidence_loss({'support_logits': logits}, targets, valid)
        loss.backward()
        self.assertLess(logits.grad[0, 1, 0, 0].item(), 0)
        self.assertEqual(logits.grad[0, 1, 0, 1:].abs().sum().item(), 0)
        self.assertEqual(logits.grad[0, 0, 0, 2].item(), 0)
        self.assertEqual(shallow_evidence_loss({'support_logits': logits}).item(), 0)

    def test_shallow_requires_explicit_validity_and_checks_annotated_values(self):
        logits = torch.zeros(1, 2, 2, 2)
        targets = torch.ones_like(logits)
        with self.assertRaises(ValueError):
            shallow_evidence_loss({'support_logits': logits}, targets)
        targets[0, 0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            shallow_evidence_loss({'support_logits': logits}, targets, torch.ones_like(targets))

class TrainingBridgeContractTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(52)
        self.adapter = TransparentTrainingAdapter().train()
        self.criterion = TransparentNativeCriterion()
        self.batch = unsorted_batch()

    def test_unsorted_batch_sorted_for_native_then_original_grades_restored(self):
        original_boxes = self.batch['bboxes'].clone()
        out = baseline_training_loss(self.adapter, self.batch, self.criterion)
        torch.testing.assert_close(self.criterion.last_batch['batch_idx'], torch.tensor([0.0, 0.0, 1.0, 1.0]))
        torch.testing.assert_close(self.criterion.last_batch['bboxes'], original_boxes[[1, 3, 0, 2]])
        torch.testing.assert_close(self.batch['bboxes'], original_boxes)
        torch.testing.assert_close(self.batch['batch_idx'], torch.tensor([1.0, 0.0, 1.0, 0.0]))
        self.assertEqual(out['positive_anchors'].item(), 6)
        self.assertEqual(out['graded_instances'], 4)
        expected_rois = torch.tensor([[1.0, 2.0, 2.0, 9.0, 9.0], [0.0, 4.0, 4.0, 12.0, 12.0], [1.0, 8.0, 8.0, 14.0, 14.0], [0.0, 10.0, 2.0, 14.0, 8.0]])
        torch.testing.assert_close(self.adapter.observed_rois, expected_rois)
        expected = F.cross_entropy(out['grade_output']['grade_logits'], torch.tensor([2, 0, 1, 2]))
        torch.testing.assert_close(out['grade_loss'], expected)
        self.assertEqual(self.criterion.calls, 1)

    def test_native_vector_items_aux_and_gradients_are_preserved(self):

        def compose_native(master, native_loss, native_items):
            auxiliary = master.routing_aux_parameter.square()
            return (native_loss + auxiliary, torch.cat((native_items, auxiliary.detach()[None])))
        out = baseline_training_loss(self.adapter, self.batch, self.criterion, retention_weight=0, grade_weight=0, evidence_weight=0, shallow_weight=0, compose_native=compose_native)
        expected = (2 * self.criterion.last_vector + self.adapter.master.routing_aux_parameter.square()).sum()
        torch.testing.assert_close(out['loss'], expected)
        torch.testing.assert_close(out['native_items'][:3], self.criterion.last_vector.detach())
        self.assertEqual(out['native_items'].shape, (4,))
        out['loss'].backward()
        torch.testing.assert_close(self.adapter.master.routing_aux_parameter.grad, torch.tensor(1.8))
        self.assertGreater(self.adapter.boxes.grad.abs().sum().item(), 0)
        self.assertGreater(self.adapter.scores.grad.abs().sum().item(), 0)
        self.assertEqual(self.criterion.calls, 1)

    def test_absent_evidence_does_not_train_evidence_heads_as_background(self):
        out = baseline_training_loss(self.adapter, self.batch, self.criterion)
        self.assertEqual(out['evidence_loss'].item(), 0)
        self.assertEqual(out['retention_loss'].item(), 0)
        self.assertEqual(out['shallow_loss'].item(), 0)
        out['loss'].backward()
        self.assertGreater(self.adapter.grade_head.classifier.weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.adapter.feature.grad.abs().sum().item(), 0)
        for head in self.adapter.grade_head.evidence_heads.children():
            self.assertEqual(head.weight.grad.abs().sum().item(), 0)
        self.assertEqual(self.adapter.support_logits.grad.abs().sum().item(), 0)

    def test_known_damage_without_visible_retains_boxes_but_skips_conditional_bce(self):
        evidence = torch.full((4, 3, 4, 4), float('nan'))
        evidence[:, 1] = 1.0
        self.batch.update(evidence=evidence, evidence_valid=torch.tensor([[False, True, False]]).expand(4, 3))
        out = baseline_training_loss(self.adapter, self.batch, self.criterion)
        self.assertTrue(torch.isfinite(out['loss']))
        self.assertGreater(out['retention_loss'].item(), 0)
        self.assertEqual(out['evidence_loss'].item(), 0)
        grad = torch.autograd.grad(out['retention_loss'], self.adapter.boxes)[0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)

    def test_annotated_evidence_and_shallow_have_gradients(self):
        evidence = torch.ones(4, 3, 4, 4)
        evidence[:, 1] *= 0.6
        evidence[:, 2] *= 0.2
        self.batch.update(evidence=evidence, evidence_valid=torch.ones(4, 3, dtype=torch.bool), shallow_targets=torch.ones(2, 2, 16, 16), shallow_valid=torch.ones(2, 2, 16, 16))
        out = baseline_training_loss(self.adapter, self.batch, self.criterion)
        out['loss'].backward()
        for head in self.adapter.grade_head.evidence_heads.children():
            self.assertTrue(torch.isfinite(head.weight.grad).all())
            self.assertGreater(head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.adapter.support_logits.grad.abs().sum().item(), 0)

    def test_no_positive_anchors_keeps_differentiable_native_loss(self):
        criterion = TransparentNativeCriterion(no_positive=True)
        self.batch.update(batch_idx=torch.empty(0), cls=torch.empty(0, 1), bboxes=torch.empty(0, 4), evidence=torch.empty(0, 3, 4, 4), evidence_valid=torch.empty(0, 3, dtype=torch.bool))
        out = baseline_training_loss(self.adapter, self.batch, criterion)
        self.assertEqual(out['graded_instances'], 0)
        self.assertEqual(out['positive_anchors'].item(), 0)
        for name in ('grade_loss', 'evidence_loss', 'retention_loss'):
            self.assertEqual(out[name].item(), 0)
        self.assertEqual(out['grade_output']['grade_logits'].shape, (0, 3))
        self.assertTrue(torch.isfinite(out['loss']))
        out['loss'].backward()
        self.assertGreater(self.adapter.boxes.grad.abs().sum().item(), 0)
        self.assertEqual(criterion.calls, 1)

    def test_missing_validity_and_invalid_grade_or_image_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, 'evidence_valid required'):
            baseline_training_loss(self.adapter, dict(self.batch, evidence=torch.ones(4, 3, 2, 2)), self.criterion)
        with self.assertRaisesRegex(ValueError, 'grade IDs'):
            baseline_training_loss(self.adapter, dict(self.batch, grades=torch.tensor([0.5, 0, 1, 2])), self.criterion)
        with self.assertRaisesRegex(ValueError, 'batch_idx'):
            baseline_training_loss(self.adapter, dict(self.batch, batch_idx=torch.tensor([0.5, 0, 1, 0])), self.criterion)

    def test_native_ciou_configuration_uses_actual_source_attributes(self):
        for name, value in (('iou_loss', 'giou'), ('iou_aux', 'nwd'), ('use_wiseiou', True)):
            criterion = TransparentNativeCriterion()
            setattr(criterion.bbox_loss, name, value)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'native CIoU'):
                baseline_training_loss(self.adapter, self.batch, criterion)
        criterion = TransparentNativeCriterion()
        del criterion.bbox_loss.iou_loss
        with self.assertRaises(ValueError):
            baseline_training_loss(self.adapter, self.batch, criterion)

    def test_invalid_loss_weight_rejected(self):
        for value in (float('nan'), float('inf'), -0.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                baseline_training_loss(self.adapter, self.batch, self.criterion, retention_weight=value)
if __name__ == '__main__':
    unittest.main()
