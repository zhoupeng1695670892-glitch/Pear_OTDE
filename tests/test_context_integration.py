import unittest
import torch
from pear_innovations.master_adapter import PearMasterAdapter, PearConfig, expanded_context_rois
from pear_innovations.support_grading import CSEGGradeHead
from pear_innovations.training import master_training_loss, fixed_context_response_loss, response_query_boxes
from test_roi_adapter import Toy25NodeContractGraph
from test_feature_integration import TransparentFeatureAdapter
from test_training import TransparentNativeCriterion, unsorted_batch

class TransparentContextAdapter(TransparentFeatureAdapter):
    grade = PearMasterAdapter.grade
    grade_in_context = PearMasterAdapter.grade_in_context

    def __init__(self):
        super().__init__()
        self.config = PearConfig(roi_size=4, fine_roi_size=7, grade_dim=5)
        self.grade_head = CSEGGradeHead(4, feature_dim=5, fine_channels=3, use_fine=True)

def annotated_batch():
    batch = unsorted_batch()
    batch['evidence'] = torch.zeros(4, 3, 12, 12)
    batch['evidence'][:, 0] = 1
    batch['evidence'][:, 1, 2:10, 2:10] = 1
    batch['evidence'][:, 2, 3:6, 4:8] = 1
    batch['evidence_valid'] = torch.ones(4, 3, dtype=torch.bool)
    batch['shallow_targets'] = torch.ones(2, 2, 16, 16)
    batch['shallow_valid'] = torch.ones_like(batch['shallow_targets'])
    return batch

class ContextIntegrationTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(71)
        torch.set_num_threads(2)

    def test_context_expands_without_clipping_and_retains_gradient(self):
        rois = torch.tensor([[0.0, 0.0, 0.0, 8.0, 8.0]], requires_grad=True)
        context = expanded_context_rois(rois, 1.25)
        torch.testing.assert_close(context, torch.tensor([[0.0, -1.0, -1.0, 9.0, 9.0]]))
        context[:, 1:].sum().backward()
        self.assertTrue(torch.isfinite(rois.grad).all())

    def test_native_graph_query_api_and_prediction(self):
        model = PearMasterAdapter(Toy25NodeContractGraph(), PearConfig(grade_dim=8, phase_rank=4, roi_size=5, fine_roi_size=9)).eval()
        f = model.extract(torch.randn(1, 3, 64, 64))
        boxes = torch.tensor([[8.0, 8.0, 24.0, 24.0]])
        ref = expanded_context_rois(torch.cat((torch.zeros(1, 1), boxes), -1))
        output = model.grade_in_context(f, ref, response_query_boxes(boxes))
        self.assertEqual(output['grade_probs'].shape, (1, 5, 3))
        self.assertEqual(output['aux']['evidence_logits']['visible'].shape, (1, 1, 9, 9))
        result = model.predict(torch.randn(1, 3, 64, 64))[0]
        self.assertGreater(result['boxes'].shape[0], 0)
        torch.testing.assert_close(result['grade_probabilities'].sum(-1), torch.ones(result['boxes'].shape[0]))

    def test_native_sum_fallback_pair_annotations_and_gradients(self):
        model = TransparentContextAdapter().train()
        batch = annotated_batch()
        criterion = TransparentNativeCriterion()
        out = master_training_loss(model, batch, criterion)
        expected = criterion.last_vector.sum() * 2 + out['grade_loss'] + 0.25 * out['retention_loss'] + 0.1 * out['evidence_loss'] + 0.1 * out['shallow_loss'] + 0.1 * out['fusion_support_loss'] + 0.1 * out['response_loss']
        torch.testing.assert_close(out['loss'], expected)
        self.assertEqual(criterion.calls, 1)
        self.assertEqual(out['graded_instances'], 4)
        self.assertEqual(out['response_details']['pairs'], 16)
        self.assertEqual(out['grade_output']['aux']['evidence_logits']['visible'].shape[-1], 7)
        out['loss'].backward()
        for param in (model.boxes, model.feature, model.fine_feature, model.fusion_logits):
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())
            self.assertGreater(param.grad.abs().sum().item(), 0)

    def test_missing_maps_falls_back_to_each_gt_without_fabricated_supervision(self):
        model = TransparentContextAdapter().train()
        out = master_training_loss(model, unsorted_batch(), TransparentNativeCriterion())
        self.assertTrue(out['selection']['fallback'].all())
        self.assertEqual(out['graded_instances'], 4)
        for key in ('retention_loss', 'evidence_loss', 'response_loss'):
            self.assertEqual(out[key].item(), 0)
        self.assertGreater(out['grade_loss'].item(), 0)

    def test_no_positive_anchors_still_supervises_gt_grades(self):
        model = TransparentContextAdapter().train()
        out = master_training_loss(model, annotated_batch(), TransparentNativeCriterion(no_positive=True))
        self.assertTrue(out['selection']['fallback'].all())
        self.assertEqual(out['graded_instances'], 4)
        self.assertEqual(out['retention_loss'].item(), 0)
        self.assertTrue(torch.isfinite(out['loss']))

    def test_response_detaches_calibration_but_trains_actual_field(self):
        model = TransparentContextAdapter().train()
        batch = annotated_batch()
        from pear_innovations.training import normalized_gt_xyxy
        f = model(batch['img'])
        out = fixed_context_response_loss(model, f, normalized_gt_xyxy(batch), batch['batch_idx'].flatten(), batch['evidence'], batch['evidence_valid'])
        self.assertEqual(out['pairs'], 16)
        out['loss'].backward()
        for param in (model.grade_head.raw_slope, model.grade_head.threshold_base, model.grade_head.threshold_gap):
            self.assertIsNone(param.grad)
        self.assertIsNotNone(model.fine_feature.grad)
        self.assertGreater(model.fine_feature.grad.abs().sum().item(), 0)
        active = out['valid']
        self.assertTrue((out['actual_tv'][active] <= out['bound'][active] + 2e-06).all())

    def test_partial_labels_do_not_create_response_pairs(self):
        model = TransparentContextAdapter().train()
        batch = annotated_batch()
        batch['evidence_valid'][:, 2] = False
        out = master_training_loss(model, batch, TransparentNativeCriterion())
        self.assertTrue(out['selection']['fallback'].all())
        self.assertEqual(out['response_details']['pairs'], 0)
        self.assertEqual(out['response_loss'].item(), 0)
        self.assertGreater(out['evidence_loss'].item(), 0)

    def test_response_weight_ablation_preserves_annotation_budget(self):
        model = TransparentContextAdapter().train()
        batch = annotated_batch()
        off = master_training_loss(model, batch, TransparentNativeCriterion(), response_weight=0.0)
        on = master_training_loss(model, batch, TransparentNativeCriterion(), response_weight=0.1)
        torch.testing.assert_close(off['evidence_loss'], on['evidence_loss'])
        torch.testing.assert_close(on['loss'] - off['loss'], 0.1 * on['response_loss'])
        self.assertGreater(off['response_context_evidence_loss'].item(), 0)
if __name__ == '__main__':
    unittest.main()
