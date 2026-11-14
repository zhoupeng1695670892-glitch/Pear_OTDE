from types import SimpleNamespace
import unittest
import torch
from torch import nn
from pear_innovations.feature_adapter import FeatureAdapter, FeatureConfig
from pear_innovations.ordinal_readout import OrdinalEvidenceReadout
from pear_innovations.training_helpers import point_readout_training_loss
from pear_innovations.roi import sample_rois
from test_roi_adapter import Toy25NodeContractGraph
from test_training import TransparentTrainingAdapter, TransparentNativeCriterion, unsorted_batch

class TransparentFeatureAdapter(TransparentTrainingAdapter):

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(roi_size=4, fine_roi_size=7, max_slope=8.0, extent_eps=0.0001, coverage_weight=1.0)
        self.fine_feature = nn.Parameter(torch.randn(2, 3, 16, 16))
        self.fusion_logits = nn.Parameter(torch.randn(2, 2, 8, 8))
        self.grade_head = OrdinalEvidenceReadout(4, feature_dim=5, fine_channels=3, use_fine=True)

    def forward(self, images):
        out = super().forward(images)
        out['fine_roi_feature'] = self.fine_feature
        out['fusion_aux'] = {'support_logits': self.fusion_logits}
        return out

    def grade(self, features, rois):
        self.observed_rois = rois.detach().clone()
        coarse = sample_rois(features['roi_feature'], rois, features['image_size'], 4)
        fine = sample_rois(features['fine_roi_feature'], rois, features['image_size'], 7)
        return self.grade_head(coarse, fine)

class FeatureIntegrationTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(45)
        torch.set_num_threads(2)

    def test_shared_fine_source_and_prediction_probabilities(self):
        model = FeatureAdapter(Toy25NodeContractGraph(), FeatureConfig(grade_dim=8, phase_rank=4, roi_size=5, fine_roi_size=9)).eval()
        features = model.extract(torch.randn(1, 3, 64, 64))
        self.assertEqual(features['fine_roi_feature'].shape[-2:], (16, 16))
        self.assertEqual(features['roi_feature'].shape[-2:], (8, 8))
        output = model.grade(features, torch.tensor([[0.0, 8.0, 8.0, 24.0, 24.0]]))
        self.assertEqual(output['aux']['evidence_logits']['visible'].shape[-2:], (9, 9))
        result = model.predict(torch.randn(1, 3, 64, 64))[0]
        self.assertGreater(result['boxes'].shape[0], 0)
        torch.testing.assert_close(result['grade_probabilities'].sum(-1), torch.ones(result['boxes'].shape[0]))

    def test_disabled_fusion_matches_native_channel_order(self):
        model = FeatureAdapter(Toy25NodeContractGraph(), FeatureConfig(lecf=False, rcpm=False, grade_dim=8, phase_rank=4)).eval()
        features = model.extract(torch.randn(1, 3, 64, 64))
        aux = features['fusion_aux']
        self.assertEqual(aux['support_logits'].shape[1], 2)
        self.assertEqual(aux['enabled'].item(), 0)
        self.assertIs(model.downsample.main, model.master.model[3])

    def test_training_bridge_preserves_native_and_supervises_fine_grid(self):
        model = TransparentFeatureAdapter().train()
        batch = unsorted_batch()
        batch['evidence'] = torch.zeros(4, 3, 8, 8)
        batch['evidence'][:, 0] = 1
        batch['evidence'][:, 1, 1:4, 2:5] = 1
        batch['evidence'][:, 2, 2:3, 2:4] = 1
        batch['evidence_valid'] = torch.ones(4, 3, dtype=torch.bool)
        batch['shallow_targets'] = torch.ones(2, 2, 16, 16)
        batch['shallow_valid'] = torch.ones_like(batch['shallow_targets'])
        criterion = TransparentNativeCriterion()
        out = point_readout_training_loss(model, batch, criterion)
        expected = criterion.last_vector.sum() * 2 + out['grade_loss'] + 0.25 * out['retention_loss'] + 0.1 * out['evidence_loss'] + 0.1 * out['shallow_loss'] + 0.1 * out['fusion_support_loss']
        torch.testing.assert_close(out['loss'], expected)
        self.assertEqual(criterion.calls, 1)
        self.assertEqual(out['grade_output']['aux']['evidence_logits']['visible'].shape[-1], 7)
        self.assertEqual(out['graded_instances'], 4)
        out['loss'].backward()
        for parameter in (model.boxes, model.feature, model.fine_feature, model.fusion_logits):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_no_annotations_are_not_negative_supervision(self):
        model = TransparentFeatureAdapter().train()
        out = point_readout_training_loss(model, unsorted_batch(), TransparentNativeCriterion())
        for key in ('retention_loss', 'evidence_loss', 'shallow_loss', 'fusion_support_loss'):
            self.assertEqual(out[key].item(), 0)
        self.assertTrue(torch.isfinite(out['loss']))

    def test_no_matches_keeps_native_and_shallow_terms(self):
        model = TransparentFeatureAdapter().train()
        batch = unsorted_batch()
        batch['shallow_targets'] = torch.ones(2, 2, 16, 16)
        batch['shallow_valid'] = torch.ones_like(batch['shallow_targets'])
        out = point_readout_training_loss(model, batch, TransparentNativeCriterion(no_positive=True))
        self.assertEqual(out['graded_instances'], 0)
        self.assertEqual(out['grade_loss'].item(), 0)
        self.assertGreater(out['fusion_support_loss'].item(), 0)
        out['loss'].backward()
        self.assertTrue(torch.isfinite(model.fusion_logits.grad).all())
if __name__ == '__main__':
    unittest.main()
