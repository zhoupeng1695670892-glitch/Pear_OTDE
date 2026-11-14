import unittest
import torch
from pear_innovations.training_helpers import _sanitize_joint_maps
from pear_innovations.detection_training import _sanitize_joint_maps as sanitize_coverage_control
from pear_innovations.grading_iou import GSFIoULoss
from pear_innovations.statistic_baseline import PointStatisticIoULoss
from pear_innovations import PearConfig, PearMasterAdapter
from test_roi_adapter import Toy25NodeContractGraph


class PartialAnnotationRegressionTests(unittest.TestCase):
    def setUp(self):
        self.box = torch.tensor([[0.0, 0.0, 4.0, 4.0]])
        self.available = torch.tensor([[True, False, True]])
        self.joint = torch.zeros(1, 3, 4, 4)
        self.joint[:, 0] = 0.2
        self.joint[:, 2] = 0.8

    def test_nonadjacent_known_supports_are_checked_in_training(self):
        for sanitize in (_sanitize_joint_maps, sanitize_coverage_control):
            with self.subTest(sanitize=sanitize.__module__):
                with self.assertRaises(ValueError):
                    sanitize(self.joint, self.available)

    def test_standalone_losses_reject_critical_outside_visible(self):
        for criterion in (GSFIoULoss(), PointStatisticIoULoss()):
            with self.subTest(criterion=type(criterion).__name__):
                with self.assertRaises(ValueError):
                    criterion.statistics(self.box, self.box, self.joint, self.available)

    def test_unknown_damage_remains_unknown_with_valid_critical_support(self):
        self.joint[:, 0] = 1.0
        self.joint[:, 1] = float('nan')
        clean, valid = _sanitize_joint_maps(self.joint, self.available)
        self.assertFalse(valid[0, 1])
        torch.testing.assert_close(clean[:, 1], torch.zeros_like(clean[:, 1]))
        details = GSFIoULoss().statistics(self.box, self.box, self.joint, self.available)
        self.assertTrue(details['valid_critical'][0])
        self.assertFalse(details['valid_extent'][0])
        self.assertEqual(details['retention'].item(), 0.0)

    def test_critical_only_annotation_does_not_require_visible_channel(self):
        available = torch.tensor([[False, False, True]])
        self.joint[:, :2] = float('nan')
        clean, valid = _sanitize_joint_maps(self.joint, available)
        self.assertTrue(valid[0, 2])
        details = GSFIoULoss().statistics(self.box, self.box, clean, valid)
        self.assertTrue(details['valid_critical'][0])
        self.assertFalse(details['valid_visible'][0])

    def test_empty_prediction_path_preserves_output_shapes(self):
        model = PearMasterAdapter(Toy25NodeContractGraph(), PearConfig(score_threshold=1.0, grade_dim=8, phase_rank=4, roi_size=5, fine_roi_size=9)).eval()
        output = model.predict(torch.zeros(1, 3, 64, 64))[0]
        self.assertEqual(output['boxes'].shape, (0, 4))
        self.assertEqual(output['grade_probabilities'].shape, (0, 3))
        self.assertEqual(output['grades'].shape, (0,))


if __name__ == '__main__':
    unittest.main()
