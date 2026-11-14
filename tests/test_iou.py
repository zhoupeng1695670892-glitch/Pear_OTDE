import unittest
import torch
from pear_innovations.box_geometry import SupportCoverageIoULoss, aligned_iou, instance_mean

class TestEvidenceIoU(unittest.TestCase):

    def setUp(self):
        self.loss = SupportCoverageIoULoss(reduction='none')
        self.gt = torch.tensor([[0.0, 0.0, 20.0, 10.0]])
        self.e = torch.ones(1, 3, 16, 16)
        self.e[:, 1:] = 0
        self.e[:, 1, 4:8, 1:3] = 1
        self.e[:, 2, 5:7, 1:2] = 1

    def test_perfect_match_zero_no_inflation_gradient(self):
        p = self.gt.clone().requires_grad_()
        loss, info = self.loss(p, self.gt, self.e, return_details=True)
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(info['retention'].item(), 0.0)
        loss.sum().backward()
        self.assertTrue(torch.allclose(p.grad, torch.zeros_like(p)))

    def test_cutting_critical_side_is_more_costly_at_equal_iou(self):
        left = torch.tensor([[4.0, 0.0, 20.0, 10.0]], requires_grad=True)
        right = torch.tensor([[0.0, 0.0, 16.0, 10.0]])
        self.assertTrue(torch.allclose(aligned_iou(left, self.gt), aligned_iou(right, self.gt)))
        a, ai = self.loss(left, self.gt, self.e, return_details=True)
        b, bi = self.loss(right, self.gt, self.e, return_details=True)
        self.assertGreater(a.item(), b.item())
        self.assertGreater(ai['retention'].item(), bi['retention'].item())
        ai['retention'].sum().backward()
        self.assertGreater(left.grad[0, 0].item(), 0.0)

    def test_expansion_zero_retention_but_positive_geometry(self):
        loss, info = self.loss(torch.tensor([[-2.0, -2.0, 22.0, 12.0]]), self.gt, self.e, return_details=True)
        self.assertEqual(info['retention'].item(), 0.0)
        self.assertGreater(loss.item(), 0.0)

    def test_common_axis_scaling_translation_invariance(self):
        p = torch.tensor([[3.0, 1.0, 19.0, 9.0]])
        scale = torch.tensor([3.0, 7.0, 3.0, 7.0])
        offset = torch.tensor([31.0, -20.0, 31.0, -20.0])
        a = self.loss.retention(p, self.gt, self.e)['retention']
        b = self.loss.retention(p * scale + offset, self.gt * scale + offset, self.e)['retention']
        self.assertTrue(torch.allclose(a, b, atol=1e-06))

    def test_missing_annotation_falls_back_and_detaches_evidence(self):
        p = torch.tensor([[3.0, 1.0, 19.0, 9.0]], requires_grad=True)
        e = self.e.clone().requires_grad_()
        g = self.gt.clone().requires_grad_()
        value = self.loss(p, g, e, torch.zeros(1, 3, dtype=torch.bool))
        self.assertTrue(torch.allclose(value, 1 - aligned_iou(p, g)))
        self.loss(p, g, e).sum().backward()
        self.assertIsNone(e.grad)
        self.assertIsNone(g.grad)
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_group_replication_and_anchor_multiplicity(self):
        p = torch.tensor([[3.0, 1.0, 19.0, 9.0]])
        self.assertTrue(torch.allclose(self.loss(p, self.gt, self.e), self.loss(p, self.gt, self.e.repeat(1, 2, 1, 1))))
        a = instance_mean(torch.tensor([1.0, 3.0]), torch.tensor([10, 20]))
        b = instance_mean(torch.tensor([1.0, 1.0, 1.0, 3.0]), torch.tensor([10, 10, 10, 20]))
        self.assertEqual(a.item(), b.item())

    def test_empty_and_low_precision(self):
        p = torch.empty(0, 4, requires_grad=True)
        loss = SupportCoverageIoULoss()(p, p.detach(), torch.empty(0, 3, 8, 8))
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        val = self.loss(self.gt.half(), self.gt.half(), self.e.half())
        self.assertEqual(val.dtype, torch.float32)
        self.assertTrue(torch.isfinite(val).all())

    def test_gradient_finite_difference_away_from_kinks(self):
        p = torch.tensor([[3.11, 0.71, 18.21, 9.31]], dtype=torch.double, requires_grad=True)
        fn = lambda x: self.loss.retention(x, self.gt.double(), self.e.double())['retention']
        self.assertTrue(torch.autograd.gradcheck(fn, (p,), eps=1e-05, atol=0.0001))

    def test_invalid_hyperparameters_and_availability_are_rejected(self):
        for kw in ({'temperature': float('nan')}, {'weight': float('inf')}, {'eps': float('nan')}):
            with self.assertRaises(ValueError):
                SupportCoverageIoULoss(**kw)
        for valid in (torch.tensor([[1.0, float('nan'), 0.0]]), torch.tensor([[1.0, 2.0, 0.0]])):
            with self.assertRaises(ValueError):
                self.loss(self.gt, self.gt, self.e, valid)
if __name__ == '__main__':
    unittest.main()
