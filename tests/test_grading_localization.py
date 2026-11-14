import unittest
import torch
from pear_innovations.ordinal_readout import structured_grade_probabilities
from pear_innovations.support_grading import CSEGGradeHead
from pear_innovations.grading_iou import GSFIoULoss, normalized_soft_capture
from pear_innovations.grade_selection import select_sufficient_grade_rois

class CaptureTests(unittest.TestCase):

    def setUp(self):
        self.reference = torch.tensor([[2.0, 3.0, 13.0, 20.0]], dtype=torch.float64)

    def test_exact_identity_value_and_zero_capture_gradient(self):
        p = self.reference.clone().requires_grad_()
        r = self.reference.clone().requires_grad_()
        capture = normalized_soft_capture(p, r, 13, 17)
        torch.testing.assert_close(capture, torch.ones_like(capture), rtol=0, atol=0)
        capture.sum().backward()
        torch.testing.assert_close(p.grad, torch.zeros_like(p), rtol=0, atol=0)
        self.assertIsNone(r.grad)

    def test_independent_finite_difference_interior_crop(self):
        p = torch.tensor([[3.1, 5.2, 11.7, 17.1]], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda box: normalized_soft_capture(box, self.reference, 7, 9), (p,), eps=1e-06, atol=2e-05, rtol=0.0002))
        capture = normalized_soft_capture(p, self.reference, 7, 9)
        capture.sum().backward()
        self.assertTrue((p.grad[0, :2] < 0).all())
        self.assertTrue((p.grad[0, 2:] > 0).all())

    def test_translation_axiswise_scaling_half_and_empty(self):
        p = torch.tensor([[3.0, 2.0, 12.0, 18.0]], dtype=torch.float64)
        scale = torch.tensor([2.0, 0.7, 2.0, 0.7], dtype=torch.float64)
        shift = torch.tensor([15.0, -8.0, 15.0, -8.0], dtype=torch.float64)
        baseline = normalized_soft_capture(p, self.reference, 8, 11)
        moved = normalized_soft_capture(p * scale + shift, self.reference * scale + shift, 8, 11)
        torch.testing.assert_close(baseline, moved, rtol=1e-12, atol=1e-12)
        half = normalized_soft_capture(p.half(), self.reference.half(), 8, 11)
        self.assertEqual(half.dtype, torch.float32)
        self.assertTrue(torch.isfinite(half).all())
        self.assertEqual(normalized_soft_capture(p[:0], self.reference[:0], 8, 11).shape, (0, 8, 11))
        for kwargs in ({'height': 0}, {'width': 0}, {'temp': 0.0}, {'temp': float('nan')}):
            options = dict(height=8, width=11)
            options.update(kwargs)
            with self.assertRaises(ValueError):
                normalized_soft_capture(p, self.reference, **options)

class GSFLocalizationTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(46)
        self.loss = GSFIoULoss()
        self.reference = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float64)
        self.evidence = torch.zeros(1, 3, 20, 32, dtype=torch.float64)
        self.evidence[:, 0] = 1
        self.evidence[:, 1, :, :16] = 1
        self.evidence[:, 2, 5:15, 3:13] = 1

    def test_identity_exact_zero_and_all_fixed_targets_detached(self):
        p = self.reference.clone().requires_grad_()
        r = self.reference.clone().requires_grad_()
        evidence = self.evidence.clone().requires_grad_()
        loss, info = self.loss(p, r, evidence, return_details=True)
        self.assertEqual(loss.item(), 0)
        self.assertEqual(info['retention'].item(), 0)
        self.assertEqual(info['critical_reference'].item(), 1)
        self.assertTrue(info['bound_valid'].item())
        loss.backward()
        torch.testing.assert_close(p.grad, torch.zeros_like(p), rtol=0, atol=0)
        self.assertIsNone(r.grad)
        self.assertIsNone(evidence.grad)

    def test_component_support_loss_is_not_cropped_pixel_max(self):
        p = torch.tensor([[0.0, 0.0, 0.2, 1.0]], dtype=torch.float64, requires_grad=True)
        info = self.loss.statistics(p, self.reference, self.evidence)
        expected = info['capture'][:, 5:15, 3:13].mean((-1, -2)).square()
        torch.testing.assert_close(info['critical_captured'], expected)
        cropped_max = (info['capture'] * self.evidence[:, 2]).amax((-1, -2))
        self.assertGreater((cropped_max - info['critical_captured']).item(), 0.4)
        info['critical_loss'].sum().backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertLess(p.grad[0, 2].item(), 0)

    def test_any_complete_critical_component_is_sufficient(self):
        maps = torch.ones_like(self.evidence)
        maps[:, 2] = 0
        maps[:, 2, 8:12, 3:6] = 1
        maps[:, 2, 8:12, 26:29] = 1
        info = self.loss.statistics(torch.tensor([[0.0, 0.0, 0.55, 1.0]], dtype=torch.float64), self.reference, maps)
        self.assertEqual(info['component_valid'].sum().item(), 2)
        self.assertLess(info['critical_loss'].item(), 0.001)
        self.assertGreater(info['coverage_loss'].item(), 0.4)

    def test_proportional_crop_requires_visible_coverage_even_without_stat_change(self):
        maps = self.evidence.clone()
        maps[:, 1] = 0.25
        maps[:, 2] = 0
        info = self.loss.statistics(torch.tensor([[0.2, 0.2, 0.8, 0.8]], dtype=torch.float64), self.reference, maps)
        self.assertLess(info['stat_loss'].item(), 1e-12)
        self.assertGreater(info['coverage_loss'].item(), 0.5)

    def test_partial_missing_and_known_zero_have_distinct_meaning(self):
        p = torch.tensor([[0.3, 0.0, 1.0, 1.0]], dtype=torch.float64)
        maps = self.evidence.clone()
        maps[:, 2] = float('nan')
        info = self.loss.statistics(p, self.reference, maps, torch.tensor([[1, 1, 0]]))
        self.assertFalse(info['valid_critical'].item())
        self.assertFalse(info['bound_valid'].item())
        self.assertGreater(info['extent_loss'].item(), 0)
        only_s = torch.full_like(maps, float('nan'))
        only_s[:, 2] = self.evidence[:, 2]
        partial = self.loss.statistics(p, self.reference, only_s, torch.tensor([[0, 0, 1]]))
        self.assertTrue(partial['valid_critical'].item())
        self.assertFalse(partial['valid_extent'].item())
        self.assertFalse(partial['bound_valid'].item())
        known_zero = self.evidence.clone()
        known_zero[:, 2] = 0
        full = self.loss.statistics(p, self.reference, known_zero)
        self.assertTrue(full['valid_critical'].item())
        self.assertTrue(full['bound_valid'].item())
        self.assertEqual(full['critical_reference'].item(), 0)
        missing = self.loss.statistics(p, self.reference, None)
        self.assertEqual(missing['retention'].item(), 0)
        self.assertFalse(missing['bound_valid'].item())
        with self.assertRaises(ValueError):
            self.loss.statistics(p, self.reference, maps)

    def test_invalid_component_hyperparameters_and_nesting(self):
        for options in ({'component_power': 0.5}, {'component_threshold': 0.0}, {'component_threshold': 1.1}, {'min_mass': 1e-09}):
            with self.assertRaises(ValueError):
                GSFIoULoss(**options)
        malformed = self.evidence.clone()
        malformed[:, 1] = 0
        with self.assertRaisesRegex(ValueError, 'S <= D <= V'):
            self.loss.statistics(self.reference, self.reference, malformed)

    def test_probability_certificate_for_fixed_component_context(self):
        n = 48
        v = 0.8 + 0.2 * torch.rand(n, 1, 8, 10, dtype=torch.float64)
        d = v * (0.7 + 0.3 * torch.rand_like(v))
        s = d * torch.rand_like(v)
        maps = torch.cat((v, d, s), 1)
        start = torch.rand(n, 2, dtype=torch.float64) * 0.5 - 0.1
        size = torch.rand(n, 2, dtype=torch.float64) * 0.6 + 0.3
        info = self.loss.statistics(torch.cat((start, start + size), -1), self.reference.expand(n, -1), maps)
        self.assertTrue(info['bound_valid'].all())
        self.assertTrue((info['critical_captured'] <= info['critical_reference']).all())
        for slope in (0.25, 2.0, 8.0):
            thresholds = torch.tensor([-4.2, -0.8], dtype=torch.float64)
            ref = structured_grade_probabilities(info['extent_reference'][:, None], info['critical_reference'][:, None], slope, thresholds)['grade_probs']
            cap = structured_grade_probabilities(info['extent_captured'][:, None], info['critical_captured'][:, None], slope, thresholds)['grade_probs']
            tv = (ref - cap).abs().sum(-1) * 0.5
            self.assertTrue((tv <= info['tv_bound'] + 1e-12).all())

    def test_exact_minimum_mass_excludes_empty_prior_from_certificate(self):
        p = torch.tensor([[0.0, 0.0, 0.5, 1.0]], dtype=torch.float64)
        capture = normalized_soft_capture(p, self.reference, 1, 1)
        v = torch.full((1, 1, 1, 1), 1e-06, dtype=torch.float64) / capture[:, None]
        d, s = (torch.full_like(v, 0.01), torch.zeros_like(v))
        evidence = torch.cat((v, v * d, v * d * s), 1)
        info = self.loss.statistics(p, self.reference, evidence)
        self.assertEqual(info['captured_mass'].item(), 1e-06)
        self.assertFalse(info['bound_valid'].item())
        head = CSEGGradeHead(1, 2).double()
        with torch.no_grad():
            head.empty_support_logits.copy_(torch.tensor([-5.0, -5.0, 5.0], dtype=torch.float64))
        out = head.readout_from_probabilities({'visible': v, 'damage': d, 'critical': s}, torch.stack((torch.ones_like(capture), capture), 1).reshape(1, 2, 1, 1))
        self.assertEqual(out['aux']['support_valid'].tolist(), [[True, False]])
        tv = 0.5 * (out['grade_probs'][:, 1] - out['grade_probs'][:, 0]).abs().sum(-1)
        self.assertGreater(tv.item(), 0.9)
        self.assertEqual(info['stat_loss'].item(), 0)
        at_reference = torch.cat((torch.full_like(v, 1e-06), torch.full_like(v, 1e-08), s), 1)
        exact = self.loss.statistics(self.reference, self.reference, at_reference)
        self.assertFalse(exact['valid_visible'].item())
        self.assertFalse(exact['bound_valid'].item())

    def test_full_miss_empty_half_and_instance_reduction(self):
        p = torch.tensor([[100.0, 100.0, 101.0, 101.0]], dtype=torch.float64, requires_grad=True)
        out = self.loss.statistics(p, self.reference, self.evidence)
        self.assertFalse(out['bound_valid'].item())
        self.assertTrue(torch.isfinite(out['retention']).all())
        self.assertLessEqual(out['retention'].item(), self.loss.per_box_upper_bound)
        out['retention'].sum().backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        empty_p = torch.empty(0, 4, requires_grad=True)
        empty = self.loss(empty_p, empty_p.detach(), torch.empty(0, 3, 4, 4))
        self.assertEqual(empty.item(), 0)
        empty.backward()
        self.assertTrue(torch.isfinite(self.loss(self.reference.half(), self.reference.half(), self.evidence.half())))
        p = torch.tensor([[0.0, 0.0, 0.6, 1.0], [0.0, 0.0, 0.6, 1.0], [0.3, 0.0, 1.0, 1.0]], dtype=torch.float64)
        r, e = (self.reference.expand(3, -1), self.evidence.expand(3, -1, -1, -1))
        per = GSFIoULoss(reduction='none')(p, r, e)
        mean = self.loss(p, r, e, instance_ids=torch.tensor([5, 5, 9]))
        torch.testing.assert_close(mean, (per[:2].mean() + per[2]) / 2)

class EvidenceMatchingTests(unittest.TestCase):

    def test_selects_highest_eligible_score_not_highest_score_and_falls_back(self):
        ids = torch.tensor([0, 0, 0, 1, 2])
        scores = torch.tensor([0.99, 0.8, 0.7, 0.95, 0.9], requires_grad=True)
        stats = {'bound_valid': torch.tensor([True, True, True, False, True]), 'captured_fraction': torch.tensor([0.89, 0.95, 0.99, 1.0, 0.99]), 'delta_extent': torch.tensor([0.0, 0.01, 0.0, 0.0, 0.021]), 'delta_critical': torch.tensor([0.0, 0.05, 0.0, 0.0, 0.0])}
        out = select_sufficient_grade_rois(ids, scores, stats, 4)
        self.assertEqual(out['candidate_indices'].tolist(), [1, -1, -1, -1])
        self.assertEqual(out['fallback'].tolist(), [False, True, True, True])
        self.assertEqual(out['gt_ids'].tolist(), [0, 1, 2, 3])
        self.assertEqual(out['eligible'].tolist(), [False, True, True, False, False])
        self.assertFalse(out['candidate_indices'].requires_grad)

    def test_empty_missing_and_exact_tolerance_boundaries(self):
        out = select_sufficient_grade_rois(torch.empty(0, dtype=torch.long), torch.empty(0), None, 3)
        self.assertEqual(out['candidate_indices'].tolist(), [-1, -1, -1])
        stats = {'bound_valid': torch.tensor([True]), 'captured_fraction': torch.tensor([0.9]), 'delta_extent': torch.tensor([0.02]), 'delta_critical': torch.tensor([0.1])}
        out = select_sufficient_grade_rois(torch.tensor([0]), torch.tensor([0.4]), stats, 1)
        self.assertFalse(out['fallback'].item())
        none = select_sufficient_grade_rois(torch.tensor([0]), torch.tensor([0.4]), None, 1)
        self.assertTrue(none['fallback'].item())
        zero = select_sufficient_grade_rois(torch.empty(0, dtype=torch.long), torch.empty(0), None, 0)
        self.assertEqual(zero['gt_ids'].numel(), 0)

    def test_invalid_ids_scores_and_statistics_are_rejected(self):
        for ids, scores, count in ((torch.tensor([1]), torch.tensor([0.5]), 1), (torch.tensor([0.5]), torch.tensor([0.5]), 1), (torch.tensor([0]), torch.tensor([float('nan')]), 1)):
            with self.assertRaises(ValueError):
                select_sufficient_grade_rois(ids, scores, None, count)
        with self.assertRaises(ValueError):
            select_sufficient_grade_rois(torch.tensor([0]), torch.tensor([0.5]), {}, 1)
if __name__ == '__main__':
    unittest.main()
