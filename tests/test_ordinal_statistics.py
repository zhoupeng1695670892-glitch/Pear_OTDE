import unittest
import torch
import torch.nn.functional as F
from pear_innovations.grade_support import evidence_supervision_loss
from pear_innovations.ordinal_readout import OrdinalEvidenceReadout, extent_log_odds, structured_grade_probabilities
from pear_innovations.statistic_baseline import PointStatisticIoULoss

class StructuredReadoutTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(19)
        self.head = OrdinalEvidenceReadout(4, 8)
        self.x = torch.randn(2, 4, 4, 6)

    def test_shapes_simplex_order_and_cross_entropy(self):
        out = self.head(self.x)
        self.assertEqual(out['grade_logits'].shape, (2, 3))
        self.assertEqual(out['embedding'].shape, (2, 8))
        self.assertEqual(out['aux']['evidence_logits']['visible'].shape, (2, 1, 4, 6))
        self.assertTrue(torch.isfinite(out['grade_logits']).all())
        torch.testing.assert_close(out['grade_probs'].sum(-1), torch.ones(2))
        q = out['aux']['cumulative_probs']
        self.assertTrue((q[:, 0] >= q[:, 1]).all())
        torch.testing.assert_close(out['grade_probs'][:, 2], q[:, 1])
        self.assertTrue(torch.isfinite(F.cross_entropy(out['grade_logits'], torch.tensor([0, 2]))))

    def test_fine_input_controls_evidence_grid_and_both_inputs_get_gradients(self):
        head = OrdinalEvidenceReadout(4, 8, fine_channels=3, use_fine=True)
        coarse = torch.randn(2, 4, 14, 14, requires_grad=True)
        fine = torch.randn(2, 3, 28, 28, requires_grad=True)
        out = head(coarse, fine)
        self.assertEqual(out['aux']['evidence_logits']['damage'].shape, (2, 1, 28, 28))
        self.assertTrue(out['aux']['fine_readout_used'])
        F.cross_entropy(out['grade_logits'], torch.tensor([0, 1])).backward()
        for tensor in (coarse, fine, head.stem[0].weight, head.fine_stem[0].weight, head.local_context[0].weight):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)
        with self.assertRaisesRegex(ValueError, 'fine_roi required'):
            head(coarse)
        with self.assertRaises(ValueError):
            head(coarse, torch.randn(2, 4, 28, 28))
        torch.testing.assert_close(self.head(self.x)['grade_logits'], self.head(self.x, torch.randn(2, 3, 9, 9))['grade_logits'])

    def test_grade_gradient_cannot_directly_change_visible_mask(self):
        out = self.head(self.x)
        logits = out['aux']['evidence_logits']
        for value in logits.values():
            value.retain_grad()
        F.cross_entropy(out['grade_logits'], torch.tensor([0, 1])).backward()
        self.assertIsNone(logits['visible'].grad)
        self.assertIsNone(self.head.evidence_heads.visible.weight.grad)
        for name in ('damage', 'critical'):
            self.assertGreater(logits[name].grad.abs().sum().item(), 0)
            self.assertTrue(torch.isfinite(logits[name].grad).all())
        for parameter in (self.head.raw_slope, self.head.threshold_base, self.head.threshold_gap):
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_existing_conditional_evidence_supervision_on_fine_grid(self):
        head = OrdinalEvidenceReadout(4, 8, fine_channels=3, use_fine=True)
        out = head(torch.randn(2, 4, 14, 14), torch.randn(2, 3, 28, 28))
        v = torch.ones(2, 1, 28, 28)
        d = torch.zeros_like(v)
        s = torch.zeros_like(v)
        d[:, :, :10, :10] = 1
        s[:, :, :3, :3] = 1
        loss = evidence_supervision_loss(out, {'visible': v, 'damage': d, 'critical': s})['loss']
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for parameter in head.evidence_heads.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_visible_area_and_critical_witness_have_different_dilution(self):

        def controlled(w):
            v = torch.ones(1, 1, 1, w)
            d = torch.zeros_like(v)
            s = torch.zeros_like(v)
            d[..., 0] = 0.8
            s[..., 0] = 0.9
            return self.head(torch.randn(1, 4, 1, w), evidence_override={'visible': v, 'damage': d, 'critical': s}, diagnostic=True)
        small, large = (controlled(4)['aux'], controlled(16)['aux'])
        torch.testing.assert_close(small['extent'], 4 * large['extent'])
        torch.testing.assert_close(small['critical_presence'], large['critical_presence'])
        torch.testing.assert_close(small['critical_presence'], torch.tensor([[0.72]]))

    def test_empty_visible_prior_is_finite_and_not_a_forced_grade(self):
        with torch.no_grad():
            self.head.empty_support_logits.copy_(torch.tensor([-1.0, 0.0, 2.0]))
        out = self.head(self.x, evidence_override={'visible': torch.zeros(2, 1, 4, 6)}, diagnostic=True)
        self.assertFalse(out['aux']['support_valid'].any())
        self.assertTrue(torch.isfinite(out['grade_logits']).all())
        torch.testing.assert_close(out['grade_probs'], torch.tensor([-1.0, 0.0, 2.0]).softmax(0).expand(2, -1))
        torch.testing.assert_close(out['aux']['cumulative_probs'][:, 0], out['grade_probs'][:, 1:].sum(-1))
        torch.testing.assert_close(out['aux']['cumulative_probs'][:, 1], out['grade_probs'][:, 2])
        F.cross_entropy(out['grade_logits'], torch.tensor([0, 1])).backward()
        self.assertGreater(self.head.empty_support_logits.grad.abs().sum().item(), 0)

    def test_diagnostic_override_boundaries(self):
        with self.assertRaisesRegex(ValueError, 'diagnostic-only'):
            self.head(self.x, evidence_override={})
        for masks in ({'other': torch.ones(2, 1, 4, 6)}, {'visible': torch.ones(2, 1, 3, 6)}, {'visible': torch.full((2, 1, 4, 6), float('nan'))}, {'visible': torch.full((2, 1, 4, 6), 1.1)}):
            with self.assertRaises(ValueError):
                self.head(self.x, evidence_override=masks, diagnostic=True)

    def test_empty_batch_and_autocast_are_finite(self):
        for head, coarse, fine in ((self.head, torch.empty(0, 4, 4, 6), None), (OrdinalEvidenceReadout(4, 8, fine_channels=3, use_fine=True), torch.empty(0, 4, 14, 14), torch.empty(0, 3, 28, 28))):
            out = head(coarse, fine)
            self.assertEqual(out['grade_logits'].shape, (0, 3))
            loss = out['grade_logits'].sum() + evidence_supervision_loss(out)['loss']
            self.assertEqual(loss.item(), 0)
            loss.backward()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            out = self.head(self.x)
        self.assertEqual(out['grade_logits'].dtype, torch.float32)
        self.assertTrue(torch.isfinite(out['grade_logits']).all())

    def test_probability_monotonicity_extremes_and_convex_mixture(self):
        e = torch.linspace(0, 1, 301, dtype=torch.float64)[:, None]
        c = torch.full_like(e, 0.27)
        t = torch.tensor([-4.0, -1.0], dtype=torch.float64)
        out = structured_grade_probabilities(e, c, 3.0, t)
        cumulative = out['cumulative_probs']
        self.assertTrue((cumulative[1:] >= cumulative[:-1]).all())
        self.assertTrue((cumulative[:, 0] >= cumulative[:, 1]).all())
        torch.testing.assert_close(out['grade_probs'].sum(-1), torch.ones(301, dtype=torch.float64))
        zero = structured_grade_probabilities(e, torch.zeros_like(c), 3.0, t)['grade_probs']
        corner = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        torch.testing.assert_close(out['grade_probs'], (1 - c) * zero + c * corner)
        all_critical = structured_grade_probabilities(e, torch.ones_like(c), 3.0, t)
        self.assertTrue(torch.isfinite(all_critical['grade_logits']).all())
        torch.testing.assert_close(all_critical['grade_probs'], corner.expand(301, -1))
        cs = torch.linspace(0, 1, 301, dtype=torch.float64)[:, None]
        qs = structured_grade_probabilities(torch.full_like(cs, 0.025), cs, 3.0, t)['cumulative_probs']
        self.assertTrue((qs[1:] >= qs[:-1]).all())

    def test_tiny_extent_gradient_and_saturated_middle_probability(self):
        extent = torch.tensor([[0.003]], dtype=torch.float64, requires_grad=True)
        t = extent_log_odds(torch.tensor([0.01, 0.2], dtype=torch.float64))
        out = structured_grade_probabilities(extent, torch.zeros_like(extent), 2.0, t)
        (-out['grade_logits'][0, 1]).backward()
        self.assertTrue(torch.isfinite(extent.grad).all())
        self.assertGreater(extent.grad.abs().sum().item(), 0)
        out = structured_grade_probabilities(torch.tensor([[0.5]], dtype=torch.float64), torch.zeros(1, 1, dtype=torch.float64), 1.0, torch.tensor([-100.0, -99.0], dtype=torch.float64))
        self.assertGreater(out['grade_probs'][0, 1].item(), 0)
        self.assertTrue(torch.isfinite(out['grade_logits']).all())
        equal = structured_grade_probabilities(torch.tensor([[0.0], [1.0]]), torch.tensor([[0.0], [1.0]]), 8.0, torch.tensor([0.0, 0.0]))
        self.assertTrue(torch.isfinite(equal['grade_logits']).all())

    def test_invalid_readout_hyperparameters_and_inputs(self):
        for kwargs in ({'use_fine': True}, {'num_grades': 4}, {'extent_eps': 0.0}, {'min_slope': 8.0}, {'initial_thresholds': (0.4, 0.2)}, {'max_slope': float('inf')}):
            with self.assertRaises(ValueError):
                OrdinalEvidenceReadout(4, 8, **kwargs)
        for t, c, a in (([1.0, 0.0], 0.1, 1.0), ([0.0, 1.0], -1.0, 1.0), ([0.0, 1.0], 0.1, 0.0)):
            with self.assertRaises(ValueError):
                structured_grade_probabilities(torch.tensor([[0.2]]), torch.tensor([[c]]), a, torch.tensor(t))
        with self.assertRaises(ValueError):
            extent_log_odds(torch.tensor([1.01]))

class StatisticFidelityTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(21)
        self.loss = PointStatisticIoULoss()
        self.target = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float64)
        self.maps = torch.zeros(1, 3, 12, 20, dtype=torch.float64)
        self.maps[:, 0] = 1
        self.maps[:, 1, :, :8] = 1

    def test_identity_has_exact_zero_penalty_and_gradient(self):
        pred = self.target.clone().requires_grad_()
        target = self.target.clone().requires_grad_()
        evidence = self.maps.clone().requires_grad_()
        total, info = self.loss(pred, target, evidence, return_details=True)
        self.assertEqual(total.item(), 0)
        self.assertEqual(info['retention'].item(), 0)
        torch.testing.assert_close(info['capture'], torch.ones_like(info['capture']), rtol=0, atol=0)
        total.backward()
        torch.testing.assert_close(pred.grad, torch.zeros_like(pred), rtol=0, atol=0)
        self.assertIsNone(target.grad)
        self.assertIsNone(evidence.grad)

    def test_crop_normal_and_crop_damage_move_extent_in_opposite_directions(self):
        pred = torch.tensor([[0.0, 0.0, 0.6, 1.0], [0.4, 0.0, 1.0, 1.0]], dtype=torch.float64, requires_grad=True)
        out = self.loss.statistics(pred, self.target.expand(2, -1), self.maps.expand(2, -1, -1, -1))
        delta = out['extent_captured'] - out['extent_reference']
        self.assertGreater(delta[0].item(), 0)
        self.assertLess(delta[1].item(), 0)
        self.assertTrue((out['extent_loss'] > 0).all())
        out['retention'].sum().backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(pred.grad.abs().sum().item(), 0)

    def test_proportional_crop_keeps_extent_but_loses_visible_coverage(self):
        evidence = self.maps.clone()
        evidence[:, 1] = 0.3
        out = self.loss.statistics(torch.tensor([[0.2, 0.2, 0.8, 0.8]], dtype=torch.float64), self.target, evidence)
        self.assertLess(out['delta_x'].item(), 1e-12)
        self.assertGreater(out['coverage_loss'].item(), 0.5)
        self.assertGreater(out['retention'].item(), 0.5)

    def test_expansion_is_handled_by_geometry_without_artificial_capture_gain(self):
        pred = torch.tensor([[-0.2, -0.1, 1.2, 1.1]], dtype=torch.float64, requires_grad=True)
        total, out = self.loss(pred, self.target, self.maps, return_details=True)
        self.assertEqual(out['retention'].item(), 0)
        self.assertGreater(total.item(), 0)
        torch.testing.assert_close(out['capture'], torch.ones_like(out['capture']), rtol=0, atol=0)

    def test_critical_witness_fidelity_and_extent_gate(self):
        evidence = self.maps.clone()
        evidence[:, 2, 6, 1] = 1
        pred = torch.tensor([[0.6, 0.0, 1.0, 1.0]], dtype=torch.float64)
        out = self.loss.statistics(pred, self.target, evidence)
        self.assertEqual(out['critical_reference'].item(), 1)
        self.assertGreater(out['critical_loss'].item(), 0.99)
        self.assertEqual(out['extent_loss'].item(), 0)
        self.assertTrue(out['bound_valid'].item())

    def test_full_miss_finite_upper_bound_and_unavailable_bound(self):
        pred = torch.tensor([[100.0, 100.0, 101.0, 101.0]], dtype=torch.float64, requires_grad=True)
        out = self.loss.statistics(pred, self.target, self.maps)
        self.assertTrue(torch.isfinite(out['retention']).all())
        self.assertLessEqual(out['retention'].item(), self.loss.per_box_upper_bound)
        self.assertFalse(out['bound_valid'].item())
        out['retention'].sum().backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_missing_is_not_empty_and_partial_annotations_are_masked(self):
        pred = torch.tensor([[0.3, 0.0, 1.0, 1.0]], dtype=torch.float64)
        annotated_empty = self.loss.statistics(pred, self.target, self.maps)
        self.assertTrue(annotated_empty['valid_critical'].item())
        partial = self.loss.statistics(pred, self.target, self.maps, torch.tensor([[1, 1, 0]]))
        self.assertFalse(partial['valid_critical'].item())
        self.assertFalse(partial['bound_valid'].item())
        self.assertGreater(partial['extent_loss'].item(), 0)
        noisy_placeholder = self.maps.clone()
        noisy_placeholder[:, 2] = 0.9
        partial2 = self.loss.statistics(pred, self.target, noisy_placeholder, torch.tensor([[1, 1, 0]]))
        torch.testing.assert_close(partial['retention'], partial2['retention'])
        missing = self.loss.statistics(pred, self.target, None)
        self.assertEqual(missing['retention'].item(), 0)
        self.assertFalse(missing['valid'].item())
        none_available = self.loss.statistics(pred, self.target, self.maps, torch.zeros(1, 3))
        self.assertEqual(none_available['retention'].item(), 0)
        empty = self.loss.statistics(pred, self.target, torch.zeros_like(self.maps))
        self.assertEqual(empty['retention'].item(), 0)
        self.assertFalse(empty['valid_extent'].item())
        self.assertFalse(empty['bound_valid'].item())

    def test_semantic_nesting_and_availability_validation(self):
        for channel in (0, 1):
            bad = self.maps.clone()
            if channel == 0:
                bad[:, 0] = 0
            else:
                bad[:, 2] = 1
            with self.assertRaisesRegex(ValueError, 'S <= D <= V'):
                self.loss.statistics(self.target, self.target, bad)
        for valid in (torch.tensor([[1, 1]]), torch.tensor([[1.0, 0.5, 0.0]]), torch.tensor([[1.0, float('nan'), 0.0]])):
            with self.assertRaises(ValueError):
                self.loss.statistics(self.target, self.target, self.maps, valid)
        with self.assertRaises(ValueError):
            self.loss.statistics(torch.zeros(1, 4), self.target, None)

    def test_fixed_field_probability_tv_is_bounded_for_random_boxes(self):
        n = 128
        v = torch.rand(n, 1, 7, 9, dtype=torch.float64)
        d = v * torch.rand_like(v)
        s = d * torch.rand_like(v)
        evidence = torch.cat((v, d, s), 1)
        left = torch.rand(n, 2, dtype=torch.float64) * 0.6 - 0.1
        size = torch.rand(n, 2, dtype=torch.float64) * 0.8 + 0.2
        pred = torch.cat((left, left + size), -1)
        target = self.target.expand(n, -1)
        info = self.loss.statistics(pred, target, evidence)
        self.assertTrue(info['bound_valid'].all())
        for slope in (0.25, 2.0, 8.0):
            t = torch.tensor([-4.2, -0.8], dtype=torch.float64)
            reference = structured_grade_probabilities(info['extent_reference'][:, None], info['critical_reference'][:, None], slope, t)['grade_probs']
            captured = structured_grade_probabilities(info['extent_captured'][:, None], info['critical_captured'][:, None], slope, t)['grade_probs']
            tv = 0.5 * (reference - captured).abs().sum(-1)
            self.assertTrue((tv <= info['tv_bound'] + 1e-12).all())
        self.assertTrue((info['retention'] <= self.loss.per_box_upper_bound).all())

    def test_translation_and_axiswise_scale_invariance_of_retention(self):
        pred = torch.tensor([[0.2, -0.1, 0.9, 0.8]], dtype=torch.float64)
        baseline = self.loss.statistics(pred, self.target, self.maps)
        scale = torch.tensor([1.7, 2.9, 1.7, 2.9], dtype=torch.float64)
        shift = torch.tensor([13.0, -9.0, 13.0, -9.0], dtype=torch.float64)
        moved = self.loss.statistics(pred * scale + shift, self.target * scale + shift, self.maps)
        for key in ('capture', 'extent_captured', 'critical_captured', 'retention'):
            torch.testing.assert_close(baseline[key], moved[key], rtol=1e-12, atol=1e-12)

    def test_empty_batch_half_precision_and_instance_aggregation(self):
        pred = torch.empty(0, 4, requires_grad=True)
        empty = self.loss(pred, pred.detach(), torch.empty(0, 3, 4, 4))
        self.assertEqual(empty.item(), 0)
        empty.backward()
        half = self.loss(self.target.half(), self.target.half(), self.maps.half())
        self.assertTrue(torch.isfinite(half))
        target = self.target.expand(3, -1)
        pred = torch.tensor([[0.0, 0.0, 0.8, 1.0], [0.0, 0.0, 0.8, 1.0], [0.2, 0.0, 1.0, 1.0]], dtype=torch.float64)
        maps = self.maps.expand(3, -1, -1, -1)
        per_box = PointStatisticIoULoss(reduction='none')(pred, target, maps)
        average = self.loss(pred, target, maps, instance_ids=torch.tensor([11, 11, 22]))
        torch.testing.assert_close(average, (per_box[:2].mean() + per_box[2]) / 2)
        torch.testing.assert_close(PointStatisticIoULoss(reduction='sum')(pred, target, maps), per_box.sum())

    def test_invalid_loss_hyperparameters(self):
        for kwargs in ({'temperature': 0.0}, {'max_slope': -1.0}, {'weight': -0.1}, {'coverage_weight': -0.1}, {'eps': float('nan')}, {'min_mass': 1e-09, 'eps': 1e-07}, {'reduction': 'other'}):
            with self.assertRaises(ValueError):
                PointStatisticIoULoss(**kwargs)
if __name__ == '__main__':
    unittest.main()
