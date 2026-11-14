import unittest
import torch
import torch.nn.functional as F
from pear_innovations.critical_support import build_component_labels, component_critical, critical_support_readout, prepare_component_context
from pear_innovations.grade_support import evidence_supervision_loss
from pear_innovations.support_grading import CSEGGradeHead

class CriticalSupportTests(unittest.TestCase):

    def test_four_connectivity_threshold_and_single_pixel_support(self):
        h = torch.tensor([[[[0.5, 0.1], [0.1, 0.9]]]])
        labels = build_component_labels(h)
        torch.testing.assert_close(labels, torch.tensor([[[[1, 0], [0, 2]]]]))
        self.assertFalse(labels.requires_grad)
        out = critical_support_readout(h)
        torch.testing.assert_close(out['peaks'], torch.tensor([[0.5, 0.9]]))
        torch.testing.assert_close(out['c'], torch.tensor([[0.9]]))
        torch.testing.assert_close(out['coverage'], torch.ones(1, 1, 2))

    def test_partial_component_attenuates_even_when_peak_is_retained(self):
        h = torch.tensor([[[[0.6, 0.9]]]], dtype=torch.float64)
        capture = torch.tensor([[[[0.0, 1.0]]]], dtype=torch.float64)
        out = critical_support_readout(h, capture)
        torch.testing.assert_close(out['masses'], torch.tensor([[1.5]], dtype=torch.float64))
        torch.testing.assert_close(out['coverage'], torch.tensor([[[0.6]]], dtype=torch.float64))
        torch.testing.assert_close(out['c'], torch.tensor([[0.324]], dtype=torch.float64))
        self.assertLess(out['c'].item(), 0.9)

    def test_alternative_complete_component_preserves_its_own_contribution(self):
        h = torch.tensor([[[[0.8, 0.8, 0.0, 0.9], [0.0, 0.0, 0.0, 0.9]]]])
        t = torch.zeros_like(h)
        t[..., 0, :2] = 1
        out = critical_support_readout(h, t)
        torch.testing.assert_close(out['c'], torch.tensor([[0.8]]))
        self.assertLess(out['c'].item(), critical_support_readout(h)['c'].item())
        binary = (h > 0).float()
        torch.testing.assert_close(critical_support_readout(binary, t)['c'], torch.ones(1, 1))

    def test_components_are_built_on_context_not_query_fragments(self):
        h = torch.tensor([[[[0.8, 0.8, 0.8]]]])
        capture = torch.tensor([[[[1.0, 0.0, 1.0]]]])
        out = critical_support_readout(h, capture)
        self.assertEqual(out['peaks'].shape[1], 1)
        torch.testing.assert_close(out['coverage'], torch.tensor([[[2 / 3]]]))
        torch.testing.assert_close(out['c'], torch.tensor([[0.8 * (2 / 3) ** 2]]))

    def test_multiple_queries_share_identities_and_match_individual_reads(self):
        h = torch.tensor([[[[0.6, 0.8, 0.1, 0.9]]], [[[0.1, 0.2, 0.3, 0.4]]]], requires_grad=True)
        capture = torch.tensor([[[[1.0, 1.0, 1.0, 1.0]], [[1.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0]]], [[[1.0, 1.0, 1.0, 1.0]], [[1.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0]]]])
        out = critical_support_readout(h, capture)
        self.assertEqual(out['c'].shape, (2, 3))
        self.assertEqual(out['coverage'].shape, (2, 3, 2))
        torch.testing.assert_close(out['c'][1], torch.zeros(3))
        for query in range(3):
            one = critical_support_readout(h, capture[:, query], component_labels=out['labels'])
            torch.testing.assert_close(out['c'][:, query], one['c'][:, 0])

    def test_zero_support_zero_capture_and_zero_batch(self):
        h = torch.zeros(2, 1, 3, 5, requires_grad=True)
        capture = torch.zeros(2, 4, 3, 5, requires_grad=True)
        out = critical_support_readout(h, capture)
        self.assertEqual(out['coverage'].shape, (2, 4, 0))
        self.assertEqual(out['c'].sum().item(), 0)
        out['c'].sum().backward()
        torch.testing.assert_close(h.grad, torch.zeros_like(h))
        torch.testing.assert_close(capture.grad, torch.zeros_like(capture))
        empty = critical_support_readout(torch.empty(0, 1, 3, 5), torch.empty(0, 2, 3, 5))
        self.assertEqual(empty['c'].shape, (0, 2))
        self.assertEqual(empty['labels'].shape, (0, 1, 3, 5))
        self.assertEqual(empty['masses'].shape, (0, 0))
        active = torch.ones(1, 1, 2, 2, requires_grad=True)
        no_capture = torch.zeros(1, 1, 2, 2, requires_grad=True)
        zero = critical_support_readout(active, no_capture)
        zero['c'].sum().backward()
        self.assertTrue(torch.isfinite(no_capture.grad).all())
        torch.testing.assert_close(no_capture.grad, torch.zeros_like(no_capture))

    def test_gradient_check_with_fixed_component_identities(self):
        h = torch.tensor([[[[0.63, 0.81, 0.1, 0.72]]]], dtype=torch.float64, requires_grad=True)
        capture = torch.tensor([[[[0.7, 0.4, 0.3, 0.2]]]], dtype=torch.float64, requires_grad=True)
        labels = build_component_labels(h)

        def read(values, captured):
            return critical_support_readout(values, captured, component_labels=labels)['c']
        self.assertTrue(torch.autograd.gradcheck(read, (h, capture), eps=1e-06, atol=1e-05))

    def test_capture_monotonicity_bounded_values_and_normal_padding(self):
        torch.manual_seed(31)
        h = torch.rand(4, 1, 7, 9)
        a = torch.rand(4, 2, 7, 9)
        b = a + (1 - a) * torch.rand_like(a)
        low = critical_support_readout(h, a)
        high = critical_support_readout(h, b, component_labels=low['labels'])
        self.assertTrue((high['c'] >= low['c'] - 1e-07).all())
        self.assertTrue(((low['c'] >= 0) & (high['c'] <= 1)).all())
        padded = F.pad(h, (0, 4, 0, 2))
        padded_capture = F.pad(a, (0, 4, 0, 2), value=1)
        torch.testing.assert_close(low['c'], critical_support_readout(padded, padded_capture)['c'])

    def test_fixed_identity_can_be_reused_without_rethresholding(self):
        h = torch.tensor([[[[0.8, 0.9]]]])
        context = prepare_component_context(h)
        changed = torch.tensor([[[[0.2, 0.3]]]], requires_grad=True)
        out = component_critical(changed, context=context)
        torch.testing.assert_close(out['c'], torch.tensor([[0.3]]))
        self.assertEqual(critical_support_readout(changed)['c'].item(), 0)
        out['c'].sum().backward()
        self.assertTrue(torch.isfinite(changed.grad).all())

    def test_half_input_and_weak_zero_mass_cached_component(self):
        out = critical_support_readout(torch.tensor([[[[0.75, 1.0]]]], dtype=torch.float16))
        self.assertEqual(out['c'].dtype, torch.float32)
        torch.testing.assert_close(out['c'], torch.ones(1, 1))
        zero = critical_support_readout(torch.zeros(1, 1, 1, 2), component_labels=torch.ones(1, 1, 1, 2, dtype=torch.long))
        self.assertEqual(zero['c'].item(), 0)
        self.assertFalse(zero['component_valid'].any())

    def test_invalid_input_contracts(self):
        h = torch.ones(1, 1, 2, 3)
        for kwargs in ({'power': 0.5}, {'power': float('inf')}, {'threshold': 0.0}, {'threshold': 1.1}, {'eps': 0.0}):
            with self.assertRaises(ValueError):
                critical_support_readout(h, **kwargs)
        for capture in (torch.ones(1, 1, 2, 2), torch.ones(1, 0, 2, 3), torch.full_like(h, float('nan')), torch.full_like(h, 1.01)):
            with self.assertRaises(ValueError):
                critical_support_readout(h, capture)
        for labels in (torch.ones_like(h), torch.full_like(h, -1, dtype=torch.long), torch.full_like(h, 7, dtype=torch.long)):
            with self.assertRaises(ValueError):
                critical_support_readout(h, component_labels=labels)
        with self.assertRaises(ValueError):
            critical_support_readout(torch.ones(1, 2, 2, 3))

class ContextGradeHeadTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(23)
        self.head = CSEGGradeHead(4, 8, fine_channels=3, use_fine=True)
        self.coarse = torch.randn(2, 4, 18, 18)
        self.fine = torch.randn(2, 3, 35, 35)

    def test_context_default_and_multiquery_shapes_and_simplex(self):
        context = self.head.encode_context(self.coarse, self.fine)
        out = self.head.readout_context(context)
        self.assertEqual(out['grade_logits'].shape, (2, 3))
        self.assertEqual(out['embedding'].shape, (2, 8))
        self.assertEqual(out['aux']['evidence_logits']['visible'].shape, (2, 1, 35, 35))
        torch.testing.assert_close(out['grade_probs'].sum(-1), torch.ones(2))
        t = torch.stack((torch.ones(2, 35, 35), torch.full((2, 35, 35), 0.5)), dim=1)
        multi = self.head.readout_context(context, t)
        self.assertEqual(multi['grade_logits'].shape, (2, 2, 3))
        self.assertEqual(multi['aux']['cumulative_probs'].shape, (2, 2, 2))
        torch.testing.assert_close(multi['grade_probs'][:, 0], out['grade_probs'])
        torch.testing.assert_close(multi['grade_probs'].sum(-1), torch.ones(2, 2))
        torch.testing.assert_close(multi['aux']['component_labels'], context['component_labels'])

    def test_partial_query_extent_and_alternative_support_by_hand(self):
        head = CSEGGradeHead(4, 8)
        fields = {'visible': torch.ones(1, 1, 1, 4), 'damage': torch.tensor([[[[1.0, 1.0, 0.0, 1.0]]]]), 'critical': torch.tensor([[[[0.8, 0.8, 0.0, 0.9]]]])}
        capture = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0, 1.0]]]])
        out = head.readout_from_probabilities(fields, capture)
        torch.testing.assert_close(out['aux']['extent'], torch.tensor([[0.5, 0.5]]))
        torch.testing.assert_close(out['aux']['critical_presence'], torch.tensor([[0.2, 0.9]]))
        self.assertEqual(out['aux']['component_peaks'].shape, (1, 2))

    def test_visibility_direct_gradient_detached_and_capture_gradient_active(self):
        head = CSEGGradeHead(4, 8)
        fields = {name: torch.tensor([[[[0.9, 0.8, 0.2, 0.85]]]], requires_grad=True) for name in ('visible', 'damage', 'critical')}
        capture = torch.tensor([[[[0.7, 0.3, 0.4, 0.8]]]], requires_grad=True)
        out = head.readout_from_probabilities(fields, capture)
        F.cross_entropy(out['grade_logits'], torch.tensor([1])).backward()
        self.assertIsNone(fields['visible'].grad)
        for value in (fields['damage'], fields['critical'], capture):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)

    def test_detached_calibration_keeps_field_and_capture_gradients(self):
        head = CSEGGradeHead(4, 8)
        fields = {name: torch.full((1, 1, 2, 3), 0.9, requires_grad=True) for name in ('visible', 'damage', 'critical')}
        capture = torch.full((1, 1, 2, 3), 0.7, requires_grad=True)
        out = head.readout_from_probabilities(fields, capture, calibration_detach=True)
        out['aux']['cumulative_probs'].sum().backward()
        for parameter in (head.raw_slope, head.threshold_base, head.threshold_gap, head.empty_support_logits):
            self.assertIsNone(parameter.grad)
        self.assertGreater(fields['critical'].grad.abs().sum().item(), 0)
        self.assertGreater(capture.grad.abs().sum().item(), 0)

    def test_no_component_has_zero_critical_response_not_an_implicit_max_fallback(self):
        head = CSEGGradeHead(4, 8)
        fields = {name: torch.full((1, 1, 2, 3), 0.5, requires_grad=True) for name in ('visible', 'damage', 'critical')}
        out = head.readout_from_probabilities(fields)
        self.assertEqual(out['aux']['critical_presence'].item(), 0)
        self.assertEqual(out['aux']['component_peaks'].shape, (1, 0))
        F.cross_entropy(out['grade_logits'], torch.tensor([2])).backward()
        self.assertTrue(torch.isfinite(fields['critical'].grad).all())
        self.assertEqual(fields['critical'].grad.abs().sum().item(), 0)

    def test_auxiliary_supervision_trains_support_heads_on_context_grid(self):
        out = self.head(self.coarse, self.fine)
        v = torch.ones(2, 1, 35, 35)
        d = torch.zeros_like(v)
        s = torch.zeros_like(v)
        d[:, :, :12, :12] = 1
        s[:, :, :3, :3] = 1
        loss = evidence_supervision_loss(out, {'visible': v, 'damage': d, 'critical': s})['loss']
        loss.backward()
        for parameter in self.head.evidence_heads.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertGreater(self.head.fine_stem[0].weight.grad.abs().sum().item(), 0)

    def test_empty_capture_prior_and_ideal_event_diagnostic_are_separate(self):
        with torch.no_grad():
            self.head.empty_support_logits.copy_(torch.tensor([0.0, 1.0, -1.0]))
        out = self.head(self.coarse, self.fine, capture=torch.zeros(2, 1, 35, 35))
        self.assertFalse(out['aux']['support_valid'].any())
        self.assertFalse(out['aux']['cumulative_includes_empty_prior'])
        torch.testing.assert_close(out['grade_probs'], torch.tensor([0.0, 1.0, -1.0]).softmax(0).expand(2, -1))
        self.assertTrue(torch.isfinite(out['grade_logits']).all())

    def test_context_reuse_does_not_rerun_projection_or_support_heads(self):
        calls = []
        handle = self.head.stem.register_forward_hook(lambda *_: calls.append(1))
        context = self.head.encode_context(self.coarse, self.fine)
        out1 = self.head.readout_context(context)
        out2 = self.head.readout_context(context, torch.ones(2, 3, 35, 35))
        handle.remove()
        self.assertEqual(len(calls), 1)
        for q in range(3):
            torch.testing.assert_close(out2['grade_probs'][:, q], out1['grade_probs'])

    def test_zero_batch_autocast_and_one_scale_ablation(self):
        empty = self.head(torch.empty(0, 4, 18, 18), torch.empty(0, 3, 35, 35), capture=torch.empty(0, 2, 35, 35))
        self.assertEqual(empty['grade_logits'].shape, (0, 2, 3))
        (empty['grade_logits'].sum() + evidence_supervision_loss(empty)['loss']).backward()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            out = self.head(self.coarse, self.fine)
        self.assertEqual(out['grade_logits'].dtype, torch.float32)
        self.assertTrue(torch.isfinite(out['grade_logits']).all())
        coarse_head = CSEGGradeHead(4, 8)
        single = coarse_head(self.coarse)
        self.assertEqual(single['aux']['component_labels'].shape, (2, 1, 18, 18))

    def test_override_and_probability_contracts(self):
        with self.assertRaisesRegex(ValueError, 'diagnostic-only'):
            self.head(self.coarse, self.fine, evidence_override={})
        override = {'visible': torch.ones(2, 1, 35, 35), 'damage': torch.ones(2, 1, 35, 35), 'critical': torch.ones(2, 1, 35, 35)}
        out = self.head(self.coarse, self.fine, evidence_override=override, diagnostic=True)
        torch.testing.assert_close(out['aux']['critical_presence'], torch.ones(2, 1))
        for bad in ({'visible': torch.ones(1, 1, 2, 3)}, {'visible': torch.ones(1, 1, 2, 3), 'damage': torch.ones(1, 1, 2, 3), 'critical': torch.full((1, 1, 2, 3), 1.1)}):
            with self.assertRaises(ValueError):
                self.head.readout_from_probabilities(bad)
        with self.assertRaises(ValueError):
            CSEGGradeHead(4, 8, critical_power=0.5)
if __name__ == '__main__':
    unittest.main()
