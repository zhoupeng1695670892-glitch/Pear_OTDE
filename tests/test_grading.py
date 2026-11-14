import unittest
import torch
import torch.nn.functional as F
from pear_innovations.grade_support import BasicROIGradeHead, EvidencePoolingGradeHead, evidence_supervision_loss, nested_region_weights, weighted_upper_tail

class GradingTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(7)
        self.head = EvidencePoolingGradeHead(4, feature_dim=8, tail_fraction=0.5)
        self.x = torch.randn(2, 4, 3, 5)

    def test_predicted_shapes_and_nested_conservation(self):
        result = self.head(self.x)
        aux = result['aux']
        self.assertEqual(result['grade_logits'].shape, (2, 3))
        self.assertEqual(result['embedding'].shape, (2, 8))
        self.assertEqual(aux['region_tokens'].shape, (2, 3, 8))
        self.assertFalse(aux['evidence_override_used'])
        torch.testing.assert_close(aux['region_weights'].sum(1, keepdim=True), aux['evidence_probs']['visible'])
        torch.testing.assert_close(aux['relative_mass'].sum(1), torch.ones(2))
        self.assertTrue(torch.isfinite(result['grade_logits']).all())

    def test_nested_weights_at_extremes(self):
        v = torch.tensor([[[[1.0, 0.5, 0.0]]]])
        d = torch.tensor([[[[0.0, 1.0, 1.0]]]])
        s = torch.tensor([[[[1.0, 0.25, 1.0]]]])
        weights = nested_region_weights(v, d, s)
        torch.testing.assert_close(weights, torch.tensor([[[[1.0, 0.0, 0.0]], [[0.0, 0.375, 0.0]], [[0.0, 0.125, 0.0]]]]))
        torch.testing.assert_close(weights.sum(1, keepdim=True), v)

    def test_known_regions_and_visible_relative_area(self):
        x = torch.randn(1, 4, 1, 4)
        masks = {'visible': torch.tensor([[[[1.0, 1.0, 0.0, 0.0]]]]), 'damage': torch.tensor([[[[0.0, 1.0, 1.0, 0.0]]]]), 'critical': torch.tensor([[[[0.0, 1.0, 1.0, 1.0]]]])}
        out = self.head(x, evidence_override=masks, diagnostic=True)
        aux, feat = (out['aux'], self.head.stem(x))
        torch.testing.assert_close(aux['relative_mass'], torch.tensor([[0.5, 0.0, 0.5]]))
        torch.testing.assert_close(aux['extent'], torch.tensor([[0.5]]))
        torch.testing.assert_close(aux['region_tokens'][:, 0], feat[:, :, 0, 0])
        torch.testing.assert_close(aux['region_tokens'][:, 2], feat[:, :, 0, 1])
        self.assertFalse(aux['region_valid'][0, 1])
        torch.testing.assert_close(aux['tail']['mean'], torch.ones(1, 1))
        torch.testing.assert_close(aux['tail']['token'], feat[:, :, 0, 1])

    def test_no_visibility_falls_back_without_nan_or_dropping_roi(self):
        zero = torch.zeros(2, 1, 3, 5)
        result = self.head(self.x, evidence_override={'visible': zero}, diagnostic=True)
        aux = result['aux']
        self.assertFalse(aux['visible_valid'].any())
        self.assertFalse(aux['region_valid'].any())
        self.assertEqual(aux['relative_mass'].abs().sum().item(), 0)
        self.assertEqual(aux['region_tokens'].abs().sum().item(), 0)
        self.assertEqual(aux['tail']['token'].abs().sum().item(), 0)
        torch.testing.assert_close(aux['global_token'], self.head.stem(self.x).flatten(2).mean(-1))
        self.assertTrue(torch.isfinite(result['grade_logits']).all())

    def test_no_critical_evidence_produces_zero_tail_evidence_token(self):
        result = self.head(self.x, evidence_override={'critical': torch.zeros(2, 1, 3, 5)}, diagnostic=True)
        tail = result['aux']['tail']
        self.assertEqual(tail['mean'].abs().sum().item(), 0)
        self.assertEqual(tail['token'].abs().sum().item(), 0)
        self.assertFalse(tail['evidence_valid'].any())

    def test_diagnostic_override_must_be_explicit(self):
        with self.assertRaisesRegex(ValueError, 'diagnostic'):
            self.head(self.x, evidence_override={'visible': torch.ones(2, 1, 3, 5)})
        with self.assertRaises(ValueError):
            self.head(self.x, evidence_override={'visible': torch.ones(2, 1, 2, 5)}, diagnostic=True)
        with self.assertRaises(ValueError):
            self.head(self.x, evidence_override={'visible': torch.full((2, 1, 3, 5), 2.0)}, diagnostic=True)

    def test_zero_roi_batch_all_heads(self):
        x = torch.empty(0, 4, 3, 5)
        for head in (self.head, BasicROIGradeHead(4, 8), BasicROIGradeHead(4, 8, pooling='meanmax')):
            out = head(x)
            self.assertEqual(out['grade_logits'].shape, (0, 3))
            loss = evidence_supervision_loss(out)['loss'] + out['grade_logits'].sum()
            self.assertEqual(loss.item(), 0)
            loss.backward()

    def test_fractional_weighted_tail_matches_hand_calculation(self):
        score = torch.tensor([[0.2, 0.9, 0.6, 0.1]], dtype=torch.float64, requires_grad=True)
        support = torch.tensor([[2.0, 1.0, 3.0, 4.0]], dtype=torch.float64)
        tail = weighted_upper_tail(score, support, 10 * score.detach().unsqueeze(-1), fraction=0.25)
        torch.testing.assert_close(tail['weights'], torch.tensor([[0.0, 1.0, 1.5, 0.0]], dtype=torch.float64))
        torch.testing.assert_close(tail['mean'], torch.tensor([[0.72]], dtype=torch.float64))
        torch.testing.assert_close(tail['token'], torch.tensor([[7.2]], dtype=torch.float64))
        tail['mean'].sum().backward()
        torch.testing.assert_close(score.grad, torch.tensor([[0.0, 0.4, 0.6, 0.0]], dtype=torch.float64))

    def test_tail_support_not_roi_pixel_count(self):
        scores = torch.tensor([[0.9, 0.6, 100.0, 200.0]])
        support = torch.tensor([[1.0, 3.0, 0.0, 0.0]])
        tail = weighted_upper_tail(scores, support, fraction=0.5)
        torch.testing.assert_close(tail['mean'], torch.tensor([[0.75]]))
        torch.testing.assert_close(tail['selected_mass'], torch.tensor([[2.0]]))
        repeated = weighted_upper_tail(scores.repeat_interleave(2, dim=1), support.repeat_interleave(2, dim=1) / 2, fraction=0.5)
        torch.testing.assert_close(tail['mean'], repeated['mean'])
        full = weighted_upper_tail(scores, support, fraction=1)
        torch.testing.assert_close(full['mean'], torch.tensor([[0.675]]))

    def test_tail_zero_and_tiny_support_finite(self):
        score = torch.tensor([[0.2, 0.8], [0.1, 0.9]])
        support = torch.tensor([[0.0, 0.0], [0.0, 1e-09]])
        tail = weighted_upper_tail(score, support, fraction=0.01)
        self.assertEqual(tail['mean'].abs().sum().item(), 0)
        self.assertFalse(tail['valid'].any())
        with self.assertRaises(ValueError):
            weighted_upper_tail(score, support, fraction=0)

    def test_grade_gradient_cannot_directly_train_visibility(self):
        result = self.head(self.x)
        logits = result['aux']['evidence_logits']
        for tensor in logits.values():
            tensor.retain_grad()
        result['grade_logits'].square().sum().backward()
        self.assertIsNone(logits['visible'].grad)
        self.assertIsNone(self.head.evidence_heads.visible.weight.grad)
        self.assertGreater(logits['damage'].grad.abs().sum().item(), 0)
        self.assertGreater(logits['critical'].grad.abs().sum().item(), 0)
        self.assertGreater(self.head.stem[0].weight.grad.abs().sum().item(), 0)

    def test_optional_detach_all_evidence_weights(self):
        head = EvidencePoolingGradeHead(4, 8, detach_region_weights=True)
        head(self.x)['grade_logits'].square().sum().backward()
        for parameter in head.evidence_heads.parameters():
            self.assertIsNone(parameter.grad)
        self.assertGreater(head.stem[0].weight.grad.abs().sum().item(), 0)

    def test_baseline_shares_auxiliary_head_architecture(self):
        baseline = BasicROIGradeHead(4, 8, pooling='meanmax')
        self.assertEqual(list(self.head.stem.state_dict()), list(baseline.stem.state_dict()))
        self.assertEqual(list(self.head.evidence_heads.state_dict()), list(baseline.evidence_heads.state_dict()))
        for key, tensor in self.head.evidence_heads.state_dict().items():
            self.assertEqual(tensor.shape, baseline.evidence_heads.state_dict()[key].shape)
        original = baseline(self.x)['grade_logits']
        oracle = baseline(self.x, evidence_override={'visible': torch.zeros(2, 1, 3, 5)}, diagnostic=True)['grade_logits']
        torch.testing.assert_close(original, oracle)
        original.sum().backward()
        self.assertTrue(all((p.grad is None for p in baseline.evidence_heads.parameters())))

    def test_no_masks_are_not_negative_labels(self):
        result = self.head(self.x)
        losses = evidence_supervision_loss(result)
        self.assertEqual(losses['loss'].item(), 0)
        losses['loss'].backward()
        for param in self.head.evidence_heads.parameters():
            self.assertEqual(param.grad.abs().sum().item(), 0)

    def test_conditional_supervision_masks_out_background_and_non_damage(self):
        result = self.head(torch.randn(1, 4, 1, 4))
        logits = result['aux']['evidence_logits']
        for tensor in logits.values():
            tensor.retain_grad()
        target = {'visible': torch.tensor([[[[1.0, 1.0, 0.0, 0.0]]]]), 'damage': torch.tensor([[[[0.0, 1.0, 1.0, 1.0]]]]), 'critical': torch.ones(1, 1, 1, 4)}
        loss = evidence_supervision_loss(result, target)
        self.assertEqual(loss['support_mass']['damage'].item(), 2)
        self.assertEqual(loss['support_mass']['critical'].item(), 1)
        loss['loss'].backward()
        self.assertEqual(logits['damage'].grad[..., 2:].abs().sum().item(), 0)
        self.assertEqual(logits['critical'].grad[..., [0, 2, 3]].abs().sum().item(), 0)
        self.assertGreater(logits['critical'].grad[..., 1].abs().sum().item(), 0)
        self.assertGreater(self.head.evidence_heads.visible.weight.grad.abs().sum().item(), 0)

    def test_missing_condition_requires_explicit_conditional_domain(self):
        result = self.head(self.x)
        target = torch.ones(2, 1, 3, 5)
        with self.assertRaisesRegex(ValueError, 'visible support'):
            evidence_supervision_loss(result, {'damage': target})
        with self.assertRaisesRegex(ValueError, 'damage support'):
            evidence_supervision_loss(result, {'critical': target})
        valid = torch.zeros_like(target)
        valid[..., 0, 0] = 1
        loss = evidence_supervision_loss(result, {'critical': target}, valid_masks={'critical': valid})
        self.assertEqual(loss['support_mass']['critical'].item(), 2)
        self.assertEqual(loss['losses']['damage'].item(), 0)

    def test_partial_labels_and_roi_balanced_auxiliary_loss(self):
        logits = {name: torch.tensor([[[[0.0, 0.0, 0.0]]], [[[2.0, 2.0, 2.0]]]], requires_grad=True) for name in ('visible', 'damage', 'critical')}
        target = torch.tensor([[[[1.0, -1.0, float('nan')]]], [[[0.0, 0.0, 0.0]]]])
        loss = evidence_supervision_loss({'evidence_logits': logits}, {'visible': target})
        expected = (F.softplus(torch.tensor(0.0)) + F.softplus(torch.tensor(2.0))) / 2
        torch.testing.assert_close(loss['loss'], expected)
        self.assertEqual(loss['annotated_instances']['visible'].item(), 2)
        loss['loss'].backward()
        self.assertEqual(logits['visible'].grad[0, :, :, 1:].abs().sum().item(), 0)

    def test_noncontiguous_input_and_cpu_autocast(self):
        x = self.x.transpose(-1, -2)
        self.assertFalse(x.is_contiguous())
        with torch.autocast('cpu', dtype=torch.bfloat16):
            result = self.head(x)
            loss = result['grade_logits'].float().square().mean()
            loss = loss + evidence_supervision_loss(result, {'visible': torch.ones(2, 1, 5, 3)})['loss']
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all((p.grad is None or torch.isfinite(p.grad).all() for p in self.head.parameters())))
if __name__ == '__main__':
    unittest.main()
