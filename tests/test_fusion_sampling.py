import unittest
import torch
from torch import nn
from torch.nn import functional as F
from pear_innovations.fusion_base import FixedBinomialDownsample, ScaleComparableFusion
from pear_innovations.phase_base import PhaseSupportCompressor, aggregate_evidence_reference, polyphase_gather

class ConstantSupport(nn.Module):

    def __init__(self, visible_logit, damage_logit):
        super().__init__()
        self.visible_logit = visible_logit
        self.damage_logit = damage_logit

    def forward(self, x):
        visible = x.new_full((x.shape[0], 1, *x.shape[-2:]), self.visible_logit)
        damage = x.new_full((x.shape[0], 1, *x.shape[-2:]), self.damage_logit)
        return torch.cat((visible, damage), dim=1)

class TestScaleComparableFusion(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(17)

    def test_native_even_odd_and_single_pixel_shapes(self):
        model = ScaleComparableFusion(4, 6, 5)
        for h, w in [(8, 10), (9, 11), (1, 1), (1, 7), (7, 1)]:
            with self.subTest(shape=(h, w)):
                fine = torch.randn(2, 4, h, w)
                coarse = torch.randn(2, 6, (h + 1) // 2, (w + 1) // 2)
                y, aux = model([fine, coarse], return_aux=True)
                self.assertEqual(y.shape, (2, 5, h, w))
                self.assertEqual(aux['comparison_fine'].shape[-2:], coarse.shape[-2:])
                self.assertTrue(torch.isfinite(y).all())
                self.assertTrue(torch.allclose(y, model(fine, coarse)))

    def test_rejects_resized_or_wrong_native_grid(self):
        model = ScaleComparableFusion(4, 6, 5)
        with self.assertRaisesRegex(ValueError, 'native coarse grid'):
            model(torch.randn(1, 4, 9, 11), torch.randn(1, 6, 9, 11))
        with self.assertRaises(ValueError):
            model(torch.randn(1, 5, 8, 8), torch.randn(1, 6, 4, 4))

    def test_binomial_filter_preserves_constant_including_edges(self):
        down = FixedBinomialDownsample()
        x = torch.tensor([0.0, 2.0, -3.0]).reshape(1, 3, 1, 1).expand(2, 3, 7, 9)
        y = down(x)
        self.assertEqual(y.shape[-2:], (4, 5))
        self.assertTrue(torch.equal(y, x[..., :4, :5]))
        model = ScaleComparableFusion(3, 2, 4)
        _, aux = model(x, torch.randn(2, 2, 4, 5), return_aux=True)
        self.assertLess(aux['fine_exclusive'].abs().max().item(), 1e-06)
        self.assertLess(aux['unique_update'].abs().max().item(), 1e-06)

    def test_unique_branch_has_no_coarse_computational_path(self):
        model = ScaleComparableFusion(4, 6, 5)
        fine = torch.randn(1, 4, 8, 10, requires_grad=True)
        coarse = torch.randn(1, 6, 4, 5, requires_grad=True)
        _, a = model(fine, coarse, return_aux=True)
        _, b = model(fine, 20.0 * torch.randn_like(coarse), return_aux=True)
        for key in ('fine_support', 'fine_exclusive', 'unique_gate', 'unique_update', 'unique_residual'):
            self.assertTrue(torch.equal(a[key], b[key]), key)
        dc = torch.autograd.grad(a['unique_residual'].square().sum(), coarse, allow_unused=True, retain_graph=True)[0]
        self.assertIsNone(dc)
        df = torch.autograd.grad(a['unique_residual'].square().sum(), fine)[0]
        self.assertGreater(df.abs().sum().item(), 0.0)

    def test_coarse_veto_ablation_actually_changes_unique_path(self):
        model = ScaleComparableFusion(4, 6, 5, gate_unique_with_common=True)
        fine = torch.randn(1, 4, 8, 10)
        coarse = torch.randn(1, 6, 4, 5, requires_grad=True)
        _, aux = model(fine, coarse, return_aux=True)
        grad = torch.autograd.grad(aux['unique_residual'].square().sum(), coarse)[0]
        self.assertGreater(grad.abs().sum().item(), 0.0)

    def test_disabled_updates_leave_the_projected_fine_identity(self):
        model = ScaleComparableFusion(4, 6, 5, enable_common=False, enable_unique=False)
        fine, coarse = (torch.randn(2, 4, 8, 8), torch.randn(2, 6, 4, 4))
        y, aux = model(fine, coarse, return_aux=True)
        self.assertTrue(torch.equal(y, model.fine_projection(fine)))
        self.assertEqual(aux['common_residual'].count_nonzero().item(), 0)
        self.assertEqual(aux['unique_residual'].count_nonzero().item(), 0)

    def test_naive_comparison_ablation_uses_fine_grid(self):
        model = ScaleComparableFusion(4, 6, 5, common_scale_comparison=False)
        _, aux = model(torch.randn(1, 4, 7, 9), torch.randn(1, 6, 4, 5), return_aux=True)
        self.assertEqual(aux['comparison_fine'].shape[-2:], (7, 9))

    def test_zero_features_are_finite_and_gates_coefficients_are_bounded(self):
        model = ScaleComparableFusion(3, 3, 3, residual_bound=0.25, residual_init=0.1)
        with torch.no_grad():
            model.common_scale_raw.fill_(1000.0)
            model.unique_scale_raw.fill_(-1000.0)
        y, aux = model(torch.zeros(1, 3, 3, 5), torch.zeros(1, 3, 2, 3), return_aux=True)
        self.assertTrue(torch.isfinite(y).all())
        for key in ('common_gate', 'unique_gate'):
            self.assertTrue(((aux[key] >= 0.0) & (aux[key] <= 1.0)).all())
        for key in ('common_coefficient', 'unique_coefficient'):
            self.assertLessEqual(aux[key].abs().item(), 0.25)

    def test_gradients_reach_both_features_and_support_head(self):
        model = ScaleComparableFusion(4, 6, 5)
        fine = torch.randn(2, 4, 8, 10, requires_grad=True)
        coarse = torch.randn(2, 6, 4, 5, requires_grad=True)
        y, aux = model(fine, coarse, return_aux=True)
        target = torch.full_like(aux['fine_support_logits'], 0.3)
        (y.square().mean() + F.binary_cross_entropy_with_logits(aux['fine_support_logits'], target)).backward()
        for x in (fine, coarse):
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(x.grad.abs().sum().item(), 0.0)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_training_mode_is_batch_independent(self):
        model = ScaleComparableFusion(4, 6, 5).train()
        fine, coarse = (torch.randn(3, 4, 8, 10), torch.randn(3, 6, 4, 5))
        batch = model(fine, coarse)
        singles = torch.cat([model(fine[i:i + 1], coarse[i:i + 1]) for i in range(3)])
        self.assertTrue(torch.allclose(batch, singles, atol=1e-06, rtol=1e-05))

    def test_cpu_autocast_comparison_and_gradients_remain_finite(self):
        model = ScaleComparableFusion(4, 6, 5)
        fine = torch.randn(2, 4, 7, 9, requires_grad=True)
        coarse = torch.randn(2, 6, 4, 5, requires_grad=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            output, aux = model(fine, coarse, return_aux=True)
            loss = output.float().square().mean()
        self.assertEqual(aux['comparison_fine'].dtype, torch.float32)
        self.assertTrue(torch.isfinite(output).all())
        loss.backward()
        for feature in (fine, coarse):
            self.assertTrue(torch.isfinite(feature.grad).all())

class TestPhaseSupportCompressor(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(23)

    def test_exact_phase_order(self):
        values, valid = polyphase_gather(torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]))
        self.assertTrue(torch.equal(values.flatten(), torch.tensor([1.0, 3.0, 2.0, 4.0])))
        self.assertTrue(torch.equal(valid, torch.ones_like(valid)))

    def test_odd_shape_valid_phases_and_epsilon_exclude_padding(self):
        for h, w in [(3, 5), (4, 5), (3, 4), (1, 1)]:
            with self.subTest(shape=(h, w)):
                values, valid = polyphase_gather(torch.ones(1, 2, h, w))
                supports = torch.zeros(1, 1, *values.shape[2:])
                out = aggregate_evidence_reference(values, supports, supports, valid)
                self.assertEqual(valid.sum().item(), h * w)
                for key in ('evidence_weights', 'reference_weights'):
                    self.assertEqual((out[key] * (1.0 - valid)).count_nonzero().item(), 0)
                    self.assertTrue(torch.allclose(out[key].sum(-1), torch.ones_like(out[key][..., 0])))
                self.assertTrue(torch.equal(out['evidence_slot'], torch.ones_like(out['evidence_slot'])))
                self.assertEqual(out['availability'].count_nonzero().item(), 0)

    def test_one_defect_phase_has_distinct_evidence_and_reference(self):
        values, valid = polyphase_gather(torch.tensor([[[[9.0, 1.0], [2.0, 3.0]]]]))
        visible = torch.ones_like(valid)
        damage = torch.tensor([[[[[1.0, 0.0, 0.0, 0.0]]]]])
        out = aggregate_evidence_reference(values, visible, damage, valid, eps=1e-08)
        self.assertAlmostEqual(out['evidence_slot'].item(), 9.0, places=5)
        self.assertAlmostEqual(out['reference_slot'].item(), 2.0, places=5)
        self.assertAlmostEqual(out['evidence_mass'].item(), 1.0)
        self.assertAlmostEqual(out['reference_mass'].item(), 3.0)
        reconstructed = out['evidence_slot'] - out['reference_slot'] + out['reference_slot']
        self.assertTrue(torch.allclose(reconstructed, out['evidence_slot']))

    def test_values_at_invalid_phases_do_not_enter_slots(self):
        values, valid = polyphase_gather(torch.tensor([[[[7.0]]]]))
        corrupted = values + (1.0 - valid) * 100000000.0
        visible, damage = (torch.ones_like(valid), torch.full_like(valid, 0.25))
        out = aggregate_evidence_reference(corrupted, visible, damage, valid)
        self.assertAlmostEqual(out['evidence_slot'].item(), 7.0, places=5)
        self.assertAlmostEqual(out['reference_slot'].item(), 7.0, places=5)

    def test_all_invalid_external_mask_remains_finite(self):
        values = torch.randn(1, 2, 1, 1, 4)
        valid = torch.zeros(1, 1, 1, 1, 4)
        out = aggregate_evidence_reference(values, torch.ones_like(valid), torch.ones_like(valid), valid)
        for value in out.values():
            self.assertTrue(torch.isfinite(value).all())
        self.assertEqual(out['evidence_slot'].count_nonzero().item(), 0)
        self.assertEqual(out['availability'].count_nonzero().item(), 0)

    def test_tiny_support_has_finite_slots_and_vanishing_availability(self):
        values = torch.tensor([[[[[1.0, 3.0, 2.0, 4.0]]]]], dtype=torch.float16)
        visible = torch.full((1, 1, 1, 1, 4), 1e-10, dtype=torch.float32)
        damage = torch.full_like(visible, 0.5)
        valid = torch.ones_like(visible)
        out = aggregate_evidence_reference(values, visible, damage, valid)
        self.assertEqual(out['evidence_weights'].dtype, torch.float32)
        for value in out.values():
            self.assertTrue(torch.isfinite(value).all())
        self.assertAlmostEqual(out['evidence_slot'].item(), 2.5, places=5)
        self.assertLess(out['availability'].item(), 1e-15)

    def test_empty_evidence_or_reference_recovers_original_main_path(self):
        x = torch.randn(2, 3, 7, 9)
        for visible, damage in [(-1000.0, 0.0), (1000.0, -1000.0), (1000.0, 1000.0)]:
            with self.subTest(visible=visible, damage=damage):
                model = PhaseSupportCompressor(3, 5, rank=4)
                model.support_head = ConstantSupport(visible, damage)
                y, aux = model(x, return_aux=True)
                self.assertTrue(torch.equal(y, model.main(x)))
                self.assertEqual(aux['availability'].count_nonzero().item(), 0)
                self.assertTrue(torch.isfinite(y).all())

    def test_main_path_never_receives_padded_input(self):
        model = PhaseSupportCompressor(1, 1, rank=2, enable_residual=False)
        with torch.no_grad():
            model.main.weight.fill_(1.0)
        x = torch.arange(15.0).reshape(1, 1, 3, 5)
        y = model(x)
        expected = F.conv2d(x, model.main.weight, stride=2, padding=1)
        self.assertTrue(torch.equal(y, expected))
        self.assertEqual(y.shape[-2:], (2, 3))

    def test_native_even_odd_singleton_and_one_slot_shapes(self):
        for mode in ('dual', 'evidence'):
            model = PhaseSupportCompressor(3, 5, rank=4, slot_mode=mode)
            for h, w in [(8, 10), (9, 11), (1, 1), (1, 7), (7, 1)]:
                with self.subTest(mode=mode, shape=(h, w)):
                    y = model(torch.randn(2, 3, h, w))
                    self.assertEqual(y.shape, (2, 5, (h + 1) // 2, (w + 1) // 2))
                    self.assertTrue(torch.isfinite(y).all())

    def test_pooling_autograd_matches_finite_differences(self):
        values = torch.randn(1, 2, 1, 1, 4, dtype=torch.double, requires_grad=True)
        visible = torch.full((1, 1, 1, 1, 4), 0.7, dtype=torch.double, requires_grad=True)
        damage = torch.tensor([[[[[0.2, 0.4, 0.6, 0.8]]]]], dtype=torch.double, requires_grad=True)
        valid = torch.ones_like(visible)

        def pool(z, v, d):
            result = aggregate_evidence_reference(z, v, d, valid)
            return (result['evidence_slot'], result['reference_slot'], result['availability'])
        self.assertTrue(torch.autograd.gradcheck(pool, (values, visible, damage), eps=1e-06, atol=1e-05))

    def test_all_parameters_receive_finite_training_gradients(self):
        model = PhaseSupportCompressor(3, 5, rank=4)
        x = torch.randn(2, 3, 7, 9, requires_grad=True)
        y, aux = model(x, return_aux=True)
        target = torch.full_like(aux['support_logits'], 0.3)
        (y.square().mean() + F.binary_cross_entropy_with_logits(aux['support_logits'], target)).backward()
        self.assertGreater(x.grad.abs().sum().item(), 0.0)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0, name)

    def test_detach_support_blocks_main_loss_but_allows_aux_loss(self):
        model = PhaseSupportCompressor(3, 5, rank=4, detach_support=True)
        y, aux = model(torch.randn(1, 3, 7, 9), return_aux=True)
        gradient = torch.autograd.grad(y.sum(), aux['support_logits'], allow_unused=True, retain_graph=True)[0]
        self.assertIsNone(gradient)
        loss = F.binary_cross_entropy_with_logits(aux['support_logits'], torch.zeros_like(aux['support_logits']))
        gradient = torch.autograd.grad(loss, model.support_head[-1].weight)[0]
        self.assertGreater(gradient.abs().sum().item(), 0.0)

    def test_coefficient_is_bounded_and_uniform_support_is_explicit(self):
        model = PhaseSupportCompressor(3, 5, rank=4, support_mode='uniform', alpha_max=0.2, residual_init=0.1)
        with torch.no_grad():
            model.alpha_raw.fill_(1000.0)
        _, aux = model(torch.randn(1, 3, 7, 9), return_aux=True)
        self.assertLessEqual(aux['residual_coefficient'].abs().item(), 0.2 + 1e-07)
        self.assertTrue(torch.equal(aux['visible_support'], torch.ones_like(aux['visible_support'])))
        self.assertTrue(torch.equal(aux['damage_support'], torch.full_like(aux['damage_support'], 0.5)))

    def test_training_mode_is_batch_independent(self):
        model = PhaseSupportCompressor(3, 5, rank=4).train()
        x = torch.randn(3, 3, 9, 11)
        batched = model(x)
        individual = torch.cat([model(x[i:i + 1]) for i in range(3)])
        self.assertTrue(torch.allclose(batched, individual, atol=1e-06, rtol=1e-05))

    def test_cpu_autocast_keeps_support_sums_fp32_and_backward_finite(self):
        model = PhaseSupportCompressor(3, 5, rank=4)
        x = torch.randn(2, 3, 7, 9, requires_grad=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            y, aux = model(x, return_aux=True)
            loss = y.float().square().mean()
        self.assertEqual(aux['evidence_mass'].dtype, torch.float32)
        self.assertTrue(torch.isfinite(y).all())
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_invalid_configuration_is_rejected(self):
        for kwargs in ({'tau': 0.0}, {'eps': 0.0}, {'rank': 0}, {'slot_mode': 'oops'}, {'alpha_max': 0.1, 'residual_init': 0.1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PhaseSupportCompressor(3, 5, **kwargs)
if __name__ == '__main__':
    unittest.main()
