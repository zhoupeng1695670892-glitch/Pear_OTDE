import unittest
import torch
from torch import nn
from torch.nn import functional as F
from pear_innovations.energy_fusion import LECF, constrain_reference_energy, polyphase_restore
from pear_innovations.phase_base import aggregate_evidence_reference, polyphase_gather
from pear_innovations.phase_moments import RCPM, reference_moments
from pear_innovations.detection_training import shallow_evidence_loss

def vector(values, dtype=torch.float64):
    return torch.tensor(values, dtype=dtype).reshape(1, 1, 1, 1, 4)

class FixedLogits(nn.Module):

    def __init__(self, visible, damage):
        super().__init__()
        self.visible, self.damage = (visible, damage)

    def forward(self, x):
        shape = (x.shape[0], 1, *x.shape[-2:])
        return torch.cat((x.new_full(shape, self.visible), x.new_full(shape, self.damage)), 1)

class TestReferenceMoments(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(53)

    def test_same_means_different_distribution_is_distinguished(self):
        we, wr = (vector([0.5, 0.5, 0.0, 0.0]), vector([0.0, 0.0, 0.5, 0.5]))
        patterned = reference_moments(vector([-1.0, 1.0, 0.0, 0.0]), we, wr, variance_floor=0.1)
        flat = reference_moments(vector([0.0, 0.0, 0.0, 0.0]), we, wr, variance_floor=0.1)
        for key in ('evidence_slot', 'reference_slot', 'mean_difference'):
            torch.testing.assert_close(patterned[key], flat[key])
        self.assertAlmostEqual(patterned['log_energy_contrast'].item(), torch.log(torch.tensor(11.0)).item(), places=6)
        self.assertEqual(flat['log_energy_contrast'].item(), 0.0)

    def test_identical_distributions_have_zero_contrast(self):
        z = vector([-2.0, 2.0, -2.0, 2.0])
        stats = reference_moments(z, vector([0.5, 0.5, 0.0, 0.0]), vector([0.0, 0.0, 0.5, 0.5]))
        self.assertEqual(stats['log_energy_contrast'].item(), 0.0)

    def test_translation_invariance_of_centered_statistics(self):
        z = torch.randn(2, 3, 2, 2, 4, dtype=torch.float64)
        we = torch.rand(2, 1, 2, 2, 4, dtype=torch.float64).softmax(-1)
        wr = torch.rand_like(we).softmax(-1)
        shift = torch.randn(2, 3, 2, 2, 1, dtype=torch.float64) * 20
        a, b = (reference_moments(z, we, wr), reference_moments(z + shift, we, wr))
        for key in ('mean_difference', 'reference_variance', 'evidence_energy', 'log_energy_contrast'):
            torch.testing.assert_close(a[key], b[key], atol=1e-12, rtol=1e-12)

    def test_phase_roundtrip_even_odd_and_thin_inputs(self):
        for shape in ((4, 6), (5, 7), (1, 9), (7, 1), (1, 1)):
            x = torch.arange(2 * 3 * shape[0] * shape[1]).reshape(2, 3, *shape).float()
            phases, valid = polyphase_gather(x)
            torch.testing.assert_close(polyphase_restore(phases, *shape), x, atol=0, rtol=0)
            self.assertEqual(valid.sum().item(), shape[0] * shape[1])

    def test_erpc_disabled_is_exact_main_but_head_remains_trainable(self):
        model = RCPM(3, 5, rank=4, enable_residual=False)
        x = torch.randn(2, 3, 5, 7)
        y, aux = model(x, return_aux=True)
        torch.testing.assert_close(y, model.main(x), atol=0, rtol=0)
        self.assertEqual(aux['support_logits'].shape, (2, 2, 5, 7))
        shallow_evidence_loss(aux, torch.ones(2, 2, 5, 7), torch.ones(2, 2, 5, 7)).backward()
        self.assertGreater(model.support_head[-1].weight.grad.abs().sum().item(), 0.0)

    def test_empty_one_sided_support_falls_back_to_main(self):
        model = RCPM(3, 5, rank=4)
        x = torch.randn(1, 3, 5, 7)
        for logits in ((-1000.0, 0.0), (1000.0, -1000.0), (1000.0, 1000.0)):
            model.support_head = FixedLogits(*logits)
            y, aux = model(x, return_aux=True)
            torch.testing.assert_close(y, model.main(x), atol=0, rtol=0)
            self.assertTrue(torch.isfinite(aux['log_energy_contrast']).all())

    def test_moment_gradcheck(self):
        z = torch.randn(1, 2, 1, 1, 4, dtype=torch.float64, requires_grad=True)
        we, wr = (vector([0.4, 0.3, 0.2, 0.1]), vector([0.1, 0.2, 0.3, 0.4]))
        self.assertTrue(torch.autograd.gradcheck(lambda value: reference_moments(value, we, wr)['log_energy_contrast'], (z,)))

class TestQuadraticProjection(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(59)
        self.we, self.wr = (vector([0.5, 0.5, 0.0, 0.0]), vector([0.0, 0.0, 0.5, 0.5]))

    def test_exact_counteracting_update_is_removed(self):
        base = vector([-1.0, 1.0, 0.0, 0.0])
        output, aux = constrain_reference_energy(base, -base, self.we, self.wr, variance_floor=1.0)
        torch.testing.assert_close(output, base, atol=0, rtol=0)
        self.assertEqual(aux['energy_before'].item(), 1.0)
        self.assertEqual(aux['energy_unconstrained'].item(), 0.0)
        self.assertEqual(aux['energy_after'].item(), 1.0)
        self.assertTrue(aux['projection_applied'].all())

    def test_feasible_tangent_update_is_unchanged(self):
        base, update = (vector([-1.0, 1.0, 0.0, 0.0]), vector([1.0, 1.0, 0.0, 0.0]))
        output, aux = constrain_reference_energy(base, update, self.we, self.wr, variance_floor=1.0)
        torch.testing.assert_close(output, base + update, atol=0, rtol=0)
        self.assertFalse(aux['projection_applied'].any())
        self.assertEqual(aux['energy_after'].item(), 2.0)

    def test_zero_normal_and_inactive_constraints(self):
        base, update = (vector([3.0, 3.0, 3.0, 3.0]), vector([-1.0, 2.0, 4.0, -2.0]))
        output, aux = constrain_reference_energy(base, update, self.we, self.wr)
        torch.testing.assert_close(output, base + update, atol=0, rtol=0)
        self.assertFalse(aux['constraint_active'].any())
        base = vector([-1.0, 1.0, 0.0, 0.0])
        output, aux = constrain_reference_energy(base, -base, self.we, self.wr, active=torch.zeros(1, 1, 1, 1, dtype=torch.bool))
        torch.testing.assert_close(output, torch.zeros_like(base), atol=0, rtol=0)
        self.assertFalse(aux['projection_applied'].any())

    def test_matrix_definition_and_autograd_verify_normal(self):
        base = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
        we = torch.tensor([0.55, 0.25, 0.15, 0.05], dtype=torch.float64)
        wr = torch.tensor([0.05, 0.15, 0.25, 0.55], dtype=torch.float64)
        a = torch.diag(we.sqrt()) @ (torch.eye(4, dtype=torch.float64) - torch.ones(4, 1, dtype=torch.float64) @ wr[None])
        reference = wr @ base.detach()
        variance = wr @ (base.detach() - reference).square()
        q = torch.diag((variance + 0.1).rsqrt())
        energy = (a @ base @ q).square().sum()
        gradient, = torch.autograd.grad(energy, base)
        phases = base.T.reshape(1, 3, 1, 1, 4)
        _, aux = constrain_reference_energy(phases, torch.zeros_like(phases), we.reshape(1, 1, 1, 1, 4), wr.reshape(1, 1, 1, 1, 4))
        expected_normal = gradient.T.reshape_as(phases)
        expected_normal /= expected_normal.abs().max()
        torch.testing.assert_close(aux['constraint_normal'], expected_normal, atol=1e-12, rtol=1e-12)
        self.assertAlmostEqual(aux['energy_before'].item(), energy.item(), places=12)

    def test_random_feasibility_and_minimum_distance(self):
        base = torch.randn(1, 3, 2, 2, 4, dtype=torch.float64)
        update = torch.randn_like(base)
        we = torch.randn(1, 1, 2, 2, 4, dtype=torch.float64).softmax(-1)
        wr = torch.randn_like(we).softmax(-1)
        out, aux = constrain_reference_energy(base, update, we, wr)
        self.assertTrue((aux['constraint_dot_after'] >= -1e-12).all())
        self.assertTrue((aux['energy_after'] >= aux['energy_before'] - 1e-12).all())
        projected = out - base
        normal = aux['constraint_normal']
        for _ in range(8):
            competitor = torch.randn_like(base)
            inner = (competitor * normal).sum((1, 4), keepdim=True)
            norm = normal.square().sum((1, 4), keepdim=True)
            competitor += (1.0 + inner.abs()) / norm * normal
            distance = (competitor - update).square().sum((1, 4))
            optimum = (projected - update).square().sum((1, 4))
            self.assertTrue((distance >= optimum - 1e-12).all())

    def test_projection_does_not_increase_update_norm(self):
        base = torch.randn(2, 3, 3, 2, 4, dtype=torch.float64)
        update = torch.randn_like(base)
        we = torch.rand(2, 1, 3, 2, 4, dtype=torch.float64).softmax(-1)
        wr = torch.rand_like(we).softmax(-1)
        output, _ = constrain_reference_energy(base, update, we, wr)
        self.assertTrue(((output - base).square().sum((1, 4)) <= update.square().sum((1, 4)) + 1e-12).all())

    def test_fixed_constraint_update_gradient_and_full_gradient(self):
        base = vector([-1.0, 1.0, 0.1, -0.1])
        update = (-0.8 * base + 0.02).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda u: constrain_reference_energy(base, u, self.we, self.wr)[0], (update,)))
        base = base.requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda f, u: constrain_reference_energy(f, u, self.we, self.wr, detach_constraint=False)[0], (base, update)))

    def test_detach_removes_constraint_support_gradient_only(self):
        base = torch.randn(1, 2, 1, 1, 4, dtype=torch.float64, requires_grad=True)
        logits = torch.randn(1, 1, 1, 1, 4, dtype=torch.float64, requires_grad=True)
        we, wr = (logits.softmax(-1), (-logits).softmax(-1))
        update = torch.randn_like(base, requires_grad=True)
        output, _ = constrain_reference_energy(base, update, we, wr, detach_constraint=True)
        gradients = torch.autograd.grad(output.square().sum(), (base, update, logits), allow_unused=True)
        self.assertIsNotNone(gradients[0])
        self.assertIsNotNone(gradients[1])
        self.assertIsNone(gradients[2])

class TestEncoderIntegration(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(61)

    def test_native_concat_fallback_order_and_auxiliary_training(self):
        model = LECF(3, 2, 5, enabled=False)
        fine, coarse = (torch.randn(2, 3, 5, 7), torch.randn(2, 2, 3, 4))
        output, aux = model(fine, coarse, return_aux=True)
        expected = torch.cat((F.interpolate(coarse, size=(5, 7), mode='nearest'), fine), 1)
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        self.assertEqual(aux['support_logits'].shape, (2, 2, 5, 7))
        shallow_evidence_loss(aux, torch.ones(2, 2, 5, 7), torch.ones(2, 2, 5, 7)).backward()
        self.assertGreater(model.fine_support_head.weight.grad.abs().sum().item(), 0.0)

    def test_odd_shape_energy_guarantee_and_backward(self):
        model = LECF(3, 2, 5).double()
        fine = torch.randn(2, 3, 5, 7, dtype=torch.float64, requires_grad=True)
        coarse = torch.randn(2, 2, 3, 4, dtype=torch.float64, requires_grad=True)
        output, aux = model(fine, coarse, return_aux=True)
        self.assertEqual(output.shape, (2, 5, 5, 7))
        active = aux['constraint_active']
        self.assertTrue((aux['energy_after'][active] >= aux['energy_before'][active] - 1e-10).all())
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(fine.grad).all())
        self.assertTrue(torch.isfinite(coarse.grad).all())
        self.assertIsNotNone(model.fine_support_head.weight.grad)
        self.assertGreater(model.fine_support_head.weight.grad.abs().sum().item(), 0.0)

    def test_support_absence_disables_projection(self):
        model = LECF(3, 2, 5)
        model.fine_support_head = FixedLogits(1000.0, -1000.0)
        fine, coarse = (torch.randn(1, 3, 4, 6), torch.randn(1, 2, 2, 3))
        y, aux = model(fine, coarse, return_aux=True)
        self.assertFalse(aux['constraint_active'].any())
        torch.testing.assert_close(y, aux['base'] + aux['proposed_update'], atol=1e-07, rtol=1e-06)

    def test_projection_off_uses_same_head_and_unmodified_update(self):
        model = LECF(3, 2, 5, enable_projection=False)
        y, aux = model(torch.randn(1, 3, 4, 6), torch.randn(1, 2, 2, 3), return_aux=True)
        torch.testing.assert_close(y, aux['base'] + aux['proposed_update'], atol=0, rtol=0)
        self.assertEqual(aux['projection_correction'].count_nonzero().item(), 0)

    def test_missing_annotation_is_zero_loss_not_background(self):
        model = LECF(3, 2, 5)
        _, aux = model(torch.randn(1, 3, 3, 5), torch.randn(1, 2, 2, 3), return_aux=True)
        loss = shallow_evidence_loss(aux)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertEqual(model.fine_support_head.weight.grad.count_nonzero().item(), 0)

    def test_bfloat16_forward_and_gradients_are_finite(self):
        model = LECF(3, 2, 5)
        encoder = RCPM(3, 4, rank=3)
        fine, coarse = (torch.randn(2, 3, 5, 7, requires_grad=True), torch.randn(2, 2, 3, 4, requires_grad=True))
        with torch.autocast('cpu', dtype=torch.bfloat16):
            y, aux = model(fine, coarse, return_aux=True)
            z, zaux = encoder(fine, return_aux=True)
            loss = y.float().square().mean() + z.float().square().mean()
        self.assertEqual(aux['energy_before'].dtype, torch.float32)
        self.assertEqual(zaux['reference_variance'].dtype, torch.float32)
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(torch.isfinite(z).all())
        loss.backward()
        self.assertTrue(torch.isfinite(fine.grad).all())
        self.assertTrue(torch.isfinite(coarse.grad).all())

    def test_shape_and_positive_constant_guards(self):
        with self.assertRaises(ValueError):
            LECF(3, 2, 4)
        with self.assertRaises(ValueError):
            RCPM(3, 4, variance_floor=0.0)
        model = LECF(3, 2, 5)
        with self.assertRaisesRegex(ValueError, 'native coarse grid'):
            model(torch.randn(1, 3, 5, 7), torch.randn(1, 2, 5, 7))
if __name__ == '__main__':
    unittest.main()
