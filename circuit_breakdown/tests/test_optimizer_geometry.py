from types import SimpleNamespace
import unittest

import torch

from intervene_pareto import EditParams, basis_from_differences, optimize_sample, target_margin


class LinearHarness:
    def __init__(self):
        self.device = "cpu"
        self.layers = [torch.nn.Identity()]
        self.model = self
        self.config = SimpleNamespace(hidden_size=3)

    def __call__(self, ids):
        hidden = torch.tensor([[[0.0, 2.0, 4.0]]]).expand(1, ids.shape[1], 3).clone()
        return SimpleNamespace(logits=self.layers[0](hidden))


class OptimizerGeometryTests(unittest.TestCase):
    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is not available")
    def test_mps_differences_are_moved_to_cpu_before_double_precision(self):
        basis, information = basis_from_differences(torch.eye(3, device="mps"), 2)
        self.assertEqual(basis.device.type, "cpu")
        self.assertEqual(basis.dtype, torch.float32)
        self.assertEqual(information["basis_rank"], 2)

    def test_numerical_rank_caps_redundant_training_rows(self):
        matrix = torch.tensor([[1., 0, 0, 0], [2., 0, 0, 0], [3., 0, 0, 0]])
        basis, info = basis_from_differences(matrix, 3)
        self.assertEqual(info["numerical_rank"], 1)
        self.assertEqual(len(basis), 1)
        control, metadata = basis_from_differences(matrix, 3, True, 2)
        self.assertTrue(torch.allclose(control @ basis.T, torch.zeros(2, 1), atol=1e-6))
        self.assertLess(metadata["orthonormal_error"], 1e-6)
        with self.assertRaises(ValueError):
            basis_from_differences(torch.zeros(2, 3), 1)

    def test_per_layer_budgets_are_applied_independently(self):
        params = EditParams([0, 1], 3, "cpu")
        with torch.no_grad():
            for parameter in params.parameters():
                parameter.fill_(10)
        params.clamp_norm([1, 2])
        self.assertAlmostEqual(params.vectors()[0].norm().item(), 1, places=5)
        self.assertAlmostEqual(params.vectors()[1].norm().item(), 2, places=5)
        with self.assertRaises(ValueError):
            params.clamp_norm([1])

    def test_full_vocabulary_margin_detects_third_token(self):
        logits = torch.tensor([3., 1., 5.])
        self.assertGreater(logits[0] - logits[1], 0)
        self.assertEqual(target_margin(logits, 0).item(), -2)

    def test_identity_basis_matches_full_space_optimizer(self):
        harness = LinearHarness()
        record = SimpleNamespace(corrupt_ids=torch.tensor([[0, 1]]), cid=0, kid=1, dpos=1)
        cfg = {"lr": .2, "lam_kl": 0, "guard_every": 4, "stage_a_steps": 16,
               "l2": 0, "norm_budget": [10], "max_control_drop": 1,
               "two_stage": False, "objective": "cross_entropy"}
        full = optimize_sample(harness, record, [0], [], [], None, cfg)
        identity = optimize_sample(harness, record, [0], [], [], [torch.eye(3)], cfg)
        self.assertTrue(torch.allclose(full.detached()[0], identity.detached()[0]))
        self.assertEqual(full.trace[-1]["target_success"], 1)
        self.assertEqual(len(harness.layers[0]._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()