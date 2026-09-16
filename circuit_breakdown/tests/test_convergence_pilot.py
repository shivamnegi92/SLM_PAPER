from types import SimpleNamespace
import unittest

import torch

from convergence_pilot import optimize_trajectory
from intervene_pareto import optimize_sample
from test_optimizer_geometry import LinearHarness


class ConvergencePilotTests(unittest.TestCase):
    def fixture(self, steps=128):
        record = SimpleNamespace(corrupt_ids=torch.tensor([[0, 1]]), cid=0, kid=1, dpos=1)
        cfg = {"lr": .2, "lam_kl": 0, "guard_every": 4, "stage_a_steps": steps,
               "l2": .001, "norm_budget": [10.], "max_control_drop": 1.,
               "two_stage": False, "objective": "cross_entropy"}
        return LinearHarness(), record, cfg

    def test_step32_matches_unchanged_optimizer(self):
        for bases in (None, [torch.eye(3)], [torch.tensor([[1., 0., 0.], [0., 0., 1.]])]):
            with self.subTest(subspace=bases is not None):
                harness, record, cfg = self.fixture()
                trajectory = optimize_trajectory(harness, record, [0], [], [], bases, cfg, [32, 64, 128])
                reference = optimize_sample(harness, record, [0], [], [], bases, {**cfg, "stage_a_steps": 32})
                self.assertTrue(torch.allclose(trajectory["snapshots"]["32"]["vectors"][0], reference.detached()[0]))
                self.assertAlmostEqual(trajectory["snapshots"]["32"]["metrics"]["target_margin"],
                                       reference.trace[-1]["target_margin"], places=6)
                self.assertEqual(len(harness.layers[0]._forward_hooks), 0)

    def test_snapshot_is_not_a_reference_to_final_parameters(self):
        harness, record, cfg = self.fixture()
        result = optimize_trajectory(harness, record, [0], [], [], None, cfg, [32, 64, 128])
        self.assertEqual(set(result["snapshots"]), {"0", "32", "64", "128"})
        self.assertTrue(torch.equal(result["snapshots"]["0"]["vectors"][0], torch.zeros(1, 3)))
        self.assertFalse(torch.equal(result["snapshots"]["32"]["vectors"][0], result["snapshots"]["128"]["vectors"][0]))
        self.assertEqual(len(result["trace"]), 129)
        self.assertEqual(len(result["guard_events"]), 32)

    def test_snapshot_records_the_actual_prediction(self):
        harness, record, cfg = self.fixture(32)
        result = optimize_trajectory(harness, record, [0], [], [], None, cfg, [32])
        for snapshot in result["snapshots"].values():
            self.assertEqual(snapshot["metrics"]["prediction"], snapshot["logits"].argmax().item())
            self.assertLessEqual(snapshot["vectors"][0].norm().item(), 10.00001)

    def test_guarded_trajectory_matches_original(self):
        harness, record, cfg = self.fixture(32)
        cfg["max_control_drop"] = 0.
        controls = [record.corrupt_ids]
        control_predictions = [2]
        result = optimize_trajectory(harness, record, [0], controls, control_predictions, None, cfg, [32])
        reference = optimize_sample(harness, record, [0], controls, control_predictions, None, cfg)
        self.assertTrue(any(event["shrunk"] for event in result["guard_events"]))
        self.assertTrue(torch.allclose(result["snapshots"]["32"]["vectors"][0], reference.detached()[0]))

    def test_rejects_undeclared_objective_and_checkpoints(self):
        harness, record, cfg = self.fixture(32)
        with self.assertRaises(ValueError):
            optimize_trajectory(harness, record, [0], [], [], None, cfg, [16])
        with self.assertRaises(ValueError):
            optimize_trajectory(harness, record, [0], [], [], None, {**cfg, "two_stage": True}, [32])


if __name__ == "__main__":
    unittest.main()