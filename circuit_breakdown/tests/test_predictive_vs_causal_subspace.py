import unittest

import torch

from predictive_vs_causal_subspace import (
    centroid_probe_accuracy,
    centroid_probe_fit,
    coordinate_basis_from_scores,
    energy_fraction,
    pca_basis,
    project_vectors,
    random_basis,
    summarize_decode_control,
    validate_protocol,
)


class PredictiveVsCausalSubspaceTests(unittest.TestCase):
    def test_declared_protocol_shape_is_validated(self):
        protocol = {
            "version": "predictive_vs_causal_subspace_v1",
            "scope": "development_only_protocol_skeleton_not_yet_executed",
            "model": "phi-3.5-mini",
            "task": "transfer",
            "training_seed": 1,
            "development_seed": 1,
            "development_indices": list(range(8)),
            "ranks": [1, 2, 4, 8, 16, 32, 64],
            "subspaces": {"tracking": "", "pca": "", "random": "", "causal": ""},
        }
        validate_protocol(protocol)
        protocol["subspaces"].pop("causal")
        with self.assertRaisesRegex(ValueError, "tracking, pca, random and causal"):
            validate_protocol(protocol)

    def test_projection_energy_and_coordinate_basis(self):
        vector = torch.tensor([[3., 4., 0.]])
        basis, indices = coordinate_basis_from_scores(torch.tensor([0.5, 2.0, 1.0]), 2)
        self.assertEqual(indices, [1, 2])
        projected = project_vectors([vector], [basis])
        self.assertTrue(torch.equal(projected[0], torch.tensor([[0., 4., 0.]])))
        self.assertAlmostEqual(energy_fraction(projected, [vector]), 16 / 25)

    def test_pca_and_random_bases_are_orthonormal(self):
        states = torch.tensor([[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.], [0., -1., 0.]])
        basis, info = pca_basis(states, 2)
        self.assertEqual(info["basis_rank"], 2)
        self.assertTrue(torch.allclose(basis @ basis.T, torch.eye(2), atol=1e-6))
        random = random_basis(5, 3, 17)
        self.assertTrue(torch.allclose(random @ random.T, torch.eye(3), atol=1e-6))

    def test_centroid_probe_decodes_projected_states(self):
        bases = [torch.eye(2)[:1]]
        clean_train = [torch.tensor([[2., 0.], [3., 1.]])]
        corrupt_train = [torch.tensor([[-2., 0.], [-3., 1.]])]
        probe = centroid_probe_fit(clean_train, corrupt_train, bases)
        score = centroid_probe_accuracy(probe, [torch.tensor([[4., 9.]])], [torch.tensor([[-4., 9.]])], bases)
        self.assertEqual(score["correct"], 2)
        self.assertEqual(score["accuracy"], 1.0)

    def test_summary_reports_control_recovery(self):
        rows = [
            {"baseline": {"target_success": 0, "target_probability": .1},
             "full": {"target_success": 1, "target_probability": .9},
             "conditions": {"tracking_k1": {"target_success": 0, "target_probability": .3, "energy_fraction": .25}}},
            {"baseline": {"target_success": 0, "target_probability": .2},
             "full": {"target_success": 1, "target_probability": .8},
             "conditions": {"tracking_k1": {"target_success": 1, "target_probability": .5, "energy_fraction": .5}}},
        ]
        summary = summarize_decode_control(rows)["tracking_k1"]
        self.assertEqual(summary["successes"], 1)
        self.assertAlmostEqual(summary["mean_energy_fraction"], .375)
        self.assertGreater(summary["probability_recovery_fraction_vs_full"], 0)


if __name__ == "__main__":
    unittest.main()