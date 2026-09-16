import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiment_metrics import prediction_outcome, summarize_predictions, align_records, paired_comparison, mcnemar
from intervene_pareto import eval_set


class InterventionMetricTests(unittest.TestCase):
    def test_preexisting_error_is_not_new_damage(self):
        outcome = prediction_outcome(1, 2, 3, 2, 1, 3, 3)
        self.assertEqual(outcome["negative_edit_error"], 1)
        self.assertEqual(outcome["same_sign_damage"], 0)
        self.assertEqual(outcome["negative_edit_disruption"], 0)
        self.assertEqual(outcome["baseline_clean_correct"], 0)
        self.assertEqual(outcome["new_target_success"], 1)

    def test_positive_and_negative_edits_are_separate(self):
        outcome = prediction_outcome(1, 2, 1, 2, 3, 1, 3)
        self.assertEqual(outcome["same_sign_damage"], 0)
        self.assertEqual(outcome["negative_edit_disruption"], 1)
        self.assertEqual(outcome["third_token"], 1)
        self.assertEqual(outcome["steer"], 0)

    def test_already_target_is_not_new_steering(self):
        outcome = prediction_outcome(1, 2, 1, 1, 1, 2, 1)
        self.assertEqual(outcome["steer"], 1)
        self.assertEqual(outcome["new_target_eligible"], 0)
        self.assertEqual(outcome["new_target_success"], 0)
        self.assertEqual(outcome["same_sign_damage"], 1)

    def test_summary_uses_conditional_denominators(self):
        records = [prediction_outcome(1, 2, 1, 2, 1, 2, 1),
                   prediction_outcome(1, 2, 3, 1, 1, 3, 3)]
        summary = summarize_predictions(records)
        self.assertEqual(summary["n_baseline_clean_correct"], 1)
        self.assertEqual(summary["same_sign_damage"], 1.0)
        self.assertEqual(summary["negative_edit_disruption"], 0.0)
        self.assertEqual(summary["n_new_target_eligible"], 1)
        self.assertEqual(summary["new_target_rate"], 1.0)
        self.assertGreater(summary["negative_edit_disruption_ci"][1], 0)

    def test_no_eligible_samples_returns_null_not_zero(self):
        summary = summarize_predictions([prediction_outcome(1, 2, 3, 1, 1, 3, 3)])
        self.assertIsNone(summary["same_sign_damage"])
        self.assertIsNone(summary["new_target_rate"])
        with self.assertRaises(ValueError):
            summarize_predictions([])

    def test_real_eval_path_records_positive_and_negative_predictions(self):
        clean_ids = torch.tensor([[1]])
        corrupt_ids = torch.tensor([[2]])
        def model(ids):
            prediction = 1 if ids is clean_ids else 2
            logits = torch.zeros(1, 1, 4)
            logits[0, 0, prediction] = 10
            return SimpleNamespace(logits=logits)
        harness = SimpleNamespace(model=model)
        record = SimpleNamespace(clean_ids=clean_ids, corrupt_ids=corrupt_ids,
                                 cid=1, kid=2, dpos=0, sample_id="pair", input_hash="tokens")
        def forward(harness, ids, layers, vectors, position):
            prediction = 3 if vectors[0].item() < 0 else 1
            return torch.nn.functional.one_hot(torch.tensor(prediction), 4).float() * 10
        edit = SimpleNamespace(detached=lambda: [torch.ones(1)], total_norm=lambda: 1.0)
        with patch("intervene_pareto.optimize_sample", return_value=edit), \
                patch("intervene_pareto.forward_logits", side_effect=forward), \
                patch("intervene_pareto.control_degradation", return_value=0):
            result = eval_set(harness, [record], [0], None, {}, [], [])
        self.assertEqual(result["same_sign_damage"], 0)
        self.assertEqual(result["negative_edit_disruption"], 1)
        self.assertEqual(result["records"][0]["sample_id"], "pair")
        self.assertEqual(result["records"][0]["positive_clean_prediction"], 1)

    def test_pairing_uses_ids_not_array_order(self):
        left = [{"sample_id": "a", "steer": 1}, {"sample_id": "b", "steer": 0}]
        right = [{"sample_id": "b", "steer": 0}, {"sample_id": "a", "steer": 1}]
        comparison = paired_comparison(left, right, "steer")
        self.assertEqual(comparison["difference"], 0)
        self.assertEqual(comparison["p"], 1)
        self.assertLess(comparison["ci"][0], 0)
        self.assertGreater(comparison["ci"][1], 0)

    def test_invalid_pairing_fails_closed(self):
        original = [{"sample_id": "a", "input_hash": "x"}]
        invalid = [[], [{"sample_id": "b"}], original * 2,
                   [{"sample_id": "a", "input_hash": "y"}], [{}]]
        for other in invalid:
            with self.subTest(other=other), self.assertRaises(ValueError):
                align_records(original, other)
        with self.assertRaises(ValueError):
            mcnemar([0, 1], [1])
        self.assertEqual(mcnemar([1] * 6, [0] * 6), (6, 0, 0.03125))

    def test_continuous_comparison_has_finite_monte_carlo_probability(self):
        left = [{"sample_id": str(index), "seed": index % 2, "value": index + 1}
                for index in range(12)]
        right = [{**record, "value": 0} for record in left]
        result = paired_comparison(left, right, "value", binary=False, iters=99)
        self.assertGreaterEqual(result["p"], 0.01)
        self.assertEqual(result["difference"], 6.5)


if __name__ == "__main__":
    unittest.main()