import unittest

from run_capability_study import paired_accuracy, window_summary


class CapabilityStudyTests(unittest.TestCase):
    def test_pairing_checks_ids(self):
        baseline = [{"id": "a", "correct": 1}, {"id": "b", "correct": 0}]
        result = paired_accuracy(baseline, baseline)
        self.assertEqual(result["delta"], 0)
        self.assertFalse(result["two_point_loss_bound_met"])
        with self.assertRaises(ValueError):
            paired_accuracy(baseline, baseline[::-1])

    def test_window_bootstrap_uses_windows_not_independent_tokens(self):
        result = window_summary([[1, 2]] * 5, [[2, 3]] * 5)
        self.assertAlmostEqual(result["mean_log_perplexity_change"], 1)
        self.assertFalse(result["ten_percent_ratio_bound_met"])
        self.assertEqual(len(result["per_window_log_changes"]), 5)
        with self.assertRaises(ValueError):
            window_summary([[1, 2]], [[1]])


if __name__ == "__main__":
    unittest.main()