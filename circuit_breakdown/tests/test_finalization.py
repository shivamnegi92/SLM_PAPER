import unittest

from finalize_study import replace_results, results_prose


class FinalizationTests(unittest.TestCase):
    def test_missing_evidence_cannot_be_finalized(self):
        with self.assertRaises(ValueError):
            results_prose({"matrix_complete": False, "missing": ["model"]})

    def test_generated_region_preserves_surrounding_manuscript(self):
        text = "Introduction\n<!-- validated-results:start -->old<!-- validated-results:end -->\nLimitations"
        updated = replace_results(text, "Measured results")
        self.assertTrue(updated.startswith("Introduction\n"))
        self.assertTrue(updated.endswith("\nLimitations"))
        self.assertNotIn("old", updated)
        with self.assertRaises(ValueError):
            replace_results("no region", "results")

    def test_failed_capability_bound_is_not_reported_as_preservation(self):
        ledger = {"matrix_complete": True, "missing": [], "study": [], "heads": [],
                  "capability": [{"conditions": {"active_prefix": {
                      "benchmarks": {"arc": {"two_point_loss_bound_met": False}},
                      "text": {"ten_percent_ratio_bound_met": True}}}}]}
        self.assertIn("not all established", results_prose(ledger))

    def test_accuracy_differences_are_percentage_points(self):
        ledger = {"matrix_complete": True, "missing": [], "capability": [], "heads": [],
                  "study": [{"model": "tiny", "task": "intermediate", "result": {
                      "summaries": {"full": {"steer": 1., "n": 150, "n_seeds": 3},
                                    "track8": {"steer": .25}},
                      "comparisons": {"full_minus_track8": {"steer": {
                          "difference": .75, "ci": [.60, .80]}}}}}]}
        result = results_prose(ledger)
        self.assertIn("75.0 percentage points", result)
        self.assertIn("[60.0, 80.0] percentage points", result)
        self.assertIn("RESULTS_DETAILS.md", result)


if __name__ == "__main__":
    unittest.main()