import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from audit_study import file_sha256
from experiment_metrics import rate_ci
from report_study import detailed_markdown, point_change, require_audit
import test_study_audit


class StudyReportingTests(unittest.TestCase):
    def test_accuracy_changes_use_percentage_points(self):
        self.assertEqual(point_change(.01, [-.02, .04]), "1.0 pp [-2.0, 4.0]")

    def test_incomplete_ledger_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "incomplete study"):
            detailed_markdown({"matrix_complete": False, "missing": ["absent"]})

    def test_report_contains_per_seed_and_all_controls(self):
        fixture = test_study_audit.StudyAuditTests()
        with TemporaryDirectory() as directory:
            summary = fixture.write_study(Path(directory))
        capability, _ = fixture.capability_fixture()
        capability.update(comparison={"model": "tiny"}, seed=0)
        mean, bounds = rate_ci([1, 1])
        ledger = {"matrix_complete": True, "missing": [],
                  "study": [{"model": "tiny", "task": "intermediate", "eligible": True,
                             "dev": {"n": 2, "clean_accuracy": mean, "clean_ci": bounds,
                                     "counterfactual_accuracy": mean, "counterfactual_ci": bounds},
                             "result": summary}], "capability": [capability], "heads": []}
        result = detailed_markdown(ledger)
        self.assertIn("Per-Seed Interventions", result)
        self.assertIn("diss_track8_s0.json", result)
        self.assertIn("Window 1-5", result)
        for condition in ("active_prefix", "random_prefix", "zero_prefix", "global"):
            self.assertIn(condition, result)

    def test_stale_audit_checksum_is_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.json"
            path.write_text(json.dumps({"original": True}))
            audit = {"passed": True, "snapshot_matches_sources": True, "snapshot_sha256": file_sha256(path)}
            require_audit(path, audit)
            path.write_text(json.dumps({"original": False}))
            with self.assertRaisesRegex(ValueError, "does not match"):
                require_audit(path, audit)


if __name__ == "__main__":
    unittest.main()