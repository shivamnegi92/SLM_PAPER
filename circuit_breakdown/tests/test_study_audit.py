from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analyze_dissociation import analyze
from audit_study import (CONDITIONS, audit_baseline, audit_capability, audit_heads,
                         audit_paired_summary, audit_prediction, audit_snapshot)
from experiment_metrics import prediction_outcome
from run_capability_study import paired_accuracy, window_summary
from study_data import fingerprint


class StudyAuditTests(unittest.TestCase):
    def write_study(self, root):
        for condition in ("full", "track8", "comp8"):
            records = [dict(prediction_outcome(1, 2, 1, 2, 1, 1, 2),
                            sample_id=identifier, input_hash=identifier, seed=0,
                            delta_p2way=.5) for identifier in ("first", "second")]
            run = {"completed": True, "protocol_version": "validated_v1", "seed": 0,
                   "condition": condition, "comparison": {"model": "tiny"},
                   "split": {"test": len(records)}, "test": {"records": records}}
            (root / f"diss_{condition}_s0.json").write_text(json.dumps(run))
        summary = analyze("diss", ["full", "track8", "comp8"], [0], root)
        (root / "paired_summary.json").write_text(json.dumps(summary))
        return summary

    def test_rebuilds_summary_from_saved_records(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_study(root)
            report = audit_paired_summary(root, seeds=[0])
        self.assertEqual(report["n_records"], 6)

    def test_rejects_altered_summary_rate(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            summary = self.write_study(root)
            summary["summaries"]["full"]["steer"] = .25
            (root / "paired_summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "steer: numerical mismatch"):
                audit_paired_summary(root, seeds=[0])

    def test_source_path_spelling_does_not_change_statistics(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            summary = self.write_study(root)
            for values in summary["summaries"].values():
                values["source_files"] = [Path(path).name for path in values["source_files"]]
            (root / "paired_summary.json").write_text(json.dumps(summary))
            self.assertEqual(audit_paired_summary(root, seeds=[0])["n_seeds"], 1)

    def capability_fixture(self):
        items = {"tiny": [("context", ["one", "two"], 0)]}
        baseline = [{"id": fingerprint(["context", ["one", "two"], 0]),
                     "scores": [-1., -2.], "target": 0, "prediction": 0, "correct": 1}]
        losses = [[1., 2.] for repeat in range(5)]
        conditions = {condition: {"benchmarks": {"tiny": paired_accuracy(baseline, deepcopy(baseline))},
                                  "token_nll": deepcopy(losses), "text": window_summary(losses, losses),
                                  "edit_norms": [0. if condition == "zero_prefix" else .2]}
                      for condition in CONDITIONS}
        return {"completed": True, "layers": [0], "cfg": {"norm_budget": [.3]},
                "baseline_benchmarks": {"tiny": baseline}, "baseline_token_nll": losses,
                "conditions": conditions}, items

    def test_audits_capability_without_requiring_preservation(self):
        run, items = self.capability_fixture()
        report = audit_capability(run, items)
        self.assertFalse(report["accuracy_bounds_met"]["tiny"])
        self.assertTrue(report["text_bound_met"])

    def test_rejects_incorrect_accuracy_flag(self):
        run, items = self.capability_fixture()
        run["conditions"]["active_prefix"]["benchmarks"]["tiny"]["items"][0]["correct"] = 0
        with self.assertRaisesRegex(ValueError, "correct: numerical mismatch"):
            audit_capability(run, items)

    def test_rejects_changed_zero_scores_even_if_argmax_is_unchanged(self):
        run, items = self.capability_fixture()
        run["conditions"]["zero_prefix"]["benchmarks"]["tiny"]["items"][0]["scores"][0] = -.5
        with self.assertRaisesRegex(ValueError, "Zero edit changed choice scores"):
            audit_capability(run, items)

    def test_rejects_mismatched_random_norm(self):
        run, items = self.capability_fixture()
        run["conditions"]["random_prefix"]["edit_norms"] = [.1]
        with self.assertRaisesRegex(ValueError, "edit norms do not match"):
            audit_capability(run, items)

    def test_rejects_incomplete_capability(self):
        run, items = self.capability_fixture()
        run["completed"] = False
        with self.assertRaisesRegex(ValueError, "Incomplete capability"):
            audit_capability(run, items)

    def test_prediction_flags_are_recomputed(self):
        row = dict(prediction_outcome(1, 2, 1, 2, 1, 1, 2), sample_id="first")
        audit_prediction(row)
        row["same_sign_damage"] = 1
        with self.assertRaisesRegex(ValueError, "same_sign_damage"):
            audit_prediction(row)

    def test_rejects_truncated_baseline_flags(self):
        baseline = {"n": 50, "clean_flags": [1], "counterfactual_flags": [1]}
        with self.assertRaisesRegex(ValueError, "truncated baseline"):
            audit_baseline(baseline, 50, "test")

    def test_incomplete_head_result_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Incomplete head"):
            audit_heads({"completed": False}, [])

    def test_incomplete_snapshot_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Snapshot is incomplete"):
            audit_snapshot({"matrix_complete": False, "missing": ["absent"]}, {})

    def test_snapshot_content_must_match_sources(self):
        source = {"matrix_complete": True, "missing": [], "value": 1.}
        saved = {**source, "value": .5}
        with self.assertRaisesRegex(ValueError, "snapshot.value"):
            audit_snapshot(saved, source)


if __name__ == "__main__":
    unittest.main()