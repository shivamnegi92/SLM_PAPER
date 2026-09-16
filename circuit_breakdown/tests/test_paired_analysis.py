import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analyze_dissociation import analyze, load_group
from experiment_metrics import prediction_outcome


class PairedAnalysisTests(unittest.TestCase):
    def write_run(self, root, condition, seed=0, comparison=None, identifiers=("a", "b")):
        records = [dict(prediction_outcome(1, 2, 1, 2, 1, 1, 2),
                        sample_id=identifier, input_hash=identifier, seed=seed,
                        delta_p2way=0.5) for identifier in identifiers]
        run = {"completed": True, "protocol_version": "validated_v1", "seed": seed,
               "condition": condition, "comparison": comparison or {"model": "tiny"},
               "split": {"test": len(records)}, "test": {"records": records}}
        path = Path(root) / f"diss_{condition}_s{seed}.json"
        path.write_text(json.dumps(run))
        return path

    def test_matched_runs_have_non_degenerate_zero_difference(self):
        with TemporaryDirectory() as root:
            for condition in ("full", "track8", "comp8"):
                self.write_run(root, condition)
            result = analyze("diss", ["full", "track8", "comp8"], [0], root)
        contrast = result["comparisons"]["full_minus_track8"]["steer"]
        self.assertEqual(contrast["difference"], 0)
        self.assertLess(contrast["ci"][0], 0)
        self.assertEqual(contrast["n_unique"], 2)

    def test_missing_seed_is_not_silently_dropped(self):
        with TemporaryDirectory() as root:
            self.write_run(root, "full")
            with self.assertRaises(FileNotFoundError):
                load_group("diss", "full", [0, 1], root)

    def test_duplicates_across_seeds_are_rejected(self):
        with TemporaryDirectory() as root:
            self.write_run(root, "full", 0)
            self.write_run(root, "full", 1)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_group("diss", "full", [0, 1], root)

    def test_different_configurations_are_rejected(self):
        with TemporaryDirectory() as root:
            self.write_run(root, "full")
            self.write_run(root, "track8", comparison={"model": "different"})
            with self.assertRaisesRegex(ValueError, "incompatible"):
                analyze("diss", ["full", "track8"], [0], root)

    def test_incomplete_and_legacy_runs_are_rejected(self):
        with TemporaryDirectory() as root:
            path = self.write_run(root, "full")
            saved = json.loads(path.read_text())
            saved["completed"] = False
            path.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_group("diss", "full", [0], root)

    def test_missing_token_identity_is_rejected(self):
        with TemporaryDirectory() as root:
            path = self.write_run(root, "full")
            saved = json.loads(path.read_text())
            del saved["test"]["records"][0]["input_hash"]
            path.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, "identity"):
                load_group("diss", "full", [0], root)


if __name__ == "__main__":
    unittest.main()