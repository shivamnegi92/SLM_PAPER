import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from run_validated_study import calibrate, configuration, matched_layers, read_completed


class StudyRunnerTests(unittest.TestCase):
    def test_relative_layers_are_matched_by_output_depth(self):
        self.assertEqual(matched_layers(28), [18, 20, 22, 24])
        self.assertEqual(matched_layers(32), [21, 23, 25, 28])

    def test_resume_requires_complete_identical_protocol(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "run.json"
            comparison = {"task": "example", "steps": 32}
            self.assertIsNone(read_completed(path, comparison))
            path.write_text(json.dumps({"completed": False, "comparison": comparison}))
            with self.assertRaises(ValueError):
                read_completed(path, comparison)
            path.write_text(json.dumps({"completed": True, "comparison": comparison}))
            self.assertTrue(read_completed(path, comparison)["completed"])
            with self.assertRaises(ValueError):
                read_completed(path, {"steps": 16})

    def test_primary_config_uses_full_vocabulary_and_layerwise_budgets(self):
        cfg = configuration([1, 2, 3, 4], 32, 0.05)
        self.assertEqual(cfg["objective"], "cross_entropy")
        self.assertEqual(cfg["norm_budget"], [1, 2, 3, 4])
        self.assertFalse(cfg["two_stage"])

    def test_calibration_resumes_completed_cases(self):
        record = SimpleNamespace(sample_id="example", input_hash="tokens")
        harness = SimpleNamespace(device="cpu")
        params = SimpleNamespace(trace=[{"target_margin": 1., "target_success": 1}])
        with TemporaryDirectory() as root, patch("run_validated_study.optimize_sample", return_value=params) as optimize:
            options = dict(harness=harness, splits={"dev": [record]}, layers=[0], budgets=[1],
                           bases={"full": None, "track8": None, "comp8": None}, controls=([], []),
                           steps=32, checkpoint_dir=root, comparison={"protocol": "test"})
            self.assertIsNone(calibrate(**options, max_new_cases=1))
            self.assertEqual(optimize.call_count, 1)
            result = calibrate(**options)
            self.assertEqual(optimize.call_count, 9)
            self.assertEqual(set(result["selected"]), {"full", "track8", "comp8"})


if __name__ == "__main__":
    unittest.main()