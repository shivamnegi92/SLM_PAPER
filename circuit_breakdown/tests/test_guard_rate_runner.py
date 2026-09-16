from pathlib import Path
import unittest

from audit_study import read_json
from guard_rate_pilot import cells
from run_guard_rate_pilot import summarize, validate_protocol


class GuardRateRunnerTests(unittest.TestCase):
    def protocol(self):
        return read_json(Path(__file__).resolve().parents[1] / "protocols/guard_rate_pilot_v1.json")

    def test_protocol_cannot_switch_to_test_or_change_cases(self):
        protocol = self.protocol()
        validate_protocol(protocol)
        for key, value in (("scope", "test"), ("development_indices", [4, 5, 6, 7]), ("development_seed", 0)):
            with self.subTest(field=key):
                with self.assertRaises(ValueError):
                    validate_protocol({**protocol, key: value})

    def test_summary_rejects_missing_cells(self):
        with self.assertRaisesRegex(ValueError, "four-pair factorial"):
            summarize({}, {"protocol": self.protocol(), "development_inputs": []})

    def test_contrasts_are_paired_and_directions_correct(self):
        protocol = self.protocol()
        comparison = {"protocol": protocol, "development_inputs": [{"sample_id": str(index)} for index in range(4)]}
        examples = {}
        for cell in cells(protocol):
            success = int(cell["guard"] == "observe")
            examples[cell["key"]] = [{"input": {"sample_id": str(index)},
                                      "trajectory": {"snapshots": {str(step): {"metrics": {"target_success": success,
                                                                                          "target_margin": float(success)}}
                                                                      for step in (32, 64, 128)}, "guard_events": []},
                                      "checkpoint_controls": {str(step): {"disagreements": [success] * 3} for step in (32, 64, 128)},
                                      "elapsed_seconds": 1.} for index in range(4)]
        report = summarize(examples, comparison)
        self.assertEqual(report["n_trajectories"], 32)
        self.assertEqual(len(report["contrasts"]), 8)
        for contrast in report["contrasts"]:
            self.assertEqual(contrast["override_difference"], 1. if contrast["comparison"] == "observe_minus_shrink" else 0.)
        examples[next(iter(examples))].reverse()
        with self.assertRaisesRegex(ValueError, "misordered"):
            summarize(examples, comparison)


if __name__ == "__main__":
    unittest.main()