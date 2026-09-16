from copy import deepcopy
from pathlib import Path
import unittest

from audit_study import read_json
from baseline_context_data import build_design
from run_baseline_context_pilot import summarize, validate_protocol
from swap_format_data import build_cases


class BaselineContextRunnerTests(unittest.TestCase):
    def inputs(self):
        root = Path(__file__).resolve().parents[1]
        protocol = read_json(root / "protocols/baseline_context_pilot_v1.json")
        prior = build_cases(read_json(root / "protocols/swap_format_pilot_v1.json"), set())
        design = build_design(protocol, prior, set())
        return protocol, design

    def test_protocol_rejects_posthoc_context_or_format_change(self):
        protocol, _ = self.inputs()
        validate_protocol(protocol)
        for key, value in (("answer_cue", "is"), ("generator_seed", 7), ("contexts", ["none", "fresh_a"])):
            with self.subTest(field=key):
                with self.assertRaises(ValueError):
                    validate_protocol({**protocol, key: value})

    def test_complete_summary_preserves_paired_differences(self):
        protocol, design = self.inputs()
        responses = []
        for case in design["cases"]:
            correct = int(case["context"] == "none")
            responses.append({"case": case, "elapsed_seconds": 1.,
                              "response": {"strict_correct": correct, "parsed_correct": correct, "unparsed": 0,
                                           "format_only_strict_miss": 0, "parsed_answer": case["target"] if correct else "H"}})
        summary = summarize(responses, {"protocol": protocol})
        self.assertEqual(len(summary["cells"]), 8)
        self.assertEqual(len(summary["contrasts"]), 12)
        self.assertTrue(all(row["rate_difference"] == -1. for row in summary["contrasts"]))
        altered = deepcopy(responses)
        altered[1]["case"]["query"] = "another question"
        with self.assertRaisesRegex(ValueError, "context.pairing"):
            summarize(altered, {"protocol": protocol})

    def test_missing_response_is_not_a_complete_pilot(self):
        with self.assertRaisesRegex(ValueError, "128 distinct"):
            summarize([], {})


if __name__ == "__main__":
    unittest.main()