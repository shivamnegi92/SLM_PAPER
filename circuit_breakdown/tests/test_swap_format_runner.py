from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from audit_study import read_json
from run_swap_format_pilot import interpret, measure, summarize, validate_protocol


class SwapFormatRunnerTests(unittest.TestCase):
    def test_declared_protocol_is_valid_and_rejects_extra_generation(self):
        protocol = read_json(Path(__file__).resolve().parents[1] / "protocols/swap_format_pilot_v1.json")
        validate_protocol(protocol)
        protocol["generation"]["max_new_tokens"] = 32
        with self.assertRaises(ValueError):
            validate_protocol(protocol)

    def test_parsed_box_answer_does_not_change_strict_score(self):
        result = interpret(torch.tensor([10., 2., 1.]), "box E.", "E", 1)
        self.assertEqual(result["strict_correct"], 0)
        self.assertEqual(result["parsed_correct"], 1)
        self.assertEqual(result["format_only_strict_miss"], 1)

    def test_wrong_location_is_not_a_formatting_success(self):
        result = interpret(torch.tensor([10., 2., 1.]), "box F.", "E", 1)
        self.assertEqual(result["parsed_correct"], 0)
        self.assertEqual(result["format_only_strict_miss"], 0)
        self.assertEqual(result["unparsed"], 0)

    def test_unknown_answer_stays_unparsed(self):
        result = interpret(torch.tensor([10., 2., 1.]), "The answer might be E", "E", 1)
        self.assertEqual(result["unparsed"], 1)
        self.assertEqual(result["parsed_correct"], 0)

    def test_measure_preserves_raw_response_and_scores(self):
        class TinyModel:
            def generate(self, **kwargs):
                return SimpleNamespace(sequences=torch.tensor([[0, 2, 1]]), scores=(torch.tensor([[2., 10., 1.]]),))
        tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: "E" if tokens == [1] else "other")
        harness = SimpleNamespace(device="cpu", model=TinyModel(), tok=tokenizer)
        result = measure(harness, {"input_ids": [0, 2], "target": "E", "target_token_id": 1},
                         SimpleNamespace(max_new_tokens=8))
        self.assertEqual(result["generated_token_ids"], [1])
        self.assertEqual(result["raw_response"], "E")
        self.assertEqual(result["strict_correct"], 1)
        self.assertTrue(torch.equal(result["next_token_logits"], torch.tensor([2., 10., 1.])))

    def test_summary_rejects_incomplete_design(self):
        with self.assertRaisesRegex(ValueError, "256 unique"):
            summarize([], {})


if __name__ == "__main__":
    unittest.main()