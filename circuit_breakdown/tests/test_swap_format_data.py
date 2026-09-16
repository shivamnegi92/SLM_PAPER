from collections import Counter
from pathlib import Path
import unittest

from audit_study import read_json
from dataset import BOXES, simulate_swaps
from swap_format_data import build_cases, parse_first_answer, render, sample_semantics


class SwapFormatDataTests(unittest.TestCase):
    def protocol(self):
        return read_json(Path(__file__).resolve().parents[1] / "protocols/swap_format_pilot_v1.json")

    def test_balanced_targets_and_simulator_truth(self):
        samples = sample_semantics(9072601)
        for depth in (0, 1, 3, 4):
            group = [example for example in samples if example["depth"] == depth]
            for side in ("clean", "counterfactual"):
                self.assertEqual(Counter(example[f"{side}_target"] for example in group), Counter(BOXES))
                for example in group:
                    target, moves = simulate_swaps(example[f"{side}_initial"], example["swaps"])
                    self.assertEqual(target, example[f"{side}_target"])
                    if depth >= 3:
                        self.assertGreaterEqual(moves, 2)
                        self.assertNotEqual(target, example[f"{side}_initial"])

    def test_wording_and_answer_cue_are_independent(self):
        example = sample_semantics(9072601)[0]
        for cue in ("in_box", "is"):
            first = render(example, "swap", cue, "clean")
            second = render(example, "exchange", cue, "clean")
            self.assertEqual(first.split(". ")[-1], second.split(". ")[-1])

    def test_answer_parser_is_anchored_and_does_not_accept_words_as_labels(self):
        for response, expected in ((" E.", "E"), (" box E.", "E"), ("Box A", "A"),
                                   ("Box Apple", None), ("The answer is E", None), ("Epsilon", None)):
            with self.subTest(response=response):
                self.assertEqual(parse_first_answer(response), expected)

    def test_complete_design_and_reproducible_generation(self):
        protocol = self.protocol()
        first, second = build_cases(protocol, set()), build_cases(protocol, set())
        self.assertEqual(first, second)
        self.assertEqual(first["n_semantic_pairs"], 32)
        self.assertEqual(first["n_responses"], 256)
        self.assertEqual(len({case["case_id"] for case in first["cases"]}), 256)

    def test_old_prompt_overlap_is_rejected(self):
        protocol = self.protocol()
        excluded = {render(sample_semantics(protocol["generator_seed"])[0], "swap", "in_box", "clean")}
        with self.assertRaisesRegex(ValueError, "overlaps"):
            build_cases(protocol, excluded)


if __name__ == "__main__":
    unittest.main()