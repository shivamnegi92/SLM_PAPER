from collections import Counter
from pathlib import Path
import unittest

from audit_study import read_json
from baseline_context_data import build_design
from dataset import BOXES, simulate_swaps
from swap_format_data import build_cases, render, sample_semantics


class BaselineContextDataTests(unittest.TestCase):
    def inputs(self):
        root = Path(__file__).resolve().parents[1]
        protocol = read_json(root / "protocols/baseline_context_pilot_v1.json")
        prior = build_cases(read_json(root / "protocols/swap_format_pilot_v1.json"), set())
        return protocol, prior

    def test_design_is_complete_paired_and_answer_balanced(self):
        protocol, previous = self.inputs()
        design = build_design(protocol, previous, set())
        self.assertEqual(design, build_design(protocol, previous, set()))
        self.assertEqual(len(design["cases"]), 128)
        self.assertEqual(len({row["case_id"] for row in design["cases"]}), 128)
        for depth in (0, 1):
            groups = [[row for row in design["cases"] if row["depth"] == depth and row["context"] == context]
                      for context in protocol["contexts"]]
            self.assertTrue(all([(row["sample_id"], row["side"], row["query"], row["target"]) for row in group] ==
                                [(row["sample_id"], row["side"], row["query"], row["target"]) for row in groups[0]]
                                for group in groups))
            for side in ("clean", "counterfactual"):
                self.assertEqual(Counter(row["target"] for row in groups[0] if row["side"] == side), Counter(BOXES))

    def test_only_prefix_changes_not_answer_format(self):
        protocol, previous = self.inputs()
        design = build_design(protocol, previous, set())
        self.assertEqual(design["prefixes"]["none"], "")
        self.assertEqual(design["prefixes"]["prior"], previous["prefix"])
        for row in design["cases"]:
            self.assertEqual(row["prompt"], design["prefixes"][row["context"]] + row["query"])
            self.assertTrue(row["query"].endswith("in box"))
            target, _ = simulate_swaps(row["semantic"][f"{row['side']}_initial"], row["semantic"]["swaps"])
            self.assertEqual(target, row["target"])

    def test_queries_do_not_repeat_previous_diagnostic_or_demos(self):
        protocol, previous = self.inputs()
        old = {render(row["semantic"], row["wording"], row["answer_cue"], row["side"]) for row in previous["cases"]}
        design = build_design(protocol, previous, old)
        demos = {render(example, "swap", "in_box", "clean") for group in design["demonstrations"].values() for example in group}
        self.assertFalse({row["query"] for row in design["cases"]} & (old | demos))

    def test_exhausted_fresh_prompt_space_fails_without_reuse(self):
        protocol, previous = self.inputs()
        template = sample_semantics(protocol["generator_seed"], (0,))[0]
        excluded = {render({**template, "object": obj}, "swap", "in_box", side)
                    for obj in protocol["query_objects"] for side in protocol["sides"]}
        with self.assertRaisesRegex(ValueError, "without overlap"):
            build_design(protocol, previous, excluded)


if __name__ == "__main__":
    unittest.main()