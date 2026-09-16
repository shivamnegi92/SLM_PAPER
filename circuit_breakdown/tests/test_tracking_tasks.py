from collections import Counter
from dataclasses import replace
import unittest

import dataset as ds


class TrackingTaskTests(unittest.TestCase):
    def test_simulator_composes_swaps(self):
        swaps = [["A", "B"], ["B", "C"], ["A", "D"]]
        self.assertEqual(ds.simulate_swaps("A", swaps), ("C", 2))
        self.assertEqual(ds.simulate_swaps("B", swaps), ("D", 2))
        self.assertEqual(ds.simulate_swaps("E", swaps), ("E", 0))
        with self.assertRaises(ValueError):
            ds.simulate_swaps("A", [["B", "B"]])

    def test_swap_inverse_restores_each_initial_state(self):
        swaps = [["A", "B"], ["B", "C"], ["A", "D"]]
        for initial in "ABCD":
            final, _ = ds.simulate_swaps(initial, swaps)
            restored, _ = ds.simulate_swaps(final, swaps[::-1])
            self.assertEqual(initial, restored)

    def test_generated_pairs_are_compositional_and_reproducible(self):
        pairs = ds.generate("container_swap", 200, seed=19, min_len=3, max_len=5)
        self.assertEqual(pairs, ds.generate("container_swap", 200, seed=19, min_len=3, max_len=5))
        ds.self_check(pairs)
        self.assertEqual(set(ds.shortcut_scores(pairs).values()), {0.0})
        self.assertEqual(set(Counter(pair.clean_target for pair in pairs)), set(ds.BOXES))
        self.assertEqual(set(pair.metadata["template"] for pair in pairs), {0, 1})
        self.assertEqual(len({pair.clean_prompt for pair in pairs}), 200)

    def test_incorrect_simulator_target_is_rejected(self):
        pair = ds.generate("container_swap", 1, seed=3)[0]
        with self.assertRaises(AssertionError):
            ds.self_check([replace(pair, clean_target="invalid")])

    def test_legacy_tasks_have_answer_extraction_shortcuts(self):
        for task, lengths in (("intermediate", (3, 4)), ("transfer", (1, 2))):
            pairs = ds.generate(task, 40, min_len=lengths[0], max_len=lengths[1])
            ds.self_check(pairs)
            scores = ds.shortcut_scores(pairs)
            self.assertEqual(scores["query_aware_extraction"], 1)
            self.assertEqual(scores["copy_changed_word"], 1)
            self.assertEqual(scores["last_mentioned_name"], 0)


if __name__ == "__main__":
    unittest.main()