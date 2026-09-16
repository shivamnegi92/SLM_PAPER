from contextlib import contextmanager, ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from capability import continuation_nll, ending_logprob, perplexity
from capability_deployed import hooks_at, measure, paired_delta_ci, rate_ci, run, summarize_edits


class TinyTokenizer:
    vocab = {"context": 1, "neutral": 2, "target": 3, "other": 4}

    def __call__(self, text, add_special_tokens=True):
        tokens = [self.vocab[word] for word in text.split()]
        return {"input_ids": ([0] if add_special_tokens else []) + tokens}


class TinyCausalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Identity()])

    def forward(self, input_ids):
        hidden = torch.zeros((*input_ids.shape, 5), dtype=torch.float32)
        hidden[:, :, 4] = 1.0
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=hidden.cumsum(dim=1))


class CapabilityScoringTests(unittest.TestCase):
    def setUp(self):
        model = TinyCausalModel()
        self.harness = SimpleNamespace(
            model=model, layers=model.layers, tok=TinyTokenizer(), device="cpu"
        )
        self.vectors = [torch.tensor([[0.0, 0.0, 0.0, 20.0, -20.0]])]
        self.context = "context neutral"
        self.text = "context neutral target target"

    def test_last_sequence_token_edit_is_not_scored(self):
        baseline_score = ending_logprob(self.harness, self.context, "target target")
        baseline_ppl = perplexity(self.harness, self.text)
        input_ids = torch.tensor([self.harness.tok(self.text)["input_ids"]])
        baseline_last = self.harness.model(input_ids).logits[0, -1].clone()

        handles = hooks_at(self.harness, [0], self.vectors, "last")
        try:
            self.assertEqual(
                ending_logprob(self.harness, self.context, "target target"),
                baseline_score,
            )
            self.assertEqual(perplexity(self.harness, self.text), baseline_ppl)
            self.assertFalse(torch.equal(
                self.harness.model(input_ids).logits[0, -1], baseline_last
            ))
        finally:
            for handle in handles:
                handle.remove()

    def test_active_measurement_changes_a_scored_answer(self):
        items = [(self.context, ["other other", "target target"], 1)]
        row = measure(
            self.harness, [0], self.vectors, "last", items, self.text, "active"
        )

        self.assertEqual(row["hellaswag"], 1.0)
        self.assertTrue(np.isfinite(row["ppl"]))
        self.assertEqual(len(self.harness.layers[0]._forward_hooks), 0)

    def test_measurement_records_scored_positions_and_paired_outcomes(self):
        items = [(self.context, ["other other", "target target"], 1)]
        baseline_nll = continuation_nll(self.harness, self.text, prefix_tokens=3)
        baseline = {
            "hellaswag_flags": [0], "token_nll": baseline_nll.tolist(),
            "mean_nll": float(baseline_nll.mean()), "ppl_prefix_tokens": 3,
        }
        row = measure(self.harness, [0], self.vectors, "prefix", items, self.text,
                      "active", baseline=baseline, ppl_prefix_tokens=3)

        self.assertEqual(row["exposure"]["hellaswag_context_positions"], [2, 2])
        self.assertEqual(row["exposure"]["ppl_position"], 2)
        self.assertEqual(row["hellaswag_flags"], [1])
        self.assertEqual(row["delta_hs"], 1.0)
        self.assertLess(row["ppl_ratio"], 1.0)
        self.assertEqual(row["n_ppl_tokens"], 2)
        summary = summarize_edits([row, row], baseline)
        self.assertEqual(summary["n_hs_items"], 1)
        self.assertEqual(summary["n_edits"], 2)
        self.assertNotIn("hellaswag_ci", summary)

    def test_run_saves_versioned_paired_results_and_refuses_overwrite(self):
        with TemporaryDirectory() as output_dir, ExitStack() as mocks:
            args = SimpleNamespace(
                model="tiny", outdir=output_dir, seed=0, n_hs=1, n_edits=1,
                n_eval=1, n=5, fewshot=0, stage_a_steps=1, ppl_chars=200,
                ppl_prefix_tokens=3, ppl_max_tokens=20, layers=[0], lr=0.05,
                norm_budget=4.0, max_control_drop=0.2, device="cpu",
                hellaswag="unused", text="unused", task="intermediate",
            )
            record = SimpleNamespace(
                corrupt_ids=torch.tensor([[0, 1, 2, 3]]), cid=3, dpos=1,
            )
            fixtures = {
                "Harness": self.harness,
                "load_hellaswag": [(self.context, ["other other", "target target"], 1)],
                "load_text": self.text,
                "ds.restrict_to_single_token": None,
                "prepare_pairs": [record],
                "split_records": ([], [], [record]),
                "get_control_baseline": ([], []),
                "optimize_sample": SimpleNamespace(detached=lambda: self.vectors),
            }
            for name, value in fixtures.items():
                mocks.enter_context(patch(f"capability_deployed.{name}", return_value=value))
            run(args)
            artifacts = list(Path(output_dir).glob("*.json"))
            self.assertEqual(len(artifacts), 1)
            self.assertEqual(artifacts[0].name, "capability_active_prefix_tiny_s0.json")
            saved_text = artifacts[0].read_text()
            summary = json.loads(saved_text)
            self.assertEqual(summary["protocol_version"], "active_prefix_v1")
            self.assertEqual(len(summary["hellaswag_item_ids"]), 1)
            self.assertEqual(set(summary["conditions"]), {
                "active_prefix", "global", "random_prefix", "zero_prefix"
            })
            self.assertEqual(summary["conditions"]["zero_prefix"]["delta_hs"], 0)
            self.assertEqual(summary["conditions"]["active_prefix"]["delta_hs"], 1)
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                run(args)
            self.assertEqual(artifacts[0].read_text(), saved_text)
            self.assertEqual(len(self.harness.layers[0]._forward_hooks), 0)

    def test_zero_edit_preserves_scores(self):
        items = [(self.context, ["other other", "target target"], 1)]
        row = measure(self.harness, [0], [torch.zeros_like(self.vectors[0])],
                      "prefix", items, self.text, "zero")
        self.assertEqual(row["hellaswag"], 0.0)
        self.assertEqual(row["ppl"], perplexity(self.harness, self.text))

    def test_continuation_length_does_not_move_context_edit(self):
        positions = []

        @contextmanager
        def intervention(position):
            positions.append(position)
            yield

        ending_logprob(self.harness, self.context, "target", intervention)
        ending_logprob(self.harness, self.context, "target target target", intervention)
        ending_logprob(self.harness, "context", "target target", intervention)
        self.assertEqual(positions, [2, 2, 1])

    def test_misaligned_context_tokenization_is_rejected(self):
        class MisalignedTokenizer(TinyTokenizer):
            def __call__(self, text, add_special_tokens=True):
                encoded = super().__call__(text, add_special_tokens)
                if "target" in text:
                    encoded["input_ids"][1] = 4
                return encoded

        self.harness.tok = MisalignedTokenizer()
        with self.assertRaisesRegex(ValueError, "exact prefix"):
            ending_logprob(self.harness, self.context, "target")

    def test_invalid_prefix_and_token_limits_are_rejected(self):
        for prefix in (0, 5, 10):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                perplexity(self.harness, self.text, prefix_tokens=prefix)
        with self.assertRaises(ValueError):
            perplexity(self.harness, self.text, max_tokens=1)

    def test_invalid_hook_configuration_is_rejected_without_hooks(self):
        configurations = [
            ([0], self.vectors, "unknown", 1),
            ([0], self.vectors, "prefix", None),
            ([0, 0], self.vectors * 2, "prefix", 1),
            ([0], [], "prefix", 1),
            ([1], self.vectors, "prefix", 1),
            ([0], [torch.full((1, 5), float("nan"))], "prefix", 1),
        ]
        for layers, vectors, mode, position in configurations:
            with self.subTest(mode=mode, layers=layers), self.assertRaises(ValueError):
                hooks_at(self.harness, layers, vectors, mode, position=position)
            self.assertEqual(len(self.harness.layers[0]._forward_hooks), 0)

    def test_explicit_unscored_position_is_rejected(self):
        input_ids = torch.tensor([self.harness.tok(self.text)["input_ids"]])
        handles = hooks_at(self.harness, [0], self.vectors, "prefix", position=4)
        try:
            with self.assertRaisesRegex(ValueError, "precede"):
                self.harness.model(input_ids)
        finally:
            for handle in handles:
                handle.remove()

    def test_forward_failure_removes_hooks(self):
        items = [(self.context, ["target"], 0)]
        with patch.object(self.harness.model, "forward", side_effect=RuntimeError("fixture")):
            with self.assertRaisesRegex(RuntimeError, "fixture"):
                measure(self.harness, [0], self.vectors, "prefix", items, self.text, "error")
        self.assertEqual(len(self.harness.layers[0]._forward_hooks), 0)

    def test_partial_hook_registration_failure_is_cleaned_up(self):
        self.harness.layers.append(torch.nn.Identity())
        with patch.object(self.harness.layers[1], "register_forward_hook",
                          side_effect=RuntimeError("registration")):
            with self.assertRaisesRegex(RuntimeError, "registration"):
                hooks_at(self.harness, [0, 1], self.vectors * 2, "prefix", position=1)
        self.assertEqual(len(self.harness.layers[0]._forward_hooks), 0)

    def test_rate_intervals_are_not_degenerate_at_zero_or_one(self):
        mean, interval = rate_ci([0] * 78)
        self.assertEqual(mean, 0)
        self.assertGreater(interval[1], 0.04)
        self.assertLess(interval[1], 0.05)
        mean, interval = rate_ci([1] * 78)
        self.assertEqual(mean, 1)
        self.assertLess(interval[0], 0.96)
        self.assertGreater(interval[0], 0.95)

    def test_invalid_rate_inputs_are_rejected(self):
        for flags in ([], [float("nan")], [0, 2], [[0, 1]]):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                rate_ci(flags)

    def test_paired_intervals_keep_uncertainty_for_identical_outcomes(self):
        delta, interval = paired_delta_ci([0, 1] * 20, [0, 1] * 20)
        self.assertEqual(delta, 0)
        self.assertLess(interval[0], 0)
        self.assertGreater(interval[1], 0)
        delta, interval = paired_delta_ci([1, 0, 1, 0], [1, 1, 0, 1])
        self.assertEqual(delta, 0.25)
        self.assertLess(interval[0], delta)
        self.assertGreater(interval[1], delta)
        with self.assertRaisesRegex(ValueError, "identical lengths"):
            paired_delta_ci([1, 0], [1])


if __name__ == "__main__":
    unittest.main()