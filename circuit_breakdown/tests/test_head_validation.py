from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from validate_heads import activation_gradients, matched_random, patched_logits, projected_patch, replace_heads


class HeadValidationTests(unittest.TestCase):
    def setUp(self):
        projection = torch.nn.Identity()
        def model(ids):
            hidden = projection(torch.ones(1, ids.shape[1], 4))
            return SimpleNamespace(logits=hidden)
        self.harness = SimpleNamespace(oprojs=[projection], n_heads=2, head_dim=2, model=model)
        self.ids = torch.tensor([[0, 1]])

    def test_only_selected_head_slice_is_replaced(self):
        logits = patched_logits(self.harness, self.ids, {(0, 1): torch.tensor([3., 4.])})
        self.assertTrue(torch.equal(logits, torch.tensor([1., 1., 3., 4.])))
        self.assertEqual(len(self.harness.oprojs[0]._forward_pre_hooks), 0)

    def test_baseline_clamp_is_a_noop(self):
        baseline = self.harness.model(self.ids).logits[0, -1]
        frozen = patched_logits(self.harness, self.ids, {(0, 0): baseline[:2]})
        self.assertTrue(torch.equal(baseline, frozen))

    def test_error_removes_receiver_hooks(self):
        with self.assertRaises(RuntimeError):
            with replace_heads(self.harness, {(0, 0): torch.zeros(2)}):
                raise RuntimeError("fixture")
        self.assertEqual(len(self.harness.oprojs[0]._forward_pre_hooks), 0)

    def test_random_heads_match_depth_and_exclude_selected(self):
        selected = [(1, 0), (1, 1), (4, 3)]
        random = matched_random(selected, 8, np.random.default_rng(0))
        self.assertEqual(sorted(layer for layer, head in selected), sorted(layer for layer, head in random))
        self.assertFalse(set(selected) & set(random))

    def test_activation_gradients_do_not_allocate_weight_gradients(self):
        class TinyEmbeddingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(4, 3)

            def get_input_embeddings(self):
                return self.embedding

        model = TinyEmbeddingModel()
        with activation_gradients(model):
            outputs = model.embedding(torch.tensor([1, 2]))
            self.assertTrue(outputs.requires_grad)
            outputs.retain_grad()
            (outputs ** 2).sum().backward()
            self.assertIsNotNone(outputs.grad)
            self.assertIsNone(model.embedding.weight.grad)
        self.assertEqual(len(model.embedding._forward_hooks), 0)

    def test_identity_projection_recovers_full_residual_and_zero_does_nothing(self):
        layer = torch.nn.Identity()
        def model(ids):
            hidden = layer(torch.ones(1, ids.shape[1], 4))
            return SimpleNamespace(logits=hidden)
        harness = SimpleNamespace(model=model, layers=[layer])
        clean = [torch.full((1, 2, 4), 3.)]
        full = projected_patch(harness, self.ids, clean, [0], [torch.eye(4)], 1)
        zero = projected_patch(harness, self.ids, clean, [0], [torch.zeros(1, 4)], 1)
        self.assertTrue(torch.equal(full, torch.full((4,), 3.)))
        self.assertTrue(torch.equal(zero, torch.ones(4)))
        self.assertEqual(len(layer._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()