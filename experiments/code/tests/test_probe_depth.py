"""RED: pooling for probe features, and the depth-selection decision rule."""
import torch

from slmpaper.probe import pool_hidden_states, train_linear_probe
from slmpaper.depth_selection import pick_depth


def test_pool_hidden_states_masks_padding():
    # batch=1, seq=4, dim=2; last 2 tokens are padding (mask=0)
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0, 0]])
    pooled = pool_hidden_states(hidden, mask)
    assert pooled.shape == (1, 2)
    assert torch.allclose(pooled[0], torch.tensor([2.0, 2.0]))  # mean of real tokens only


def test_pool_hidden_states_batched():
    hidden = torch.zeros(2, 3, 4)
    hidden[0, :2] = 1.0  # 2 real tokens of value 1
    hidden[1, :3] = 2.0  # 3 real tokens of value 2
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    pooled = pool_hidden_states(hidden, mask)
    assert torch.allclose(pooled[0], torch.full((4,), 1.0))
    assert torch.allclose(pooled[1], torch.full((4,), 2.0))


def test_pick_depth_picks_smallest_within_epsilon_of_max():
    accs = {3: 0.80, 6: 0.91, 9: 0.92, 12: 0.925}
    assert pick_depth(accs, epsilon=0.02) == 6  # within 0.02 of 0.925, smallest depth
    assert pick_depth(accs, epsilon=0.001) == 12  # nothing close enough except the max itself


def test_pick_depth_single_candidate():
    assert pick_depth({6: 0.9}, epsilon=0.01) == 6


def test_train_linear_probe_handles_bfloat16_features():
    # Simulates modern backbone hidden-state dtype on CPU
    torch.manual_seed(0)
    train_x = torch.randn(32, 8, dtype=torch.bfloat16)
    val_x = torch.randn(8, 8, dtype=torch.bfloat16)
    train_y = torch.randint(0, 3, (32,), dtype=torch.long)
    val_y = torch.randint(0, 3, (8,), dtype=torch.long)

    acc = train_linear_probe(
        train_x, train_y, val_x, val_y,
        num_classes=3,
        epochs=3,
        lr=0.05,
    )
    assert 0.0 <= acc <= 1.0
