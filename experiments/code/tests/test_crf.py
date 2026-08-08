"""RED: linear-chain CRF layer (batched, masked)."""
import torch

from slmpaper.crf import LinearChainCRF


def test_nll_is_positive_scalar():
    torch.manual_seed(0)
    crf = LinearChainCRF(num_tags=4)
    emissions = torch.randn(2, 5, 4)
    tags = torch.randint(0, 4, (2, 5))
    mask = torch.ones(2, 5, dtype=torch.bool)
    nll = crf.neg_log_likelihood(emissions, tags, mask)
    assert nll.ndim == 0
    assert nll.item() > 0


def test_decode_returns_valid_tag_sequences_respecting_mask():
    torch.manual_seed(0)
    crf = LinearChainCRF(num_tags=3)
    emissions = torch.randn(2, 4, 3)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    paths = crf.decode(emissions, mask)
    assert len(paths) == 2
    assert len(paths[0]) == 3  # first example has 3 valid positions
    assert len(paths[1]) == 2  # second has 2
    assert all(0 <= t < 3 for t in paths[0] + paths[1])


def test_decode_prefers_high_emission_tag_when_transitions_small():
    crf = LinearChainCRF(num_tags=3)
    # zero out transitions so decoding is driven purely by emissions
    with torch.no_grad():
        crf.transitions.zero_()
        crf.start_transitions.zero_()
        crf.end_transitions.zero_()
    emissions = torch.full((1, 3, 3), -10.0)
    emissions[0, 0, 1] = 10.0  # pos0 -> tag1
    emissions[0, 1, 2] = 10.0  # pos1 -> tag2
    emissions[0, 2, 0] = 10.0  # pos2 -> tag0
    mask = torch.ones(1, 3, dtype=torch.bool)
    assert crf.decode(emissions, mask)[0] == [1, 2, 0]


def test_nll_decreases_when_training_toward_gold():
    torch.manual_seed(0)
    crf = LinearChainCRF(num_tags=3)
    emissions = torch.randn(1, 4, 3, requires_grad=True)
    tags = torch.tensor([[0, 1, 2, 1]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    opt = torch.optim.SGD([emissions] + list(crf.parameters()), lr=0.5)
    first = crf.neg_log_likelihood(emissions, tags, mask).item()
    for _ in range(20):
        opt.zero_grad()
        loss = crf.neg_log_likelihood(emissions, tags, mask)
        loss.backward()
        opt.step()
    last = crf.neg_log_likelihood(emissions, tags, mask).item()
    assert last < first
