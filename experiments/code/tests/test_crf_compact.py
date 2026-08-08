"""RED: compact subword emissions/labels (with -100) into word-level CRF inputs."""
import torch

from slmpaper.crf_compact import compact_to_word_level


def test_compaction_gathers_valid_positions_and_pads():
    # batch=2, seq=4, tags=3
    emissions = torch.arange(2 * 4 * 3, dtype=torch.float).reshape(2, 4, 3)
    # ex0: valid at positions 0,2 (2 words); ex1: valid at 0,1,3 (3 words)
    slot_labels = torch.tensor([
        [1, -100, 2, -100],
        [0, 1, -100, 2],
    ])
    w_emit, w_tags, w_mask = compact_to_word_level(emissions, slot_labels)
    assert w_emit.shape == (2, 3, 3)   # max 3 words
    assert w_mask.tolist() == [[True, True, False], [True, True, True]]
    assert w_tags.tolist()[0][:2] == [1, 2]
    assert w_tags.tolist()[1] == [0, 1, 2]
    # emissions at ex0 word0 == original position 0; word1 == original position 2
    assert torch.allclose(w_emit[0, 0], emissions[0, 0])
    assert torch.allclose(w_emit[0, 1], emissions[0, 2])
