"""RED: generative baseline (B1) formatting -- prompt, target JSON, label mask."""
import json

from slmpaper.datasets import Example
from slmpaper.gen_format import format_prompt, format_target, build_causal_labels


def test_format_target_is_valid_parseable_json():
    ex = Example(text="find a flight to boston", tokens="find a flight to boston".split(),
                 intent="flight", slots={"toloc.city_name": ["boston"]},
                 bio_tags=["O", "O", "O", "O", "B-toloc.city_name"])
    target = format_target(ex)
    obj = json.loads(target)
    assert obj == {"intent": "flight", "slots": {"toloc.city_name": ["boston"]}}


def test_format_prompt_contains_utterance_and_no_answer():
    ex = Example(text="hi there", tokens=["hi", "there"], intent="greeting",
                 slots={}, bio_tags=["O", "O"])
    prompt = format_prompt(ex)
    assert "hi there" in prompt
    assert "flight" not in prompt  # no leakage of any answer


def test_build_causal_labels_masks_prompt_portion():
    # prompt = 3 tokens, completion = 2 tokens -> labels [-100,-100,-100, id, id]
    input_ids = [10, 11, 12, 20, 21]
    labels = build_causal_labels(input_ids, prompt_len=3)
    assert labels == [-100, -100, -100, 20, 21]
