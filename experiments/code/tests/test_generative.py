"""RED: structured-output parsing for the generative baseline.

This function is what turns "does the model hallucinate malformed output" into a
measurable parse-failure rate (the C2 reliability claim).
"""
from slmpaper.generative import parse_structured_output


def test_parses_clean_json():
    text = '{"intent": "PlayMusic", "slots": {"genre": "jazz"}}'
    result = parse_structured_output(text)
    assert result == {"intent": "PlayMusic", "slots": {"genre": "jazz"}}


def test_parses_json_with_surrounding_chatter():
    text = 'Sure thing!\n{"intent": "PlayMusic", "slots": {}}\nHope that helps.'
    result = parse_structured_output(text)
    assert result == {"intent": "PlayMusic", "slots": {}}


def test_missing_slots_key_defaults_to_empty_dict():
    text = '{"intent": "GetWeather"}'
    result = parse_structured_output(text)
    assert result == {"intent": "GetWeather", "slots": {}}


def test_malformed_json_is_parse_failure():
    text = '{"intent": "PlayMusic", "slots": {'  # truncated
    assert parse_structured_output(text) is None


def test_missing_intent_key_is_parse_failure():
    text = '{"slots": {"genre": "jazz"}}'
    assert parse_structured_output(text) is None


def test_wrong_type_for_intent_is_parse_failure():
    text = '{"intent": 42, "slots": {}}'
    assert parse_structured_output(text) is None


def test_wrong_type_for_slots_is_parse_failure():
    text = '{"intent": "PlayMusic", "slots": ["genre", "jazz"]}'
    assert parse_structured_output(text) is None


def test_empty_string_is_parse_failure():
    assert parse_structured_output("") is None


def test_no_json_object_at_all_is_parse_failure():
    assert parse_structured_output("I think the intent is PlayMusic") is None


def test_parse_failure_rate_helper():
    from slmpaper.generative import parse_failure_rate
    outputs = [
        '{"intent": "a", "slots": {}}',
        'garbage',
        '{"intent": "b", "slots": {}}',
        '{"slots": {}}',
    ]
    assert parse_failure_rate(outputs) == 0.5
