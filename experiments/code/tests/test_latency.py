"""RED: latency measurement harness (P50/P95), deterministic via injectable timer."""
from slmpaper.latency import percentile, measure_latency


def test_percentile_nearest_rank():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(values, 50) == 5
    assert percentile(values, 95) == 10
    assert percentile(values, 100) == 10


def test_percentile_single_value():
    assert percentile([42], 50) == 42


def test_measure_latency_skips_warmup_and_reports_stats():
    # timer is called twice per measured fn call (start, end); warmup calls fn
    # but never touches the timer.
    timer_values = iter([0, 10, 10, 30, 30, 60])

    def fake_timer():
        return next(timer_values)

    calls = []
    result = measure_latency(
        fn=lambda x: calls.append(x),
        inputs=["warmup_item", "a", "b", "c"],
        warmup=1,
        timer=fake_timer,
    )

    assert calls == ["warmup_item", "a", "b", "c"]
    assert result["n"] == 3
    assert result["mean"] == 20.0
    assert result["p50"] == 20
    assert result["p95"] == 30


def test_measure_latency_empty_after_warmup_is_zeroed():
    result = measure_latency(fn=lambda x: None, inputs=["only_warmup"], warmup=1)
    assert result == {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0}
