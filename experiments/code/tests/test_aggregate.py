"""RED: mean/std helper for multi-seed aggregation."""
import math

from slmpaper.aggregate import mean_std


def test_mean_std_basic():
    m, s = mean_std([1.0, 2.0, 3.0])
    assert abs(m - 2.0) < 1e-9
    assert abs(s - math.sqrt(2.0 / 3.0)) < 1e-9  # population std


def test_mean_std_single_value_zero_std():
    m, s = mean_std([0.95])
    assert m == 0.95
    assert s == 0.0


def test_mean_std_empty_raises():
    try:
        mean_std([])
        assert False, "expected ValueError"
    except ValueError:
        pass
