import numpy as np
import pytest

from earshot.active import acquire

PROBS = np.array(
    [
        [0.98, 0.01, 0.01],  # confident
        [0.40, 0.35, 0.25],  # uncertain overall
        [0.50, 0.49, 0.01],  # torn between two classes
        [0.34, 0.33, 0.33],  # maximally uncertain
    ]
)


def test_entropy_picks_flattest_predictions():
    assert acquire("entropy", PROBS, 2, None).tolist() == [3, 1]


def test_margin_picks_smallest_top2_gap():
    assert set(acquire("margin", PROBS, 2, None).tolist()) == {2, 3}
    assert 0 not in acquire("margin", PROBS, 3, None)


def test_random_is_without_replacement():
    picks = acquire("random", np.zeros((100, 1)), 50, np.random.default_rng(0))
    assert len(set(picks.tolist())) == 50 and picks.max() < 100


def test_unknown_strategy():
    with pytest.raises(ValueError):
        acquire("coreset", PROBS, 1, None)
