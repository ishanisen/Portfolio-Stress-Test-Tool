import numpy as np
import pytest

from src.metrics import (expected_shortfall, max_drawdown, percentile_bands,
                         risk_contribution, risk_metrics, value_at_risk)


def paths_from_finals(finals):
    """Build simple 2-column paths: day 0 = 0, day 1 = the given final P&L."""
    finals = np.asarray(finals, dtype=float)
    return np.column_stack([np.zeros(len(finals)), finals])


FINALS = np.linspace(-0.5, 0.5, 1001)      # evenly spread outcomes from -50% to +50%


def test_var_matches_percentile():
    assert value_at_risk(FINALS, 0.95) == pytest.approx(0.45)
    assert value_at_risk(FINALS, 0.99) == pytest.approx(0.49)


def test_expected_shortfall_is_average_of_worst_tail():
    # worst 5% runs from -0.50 to -0.45, so its average is -0.475
    assert expected_shortfall(FINALS, 0.95) == pytest.approx(0.475)


def test_es_is_at_least_var():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.1, 10_000)
    assert expected_shortfall(x, 0.95) >= value_at_risk(x, 0.95)
    assert value_at_risk(x, 0.99) >= value_at_risk(x, 0.95)


def test_max_drawdown_known_path():
    # value goes 1.0 -> 1.1 -> 0.9 -> 1.0: the fall from 1.1 to 0.9 is 0.2 / 1.1
    paths = np.array([[0.0, 0.1, -0.1, 0.0]])
    assert max_drawdown(paths)[0] == pytest.approx(0.2 / 1.1)


def test_max_drawdown_zero_when_always_rising():
    assert max_drawdown(np.array([[0.0, 0.01, 0.02, 0.05]]))[0] == 0


def test_percentile_bands_shape_and_order():
    rng = np.random.default_rng(1)
    paths = np.cumsum(rng.normal(0, 0.01, (500, 21)), axis=1)
    bands = percentile_bands(paths)
    assert bands.shape == (5, 21)
    assert (bands[0] <= bands[2]).all() and (bands[2] <= bands[4]).all()


def test_risk_contribution_sums_to_one_and_equal_split():
    rc = risk_contribution([0.5, 0.5], [0.2, 0.2], np.eye(2))
    assert rc.sum() == pytest.approx(1.0)
    assert np.allclose(rc, [0.5, 0.5])


def test_hedge_has_negative_contribution():
    corr = np.array([[1.0, 0.8], [0.8, 1.0]])
    rc = risk_contribution([1.0, -0.3], [0.2, 0.2], corr)
    assert rc.sum() == pytest.approx(1.0) and rc[1] < 0


def test_risk_metrics_summary_keys_and_values():
    m = risk_metrics(paths_from_finals(FINALS))
    assert m["var_95"] == pytest.approx(0.45)
    assert m["worst_loss"] == pytest.approx(0.5)
    assert m["prob_loss"] == pytest.approx(0.5, abs=0.01)
    assert {"es_95", "es_99", "median_max_drawdown"} <= set(m)