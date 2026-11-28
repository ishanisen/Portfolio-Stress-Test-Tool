"""Risk metrics computed from simulated portfolio P&L paths.

All P&L values are fractions of NAV (0.05 = +5%). Losses are reported as
positive numbers (a Value at Risk of 0.12 means a 12% loss of NAV).
"""
import numpy as np


def value_at_risk(final_pnl, confidence=0.95):
    """Loss that is exceeded in only (1 - confidence) of the simulated futures."""
    return -np.percentile(final_pnl, (1 - confidence) * 100)


def expected_shortfall(final_pnl, confidence=0.95):
    """Average loss in the worst (1 - confidence) of futures (the tail beyond VaR)."""
    cutoff = np.percentile(final_pnl, (1 - confidence) * 100)
    return -final_pnl[final_pnl <= cutoff].mean()


def max_drawdown(paths):
    """Largest peak-to-trough fall of portfolio value along each path (fraction, >= 0).

    Portfolio value is 1 + P&L, starting at 1. Returns one number per simulation.
    """
    nav = 1 + np.asarray(paths)
    peak = np.maximum.accumulate(nav, axis=1)
    return ((peak - nav) / peak).max(axis=1)


def percentile_bands(paths, percentiles=(5, 25, 50, 75, 95)):
    """P&L percentiles on each day, shape (len(percentiles), days + 1). Used for the fan chart."""
    return np.percentile(paths, percentiles, axis=0)


def risk_contribution(weights, sigma, corr):
    """Share of total portfolio variance coming from each position (sums to 1).

    Uses the covariance implied by volatilities and correlations. Positions that
    hedge the rest of the book can have a negative contribution.
    """
    w, s, c = (np.asarray(a, dtype=float) for a in (weights, sigma, corr))
    cov = c * np.outer(s, s)
    marginal = cov @ w
    total_var = w @ marginal
    if total_var <= 0:
        raise ValueError("portfolio variance is zero, so risk contribution is undefined")
    return w * marginal / total_var


def risk_metrics(paths, confidences=(0.95, 0.99)):
    """Summary of the simulated outcomes for one portfolio and one scenario."""
    paths = np.asarray(paths)
    final = paths[:, -1]
    dd = max_drawdown(paths)
    out = {}
    for c in confidences:
        label = int(round(c * 100))
        out[f"var_{label}"] = value_at_risk(final, c)
        out[f"es_{label}"] = expected_shortfall(final, c)
    out.update(
        worst_loss=-final.min(),
        median_pnl=float(np.median(final)),
        prob_loss=float((final < 0).mean()),
        median_max_drawdown=float(np.median(dd)),
        p95_max_drawdown=float(np.percentile(dd, 95)),
        worst_max_drawdown=float(dd.max()),
    )
    return out