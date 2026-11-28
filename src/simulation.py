"""Monte Carlo simulation of portfolio P&L using correlated geometric Brownian motion.

Works for any portfolio of price-based assets (long-only, long/short, leveraged):
weights are signed fractions of NAV and need not sum to 1. Whatever is not
invested is treated as cash earning zero.
"""
import numpy as np

TRADING_DAYS = 252


def _validate(weights, mu, sigma, corr, days, n_sims):
    """Check the inputs and fail with a clear message instead of a cryptic numpy error."""
    n = len(weights)
    if weights.ndim != 1 or n == 0:
        raise ValueError("weights must be a non-empty 1-D list of numbers")
    if mu.shape != (n,) or sigma.shape != (n,):
        raise ValueError(f"mu and sigma must each have {n} values (one per asset)")
    if (sigma < 0).any():
        raise ValueError("sigma (volatility) cannot be negative")
    if corr.shape != (n, n):
        raise ValueError(f"corr must be {n} x {n}, got {corr.shape}")
    if not np.allclose(corr, corr.T):
        raise ValueError("corr must be symmetric")
    if not np.allclose(np.diag(corr), 1.0):
        raise ValueError("corr must have 1.0 on the diagonal")
    if days < 1 or n_sims < 1:
        raise ValueError("days and n_sims must be at least 1")


def simulate(weights, mu, sigma, corr, days=252, n_sims=10_000, seed=42, batch_size=2_000):
    """Simulate portfolio P&L paths.

    Parameters
    ----------
    weights : signed weight of each asset as a fraction of NAV (shorts negative)
    mu      : annualized expected return of each asset
    sigma   : annualized volatility of each asset
    corr    : correlation matrix of the assets (symmetric, positive definite)
    days    : number of trading days to simulate
    n_sims  : number of simulated futures
    seed    : random seed, so the same inputs always give the same output
    batch_size : simulations processed at a time, which caps memory use.
                 It does not change the results.

    Returns
    -------
    ndarray of shape (n_sims, days + 1): portfolio profit or loss as a fraction of NAV
    on each day, starting at 0 on day 0. Buy-and-hold: weights are set on day 0
    and are not rebalanced.
    """
    weights, mu, sigma, corr = (np.asarray(a, dtype=float) for a in (weights, mu, sigma, corr))
    _validate(weights, mu, sigma, corr, days, n_sims)
    n_assets = len(weights)
    dt = 1 / TRADING_DAYS

    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        raise ValueError(
            "corr is not positive definite. This is common with short crisis windows; "
            "shrink the matrix toward the identity or use Ledoit-Wolf shrinkage."
        )

    drift = (mu - 0.5 * sigma**2) * dt      # steady part of each day's log return
    vol = sigma * np.sqrt(dt)               # one day's worth of volatility
    rng = np.random.default_rng(seed)

    paths = np.zeros((n_sims, days + 1))    # column 0 stays 0: nothing gained or lost on day 0
    for start in range(0, n_sims, batch_size):
        b = min(batch_size, n_sims - start)
        z = rng.standard_normal((b, days, n_assets)) @ L.T      # correlated random shocks
        log_ret = drift + vol * z                               # daily log returns
        growth = np.exp(np.cumsum(log_ret, axis=1))             # each asset's value, starting at 1
        paths[start:start + b, 1:] = (growth - 1) @ weights     # portfolio P&L (signed weights)
    return paths