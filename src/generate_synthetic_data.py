"""
generate_synthetic_data.py
==========================
Generate synthetic SPY-like price data for environments where Yahoo Finance
is not reachable (e.g. CI, sandboxes, offline demos).

This module is NOT part of the production pipeline.  It is only used when
yfinance returns no data.  The synthetic series mimics realistic SPY
characteristics: three-regime structure with calibrated mean and volatility.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Regime parameters calibrated to historical SPY (2010–2026)
_REGIMES = [
    # (weight, daily_mean, daily_vol, label)
    (0.56, 0.00055, 0.0048,  "Low Volatility Bull"),
    (0.33, 0.00090, 0.0115,  "High Volatility Bull"),
    (0.11, -0.0035, 0.0280,  "Crash / Bear Regime"),
]

_TRANSITION = np.array([
    [0.975, 0.020, 0.005],
    [0.030, 0.950, 0.020],
    [0.050, 0.100, 0.850],
])


def generate_spy_like_data(
    start: str = "2010-01-01",
    end: str = "2026-06-01",
    seed: int = 42,
    save_dir: Path | None = None,
) -> pd.DataFrame:
    """Generate a synthetic SPY-like OHLCV DataFrame.

    Parameters
    ----------
    start, end:
        Date range strings.
    seed:
        Random seed for reproducibility.
    save_dir:
        If given, CSV is written to ``<save_dir>/SPY_<start>_<end>.csv``.

    Returns
    -------
    pd.DataFrame
        OHLCV with a ``DatetimeIndex`` named ``Date``.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    # ── Simulate hidden Markov chain ──────────────────────────────────────────
    states = np.empty(n, dtype=int)
    states[0] = 0
    for t in range(1, n):
        s = states[t - 1]
        states[t] = rng.choice(3, p=_TRANSITION[s])

    # ── Simulate log returns ──────────────────────────────────────────────────
    log_rets = np.empty(n)
    for t in range(n):
        _, mu, sig, _ = _REGIMES[states[t]]
        log_rets[t] = rng.normal(mu, sig)

    # ── Build closing price from a starting level of $100 ────────────────────
    close = 100.0 * np.exp(np.cumsum(log_rets))

    # ── Build OHLV columns with realistic intraday range ─────────────────────
    daily_range = np.abs(rng.normal(0, 0.006, n)) * close
    high  = close + daily_range * 0.6
    low   = close - daily_range * 0.4
    open_ = low + rng.uniform(0, 1, n) * (high - low)
    vol   = rng.integers(50_000_000, 150_000_000, n)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )
    df.index.name = "Date"

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"SPY_{start}_{end}.csv"
        df.to_csv(out)
        print(f"Synthetic data saved → {out}")

    return df
