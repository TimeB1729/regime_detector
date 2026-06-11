"""
data_loader.py
==============
Download and persist historical OHLCV data from Yahoo Finance via yfinance.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def download_price_data(
    ticker: str,
    start: str,
    end: str,
    data_dir: Path,
) -> pd.DataFrame:
    """Download daily OHLCV data and cache it locally as a CSV.

    Parameters
    ----------
    ticker:
        Yahoo Finance ticker symbol, e.g. ``"SPY"``.
    start:
        Start date string in ``YYYY-MM-DD`` format.
    end:
        End date string in ``YYYY-MM-DD`` format.
    data_dir:
        Directory where the raw CSV will be written.

    Returns
    -------
    pd.DataFrame
        DataFrame with a ``DatetimeIndex`` and at minimum a ``Close`` column.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir / f"{ticker}_{start}_{end}.csv"

    if cache_path.exists():
        logger.info("Loading cached data from %s", cache_path)
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    logger.info("Downloading %s from %s to %s", ticker, start, end)
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        logger.warning(
            "yfinance returned no data for '%s'. "
            "If you are offline or in a restricted network, consider seeding "
            "the data/raw/ directory manually or using generate_synthetic_data.py.",
            ticker,
        )
        raise ValueError(f"No data returned for ticker '{ticker}' between {start} and {end}.")

    # Flatten multi-level columns produced by newer yfinance versions
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.index.name = "Date"
    raw.to_csv(cache_path)
    logger.info("Saved raw data to %s", cache_path)
    return raw


def load_close_series(df: pd.DataFrame) -> pd.Series:
    """Extract and return the adjusted closing price series.

    Parameters
    ----------
    df:
        DataFrame returned by :func:`download_price_data`.

    Returns
    -------
    pd.Series
        Closing price indexed by date, named ``"Close"``.
    """
    if "Close" not in df.columns:
        raise KeyError("DataFrame does not contain a 'Close' column.")
    return df["Close"].rename("Close").dropna()
