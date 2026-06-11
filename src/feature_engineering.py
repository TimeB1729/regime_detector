"""
feature_engineering.py
=======================
Compute and standardize features used to train the Hidden Markov Model.

Level-1 feature: daily log return.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def compute_log_returns(close: pd.Series) -> pd.Series:
    """Compute daily log returns from an adjusted closing price series.

    The return at time *t* is defined as::

        r_t = log(Close_t / Close_{t-1})

    The first observation is ``NaN`` and is dropped automatically.

    Parameters
    ----------
    close:
        Series of closing prices with a ``DatetimeIndex``.

    Returns
    -------
    pd.Series
        Log-return series (length ``len(close) - 1``), named ``"log_return"``.
    """
    if close.empty:
        raise ValueError("Closing price series is empty.")

    log_returns = np.log(close / close.shift(1)).dropna()
    log_returns.name = "log_return"
    logger.info(
        "Computed %d log-return observations (%.4f mean, %.6f std)",
        len(log_returns),
        log_returns.mean(),
        log_returns.std(),
    )
    return log_returns


def standardize_features(returns: pd.Series) -> tuple[np.ndarray, StandardScaler]:
    """Standardise a return series to zero mean and unit variance.

    Parameters
    ----------
    returns:
        Raw log-return series.

    Returns
    -------
    scaled_array:
        2-D NumPy array of shape ``(n_samples, 1)`` required by *hmmlearn*.
    scaler:
        Fitted :class:`~sklearn.preprocessing.StandardScaler` instance that
        can be used to inverse-transform the scaled values later.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(returns.values.reshape(-1, 1))
    logger.info("Features standardised — mean≈%.4f, std≈%.4f", scaled.mean(), scaled.std())
    return scaled, scaler


def build_feature_matrix(close: pd.Series) -> tuple[pd.Series, np.ndarray, StandardScaler]:
    """Convenience wrapper: compute returns, standardise, and return all artefacts.

    Parameters
    ----------
    close:
        Adjusted closing price series.

    Returns
    -------
    log_returns:
        Raw (unstandardised) log-return series aligned to the model's dates.
    X:
        Standardised feature matrix of shape ``(n, 1)`` ready for *hmmlearn*.
    scaler:
        Fitted :class:`~sklearn.preprocessing.StandardScaler`.
    """
    log_returns = compute_log_returns(close)
    X, scaler = standardize_features(log_returns)
    return log_returns, X, scaler
