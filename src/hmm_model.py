"""
hmm_model.py
============
Fit a Gaussian HMM to standardised log-return observations, predict hidden
market regimes, and derive human-readable regime labels from empirical
statistics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)

# ── Model hyper-parameters ────────────────────────────────────────────────────
N_COMPONENTS: int = 3
COVARIANCE_TYPE: str = "full"
N_ITER: int = 1000
RANDOM_STATE: int = 42


def build_hmm(
    n_components: int = N_COMPONENTS,
    covariance_type: str = COVARIANCE_TYPE,
    n_iter: int = N_ITER,
    random_state: int = RANDOM_STATE,
) -> GaussianHMM:
    """Instantiate an untrained :class:`~hmmlearn.hmm.GaussianHMM`.

    Parameters
    ----------
    n_components:
        Number of hidden states (market regimes).
    covariance_type:
        Structure of the emission covariance matrix.
    n_iter:
        Maximum EM iterations.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    GaussianHMM
        Configured but unfitted model.
    """
    return GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
    )


def fit_hmm(model: GaussianHMM, X: np.ndarray) -> GaussianHMM:
    """Fit the HMM via the Baum-Welch (EM) algorithm.

    Parameters
    ----------
    model:
        Unfitted :class:`~hmmlearn.hmm.GaussianHMM` instance.
    X:
        Standardised feature matrix of shape ``(n_samples, n_features)``.

    Returns
    -------
    GaussianHMM
        Fitted model (mutated in place and returned for convenience).
    """
    logger.info("Fitting HMM with %d components …", model.n_components)
    model.fit(X)
    logger.info("HMM fitted — log-likelihood: %.4f", model.score(X))
    return model


def predict_states(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Decode the most-likely hidden state sequence via the Viterbi algorithm.

    Parameters
    ----------
    model:
        Fitted model.
    X:
        Feature matrix used during training.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(n_samples,)`` with a state label per day.
    """
    states = model.predict(X)
    unique, counts = np.unique(states, return_counts=True)
    for s, c in zip(unique, counts):
        logger.info("  State %d — %d observations (%.1f%%)", s, c, 100 * c / len(states))
    return states


def compute_state_statistics(
    log_returns: pd.Series,
    states: np.ndarray,
) -> pd.DataFrame:
    """Compute per-state summary statistics on the *raw* (unstandardised) returns.

    Parameters
    ----------
    log_returns:
        Raw daily log-return series.
    states:
        Integer state array aligned to ``log_returns``.

    Returns
    -------
    pd.DataFrame
        Table indexed by integer state with columns:
        ``mean_return``, ``volatility``, ``count``, ``frequency``.
    """
    df = pd.DataFrame({"log_return": log_returns.values, "state": states})
    stats: list[dict[str, Any]] = []
    for state in sorted(df["state"].unique()):
        mask = df["state"] == state
        subset = df.loc[mask, "log_return"]
        stats.append(
            {
                "state": int(state),
                "mean_return": subset.mean(),
                "volatility": subset.std(),
                "count": int(mask.sum()),
                "frequency": mask.mean(),
            }
        )
    return pd.DataFrame(stats).set_index("state")


def label_states(stats: pd.DataFrame) -> dict[int, str]:
    """Derive descriptive regime labels from empirical statistics.

    The labelling logic is fully data-driven:

    1. The state with the **lowest volatility and positive mean** is labelled
       *"Low Volatility Bull"*.
    2. Among the remaining states the one with the **highest mean** receives
       *"High Volatility Bull"* (positive drift but elevated risk).
    3. The state with the **most negative mean** (or highest volatility if
       means are close) is labelled *"Crash / Bear Regime"*.

    If only two states are found the same logic applies with a reduced label
    set.

    Parameters
    ----------
    stats:
        Output of :func:`compute_state_statistics`.

    Returns
    -------
    dict[int, str]
        Mapping from integer state index to descriptive label string.
    """
    sorted_by_vol = stats.sort_values("volatility")
    labels: dict[int, str] = {}

    remaining = list(sorted_by_vol.index)

    # ── Lowest-vol state ──────────────────────────────────────────────────────
    low_vol_state = remaining[0]
    if stats.loc[low_vol_state, "mean_return"] >= 0:
        labels[low_vol_state] = "Low Volatility Bull"
    else:
        labels[low_vol_state] = "Low Volatility Bear"
    remaining.remove(low_vol_state)

    if not remaining:
        return labels

    # ── Among remaining: most negative mean → crash, most positive → bull ────
    means = stats.loc[remaining, "mean_return"]
    crash_state = int(means.idxmin())
    labels[crash_state] = "Crash / Bear Regime"
    remaining.remove(crash_state)

    for s in remaining:
        if stats.loc[s, "mean_return"] >= 0:
            labels[s] = "High Volatility Bull"
        else:
            labels[s] = "High Volatility Bear"

    logger.info("Derived regime labels: %s", labels)
    return labels


def build_regime_dataframe(
    log_returns: pd.Series,
    states: np.ndarray,
    labels: dict[int, str],
) -> pd.DataFrame:
    """Combine returns, integer states, and string labels into one DataFrame.

    Parameters
    ----------
    log_returns:
        Raw log-return series with ``DatetimeIndex``.
    states:
        Integer state array of the same length.
    labels:
        Mapping from integer state to descriptive string.

    Returns
    -------
    pd.DataFrame
        Columns: ``log_return``, ``state``, ``regime``.
    """
    df = pd.DataFrame(
        {
            "log_return": log_returns.values,
            "state": states,
        },
        index=log_returns.index,
    )
    df["regime"] = df["state"].map(labels)
    return df


def print_model_diagnostics(model: GaussianHMM) -> None:
    """Pretty-print model parameters with explanations.

    Parameters
    ----------
    model:
        Fitted :class:`~hmmlearn.hmm.GaussianHMM`.
    """
    print("\n" + "=" * 60)
    print("HMM MODEL DIAGNOSTICS")
    print("=" * 60)

    print("\n▸ Initial state probabilities  (model.startprob_)")
    print("  The probability that the Markov chain starts in each hidden")
    print("  state on the very first observation.  A high value for one")
    print("  state implies the market is most commonly in that regime at")
    print("  the beginning of the time series.")
    for i, p in enumerate(model.startprob_):
        print(f"    State {i}: {p:.4f}")

    print("\n▸ State transition matrix  (model.transmat_)")
    print("  transmat_[i, j] is the probability of transitioning from")
    print("  state i to state j on the next time step.  Values close to")
    print("  1 on the diagonal indicate persistent (sticky) regimes.")
    n = model.n_components
    header = "       " + "  ".join(f"→ S{j}" for j in range(n))
    print(f"  {header}")
    for i in range(n):
        row = "  ".join(f"{model.transmat_[i, j]:.4f}" for j in range(n))
        print(f"  S{i} →  {row}")

    print("\n▸ Emission means  (model.means_)")
    print("  The expected value of the standardised log-return feature")
    print("  in each hidden state.  Negative values indicate a bear/crash")
    print("  regime; positive values indicate a bull regime.")
    for i, mu in enumerate(model.means_.flatten()):
        print(f"    State {i}: {mu:.6f}")

    print("=" * 60 + "\n")


def save_model(model: GaussianHMM, path: Path) -> None:
    """Persist the fitted model with :mod:`joblib`.

    Parameters
    ----------
    model:
        Fitted model to serialise.
    path:
        Destination file path (e.g. ``outputs/models/hmm_spy.pkl``).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)


def load_model(path: Path) -> GaussianHMM:
    """Load a previously saved HMM from disk.

    Parameters
    ----------
    path:
        Path to the serialised model file.

    Returns
    -------
    GaussianHMM
        Fitted model ready for inference.
    """
    model = joblib.load(path)
    logger.info("Model loaded from %s", path)
    return model
