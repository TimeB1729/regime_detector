"""
hmm_model.py
============
Fit a Gaussian HMM to standardized log-return observations, predict hidden
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
import itertools

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
    """Instantiate an untrained :class:`~hmmlearn.hmm.GaussianHMM`."""
    return GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
    )


def fit_hmm(model: GaussianHMM, X: np.ndarray) -> GaussianHMM:
    """Fit the HMM via the Baum-Welch (EM) algorithm."""
    logger.info("Fitting HMM with %d components …", model.n_components)
    model.fit(X)
    logger.info("HMM fitted — log-likelihood: %.4f", model.score(X))
    return model


def predict_states(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Decode the most-likely hidden state sequence via the Viterbi algorithm."""
    states = model.predict(X)
    unique, counts = np.unique(states, return_counts=True)
    for s, c in zip(unique, counts):
        logger.info("  State %d — %d observations (%.1f%%)", s, c, 100 * c / len(states))
    return states


def compute_state_statistics(
        log_returns: pd.Series,
        states: np.ndarray,
) -> pd.DataFrame:
    """Compute per-state summary statistics on the *raw* (unstandardised) returns."""
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
    """Derive descriptive regime labels from empirical statistics."""
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
    """Combine returns, integer states, and string labels into one DataFrame."""
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
    """Pretty-print model parameters with explanations."""
    print("\n" + "=" * 60)
    print("HMM MODEL DIAGNOSTICS")
    print("=" * 60)

    print("\n▸ Initial state probabilities  (model.startprob_)")
    for i, p in enumerate(model.startprob_):
        print(f"    State {i}: {p:.4f}")

    print("\n▸ State transition matrix  (model.transmat_)")
    n = model.n_components
    header = "       " + "  ".join(f"→ S{j}" for j in range(n))
    print(f"  {header}")
    for i in range(n):
        row = "  ".join(f"{model.transmat_[i, j]:.4f}" for j in range(n))
        print(f"  S{i} →  {row}")

    print("\n▸ Emission means  (model.means_)")
    for i, mu in enumerate(model.means_.flatten()):
        print(f"    State {i}: {mu:.6f}")

    print("=" * 60 + "\n")


def save_model(model: GaussianHMM, path: Path) -> None:
    """Persist the fitted model with :mod:`joblib`."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)


def load_model(path: Path) -> GaussianHMM:
    """Load a previously saved HMM from disk."""
    model = joblib.load(path)
    logger.info("Model loaded from %s", path)
    return model


def compute_extended_statistics(regime_df: pd.DataFrame, labels: dict[int, str]) -> pd.DataFrame:
    """Computes risk/return statistics for each hidden state."""
    stats = []
    for state_id, label in labels.items():
        state_data = regime_df[regime_df['state'] == state_id]['log_return']
        if state_data.empty:
            continue

        stats.append({
            'Regime': label,
            'Mean Daily Return': state_data.mean(),
            'Median Daily Return': state_data.median(),
            'Daily Volatility': state_data.std(),
            'Annualized Volatility': state_data.std() * np.sqrt(252),
            'Sharpe Ratio': (state_data.mean() / state_data.std()) * np.sqrt(252),
            'Max Daily Gain': state_data.max(),
            'Max Daily Loss': state_data.min(),
            'Number of Observations': len(state_data),
            'Frequency Percentage': (len(state_data) / len(regime_df)) * 100
        })
    return pd.DataFrame(stats)


def analyze_regime_durations(regime_df: pd.DataFrame, labels: dict[int, str]) -> pd.DataFrame:
    """Computes empirical durations of consecutive regime runs."""
    regime_series = regime_df['state'].map(labels)

    durations = {label: [] for label in labels.values()}
    for key, group in itertools.groupby(regime_series):
        if pd.notna(key):
            durations[key].append(len(list(group)))

    duration_stats = []
    for regime, dur_list in durations.items():
        if not dur_list:
            continue
        duration_stats.append({
            'Regime': regime,
            'Mean Duration': np.mean(dur_list),
            'Median Duration': np.median(dur_list),
            'Maximum Duration': np.max(dur_list),
            'Std Dev of Duration': np.std(dur_list)
        })
    return pd.DataFrame(duration_stats)


def compute_persistence_score(transmat: np.ndarray, duration_df: pd.DataFrame, labels: dict[int, str]) -> pd.DataFrame:
    """Compares theoretical Markov duration against empirical duration."""
    persistence = []
    for i, label in labels.items():
        p_ii = transmat[i, i]
        theoretical_duration = 1 / (1 - p_ii) if p_ii < 1 else np.inf

        # Safely fetch empirical duration
        emp_match = duration_df[duration_df['Regime'] == label]
        empirical_duration = emp_match['Mean Duration'].values[0] if not emp_match.empty else np.nan

        persistence.append({
            'State': label,
            'Expected Duration': theoretical_duration,
            'Empirical Duration': empirical_duration
        })
    return pd.DataFrame(persistence)


def profile_regimes(stats_df: pd.DataFrame, persistence_df: pd.DataFrame, labels: dict[int, str],
                    save_path: Path) -> None:
    """Generates an automatic text profile based on the computed statistics."""
    lines = []
    inv_labels = {v: k for k, v in labels.items()}

    for _, row in stats_df.iterrows():
        regime = row['Regime']
        state_id = inv_labels[regime]
        lines.append(f"State {state_id}:")

        # Mean return parsing
        if row['Mean Daily Return'] > 0:
            lines.append("Positive average return")
        else:
            lines.append("Negative average return")

        # Volatility parsing
        vol = row['Daily Volatility']
        all_vols = stats_df['Daily Volatility']
        if vol <= all_vols.quantile(0.33):
            lines.append("Low volatility")
        elif vol >= all_vols.quantile(0.66):
            lines.append("High volatility")
        else:
            lines.append("Medium volatility")

        # Persistence parsing
        pers_row = persistence_df[persistence_df['State'] == regime]
        if not pers_row.empty:
            emp_dur = pers_row.iloc[0]['Empirical Duration']
            all_durs = persistence_df['Empirical Duration']
            if emp_dur >= all_durs.quantile(0.66):
                lines.append("Long persistence")
            elif emp_dur <= all_durs.quantile(0.33):
                lines.append("Short persistence")
            else:
                lines.append("Medium persistence")

        lines.append("")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        f.write("\n".join(lines))