"""
visualization.py
================
All plotting utilities for the Stock Market Regime Detector.

Four plots are produced:
    1. SPY closing price time series.
    2. Closing price coloured by detected regime.
    3. Histogram of log-returns per regime.
    4. State-transition matrix heatmap.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ── Palette & style ───────────────────────────────────────────────────────────
REGIME_PALETTE: list[str] = ["#2196F3", "#4CAF50", "#F44336"]   # blue, green, red
FIGURE_DPI: int = 150
TITLE_FONTSIZE: int = 14
LABEL_FONTSIZE: int = 11


def _save(fig: plt.Figure, path: Path, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    logger.info("Figure saved → %s", path)
    plt.close(fig)


# ── Plot 1 ────────────────────────────────────────────────────────────────────

def plot_closing_price(
    close: pd.Series,
    ticker: str,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot the full closing-price time series.

    Parameters
    ----------
    close:
        Adjusted closing price series with a ``DatetimeIndex``.
    ticker:
        Ticker symbol used in the title.
    save_path:
        Optional file path; if given the figure is saved and closed.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(close.index, close.values, color="#1565C0", linewidth=0.9, label=ticker)
    ax.set_title(f"{ticker} Adjusted Closing Price", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.set_xlabel("Date", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Price (USD)", fontsize=LABEL_FONTSIZE)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=10)
    if save_path:
        _save(fig, save_path)
    return fig


# ── Plot 2 ────────────────────────────────────────────────────────────────────

def plot_price_by_regime(
    close: pd.Series,
    regime_df: pd.DataFrame,
    labels: dict[int, str],
    ticker: str,
    save_path: Path | None = None,
) -> plt.Figure:
    """Closing price coloured by detected HMM regime.

    Parameters
    ----------
    close:
        Full closing price series.
    regime_df:
        DataFrame with columns ``state`` and ``regime`` indexed by date.
    labels:
        Mapping from integer state to descriptive string.
    ticker:
        Ticker symbol.
    save_path:
        Optional save path.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(13, 5))

    # Build a colour array aligned to the return dates (one day offset vs close)
    colours = {state: REGIME_PALETTE[i % len(REGIME_PALETTE)] for i, state in enumerate(sorted(labels))}

    # Plot each regime segment separately so the legend works correctly
    for state, label in labels.items():
        mask = regime_df["state"] == state
        dates = regime_df.index[mask]
        # Align close prices — regime_df dates are return dates (skip first close)
        price_slice = close.reindex(dates).dropna()
        ax.scatter(
            price_slice.index,
            price_slice.values,
            s=3,
            color=colours[state],
            label=label,
            alpha=0.85,
            linewidths=0,
        )

    # Light baseline line
    ax.plot(close.index, close.values, color="grey", linewidth=0.4, alpha=0.4, zorder=0)

    ax.set_title(
        f"{ticker} Closing Price Coloured by Detected Market Regime",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
    )
    ax.set_xlabel("Date", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Price (USD)", fontsize=LABEL_FONTSIZE)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
    ax.legend(fontsize=10, markerscale=4)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    if save_path:
        _save(fig, save_path)
    return fig


# ── Plot 3 ────────────────────────────────────────────────────────────────────

def plot_return_histograms(
    regime_df: pd.DataFrame,
    labels: dict[int, str],
    save_path: Path | None = None,
) -> plt.Figure:
    """Overlapping histogram of log-returns separated by regime.

    Parameters
    ----------
    regime_df:
        DataFrame with columns ``log_return``, ``state``, ``regime``.
    labels:
        State → label mapping.
    save_path:
        Optional save path.

    Returns
    -------
    plt.Figure
    """
    colours = {state: REGIME_PALETTE[i % len(REGIME_PALETTE)] for i, state in enumerate(sorted(labels))}
    fig, ax = plt.subplots(figsize=(9, 5))

    for state, label in labels.items():
        subset = regime_df.loc[regime_df["state"] == state, "log_return"]
        ax.hist(
            subset,
            bins=80,
            density=True,
            alpha=0.55,
            color=colours[state],
            label=f"{label} (n={len(subset):,})",
            edgecolor="none",
        )
        # Vertical mean line
        ax.axvline(subset.mean(), color=colours[state], linewidth=1.5, linestyle="--")

    ax.set_title("Distribution of Daily Log-Returns by Market Regime", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.set_xlabel("Daily Log Return", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Density", fontsize=LABEL_FONTSIZE)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    if save_path:
        _save(fig, save_path)
    return fig


# ── Plot 4 ────────────────────────────────────────────────────────────────────

def plot_transition_matrix(
    transmat: np.ndarray,
    labels: dict[int, str],
    save_path: Path | None = None,
) -> plt.Figure:
    """Heatmap of the HMM state-transition probability matrix.

    Parameters
    ----------
    transmat:
        Transition matrix of shape ``(n_states, n_states)``.
    labels:
        State → label mapping.
    save_path:
        Optional save path.

    Returns
    -------
    plt.Figure
    """
    n = transmat.shape[0]
    tick_labels = [labels.get(i, f"State {i}") for i in range(n)]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        transmat,
        annot=True,
        fmt=".4f",
        cmap="Blues",
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        linewidths=0.5,
        linecolor="white",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Transition Probability"},
    )
    ax.set_title("HMM State Transition Matrix", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.set_xlabel("To State", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("From State", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="x", rotation=15)
    ax.tick_params(axis="y", rotation=0)
    if save_path:
        _save(fig, save_path)
    return fig


# ── Convenience wrapper ───────────────────────────────────────────────────────

def generate_all_figures(
    close: pd.Series,
    regime_df: pd.DataFrame,
    labels: dict[int, str],
    transmat: np.ndarray,
    ticker: str,
    figures_dir: Path,
) -> None:
    """Generate and save all four standard figures.

    Parameters
    ----------
    close, regime_df, labels, transmat, ticker:
        See individual plot functions.
    figures_dir:
        Directory where PNG files will be written.
    """
    figures_dir = Path(figures_dir)
    plot_closing_price(close, ticker, save_path=figures_dir / "01_closing_price.png")
    plot_price_by_regime(close, regime_df, labels, ticker, save_path=figures_dir / "02_price_by_regime.png")
    plot_return_histograms(regime_df, labels, save_path=figures_dir / "03_return_histograms.png")
    plot_transition_matrix(transmat, labels, save_path=figures_dir / "04_transition_matrix.png")
    logger.info("All figures saved to %s", figures_dir)
