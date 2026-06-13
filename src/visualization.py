"""
visualization.py
================
All plotting utilities for the Stock Market Regime Detector.
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
REGIME_PALETTE: list[str] = ["#2196F3", "#4CAF50", "#F44336"]  # blue, green, red
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


def plot_closing_price(
        close: pd.Series,
        ticker: str,
        save_path: Path | None = None,
) -> plt.Figure:
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


def plot_price_by_regime(
        close: pd.Series,
        regime_df: pd.DataFrame,
        labels: dict[int, str],
        ticker: str,
        save_path: Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13, 5))
    colours = {state: REGIME_PALETTE[i % len(REGIME_PALETTE)] for i, state in enumerate(labels.keys())}
    dates = regime_df.index

    for state, label in labels.items():
        price_slice = close.reindex(dates).dropna()
        idx = regime_df["state"] == state
        ax.scatter(
            dates[idx], price_slice.values[idx], s=3, color=colours[state], label=label, alpha=0.85, linewidths=0,
        )
    ax.plot(close.index, close.values, color="grey", linewidth=0.4, alpha=0.4, zorder=0)
    ax.set_title(f"{ticker} Closing Price Coloured by Detected Market Regime", fontsize=TITLE_FONTSIZE,
                 fontweight="bold")
    ax.set_xlabel("Date", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Price (USD)", fontsize=LABEL_FONTSIZE)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    leg = ax.legend(fontsize=10, markerscale=3.0)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    if save_path:
        _save(fig, save_path)
    return fig


def plot_return_histograms(
        regime_df: pd.DataFrame,
        labels: dict[int, str],
        save_path: Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    for state, label in labels.items():
        subset = regime_df.loc[regime_df["state"] == state, "log_return"]
        color = REGIME_PALETTE[state % len(REGIME_PALETTE)]
        sns.kdeplot(subset, ax=ax, fill=True, color=color, label=label, alpha=0.3, linewidth=1.5)
    ax.set_title("Density of Daily Log Returns by Regime", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.set_xlabel("Log Return", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Density", fontsize=LABEL_FONTSIZE)
    ax.legend(fontsize=10)
    if save_path:
        _save(fig, save_path)
    return fig


def plot_transition_matrix(
        transmat: np.ndarray,
        save_path: Path | None = None
) -> plt.Figure:
    """Plots the HMM state-transition matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(transmat, annot=True, cmap="Blues", fmt=".4f", ax=ax, cbar=True)
    ax.set_title("Transition Matrix", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.set_xlabel("To State", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("From State", fontsize=LABEL_FONTSIZE)
    if save_path:
        _save(fig, save_path)
    return fig


def plot_regime_timeline(
        close: pd.Series,
        regime_df: pd.DataFrame,
        labels: dict[int, str],
        save_path: Path | None = None
) -> plt.Figure:
    """Creates a chart showing standard price history with colored regimes and event annotations."""
    fig, ax = plt.subplots(figsize=(15, 6))

    # Baseline
    ax.plot(close.index, close.values, color="black", linewidth=0.5, alpha=0.5, zorder=1)

    # Colored Scatter
    colours = {state: REGIME_PALETTE[i % len(REGIME_PALETTE)] for i, state in enumerate(labels.keys())}
    for state, label in labels.items():
        idx = regime_df["state"] == state
        common_idx = close.index.intersection(regime_df.index[idx])
        ax.scatter(common_idx, close.loc[common_idx], color=colours[state], label=label, s=5, zorder=2)

    # Annotate Historical Events
    events = {
        "2020-02-19": "COVID crash",
        "2022-01-03": "2022 bear market",
        "2023-01-01": "2023 AI rally"
    }

    for date_str, text in events.items():
        dt = pd.to_datetime(date_str)
        if dt >= close.index.min() and dt <= close.index.max():
            ax.axvline(x=dt, color="purple", linestyle="--", linewidth=1.5, alpha=0.8)
            ax.text(dt, close.max() * 0.95, f" {text}", rotation=90, va="top", ha="right", fontsize=10, color="purple")

    ax.set_title("Historical Regime Timeline", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.set_xlabel("Date", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Price (USD)", fontsize=LABEL_FONTSIZE)
    ax.legend(fontsize=10, markerscale=3.0)

    if save_path:
        _save(fig, save_path)
    return fig