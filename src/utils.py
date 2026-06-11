"""
utils.py
========
Shared helper utilities: logging configuration, path resolution, and
pretty-print helpers.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a simple console handler.

    Parameters
    ----------
    level:
        Logging level (e.g. ``logging.DEBUG``).
    """
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def project_root() -> Path:
    """Return the absolute path to the project root directory.

    The root is defined as the parent of the ``src/`` directory that contains
    this module.

    Returns
    -------
    Path
        Absolute project-root path.
    """
    return Path(__file__).resolve().parent.parent


def resolve_path(*parts: str | Path) -> Path:
    """Build an absolute path relative to the project root.

    Parameters
    ----------
    *parts:
        Path components, e.g. ``"outputs", "figures"``.

    Returns
    -------
    Path
        Absolute path ``<project_root>/<parts…>``.
    """
    return project_root().joinpath(*parts)


def print_state_statistics(stats: pd.DataFrame, labels: dict[int, str]) -> None:
    """Pretty-print the per-regime summary table.

    Parameters
    ----------
    stats:
        DataFrame returned by :func:`~src.hmm_model.compute_state_statistics`.
    labels:
        Mapping from integer state to descriptive label.
    """
    display = stats.copy()
    display.index = [labels.get(i, f"State {i}") for i in display.index]
    display.index.name = "Regime"

    display["mean_return"] = display["mean_return"].map("{:.6f}".format)
    display["volatility"] = display["volatility"].map("{:.6f}".format)
    display["count"] = display["count"].map("{:,}".format)
    display["frequency"] = display["frequency"].map("{:.2%}".format)
    display.columns = ["Mean Return", "Volatility", "Count", "Frequency"]

    print("\n" + "=" * 60)
    print("REGIME SUMMARY STATISTICS")
    print("=" * 60)
    print(display.to_string())
    print("=" * 60 + "\n")
