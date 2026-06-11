"""
main.py
=======
End-to-end entry point for the Stock Market Regime Detector.

Usage
-----
    python main.py [--ticker TICKER] [--start YYYY-MM-DD] [--end YYYY-MM-DD]

Defaults: SPY, 2010-01-01, today.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

# Allow running from the project root without installing the package
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import download_price_data, load_close_series
from src.feature_engineering import build_feature_matrix
from src.hmm_model import (
    build_hmm,
    build_regime_dataframe,
    compute_state_statistics,
    fit_hmm,
    label_states,
    predict_states,
    print_model_diagnostics,
    save_model,
)
from src.utils import configure_logging, print_state_statistics, resolve_path
from src.visualization import generate_all_figures

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stock Market Regime Detector using a Gaussian HMM."
    )
    parser.add_argument("--ticker", default="SPY", help="Yahoo Finance ticker (default: SPY)")
    parser.add_argument("--start", default="2010-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=str(date.today()), help="End date YYYY-MM-DD (default: today)")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    ticker: str = args.ticker.upper()
    start: str = args.start
    end: str = args.end

    logger.info("=" * 55)
    logger.info("Stock Market Regime Detector")
    logger.info("  Ticker : %s", ticker)
    logger.info("  Period : %s → %s", start, end)
    logger.info("=" * 55)

    # ── 1. Download data ──────────────────────────────────────────────────────
    data_dir = resolve_path("data", "raw")
    raw_df = download_price_data(ticker, start, end, data_dir)
    close = load_close_series(raw_df)
    logger.info("Loaded %d closing-price observations.", len(close))

    # ── 2. Feature engineering ────────────────────────────────────────────────
    log_returns, X, scaler = build_feature_matrix(close)
    logger.info("Feature matrix shape: %s", X.shape)

    # ── 3. Train HMM ──────────────────────────────────────────────────────────
    model = build_hmm()
    model = fit_hmm(model, X)

    # ── 4. Decode regimes ─────────────────────────────────────────────────────
    states = predict_states(model, X)

    # ── 5. Compute statistics & derive labels ─────────────────────────────────
    stats = compute_state_statistics(log_returns, states)
    labels = label_states(stats)

    # ── 6. Build labelled DataFrame ───────────────────────────────────────────
    regime_df = build_regime_dataframe(log_returns, states, labels)

    # ── 7. Print diagnostics ──────────────────────────────────────────────────
    print_model_diagnostics(model)
    print_state_statistics(stats, labels)

    # ── 8. Save artefacts ─────────────────────────────────────────────────────
    outputs = resolve_path("outputs")
    figures_dir = outputs / "figures"
    models_dir = outputs / "models"

    model_path = models_dir / f"hmm_{ticker.lower()}.pkl"
    save_model(model, model_path)

    regime_csv = outputs / f"regime_labels_{ticker.lower()}.csv"
    regime_df.to_csv(regime_csv)
    logger.info("Regime-labelled DataFrame saved → %s", regime_csv)

    stats_csv = outputs / f"state_statistics_{ticker.lower()}.csv"
    stats.to_csv(stats_csv)
    logger.info("State statistics saved → %s", stats_csv)

    # ── 9. Visualise ──────────────────────────────────────────────────────────
    generate_all_figures(
        close=close,
        regime_df=regime_df,
        labels=labels,
        transmat=model.transmat_,
        ticker=ticker,
        figures_dir=figures_dir,
    )

    logger.info("Done. All outputs written to: %s", outputs)


if __name__ == "__main__":
    main()
