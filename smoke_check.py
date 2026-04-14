"""
Low-cost smoke checks for the local codebase.

Usage:
    python smoke_check.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import backtest
import backtest_raw_probability
import predict_today  # noqa: F401 - import smoke check
import prediction_betting
import prediction_reporting
from model_runtime import (
    DATA_DIR,
    estimate_prediction_std,
    load_model_bundle,
    predict_high_tail_prob,
    predict_low_tail_prob,
    predict_point_outputs,
)


def require(condition: bool, message: str):
    if not condition:
        raise RuntimeError(message)


def main():
    print("Smoke: loading shared model bundle...")
    bundle = load_model_bundle()
    require(bundle["model"] is not None, "model failed to load")
    require(bundle["meta"], "model metadata missing")

    print("Smoke: running tiny point-inference sample...")
    sample = pd.read_csv(
        Path(DATA_DIR) / "mlb_model_data.tsv",
        sep="\t",
        parse_dates=["date"],
        nrows=3,
    )
    X, preds, point_meta = predict_point_outputs(bundle["model"], bundle["meta"], sample)
    require(len(preds) == len(sample), "point predictions length mismatch")
    require("driver_summary" in point_meta, "driver summary missing from point output")

    sigma = estimate_prediction_std(
        X.iloc[[0]],
        float(preds[0]),
        bundle["uncertainty_model"],
        bundle["uncertainty_cfg"],
        fallback_std=4.0,
    )
    require(sigma > 0, "prediction sigma should be positive")

    high_tail = predict_high_tail_prob(
        X.iloc[[0]],
        float(preds[0]),
        bundle["high_tail_model"],
        bundle["high_tail_cfg"],
    )
    low_tail = predict_low_tail_prob(
        X.iloc[[0]],
        float(preds[0]),
        bundle["low_tail_model"],
        bundle["low_tail_cfg"],
    )
    if high_tail is not None:
        require(0.0 <= high_tail <= 1.0, "high-tail probability out of range")
    if low_tail is not None:
        require(0.0 <= low_tail <= 1.0, "low-tail probability out of range")

    print("Smoke: checking historical line loaders...")
    lines = backtest.load_lines()
    require(isinstance(lines, pd.DataFrame), "historical lines loader did not return a DataFrame")

    proxy_df, proxy_stats = backtest_raw_probability.load_year(
        2025,
        "half_only",
        bundle["high_tail_cfg"],
        bundle["low_tail_cfg"],
    )
    require(proxy_stats is not None, "proxy backtest stats missing")
    require(not proxy_df.empty, "proxy backtest year returned no rows")
    proxy_result = backtest_raw_probability.evaluate_threshold(proxy_df, 60)
    require(proxy_result is not None, "proxy threshold evaluation failed")

    print("Smoke: checking betting/report export helpers...")
    pred = {
        "game_id": 1,
        "away_team": "Away",
        "home_team": "Home",
        "predicted_total": 8.2,
        "prediction_std": sigma,
        "high_tail_prob_9p5": None,
        "low_tail_prob_7p5": None,
        "_high_tail_cfg": bundle["high_tail_cfg"],
        "_low_tail_cfg": bundle["low_tail_cfg"],
    }
    prediction_betting.add_edge_to_prediction(pred, posted_line=7.5, residual_std=sigma)
    require(pred["bet_signal"] in {"OVER", "UNDER", "NO EDGE"}, "bet signal missing")

    original_dir = prediction_reporting.PREDICTIONS_DIR
    tmpdir = Path(__file__).resolve().parent / "predictions"
    tmpdir.mkdir(exist_ok=True)
    target_date = "2099-01-01-smoke"
    try:
        prediction_reporting.PREDICTIONS_DIR = str(tmpdir)
        board_path, picks_path = prediction_reporting.export_daily_prediction_reports(
            [pred],
            target_date=target_date,
            include_all_games=False,
        )
        require(Path(board_path).exists(), "board export file missing")
        require(Path(picks_path).exists(), "picks export file missing")
    finally:
        prediction_reporting.PREDICTIONS_DIR = original_dir
        for path in [
            tmpdir / f"{target_date}-board.tsv",
            tmpdir / f"{target_date}-picks.tsv",
        ]:
            if path.exists():
                path.unlink()

    print("Smoke check passed.")


if __name__ == "__main__":
    main()
