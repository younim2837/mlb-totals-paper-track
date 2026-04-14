"""
MLB Walk-Forward Betting Backtest

Runs a month-by-month walk-forward backtest for a target season using only data
available before each evaluation month, then compares predictions to historical
closing totals.

Example:
    python walk_forward_betting_backtest.py --year 2021
"""

import argparse
import json
import os
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, r2_score

from backtest import load_lines, match_with_lines, roi_at_110, run_betting_sim, safe_log_loss_binary
from market_adjustment import (
    fit_market_edge_model,
    fit_market_shrinkage,
    load_historical_line_snapshots,
    merge_predictions_with_lines,
    merge_predictions_with_snapshots,
)
from train_model import (
    DATA_DIR,
    MODEL_DIR,
    TOTAL_TARGET,
    apply_total_calibration,
    build_point_oof_artifacts,
    fit_total_calibration,
    get_feature_cols,
    load_data,
    predict_low_tail_probs,
    predict_high_tail_probs,
    predict_sigmas,
    predict_team_split,
    train_high_tail_model,
    train_low_tail_model,
    train_side_model,
    train_uncertainty_model,
)


def month_windows(df: pd.DataFrame, year: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    test = df[df["date"].dt.year == year].copy()
    if test.empty:
        return []

    windows = []
    for period in sorted(test["date"].dt.to_period("M").unique()):
        month_start = period.start_time.normalize()
        month_end = period.end_time.normalize() + pd.Timedelta(days=1)
        windows.append((month_start, month_end))
    return windows


def fit_point_models(train_df: pd.DataFrame, feature_cols: list[str]):
    side_models = {}
    side_meta = {}
    for side in ["home", "away"]:
        model, meta = train_side_model(train_df, feature_cols, side)
        side_models[side] = model
        side_meta[side] = meta
    return side_models, side_meta


def evaluate_accuracy(test_df: pd.DataFrame) -> dict:
    errors = test_df[TOTAL_TARGET] - test_df["predicted_total"]
    mae = float(mean_absolute_error(test_df[TOTAL_TARGET], test_df["predicted_total"]))
    rmse = float(sqrt(mean_squared_error(test_df[TOTAL_TARGET], test_df["predicted_total"])))
    r2 = float(r2_score(test_df[TOTAL_TARGET], test_df["predicted_total"]))
    abs_err = errors.abs()
    mean_sigma = float(test_df["prediction_std"].mean())
    within_1sigma = float((abs_err <= test_df["prediction_std"]).mean())
    within_2sigma = float((abs_err <= 2 * test_df["prediction_std"]).mean())
    bias = float(errors.mean())

    return {
        "games": int(len(test_df)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "bias": bias,
        "mean_predicted": float(test_df["predicted_total"].mean()),
        "mean_actual": float(test_df[TOTAL_TARGET].mean()),
        "mean_sigma": mean_sigma,
        "within_1sigma": within_1sigma,
        "within_2sigma": within_2sigma,
        "mean_high_tail_prob": float(test_df["high_tail_prob_9p5"].mean()) if "high_tail_prob_9p5" in test_df.columns else np.nan,
        "mean_low_tail_prob": float(test_df["low_tail_prob_7p5"].mean()) if "low_tail_prob_7p5" in test_df.columns else np.nan,
    }


def summarize_betting(results: pd.DataFrame) -> dict:
    if results.empty:
        return {
            "matched_games": 0,
            "win_rate": np.nan,
            "roi": np.nan,
            "brier": np.nan,
            "log_loss": np.nan,
        }

    wins = int(results["bet_won"].sum())
    total = int(len(results))
    return {
        "matched_games": total,
        "win_rate": float(wins / total),
        "roi": float(roi_at_110(wins, total)),
        "brier": float(brier_score_loss(results["actual_over"].astype(int), results["p_over"].astype(float))),
        "log_loss": float(safe_log_loss_binary(results["actual_over"].astype(int), results["p_over"].astype(float))),
    }


def main():
    parser = argparse.ArgumentParser(description="Walk-forward betting backtest for one MLB season.")
    parser.add_argument("--year", type=int, default=2021, help="Target season year")
    parser.add_argument(
        "--min-train-games",
        type=int,
        default=250,
        help="Minimum number of prior games required before evaluating a month",
    )
    args = parser.parse_args()

    target_year = args.year
    min_train_games = args.min_train_games

    print(f"Loading model data for walk-forward betting backtest {target_year}...")
    df = load_data()
    feature_cols = get_feature_cols(df)
    windows = month_windows(df, target_year)
    lines = load_lines()
    snapshots = load_historical_line_snapshots()
    if not lines.empty:
        if "season" in lines.columns:
            lines = lines[(lines["season"] == target_year) | (lines["date"] < pd.Timestamp(f"{target_year}-01-01"))].copy()
        else:
            lines = lines[(lines["date"].dt.year == target_year) | (lines["date"] < pd.Timestamp(f"{target_year}-01-01"))].copy()
    if not snapshots.empty:
        snapshots = snapshots[(snapshots["date"].dt.year == target_year) | (snapshots["date"] < pd.Timestamp(f"{target_year}-01-01"))].copy()

    if not windows:
        print(f"No model data found for {target_year}.")
        return
    if lines.empty:
        print(f"No historical lines found for {target_year}.")
        return

    month_rows = []
    detail_frames = []
    line_frames = []

    for month_start, month_end in windows:
        train_df = df[df["date"] < month_start].copy()
        test_df = df[(df["date"] >= month_start) & (df["date"] < month_end)].copy()
        label = month_start.strftime("%Y-%m")

        if len(train_df) < min_train_games or test_df.empty:
            print(f"Skipping {label}: insufficient train/test rows ({len(train_df)} train).")
            continue

        print(f"\n[{label}] train={len(train_df)} test={len(test_df)}")
        side_models, side_meta = fit_point_models(train_df, feature_cols)
        point_oof_artifacts = build_point_oof_artifacts(train_df, feature_cols, side_meta)
        calibration_cfg = fit_total_calibration(
            train_df,
            feature_cols,
            side_meta,
            point_oof_raw=point_oof_artifacts["oof_total_raw"],
        )

        X_test = test_df[feature_cols]
        side_preds, preds = predict_team_split(side_models, X_test, test_df, side_meta)
        preds = apply_total_calibration(preds, calibration_cfg)

        train_side_preds, train_preds = predict_team_split(side_models, train_df[feature_cols], train_df, side_meta)
        train_preds = apply_total_calibration(train_preds, calibration_cfg)

        unc_model, unc_cfg = train_uncertainty_model(
            train_df,
            feature_cols,
            side_meta,
            calibration_cfg=calibration_cfg,
            point_oof_raw=point_oof_artifacts["oof_total_raw"],
        )
        sigmas = predict_sigmas(unc_model, unc_cfg, X_test, preds)

        high_tail_model, high_tail_cfg = train_high_tail_model(
            train_df,
            feature_cols,
            side_meta,
            calibration_cfg=calibration_cfg,
            point_oof_artifacts=point_oof_artifacts,
            full_point_predictions=train_preds,
        )
        high_tail_probs = predict_high_tail_probs(high_tail_model, high_tail_cfg, X_test, preds)

        low_tail_model, low_tail_cfg = train_low_tail_model(
            train_df,
            feature_cols,
            side_meta,
            calibration_cfg=calibration_cfg,
            point_oof_artifacts=point_oof_artifacts,
            full_point_predictions=train_preds,
        )
        low_tail_probs = predict_low_tail_probs(low_tail_model, low_tail_cfg, X_test, preds)

        fold_df = test_df.copy()
        fold_df["predicted_total"] = preds
        fold_df["predicted_home_runs"] = side_preds["home"]
        fold_df["predicted_away_runs"] = side_preds["away"]
        fold_df["prediction_std"] = sigmas
        fold_df["high_tail_prob_9p5"] = high_tail_probs
        fold_df["low_tail_prob_7p5"] = low_tail_probs
        fold_df["evaluation_month"] = label

        accuracy = evaluate_accuracy(fold_df)

        train_sigmas = predict_sigmas(unc_model, unc_cfg, train_df[feature_cols], train_preds)
        train_high_tail_probs = predict_high_tail_probs(high_tail_model, high_tail_cfg, train_df[feature_cols], train_preds)
        train_low_tail_probs = predict_low_tail_probs(low_tail_model, low_tail_cfg, train_df[feature_cols], train_preds)
        train_market_frame = train_df[["date", "away_team", "home_team", TOTAL_TARGET]].copy()
        train_market_frame["predicted_total"] = train_preds
        train_market_frame["prediction_std"] = train_sigmas
        train_market_frame["high_tail_prob_9p5"] = train_high_tail_probs
        train_market_frame["low_tail_prob_7p5"] = train_low_tail_probs
        train_market_frame = train_market_frame.rename(columns={TOTAL_TARGET: "total_runs"})
        train_lines = lines[lines["date"] < month_start].copy()
        train_market_matched = merge_predictions_with_lines(train_market_frame, train_lines)
        market_shrink_cfg = fit_market_shrinkage(train_market_matched)
        train_snapshots = snapshots[snapshots["date"] < month_start].copy() if not snapshots.empty else pd.DataFrame()
        train_snapshot_matched = merge_predictions_with_snapshots(train_market_frame, train_snapshots)
        market_edge_model, market_edge_cfg = fit_market_edge_model(
            train_snapshot_matched,
            high_tail_cfg=high_tail_cfg,
            low_tail_cfg=low_tail_cfg,
        )

        residual_std = float((fold_df[TOTAL_TARGET] - fold_df["predicted_total"]).std())
        month_lines = lines[(lines["date"] >= month_start) & (lines["date"] < month_end)].copy()
        matched = match_with_lines(fold_df, month_lines)
        if not matched.empty:
            results = run_betting_sim(
                matched,
                residual_std=residual_std,
                high_tail_cfg=high_tail_cfg,
                low_tail_cfg=low_tail_cfg,
                market_shrink_cfg=market_shrink_cfg,
                market_edge_model=market_edge_model,
                market_edge_cfg=market_edge_cfg,
            )
            results["evaluation_month"] = label
            line_frames.append(results)
        else:
            results = pd.DataFrame()

        betting = summarize_betting(results)
        month_rows.append({
            "month": label,
            **accuracy,
            **betting,
            "train_games": int(len(train_df)),
        })

        detail_frames.append(
            fold_df[[
                "evaluation_month", "date", "away_team", "home_team",
                "predicted_away_runs", "predicted_home_runs",
                "predicted_total", "prediction_std",
                "high_tail_prob_9p5", "low_tail_prob_7p5",
                TOTAL_TARGET,
            ]]
        )

        if betting["matched_games"] > 0:
            print(
                f"  MAE {accuracy['mae']:.2f} | matched {betting['matched_games']} | "
                f"win {betting['win_rate']:.1%} | ROI {betting['roi']:.1f}%"
            )
        else:
            print(f"  MAE {accuracy['mae']:.2f} | no matched lines")

    if not month_rows:
        print("No walk-forward months were evaluated.")
        return

    month_df = pd.DataFrame(month_rows).sort_values("month").reset_index(drop=True)
    detail_df = pd.concat(detail_frames, ignore_index=True)
    lines_df = pd.concat(line_frames, ignore_index=True) if line_frames else pd.DataFrame()

    total_games = int(month_df["games"].sum())
    weighted_mae = float(np.average(month_df["mae"], weights=month_df["games"]))
    weighted_rmse = float(np.average(month_df["rmse"], weights=month_df["games"]))
    weighted_bias = float(np.average(month_df["bias"], weights=month_df["games"]))
    weighted_sigma = float(np.average(month_df["mean_sigma"], weights=month_df["games"]))
    weighted_1sigma = float(np.average(month_df["within_1sigma"], weights=month_df["games"]))
    weighted_2sigma = float(np.average(month_df["within_2sigma"], weights=month_df["games"]))

    summary = {
        "year": target_year,
        "min_train_games": min_train_games,
        "games": total_games,
        "weighted_mae": weighted_mae,
        "weighted_rmse": weighted_rmse,
        "weighted_bias": weighted_bias,
        "weighted_mean_sigma": weighted_sigma,
        "weighted_within_1sigma": weighted_1sigma,
        "weighted_within_2sigma": weighted_2sigma,
        "months": month_df.to_dict(orient="records"),
    }

    if not lines_df.empty:
        wins = int(lines_df["bet_won"].sum())
        total_bets = int(len(lines_df))
        summary["betting"] = {
            "matched_games": total_bets,
            "win_rate": float(wins / total_bets),
            "roi": float(roi_at_110(wins, total_bets)),
            "brier": float(brier_score_loss(lines_df["actual_over"].astype(int), lines_df["p_over"].astype(float))),
            "log_loss": float(safe_log_loss_binary(lines_df["actual_over"].astype(int), lines_df["p_over"].astype(float))),
        }

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    detail_path = os.path.join(DATA_DIR, f"walk_forward_betting_predictions_{target_year}.tsv")
    lines_path = os.path.join(DATA_DIR, f"walk_forward_betting_lines_{target_year}.tsv")
    month_path = os.path.join(DATA_DIR, f"walk_forward_betting_monthly_{target_year}.tsv")
    summary_path = os.path.join(DATA_DIR, f"walk_forward_betting_summary_{target_year}.json")

    detail_df.to_csv(detail_path, sep="\t", index=False)
    month_df.to_csv(month_path, sep="\t", index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if not lines_df.empty:
        lines_df.to_csv(lines_path, sep="\t", index=False)

    print("\n=== Walk-Forward Betting Summary ===")
    print(f"  Year: {target_year}")
    print(f"  Games evaluated: {total_games}")
    print(f"  Weighted MAE: {weighted_mae:.2f}")
    print(f"  Weighted RMSE: {weighted_rmse:.2f}")
    print(f"  Weighted bias: {weighted_bias:+.2f}")
    print(f"  Mean sigma: {weighted_sigma:.2f}")
    print(f"  1-sigma coverage: {weighted_1sigma:.1%}")
    print(f"  2-sigma coverage: {weighted_2sigma:.1%}")
    if not lines_df.empty:
        betting = summary["betting"]
        print(f"  Matched closing lines: {betting['matched_games']}")
        print(f"  Win rate vs closing lines: {betting['win_rate']:.1%}")
        print(f"  Flat ROI at -110: {betting['roi']:.1f}%")
        print(f"  Brier score at closing line: {betting['brier']:.4f}")
        print(f"  Log loss at closing line: {betting['log_loss']:.4f}")
        print(f"  Betting detail saved to {lines_path}")
    print(f"  Monthly detail saved to {month_path}")
    print(f"  Prediction detail saved to {detail_path}")
    print(f"  Summary JSON saved to {summary_path}")


if __name__ == "__main__":
    main()
