"""
MLB Walk-Forward Evaluation

Runs a stricter monthly walk-forward evaluation for a target season:
each month is scored using a model trained only on games before that month.

This is a better fidelity benchmark than one fixed train/test split because it
closer matches how the model would be deployed over time.
"""

import argparse
import json
import os
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train_model import (
    apply_total_calibration,
    DATA_DIR,
    MODEL_DIR,
    SIDE_SPECS,
    TOTAL_TARGET,
    fit_total_calibration,
    get_feature_cols,
    load_data,
    predict_team_split,
    predict_sigmas,
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


def evaluate_fold(test_df: pd.DataFrame) -> dict:
    errors = test_df[TOTAL_TARGET] - test_df["predicted_total"]
    mae = float(mean_absolute_error(test_df[TOTAL_TARGET], test_df["predicted_total"]))
    rmse = float(sqrt(mean_squared_error(test_df[TOTAL_TARGET], test_df["predicted_total"])))
    r2 = float(r2_score(test_df[TOTAL_TARGET], test_df["predicted_total"]))
    mean_sigma = float(test_df["prediction_std"].mean())
    abs_err = errors.abs()
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
    }


def main():
    parser = argparse.ArgumentParser(description="Walk-forward evaluation for one MLB season.")
    parser.add_argument("--year", type=int, default=2025, help="Target season year")
    parser.add_argument(
        "--min-train-games",
        type=int,
        default=1000,
        help="Minimum number of prior games required before evaluating a month",
    )
    args = parser.parse_args()

    target_year = args.year
    min_train_games = args.min_train_games

    print(f"Loading model data for walk-forward {target_year}...")
    df = load_data()
    feature_cols = get_feature_cols(df)
    windows = month_windows(df, target_year)

    if not windows:
        print(f"No data found for {target_year}.")
        return

    fold_rows = []
    detail_frames = []

    for month_start, month_end in windows:
        train_df = df[df["date"] < month_start].copy()
        test_df = df[(df["date"] >= month_start) & (df["date"] < month_end)].copy()
        label = month_start.strftime("%Y-%m")

        if len(train_df) < min_train_games or test_df.empty:
            print(f"Skipping {label}: insufficient train/test rows ({len(train_df)} train).")
            continue

        print(f"\n[{label}] train={len(train_df)} test={len(test_df)}")
        side_models, side_meta = fit_point_models(train_df, feature_cols)
        calibration_cfg = fit_total_calibration(train_df, feature_cols, side_meta)

        X_test = test_df[feature_cols]
        side_preds, preds = predict_team_split(side_models, X_test, test_df, side_meta)
        preds = apply_total_calibration(preds, calibration_cfg)

        unc_model, unc_cfg = train_uncertainty_model(
            train_df,
            feature_cols,
            side_meta,
            calibration_cfg=calibration_cfg,
        )
        sigmas = predict_sigmas(unc_model, unc_cfg, X_test, preds)

        fold_df = test_df.copy()
        fold_df["predicted_total"] = preds
        fold_df["predicted_home_runs"] = side_preds["home"]
        fold_df["predicted_away_runs"] = side_preds["away"]
        fold_df["prediction_std"] = sigmas
        fold_df["evaluation_month"] = label

        metrics = evaluate_fold(fold_df)
        metrics["month"] = label
        fold_rows.append(metrics)
        detail_frames.append(
            fold_df[[
                "evaluation_month", "date", "away_team", "home_team",
                "predicted_away_runs", "predicted_home_runs",
                "predicted_total", "prediction_std", TOTAL_TARGET,
            ]]
        )

        print(
            f"  MAE {metrics['mae']:.2f} | RMSE {metrics['rmse']:.2f} | "
            f"bias {metrics['bias']:+.2f} | 1-sigma {metrics['within_1sigma']:.1%}"
        )

    if not fold_rows:
        print("No walk-forward folds were evaluated.")
        return

    results = pd.DataFrame(fold_rows).sort_values("month").reset_index(drop=True)
    detail = pd.concat(detail_frames, ignore_index=True)

    total_games = int(results["games"].sum())
    weighted_mae = float(np.average(results["mae"], weights=results["games"]))
    weighted_rmse = float(np.average(results["rmse"], weights=results["games"]))
    weighted_bias = float(np.average(results["bias"], weights=results["games"]))
    weighted_sigma = float(np.average(results["mean_sigma"], weights=results["games"]))
    weighted_1sigma = float(np.average(results["within_1sigma"], weights=results["games"]))
    weighted_2sigma = float(np.average(results["within_2sigma"], weights=results["games"]))

    summary = {
        "year": target_year,
        "games": total_games,
        "weighted_mae": weighted_mae,
        "weighted_rmse": weighted_rmse,
        "weighted_bias": weighted_bias,
        "weighted_mean_sigma": weighted_sigma,
        "weighted_within_1sigma": weighted_1sigma,
        "weighted_within_2sigma": weighted_2sigma,
        "months": results.to_dict(orient="records"),
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    detail_path = os.path.join(DATA_DIR, f"walk_forward_predictions_{target_year}.tsv")
    summary_path = os.path.join(DATA_DIR, f"walk_forward_summary_{target_year}.json")
    month_path = os.path.join(DATA_DIR, f"walk_forward_monthly_{target_year}.tsv")

    detail.to_csv(detail_path, sep="\t", index=False)
    results.to_csv(month_path, sep="\t", index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Walk-Forward Summary ===")
    print(f"  Games: {total_games}")
    print(f"  Weighted MAE: {weighted_mae:.2f}")
    print(f"  Weighted RMSE: {weighted_rmse:.2f}")
    print(f"  Weighted bias: {weighted_bias:+.2f}")
    print(f"  Mean sigma: {weighted_sigma:.2f}")
    print(f"  1-sigma coverage: {weighted_1sigma:.1%}")
    print(f"  2-sigma coverage: {weighted_2sigma:.1%}")
    print(f"  Monthly detail saved to {month_path}")
    print(f"  Prediction detail saved to {detail_path}")
    print(f"  Summary JSON saved to {summary_path}")


if __name__ == "__main__":
    main()
