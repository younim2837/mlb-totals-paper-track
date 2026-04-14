"""
MLB Walk-Forward Snapshot Entry Backtest

Runs a month-by-month walk-forward backtest for a target season, but places
bets at a selected historical odds snapshot instead of only grading against the
final closing line.

Example:
    python walk_forward_snapshot_backtest.py --year 2025 --entry-rule closest_before_minutes --minutes-before 60 --min-confidence 0.53 --min-edge 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from backtest import load_lines, roi_at_110, safe_log_loss_binary
from market_adjustment import (
    MARKET_EDGE_MODEL_FILE,
    apply_market_context,
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
from walk_forward_betting_backtest import evaluate_accuracy, month_windows


MARKET_SHRINK_BASE_CFG = {
    "enabled": True,
    "use_for_post_model_shrinkage": True,
    "max_shrink_fraction": 0.35,
    "shrink_deadband_runs": 0.25,
    "min_books": 1,
    "fallback_max_abs_edge_runs": 1.25,
    "fallback_probability_shrink": 0.60,
    "fallback_confidence_cap": 0.62,
}


def fit_point_models(train_df: pd.DataFrame, feature_cols: list[str]):
    side_models = {}
    side_meta = {}
    for side in ["home", "away"]:
        model, meta = train_side_model(train_df, feature_cols, side)
        side_models[side] = model
        side_meta[side] = meta
    return side_models, side_meta


def build_prediction_frame(
    base_df: pd.DataFrame,
    preds: np.ndarray,
    side_preds: dict[str, np.ndarray],
    sigmas: np.ndarray,
    high_tail_probs: np.ndarray,
    low_tail_probs: np.ndarray,
    evaluation_month: str,
) -> pd.DataFrame:
    frame = base_df.copy()
    frame["predicted_total"] = preds
    frame["predicted_home_runs"] = side_preds["home"]
    frame["predicted_away_runs"] = side_preds["away"]
    frame["prediction_std"] = sigmas
    frame["high_tail_prob_9p5"] = high_tail_probs
    frame["low_tail_prob_7p5"] = low_tail_probs
    frame["evaluation_month"] = evaluation_month
    return frame


def build_market_frame(prediction_df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "date",
        "away_team",
        "home_team",
        TOTAL_TARGET,
        "predicted_total",
        "prediction_std",
        "high_tail_prob_9p5",
        "low_tail_prob_7p5",
    ]
    frame = prediction_df[keep_cols].copy()
    return frame.rename(columns={TOTAL_TARGET: "total_runs"})


def attach_closing_lines(
    snapshot_rows: pd.DataFrame,
    prediction_df: pd.DataFrame,
    closing_lines: pd.DataFrame,
) -> pd.DataFrame:
    if snapshot_rows.empty:
        return snapshot_rows.copy()

    close_frame = merge_predictions_with_lines(
        prediction_df[["date", "away_team", "home_team", TOTAL_TARGET]].copy().rename(columns={TOTAL_TARGET: "total_runs"}),
        closing_lines,
    )
    if close_frame.empty:
        out = snapshot_rows.copy()
        out["closing_total_line"] = np.nan
        return out

    close_keep = ["date_str", "away_team", "home_team", "close_total_line"]
    optional = ["open_total_line", "line_source", "num_books", "snapshot_ts", "commence_time"]
    close_keep.extend([c for c in optional if c in close_frame.columns])
    close_frame = close_frame[close_keep].copy()
    close_frame = close_frame.rename(
        columns={
            "close_total_line": "closing_total_line",
            "line_source": "closing_line_source",
            "num_books": "closing_num_books",
            "snapshot_ts": "closing_snapshot_ts",
            "commence_time": "closing_commence_time",
        }
    )
    return snapshot_rows.merge(close_frame, on=["date_str", "away_team", "home_team"], how="left")


def score_snapshot_candidates(
    snapshot_rows: pd.DataFrame,
    high_tail_cfg: dict | None,
    low_tail_cfg: dict | None,
    market_shrink_cfg: dict | None,
    market_edge_model,
    market_edge_cfg: dict | None,
) -> pd.DataFrame:
    if snapshot_rows.empty:
        return snapshot_rows.copy()

    df = snapshot_rows.copy()
    df = df.dropna(subset=["close_total_line", "predicted_total", "prediction_std", "actual_total"]).copy()
    if df.empty:
        return df

    snap_ts = pd.to_datetime(df.get("snapshot_ts"), errors="coerce", utc=True)
    commence_ts = pd.to_datetime(df.get("commence_time"), errors="coerce", utc=True)
    pregame_mask = snap_ts.isna() | commence_ts.isna() | (snap_ts <= commence_ts)
    df = df.loc[pregame_mask].copy()
    if df.empty:
        return df

    df["entry_total_line"] = pd.to_numeric(df["close_total_line"], errors="coerce")
    df["hours_to_first_pitch"] = (
        (commence_ts.loc[df.index] - snap_ts.loc[df.index]).dt.total_seconds() / 3600.0
    )

    adjusted_totals = []
    market_methods = []
    market_probs = []
    shrink_fracs = []
    for row in df.itertuples(index=False):
        market_result = apply_market_context(
            predicted_total=row.predicted_total,
            market_line=row.entry_total_line,
            cfg=MARKET_SHRINK_BASE_CFG,
            prediction_std=row.prediction_std,
            num_books=getattr(row, "num_books", None),
            market_features={
                col: getattr(row, col, None)
                for col in [
                    "snapshot_ts",
                    "commence_time",
                    "pinnacle_line",
                    "draftkings_line",
                    "fanduel_line",
                    "betmgm_line",
                    "caesars_line",
                ]
                if hasattr(row, col)
            },
            high_tail_prob=getattr(row, "high_tail_prob_9p5", None),
            high_tail_cfg=high_tail_cfg,
            low_tail_prob=getattr(row, "low_tail_prob_7p5", None),
            low_tail_cfg=low_tail_cfg,
            market_model=market_edge_model,
            market_model_cfg=market_edge_cfg,
            learned_shrink_cfg=market_shrink_cfg,
        )
        adjusted_totals.append(float(market_result["adjusted_total"]))
        market_methods.append(market_result.get("method"))
        market_probs.append(market_result.get("p_over"))
        shrink_fracs.append(float(market_result.get("shrink_fraction", 0.0)))

    df["market_adjusted_total"] = adjusted_totals
    df["market_adjustment_method"] = market_methods
    df["market_shrink_fraction"] = shrink_fracs
    df["p_over"] = np.clip(pd.to_numeric(market_probs, errors="coerce"), 1e-6, 1 - 1e-6)
    df["p_under"] = 1 - df["p_over"]
    df["edge"] = df["market_adjusted_total"] - df["entry_total_line"]
    df["abs_edge"] = df["edge"].abs()
    df["confidence"] = np.maximum(df["p_over"], df["p_under"])
    df["bet"] = np.where(df["edge"] > 0, "OVER", "UNDER")
    df["actual_over_entry"] = df["actual_total"] > df["entry_total_line"]
    df["bet_won_entry"] = (
        ((df["bet"] == "OVER") & df["actual_over_entry"])
        | ((df["bet"] == "UNDER") & ~df["actual_over_entry"])
    )
    return df


def select_entry_bets(
    candidates: pd.DataFrame,
    entry_rule: str,
    minutes_before: int,
    min_confidence: float,
    min_edge: float,
    allowed_methods: set[str] | None = None,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()

    df = candidates.copy()
    df = df.dropna(subset=["snapshot_ts", "commence_time", "entry_total_line"]).copy()
    df = df.sort_values(["date_str", "away_team", "home_team", "snapshot_ts"]).reset_index(drop=True)
    if allowed_methods:
        df = df[df["market_adjustment_method"].isin(allowed_methods)].copy()
        if df.empty:
            return df

    key_cols = ["date_str", "away_team", "home_team"]
    threshold_mask = (df["confidence"] >= float(min_confidence)) & (df["abs_edge"] >= float(min_edge))

    if entry_rule == "first_qualified":
        chosen = df.loc[threshold_mask].groupby(key_cols, as_index=False, sort=False).head(1).copy()
        chosen["selection_reason"] = f"first_qualified_conf_{min_confidence:.2f}_edge_{min_edge:.2f}"
        return chosen.reset_index(drop=True)

    cutoff_hours = float(minutes_before) / 60.0
    time_mask = df["hours_to_first_pitch"] >= cutoff_hours
    windowed = df.loc[time_mask].copy()
    if windowed.empty:
        return windowed

    chosen = windowed.groupby(key_cols, as_index=False, sort=False).tail(1).copy()
    chosen = chosen.loc[
        (chosen["confidence"] >= float(min_confidence)) & (chosen["abs_edge"] >= float(min_edge))
    ].copy()
    chosen["selection_reason"] = f"closest_before_{minutes_before}m_conf_{min_confidence:.2f}_edge_{min_edge:.2f}"
    return chosen.reset_index(drop=True)


def finalize_entry_results(selected_rows: pd.DataFrame) -> pd.DataFrame:
    if selected_rows.empty:
        return selected_rows.copy()

    df = selected_rows.copy()
    df["closing_total_line"] = pd.to_numeric(df.get("closing_total_line"), errors="coerce")
    df["close_move"] = df["closing_total_line"] - df["entry_total_line"]
    df["clv_runs"] = np.where(
        df["bet"] == "OVER",
        df["closing_total_line"] - df["entry_total_line"],
        df["entry_total_line"] - df["closing_total_line"],
    )
    df["beat_close"] = df["clv_runs"] > 0
    df["matched_closing_line"] = df["closing_total_line"].notna()
    df["actual_over_close"] = np.where(
        df["matched_closing_line"],
        df["actual_total"] > df["closing_total_line"],
        np.nan,
    )
    return df


def summarize_entry_betting(results: pd.DataFrame) -> dict:
    if results.empty:
        return {
            "bets": 0,
            "win_rate": np.nan,
            "roi": np.nan,
            "brier": np.nan,
            "log_loss": np.nan,
            "avg_confidence": np.nan,
            "avg_edge": np.nan,
            "avg_hours_to_first_pitch": np.nan,
        }

    wins = int(results["bet_won_entry"].sum())
    total = int(len(results))
    return {
        "bets": total,
        "win_rate": float(wins / total),
        "roi": float(roi_at_110(wins, total)),
        "brier": float(brier_score_loss(results["actual_over_entry"].astype(int), results["p_over"].astype(float))),
        "log_loss": float(safe_log_loss_binary(results["actual_over_entry"].astype(int), results["p_over"].astype(float))),
        "avg_confidence": float(results["confidence"].mean()),
        "avg_edge": float(results["abs_edge"].mean()),
        "avg_hours_to_first_pitch": float(results["hours_to_first_pitch"].mean()),
    }


def summarize_clv(results: pd.DataFrame) -> dict:
    if results.empty:
        return {
            "matched_closing_lines": 0,
            "beat_close_rate": np.nan,
            "mean_clv_runs": np.nan,
            "median_clv_runs": np.nan,
        }

    clv_rows = results.loc[results["matched_closing_line"]].copy()
    if clv_rows.empty:
        return {
            "matched_closing_lines": 0,
            "beat_close_rate": np.nan,
            "mean_clv_runs": np.nan,
            "median_clv_runs": np.nan,
        }

    return {
        "matched_closing_lines": int(len(clv_rows)),
        "beat_close_rate": float(clv_rows["beat_close"].mean()),
        "mean_clv_runs": float(clv_rows["clv_runs"].mean()),
        "median_clv_runs": float(clv_rows["clv_runs"].median()),
    }


def main():
    parser = argparse.ArgumentParser(description="Walk-forward snapshot-entry betting backtest for one MLB season.")
    parser.add_argument("--year", type=int, default=2025, help="Target season year")
    parser.add_argument("--min-train-games", type=int, default=250, help="Minimum prior games required before evaluating a month")
    parser.add_argument(
        "--entry-rule",
        choices=["closest_before_minutes", "first_qualified"],
        default="closest_before_minutes",
        help="How to select the entry snapshot for each game",
    )
    parser.add_argument(
        "--minutes-before",
        type=int,
        default=60,
        help="For closest_before_minutes, use the latest snapshot at least this many minutes before first pitch",
    )
    parser.add_argument("--min-confidence", type=float, default=0.53, help="Minimum max(p_over, p_under) required to place a bet")
    parser.add_argument("--min-edge", type=float, default=0.50, help="Minimum absolute edge in runs required to place a bet")
    parser.add_argument(
        "--allowed-methods",
        type=str,
        default="",
        help="Comma-separated market adjustment methods allowed for betting, e.g. edge_model",
    )
    args = parser.parse_args()

    target_year = args.year
    min_train_games = args.min_train_games

    allowed_methods = {
        method.strip()
        for method in str(args.allowed_methods).split(",")
        if method.strip()
    }

    print(f"Loading model data for walk-forward snapshot-entry backtest {target_year}...")
    df = load_data()
    feature_cols = get_feature_cols(df)
    windows = month_windows(df, target_year)
    close_lines = load_lines()
    snapshots = load_historical_line_snapshots()

    if not windows:
        print(f"No model data found for {target_year}.")
        return
    if snapshots.empty:
        print("No historical snapshots found.")
        return

    print(f"  Evaluation windows: {len(windows)} month(s)")

    if not close_lines.empty:
        close_lines = close_lines[
            (close_lines["date"].dt.year == target_year) | (close_lines["date"] < pd.Timestamp(f"{target_year}-01-01"))
        ].copy()
    snapshots = snapshots[
        (snapshots["date"].dt.year == target_year) | (snapshots["date"] < pd.Timestamp(f"{target_year}-01-01"))
    ].copy()

    month_rows = []
    prediction_frames = []
    entry_frames = []

    run_start = time.time()
    total_windows = len(windows)
    completed_windows = 0

    for month_idx, (month_start, month_end) in enumerate(windows, start=1):
        train_df = df[df["date"] < month_start].copy()
        test_df = df[(df["date"] >= month_start) & (df["date"] < month_end)].copy()
        label = month_start.strftime("%Y-%m")

        if len(train_df) < min_train_games or test_df.empty:
            print(f"Skipping {label}: insufficient train/test rows ({len(train_df)} train).")
            continue

        elapsed_min = (time.time() - run_start) / 60.0
        print(f"\n[{label} {month_idx}/{total_windows}] train={len(train_df)} test={len(test_df)}  elapsed={elapsed_min:.1f}m")

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

        fold_df = build_prediction_frame(
            test_df,
            preds,
            side_preds,
            sigmas,
            high_tail_probs,
            low_tail_probs,
            label,
        )
        accuracy = evaluate_accuracy(fold_df)
        prediction_frames.append(
            fold_df[[
                "evaluation_month",
                "date",
                "away_team",
                "home_team",
                "predicted_away_runs",
                "predicted_home_runs",
                "predicted_total",
                "prediction_std",
                "high_tail_prob_9p5",
                "low_tail_prob_7p5",
                TOTAL_TARGET,
            ]]
        )

        train_sigmas = predict_sigmas(unc_model, unc_cfg, train_df[feature_cols], train_preds)
        train_high_tail_probs = predict_high_tail_probs(high_tail_model, high_tail_cfg, train_df[feature_cols], train_preds)
        train_low_tail_probs = predict_low_tail_probs(low_tail_model, low_tail_cfg, train_df[feature_cols], train_preds)
        train_prediction_df = build_prediction_frame(
            train_df,
            train_preds,
            train_side_preds,
            train_sigmas,
            train_high_tail_probs,
            train_low_tail_probs,
            "train",
        )

        train_market_frame = build_market_frame(train_prediction_df)
        train_lines = close_lines[close_lines["date"] < month_start].copy()
        train_market_matched = merge_predictions_with_lines(train_market_frame, train_lines)
        market_shrink_cfg = fit_market_shrinkage(train_market_matched)

        train_snapshots = snapshots[snapshots["date"] < month_start].copy()
        train_snapshot_matched = merge_predictions_with_snapshots(train_market_frame, train_snapshots)
        market_edge_model, market_edge_cfg = fit_market_edge_model(
            train_snapshot_matched,
            high_tail_cfg=high_tail_cfg,
            low_tail_cfg=low_tail_cfg,
        )

        month_snapshots = snapshots[(snapshots["date"] >= month_start) & (snapshots["date"] < month_end)].copy()
        fold_market_frame = build_market_frame(fold_df)
        matched_snapshots = merge_predictions_with_snapshots(fold_market_frame, month_snapshots)
        matched_snapshots = score_snapshot_candidates(
            matched_snapshots,
            high_tail_cfg=high_tail_cfg,
            low_tail_cfg=low_tail_cfg,
            market_shrink_cfg=market_shrink_cfg,
            market_edge_model=market_edge_model,
            market_edge_cfg=market_edge_cfg,
        )
        matched_snapshots = attach_closing_lines(matched_snapshots, fold_df, close_lines)
        selected = select_entry_bets(
            matched_snapshots,
            entry_rule=args.entry_rule,
            minutes_before=args.minutes_before,
            min_confidence=args.min_confidence,
            min_edge=args.min_edge,
            allowed_methods=allowed_methods,
        )
        selected = finalize_entry_results(selected)
        if not selected.empty:
            selected["evaluation_month"] = label
            entry_frames.append(selected)

        betting = summarize_entry_betting(selected)
        clv = summarize_clv(selected)
        month_rows.append({
            "month": label,
            **accuracy,
            **betting,
            **clv,
            "train_games": int(len(train_df)),
            "snapshot_candidates": int(len(matched_snapshots)),
        })

        if betting["bets"] > 0:
            print(
                f"  MAE {accuracy['mae']:.2f} | bets {betting['bets']} | "
                f"win {betting['win_rate']:.1%} | ROI {betting['roi']:.1f}% | "
                f"CLV {clv['mean_clv_runs']:+.2f}"
            )
        else:
            print(f"  MAE {accuracy['mae']:.2f} | no qualifying snapshot bets")

        completed_windows += 1
        avg_window_sec = (time.time() - run_start) / max(completed_windows, 1)
        remaining_windows = total_windows - month_idx
        eta_min = (avg_window_sec * remaining_windows) / 60.0
        print(f"  Progress: {completed_windows}/{total_windows} months complete | ETA ~{eta_min:.1f}m")

    if not month_rows:
        print("No walk-forward months were evaluated.")
        return

    month_df = pd.DataFrame(month_rows).sort_values("month").reset_index(drop=True)
    prediction_df = pd.concat(prediction_frames, ignore_index=True)
    entry_df = pd.concat(entry_frames, ignore_index=True) if entry_frames else pd.DataFrame()

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
        "entry_rule": args.entry_rule,
        "minutes_before": int(args.minutes_before),
        "min_confidence": float(args.min_confidence),
        "min_edge": float(args.min_edge),
        "allowed_methods": sorted(allowed_methods),
        "games": total_games,
        "weighted_mae": weighted_mae,
        "weighted_rmse": weighted_rmse,
        "weighted_bias": weighted_bias,
        "weighted_mean_sigma": weighted_sigma,
        "weighted_within_1sigma": weighted_1sigma,
        "weighted_within_2sigma": weighted_2sigma,
        "months": month_df.to_dict(orient="records"),
    }

    if not entry_df.empty:
        betting = summarize_entry_betting(entry_df)
        clv = summarize_clv(entry_df)
        summary["entry_betting"] = {
            **betting,
            **clv,
        }

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    method_suffix = ""
    if allowed_methods:
        method_suffix = "_methods_" + "_".join(sorted(allowed_methods))
    suffix = f"{target_year}_{args.entry_rule}_{args.minutes_before}m_conf{args.min_confidence:.2f}_edge{args.min_edge:.2f}{method_suffix}"
    suffix = suffix.replace(".", "p")
    prediction_path = os.path.join(DATA_DIR, f"walk_forward_snapshot_predictions_{suffix}.tsv")
    entry_path = os.path.join(DATA_DIR, f"walk_forward_snapshot_entries_{suffix}.tsv")
    month_path = os.path.join(DATA_DIR, f"walk_forward_snapshot_monthly_{suffix}.tsv")
    summary_path = os.path.join(DATA_DIR, f"walk_forward_snapshot_summary_{suffix}.json")

    prediction_df.to_csv(prediction_path, sep="\t", index=False)
    month_df.to_csv(month_path, sep="\t", index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if not entry_df.empty:
        entry_df.to_csv(entry_path, sep="\t", index=False)

    print("\n=== Walk-Forward Snapshot Entry Summary ===")
    print(f"  Year: {target_year}")
    print(f"  Entry rule: {args.entry_rule}")
    print(f"  Min confidence: {args.min_confidence:.2f}")
    print(f"  Min edge: {args.min_edge:.2f}")
    if allowed_methods:
        print(f"  Allowed methods: {', '.join(sorted(allowed_methods))}")
    if args.entry_rule == "closest_before_minutes":
        print(f"  Minutes before first pitch: {args.minutes_before}")
    print(f"  Games evaluated: {total_games}")
    print(f"  Weighted MAE: {weighted_mae:.2f}")
    print(f"  Weighted RMSE: {weighted_rmse:.2f}")
    print(f"  Weighted bias: {weighted_bias:+.2f}")
    print(f"  Mean sigma: {weighted_sigma:.2f}")
    print(f"  1-sigma coverage: {weighted_1sigma:.1%}")
    print(f"  2-sigma coverage: {weighted_2sigma:.1%}")
    if not entry_df.empty:
        entry_betting = summary["entry_betting"]
        print(f"  Snapshot-entry bets: {entry_betting['bets']}")
        print(f"  Win rate at entry line: {entry_betting['win_rate']:.1%}")
        print(f"  Flat ROI at -110: {entry_betting['roi']:.1f}%")
        print(f"  Entry-line Brier: {entry_betting['brier']:.4f}")
        print(f"  Beat-close rate: {entry_betting['beat_close_rate']:.1%}")
        print(f"  Mean CLV (runs): {entry_betting['mean_clv_runs']:+.3f}")
        print(f"  Median CLV (runs): {entry_betting['median_clv_runs']:+.3f}")
        print(f"  Entry detail saved to {entry_path}")
    print(f"  Monthly detail saved to {month_path}")
    print(f"  Prediction detail saved to {prediction_path}")
    print(f"  Summary JSON saved to {summary_path}")


if __name__ == "__main__":
    main()
