"""
MLB Betting Model Backtest
Two-section report:
  1. Betting simulation against real sportsbook closing lines (in-sample seasons)
  2. Pure accuracy on 2025 out-of-sample test set (no lines required)

Usage:
    python backtest.py
"""

import pandas as pd
import numpy as np
import os
from sklearn.metrics import brier_score_loss, log_loss
from market_adjustment import apply_market_context
from model_runtime import (
    load_model_bundle,
)
from modeling_utils import (
    apply_high_tail_calibration,
    apply_low_tail_calibration,
    apply_total_calibration,
    build_high_tail_features,
    build_low_tail_features,
    clip_probabilities,
    probability_over_line,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

TRAIN_YEARS = list(range(2021, 2025))   # in-sample
TEST_YEAR   = 2025                       # out-of-sample

# Break-even win rate at -110 odds
BREAKEVEN = 110 / 210
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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model_data():
    return pd.read_csv(os.path.join(DATA_DIR, "mlb_model_data.tsv"),
                       sep="\t", parse_dates=["date"])


def load_lines():
    frames = []

    legacy_path = os.path.join(DATA_DIR, "lines_historical.tsv")
    if os.path.exists(legacy_path):
        legacy = pd.read_csv(legacy_path, sep="\t", parse_dates=["date"])
        legacy["line_source"] = "legacy"
        frames.append(legacy)

    oddsapi_path = os.path.join(DATA_DIR, "lines_historical_oddsapi.tsv")
    if os.path.exists(oddsapi_path):
        oddsapi = pd.read_csv(oddsapi_path, sep="\t", parse_dates=["date"])
        oddsapi["line_source"] = "oddsapi"
        frames.append(oddsapi)

    if not frames:
        return pd.DataFrame()

    lines = pd.concat(frames, ignore_index=True, sort=False)
    source_rank = {"oddsapi": 0, "legacy": 1}
    lines["_source_rank"] = lines["line_source"].map(source_rank).fillna(9)
    lines = (
        lines.sort_values(["date", "away_team", "home_team", "_source_rank"])
             .drop_duplicates(subset=["date", "away_team", "home_team"], keep="first")
             .drop(columns=["_source_rank"])
             .reset_index(drop=True)
    )
    return lines


def predict_high_tail_probs(model_data, X, high_tail_model, high_tail_cfg):
    if high_tail_model is None or not high_tail_cfg or not high_tail_cfg.get("enabled"):
        model_data = model_data.copy()
        model_data["high_tail_prob_9p5"] = np.nan
        return model_data

    tail_X = build_high_tail_features(X, model_data["predicted_total"].to_numpy())
    tail_X = tail_X.reindex(columns=high_tail_cfg.get("features", tail_X.columns.tolist()), fill_value=0.0)
    probs = high_tail_model.predict_proba(tail_X)[:, 1]
    model_data = model_data.copy()
    model_data["high_tail_prob_9p5"] = apply_high_tail_calibration(probs, high_tail_cfg)
    return model_data


def predict_low_tail_probs(model_data, X, low_tail_model, low_tail_cfg):
    if low_tail_model is None or not low_tail_cfg or not low_tail_cfg.get("enabled"):
        model_data = model_data.copy()
        model_data["low_tail_prob_7p5"] = np.nan
        return model_data

    tail_X = build_low_tail_features(X, model_data["predicted_total"].to_numpy())
    tail_X = tail_X.reindex(columns=low_tail_cfg.get("features", tail_X.columns.tolist()), fill_value=0.0)
    probs = low_tail_model.predict_proba(tail_X)[:, 1]
    model_data = model_data.copy()
    model_data["low_tail_prob_7p5"] = apply_low_tail_calibration(probs, low_tail_cfg)
    return model_data


def predict_all(model_data, model, meta):
    """Run point model on all rows, return model_data with predictions attached."""
    features = meta["features"]
    X = model_data.reindex(columns=features, fill_value=0.0)
    model_data = model_data.copy()
    model_family = meta.get("model_family", "single_total")

    if model_family == "team_split":
        total_pred = np.zeros(len(X), dtype=float)
        for side, side_meta in meta.get("side_models", {}).items():
            side_pred = model[side].predict(X)
            prediction_mode = side_meta.get("prediction_mode", "direct")
            base_feature = side_meta.get("base_feature")
            if prediction_mode == "dc_residual" and base_feature in model_data.columns:
                side_pred = side_pred + model_data[base_feature].to_numpy()
            model_data[f"predicted_{side}_runs"] = side_pred
            total_pred += side_pred
        calibrated_total = apply_total_calibration(total_pred, meta.get("total_calibration"))
        scale = np.divide(
            calibrated_total,
            total_pred,
            out=np.ones_like(calibrated_total),
            where=np.abs(total_pred) > 1e-9,
        )
        for side in meta.get("side_models", {}):
            model_data[f"predicted_{side}_runs"] = model_data[f"predicted_{side}_runs"] * scale
        model_data["predicted_total_raw"] = total_pred
        model_data["predicted_total"] = calibrated_total
        return model_data, X

    prediction_mode = meta.get("prediction_mode", "direct")
    base_feature = meta.get("base_feature")
    preds = model.predict(X)
    if prediction_mode == "dc_residual" and base_feature in model_data.columns:
        preds = preds + model_data[base_feature].to_numpy()
    model_data["predicted_total_raw"] = preds
    model_data["predicted_total"] = apply_total_calibration(preds, meta.get("total_calibration"))
    return model_data, X


def predict_sigmas(model_data, X, uncertainty_model, uncertainty_cfg, fallback_std):
    """Attach a per-game sigma column using the trained uncertainty model."""
    if uncertainty_model is None or not uncertainty_cfg:
        model_data = model_data.copy()
        model_data["prediction_std"] = float(fallback_std)
        return model_data

    unc_features = uncertainty_cfg.get("features") or []
    unc_X = X.copy()
    unc_X["point_prediction"] = model_data["predicted_total"].to_numpy()
    unc_X = unc_X.reindex(columns=unc_features, fill_value=0.0)
    raw_abs = np.clip(uncertainty_model.predict(unc_X), 0.1, None)
    sigmas = raw_abs * float(uncertainty_cfg.get("mae_to_sigma_scale", 1.2533))
    sigmas = np.clip(
        sigmas,
        float(uncertainty_cfg.get("min_sigma", fallback_std)),
        float(uncertainty_cfg.get("max_sigma", fallback_std)),
    )
    model_data = model_data.copy()
    model_data["prediction_std"] = sigmas
    return model_data


def compute_residual_std(model_data):
    """Std of prediction errors across all data (used for confidence intervals)."""
    return (model_data["total_runs"] - model_data["predicted_total"]).std()


def roi_at_110(wins, total):
    if total == 0:
        return 0.0
    profit = wins * (100 / 110) - (total - wins) * 1.0
    return profit / total * 100


def safe_log_loss_binary(y_true, probs):
    probs = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)
    return float(log_loss(y_true, probs, labels=[0, 1]))


def print_probability_quality(results, prob_col="p_over", actual_col="actual_over", label="market line"):
    probs = results[prob_col].astype(float)
    actuals = results[actual_col].astype(int)
    brier = float(brier_score_loss(actuals, probs))
    ll = safe_log_loss_binary(actuals, probs)
    print(f"\n  Probability quality at {label}:")
    print(f"  Brier score                  : {brier:.4f}  (lower is better)")
    print(f"  Log loss                     : {ll:.4f}  (lower is better)")

    print(f"\n  Confidence bucket calibration ({label}):")
    print(f"  {'P(over) range':<18} {'Games':>7} {'Pred avg':>10} {'Actual':>8} {'Brier':>8}")
    print(f"  {'-'*56}")
    for lo, hi in [(0.00, 0.40), (0.40, 0.45), (0.45, 0.50), (0.50, 0.55),
                   (0.55, 0.60), (0.60, 0.65), (0.65, 1.00)]:
        sub = results[results[prob_col].between(lo, hi, inclusive="left" if hi < 1.0 else "both")]
        if len(sub) < 10:
            continue
        sub_brier = float(brier_score_loss(sub[actual_col].astype(int), sub[prob_col].astype(float)))
        print(
            f"  {lo:.0%}–{hi:.0%}              {len(sub):>7} "
            f"{sub[prob_col].mean():>9.1%} {sub[actual_col].mean():>7.1%} {sub_brier:>7.4f}"
        )


def print_total_bucket_bias(df, label="2025 test set"):
    bins = [0, 7.5, 8.5, 9.5, 10.5, np.inf]
    labels = ["<=7.5", "7.6-8.5", "8.6-9.5", "9.6-10.5", "10.6+"]
    bucketed = df.copy()
    bucketed["pred_bucket"] = pd.cut(
        bucketed["predicted_total"], bins=bins, labels=labels, include_lowest=True
    )
    summary = (
        bucketed.groupby("pred_bucket", observed=False)
        .agg(
            games=("total_runs", "size"),
            pred_mean=("predicted_total", "mean"),
            actual_mean=("total_runs", "mean"),
        )
        .reset_index()
    )
    summary["bias"] = summary["pred_mean"] - summary["actual_mean"]

    print(f"\n  Predicted-total bucket bias ({label}):")
    print(f"  {'Bucket':<10} {'Games':>7} {'Pred':>8} {'Actual':>8} {'Bias':>8}")
    print(f"  {'-'*46}")
    for _, row in summary.iterrows():
        print(
            f"  {str(row['pred_bucket']):<10} {int(row['games']):>7} "
            f"{row['pred_mean']:>8.2f} {row['actual_mean']:>8.2f} {row['bias']:>8.2f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Betting simulation against closing lines
# ─────────────────────────────────────────────────────────────────────────────

def match_with_lines(model_data, lines):
    """Join model predictions to historical closing lines."""
    if lines.empty:
        return pd.DataFrame()
    lines_clean = lines.dropna(subset=["close_total_line"]).copy()
    lines_clean["date_str"] = lines_clean["date"].dt.strftime("%Y-%m-%d")
    model_data = model_data.copy()
    model_data["date_str"] = model_data["date"].dt.strftime("%Y-%m-%d")

    line_cols = ["date_str", "home_team", "away_team", "close_total_line"]
    optional_cols = ["actual_total", "open_total_line", "line_source", "num_books", "commence_time", "snapshot_ts"]
    line_cols.extend([c for c in optional_cols if c in lines_clean.columns])

    merged = model_data.merge(
        lines_clean[line_cols],
        on=["date_str", "home_team", "away_team"],
        how="inner",
    )
    if "actual_total" not in merged.columns and "total_runs" in merged.columns:
        merged["actual_total"] = merged["total_runs"]
    elif "actual_total" in merged.columns and "total_runs" in merged.columns:
        merged["actual_total"] = merged["actual_total"].fillna(merged["total_runs"])
    return merged


def run_betting_sim(
    merged,
    residual_std,
    high_tail_cfg=None,
    low_tail_cfg=None,
    market_shrink_cfg=None,
    market_edge_model=None,
    market_edge_cfg=None,
):
    """Add betting columns to matched rows."""
    df = merged.copy()
    adjusted_totals = []
    market_methods = []
    shrink_fracs = []
    market_probs = []
    for row in df.itertuples(index=False):
        market_result = apply_market_context(
            predicted_total=row.predicted_total,
            market_line=row.close_total_line,
            cfg=MARKET_SHRINK_BASE_CFG,
            prediction_std=getattr(row, "prediction_std", residual_std),
            num_books=getattr(row, "num_books", None),
            market_features={
                col: getattr(row, col, None)
                for col in ["snapshot_ts", "commence_time", "pinnacle_line", "draftkings_line", "fanduel_line", "betmgm_line", "caesars_line"]
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
        shrink_fracs.append(float(market_result.get("shrink_fraction", 0.0)))
        market_methods.append(market_result.get("method"))
        market_probs.append(market_result.get("p_over"))
    df["market_adjusted_total"] = adjusted_totals
    df["market_shrink_fraction"] = shrink_fracs
    df["market_adjustment_method"] = market_methods
    df["edge"] = df["market_adjusted_total"] - df["close_total_line"]
    df["p_over"] = np.clip(pd.to_numeric(market_probs, errors="coerce"), 1e-6, 1 - 1e-6)
    df["p_under"] = 1 - df["p_over"]
    df["bet"] = np.where(df["edge"] > 0, "OVER", "UNDER")
    df["actual_over"] = df["actual_total"] > df["close_total_line"]
    df["bet_won"] = (
        ((df["bet"] == "OVER") &  df["actual_over"]) |
        ((df["bet"] == "UNDER") & ~df["actual_over"])
    )
    return df


def print_lines_report(results, years_covered):
    years_str = ", ".join(str(y) for y in sorted(years_covered))
    in_out = "IN-SAMPLE (model trained on this data — results are optimistic)"

    print("\n" + "=" * 68)
    print(f"  SECTION 1: BETTING SIMULATION vs CLOSING LINES")
    print(f"  Seasons: {years_str}  |  {in_out}")
    print("=" * 68)

    total = len(results)
    wins  = results["bet_won"].sum()
    print(f"\n  Games matched to closing lines : {total}")
    print(f"  Overall win rate               : {wins/total:.1%}  (break-even: {BREAKEVEN:.1%})")
    print(f"  Overall ROI (flat -110)        : {roi_at_110(wins, total):.1f}%")
    print_probability_quality(results, prob_col="p_over", actual_col="actual_over", label="the closing line")

    # Edge buckets
    print(f"\n  {'Edge size':<18} {'Games':>7} {'Win%':>7} {'ROI':>8}  {'':>6}")
    print(f"  {'-'*50}")
    for label, min_edge in [
        ("All",       0.0),
        ("|edge|>=0.5", 0.5),
        ("|edge|>=1.0", 1.0),
        ("|edge|>=1.5", 1.5),
        ("|edge|>=2.0", 2.0),
        ("|edge|>=2.5", 2.5),
    ]:
        sub = results[results["edge"].abs() >= min_edge]
        if len(sub) < 10:
            continue
        w, n = sub["bet_won"].sum(), len(sub)
        flag = "+" if w / n > BREAKEVEN else ""
        print(f"  {label:<18} {n:>7} {w/n:>6.1%} {roi_at_110(w,n):>7.1f}%  {flag}")

    # Over vs Under
    print(f"\n  {'Direction':<18} {'Games':>7} {'Win%':>7} {'ROI':>8}")
    print(f"  {'-'*44}")
    for direction in ["OVER", "UNDER"]:
        sub = results[results["bet"] == direction]
        w, n = sub["bet_won"].sum(), len(sub)
        if n == 0:
            continue
        print(f"  {direction:<18} {n:>7} {w/n:>6.1%} {roi_at_110(w,n):>7.1f}%")

    # Calibration
    print(f"\n  Calibration (predicted P(over) vs actual hit rate):")
    print(f"  {'P(over) range':<18} {'Games':>7} {'Pred avg':>10} {'Actual':>8}")
    print(f"  {'-'*46}")
    for lo, hi in [(0.40, 0.45), (0.45, 0.50), (0.50, 0.55),
                   (0.55, 0.60), (0.60, 0.65), (0.65, 1.00)]:
        sub = results[results["p_over"].between(lo, hi)]
        if len(sub) < 10:
            continue
        print(f"  {lo:.0%}–{hi:.0%}              {len(sub):>7} {sub['p_over'].mean():>9.1%} "
              f"{sub['actual_over'].mean():>7.1%}")

    # Line movement check: are we on the right side of line movement?
    if "open_total_line" in results.columns:
        moved_with  = results[results["edge"] * (results["close_total_line"] - results["open_total_line"]) > 0]
        moved_against = results[results["edge"] * (results["close_total_line"] - results["open_total_line"]) < 0]
        if len(moved_with) > 20 and len(moved_against) > 20:
            print(f"\n  Line-movement alignment:")
            print(f"  Edge agrees w/ line move  : {len(moved_with):>5} games, "
                  f"win {moved_with['bet_won'].mean():.1%}")
            print(f"  Edge disagrees w/ line move: {len(moved_against):>5} games, "
                  f"win {moved_against['bet_won'].mean():.1%}")

    # Year-by-year breakdown
    print(f"\n  Year-by-year:")
    print(f"  {'Year':<8} {'Games':>7} {'Win%':>7} {'ROI':>8}")
    print(f"  {'-'*34}")
    for yr, grp in results.groupby(results["date"].dt.year):
        w, n = grp["bet_won"].sum(), len(grp)
        label = "(in-sample)" if yr in TRAIN_YEARS else "(OUT-OF-SAMPLE)"
        print(f"  {yr:<8} {n:>7} {w/n:>6.1%} {roi_at_110(w,n):>7.1f}%  {label}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Pure accuracy on 2025 out-of-sample test set
# ─────────────────────────────────────────────────────────────────────────────

def print_accuracy_report(model_data, residual_std, high_tail_cfg=None, low_tail_cfg=None):
    test = model_data[model_data["date"].dt.year == TEST_YEAR].copy()
    train = model_data[model_data["date"].dt.year < TEST_YEAR].copy()

    if test.empty:
        print("\n  No 2025 data found in mlb_model_data.tsv")
        return

    print("\n" + "=" * 68)
    print(f"  SECTION 2: OUT-OF-SAMPLE ACCURACY — {TEST_YEAR} SEASON")
    print(f"  (Model never saw these games during training)")
    print("=" * 68)

    # Core metrics
    mae_test  = (test["total_runs"] - test["predicted_total"]).abs().mean()
    rmse_test = np.sqrt(((test["total_runs"] - test["predicted_total"]) ** 2).mean())
    mae_train = (train["total_runs"] - train["predicted_total"]).abs().mean()
    sigma_mean = test["prediction_std"].mean() if "prediction_std" in test.columns else residual_std
    if "prediction_std" in test.columns:
        abs_err = (test["total_runs"] - test["predicted_total"]).abs()
        within_1sigma = (abs_err <= test["prediction_std"]).mean()
        within_2sigma = (abs_err <= 2 * test["prediction_std"]).mean()
    else:
        within_1sigma = np.nan
        within_2sigma = np.nan

    print(f"\n  Test  MAE  ({TEST_YEAR}) : {mae_test:.2f} runs  <-- trust this number")
    print(f"  Train MAE (2021-2024): {mae_train:.2f} runs  (in-sample, lower by design)")
    print(f"  Residual std         : {residual_std:.2f} runs")
    print(f"  Mean predicted sigma : {sigma_mean:.2f} runs")
    if not np.isnan(within_1sigma):
        print(f"  Coverage within 1-sigma: {within_1sigma:.1%}")
        print(f"  Coverage within 2-sigma: {within_2sigma:.1%}")
    print(f"  Games in test set    : {len(test)}")
    print(f"  Over-predicted games : {(test['predicted_total'] > test['total_runs']).mean():.1%}")
    print(f"  Pred > 9.5           : {(test['predicted_total'] > 9.5).mean():.1%}")
    print(f"  Actual > 9.5         : {(test['total_runs'] > 9.5).mean():.1%}")
    if "high_tail_prob_9p5" in test.columns:
        print(f"  Mean P(>9.5)         : {test['high_tail_prob_9p5'].mean():.1%}")
    print(f"  Pred < 7.5           : {(test['predicted_total'] < 7.5).mean():.1%}")
    print(f"  Actual < 7.5         : {(test['total_runs'] < 7.5).mean():.1%}")
    if "low_tail_prob_7p5" in test.columns:
        print(f"  Mean P(<7.5)         : {test['low_tail_prob_7p5'].mean():.1%}")
    print(f"  Mean predicted       : {test['predicted_total'].mean():.2f}")
    print(f"  Mean actual          : {test['total_runs'].mean():.2f}")
    print(f"  Prediction range     : {test['predicted_total'].min():.1f} – {test['predicted_total'].max():.1f}")
    print_total_bucket_bias(test, label=f"{TEST_YEAR} holdout")

    # Over/Under accuracy against fixed lines (no sportsbook needed)
    print(f"\n  Direction accuracy vs fixed thresholds (no lines required):")
    print(f"  {'Line':<8} {'Predict over':>13} {'Actual over':>12} {'Match%':>8}  {'n games':>8}")
    print(f"  {'-'*55}")
    for line in [7.5, 8.0, 8.5, 9.0, 9.5, 10.0]:
        pred_over   = test["predicted_total"] > line
        actual_over = test["total_runs"] > line
        match = (pred_over == actual_over).mean()
        n_over = actual_over.sum()
        print(f"  {line:<8} {pred_over.mean():>12.1%} {actual_over.mean():>11.1%} "
              f"{match:>7.1%}  {n_over:>8}")

    print(f"\n  Probability quality at fixed benchmark lines:")
    print(f"  {'Line':<8} {'Brier':>8} {'LogLoss':>9} {'Mean P(over)':>14} {'Actual over':>12}")
    print(f"  {'-'*60}")
    for line in [7.5, 8.5, 9.5]:
        probs = test.apply(
            lambda r: probability_over_line(
                r["predicted_total"],
                r["prediction_std"],
                line,
                high_tail_prob=r.get("high_tail_prob_9p5"),
                high_tail_cfg=high_tail_cfg,
                low_tail_prob=r.get("low_tail_prob_7p5"),
                low_tail_cfg=low_tail_cfg,
            ),
            axis=1,
        )
        actual_over = (test["total_runs"] > line).astype(int)
        brier = brier_score_loss(actual_over, probs)
        ll = safe_log_loss_binary(actual_over, probs)
        print(
            f"  {line:<8} {brier:>8.4f} {ll:>9.4f} "
            f"{probs.mean():>13.1%} {actual_over.mean():>11.1%}"
        )

    # Error distribution
    errors = (test["total_runs"] - test["predicted_total"]).abs()
    print(f"\n  Prediction error distribution:")
    print(f"  Within 1 run  : {(errors <= 1).mean():.1%}  ({(errors <= 1).sum()} games)")
    print(f"  Within 2 runs : {(errors <= 2).mean():.1%}  ({(errors <= 2).sum()} games)")
    print(f"  Within 3 runs : {(errors <= 3).mean():.1%}  ({(errors <= 3).sum()} games)")
    print(f"  > 5 runs off  : {(errors > 5).mean():.1%}  ({(errors > 5).sum()} games)")

    # Simulated ROI: if we bet every game assuming sportsbook = 8.5 (most common line)
    print(f"\n  Simulated ROI — assuming all lines posted at 8.5 (common benchmark):")
    sim_line = 8.5
    test = test.copy()
    test["sim_bet"] = np.where(test["predicted_total"] > sim_line, "OVER", "UNDER")
    test["sim_actual_over"] = test["total_runs"] > sim_line
    test["sim_won"] = (
        ((test["sim_bet"] == "OVER")  &  test["sim_actual_over"]) |
        ((test["sim_bet"] == "UNDER") & ~test["sim_actual_over"])
    )
    wins, n = test["sim_won"].sum(), len(test)
    print(f"  All games : {wins}/{n} = {wins/n:.1%} win rate, ROI {roi_at_110(wins,n):.1f}%")

    for min_edge in [0.25, 0.5, 0.75, 1.0]:
        sub = test[abs(test["predicted_total"] - sim_line) >= min_edge]
        if len(sub) < 20:
            continue
        w, cnt = sub["sim_won"].sum(), len(sub)
        print(f"  |edge|>={min_edge:.2f}: {w}/{cnt} = {w/cnt:.1%} win rate, "
              f"ROI {roi_at_110(w,cnt):.1f}%  ({cnt} games)")

    # Month-by-month accuracy
    test["month"] = test["date"].dt.month
    month_names = {4:"Apr", 5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct"}
    print(f"\n  Monthly breakdown (MAE):")
    print(f"  {'Month':<8} {'Games':>7} {'MAE':>7} {'Mean pred':>10} {'Mean actual':>12}")
    print(f"  {'-'*50}")
    for mo, grp in test.groupby("month"):
        mae = (grp["total_runs"] - grp["predicted_total"]).abs().mean()
        print(f"  {month_names.get(mo, mo):<8} {len(grp):>7} {mae:>7.2f} "
              f"{grp['predicted_total'].mean():>10.2f} {grp['total_runs'].mean():>12.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading model and data...")
    model_bundle = load_model_bundle()
    model = model_bundle["model"]
    meta = model_bundle["meta"]
    uncertainty_model = model_bundle["uncertainty_model"]
    uncertainty_cfg = model_bundle["uncertainty_cfg"]
    high_tail_model = model_bundle["high_tail_model"]
    high_tail_cfg = model_bundle["high_tail_cfg"]
    low_tail_model = model_bundle["low_tail_model"]
    low_tail_cfg = model_bundle["low_tail_cfg"]
    market_edge_model = model_bundle["market_edge_model"]
    market_edge_cfg = model_bundle["market_edge_cfg"]
    model_data    = load_model_data()
    lines         = load_lines()

    print(f"  {len(model_data)} model rows ({model_data['date'].dt.year.min()}–"
          f"{model_data['date'].dt.year.max()})")
    print(f"  Lines available for: "
          f"{sorted(lines['season'].unique().tolist()) if not lines.empty else 'none'}")

    print("Running predictions on all games...")
    model_data, X = predict_all(model_data, model, meta)
    residual_std  = compute_residual_std(model_data)
    print(f"  Residual std: {residual_std:.2f} runs")
    model_data = predict_sigmas(model_data, X, uncertainty_model, uncertainty_cfg, residual_std)
    if uncertainty_model is not None:
        print(f"  Dynamic uncertainty active (mean sigma {model_data['prediction_std'].mean():.2f})")
    else:
        print("  Dynamic uncertainty unavailable - using global sigma")
    model_data = predict_high_tail_probs(model_data, X, high_tail_model, high_tail_cfg)
    if high_tail_model is not None:
        print(
            f"  High-tail model active at {float(high_tail_cfg.get('line', 9.5)):.1f} "
            f"(mean prob {model_data['high_tail_prob_9p5'].mean():.1%})"
        )
    model_data = predict_low_tail_probs(model_data, X, low_tail_model, low_tail_cfg)
    if low_tail_model is not None:
        print(
            f"  Low-tail model active at {float(low_tail_cfg.get('line', 7.5)):.1f} "
            f"(mean prob {model_data['low_tail_prob_7p5'].mean():.1%})"
        )
    if market_edge_model is not None:
        print(
            f"  Market edge model active on {market_edge_cfg.get('samples', '?')} snapshot rows "
            f"(OOF Brier {market_edge_cfg.get('oof_brier_after', float('nan')):.4f})"
        )

    # ── Section 1: Betting simulation ────────────────────────────────────────
    matched = match_with_lines(model_data, lines)
    if matched.empty:
        print("\nNo games matched with closing lines — skipping Section 1.")
    else:
        years_in_lines = sorted(matched["date"].dt.year.unique().tolist())
        results = run_betting_sim(
            matched,
            residual_std,
            high_tail_cfg=high_tail_cfg,
            low_tail_cfg=low_tail_cfg,
            market_shrink_cfg=meta.get("market_shrinkage"),
            market_edge_model=market_edge_model,
            market_edge_cfg=market_edge_cfg,
        )
        print_lines_report(results, years_in_lines)

        out_path = os.path.join(DATA_DIR, "backtest_results_lines.tsv")
        results[["date", "away_team", "home_team", "predicted_total", "market_adjusted_total",
                 "prediction_std", "close_total_line", "edge", "p_over", "bet",
                 "actual_total", "bet_won"]].to_csv(out_path, sep="\t", index=False)
        print(f"\n  Detailed betting results saved to {out_path}")

    # ── Section 2: Out-of-sample accuracy ────────────────────────────────────
    print_accuracy_report(
        model_data,
        residual_std,
        high_tail_cfg=high_tail_cfg,
        low_tail_cfg=low_tail_cfg,
    )

    out_path2 = os.path.join(DATA_DIR, "backtest_results_2025.tsv")
    test = model_data[model_data["date"].dt.year == TEST_YEAR]
    if not test.empty:
        test[["date", "away_team", "home_team", "predicted_total",
              "prediction_std", "high_tail_prob_9p5", "low_tail_prob_7p5", "total_runs"]].to_csv(out_path2, sep="\t", index=False)
        print(f"\n  2025 prediction detail saved to {out_path2}")

    print("\nDone.")


if __name__ == "__main__":
    main()
