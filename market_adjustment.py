"""
Market-aware post-model total adjustment helpers.

The core baseball model remains independent. This module now supports two
optional second-stage market layers:

1. A legacy shrinkage policy that nudges the point total toward the market line.
2. A snapshot-trained classifier that predicts P(over | current market context)
   and converts that probability back into an implied total for downstream use.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from modeling_utils import adjusted_sigma_for_line, clip_probabilities
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LEGACY_LINES_PATH = os.path.join(DATA_DIR, "lines_historical.tsv")
ODDSAPI_LINES_PATH = os.path.join(DATA_DIR, "lines_historical_oddsapi.tsv")
ODDSAPI_SNAPSHOTS_PATH = os.path.join(DATA_DIR, "lines_historical_oddsapi_snapshots.tsv")

DEFAULT_GAP_BUCKETS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 99.0]
DEFAULT_SIGMA_BUCKETS = [0.0, 4.0, 4.5, 5.0, 99.0]
DEFAULT_ALPHA_GRID = [round(x, 2) for x in np.linspace(0.0, 0.35, 8)]
MIN_BUCKET_SAMPLES = 80

BOOK_LINE_COLUMNS = [
    "pinnacle_line",
    "draftkings_line",
    "fanduel_line",
    "betmgm_line",
    "caesars_line",
]

MARKET_EDGE_MODEL_FILE = "market_edge_xgb.json"
MARKET_EDGE_MIN_SAMPLES = 1500
MARKET_EDGE_PARAMS = {
    "n_estimators": 250,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "reg_alpha": 0.2,
    "reg_lambda": 2.0,
    "random_state": 42,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
}

SHRINKAGE_FALLBACK_MAX_ABS_EDGE = 1.25
SHRINKAGE_FALLBACK_PROB_SHRINK = 0.60
SHRINKAGE_FALLBACK_CONFIDENCE_CAP = 0.62

def _coalesce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = pd.Series(np.nan, index=frame.index, dtype=float)
    for col in columns:
        if col in frame.columns:
            series = pd.to_numeric(frame[col], errors="coerce")
            values = values.fillna(series)
    return values


def _numeric_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(default, index=frame.index, dtype=float)


def _normalize_market_columns(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()

    if "close_total_line" not in df.columns:
        df["close_total_line"] = np.nan
    df["close_total_line"] = _coalesce_numeric(
        df,
        ["close_total_line", "close_total_line.1", "consensus_total_line"],
    )

    if "num_books" in df.columns:
        df["num_books"] = pd.to_numeric(df["num_books"], errors="coerce")

    for col in ["date", "snapshot_ts", "requested_snapshot_ts", "previous_snapshot_ts", "commence_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=col != "date")

    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
        except TypeError:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in BOOK_LINE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "actual_total" in df.columns:
        df["actual_total"] = pd.to_numeric(df["actual_total"], errors="coerce")

    return df


def load_historical_lines() -> pd.DataFrame:
    frames = []

    if os.path.exists(LEGACY_LINES_PATH):
        legacy = pd.read_csv(LEGACY_LINES_PATH, sep="\t", parse_dates=["date"])
        legacy["line_source"] = legacy.get("line_source", "legacy")
        frames.append(_normalize_market_columns(legacy))

    if os.path.exists(ODDSAPI_LINES_PATH):
        oddsapi = pd.read_csv(ODDSAPI_LINES_PATH, sep="\t", parse_dates=["date"])
        oddsapi["line_source"] = oddsapi.get("line_source", "oddsapi")
        frames.append(_normalize_market_columns(oddsapi))

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


def load_historical_line_snapshots() -> pd.DataFrame:
    if not os.path.exists(ODDSAPI_SNAPSHOTS_PATH):
        return pd.DataFrame()

    snapshots = pd.read_csv(
        ODDSAPI_SNAPSHOTS_PATH,
        sep="\t",
        parse_dates=["requested_snapshot_ts", "snapshot_ts", "previous_snapshot_ts", "commence_time"],
    )
    snapshots["line_source"] = snapshots.get("line_source", "oddsapi_snapshot")
    snapshots = _normalize_market_columns(snapshots)

    if "date" not in snapshots.columns or snapshots["date"].isna().all():
        if "commence_time" in snapshots.columns:
            snapshots["date"] = snapshots["commence_time"].dt.tz_convert(None).dt.normalize()

    return snapshots


def _merge_predictions_with_market_rows(pred_df: pd.DataFrame, market_rows: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty or market_rows.empty:
        return pd.DataFrame()

    pred = pred_df.copy()
    pred["date_str"] = pd.to_datetime(pred["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    market = _normalize_market_columns(market_rows)
    market = market.dropna(subset=["close_total_line"]).copy()
    market["date_str"] = pd.to_datetime(market["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    keep_cols = ["date_str", "home_team", "away_team", "close_total_line"]
    optional = [
        "actual_total",
        "num_books",
        "line_source",
        "open_total_line",
        "snapshot_ts",
        "requested_snapshot_ts",
        "commence_time",
        *BOOK_LINE_COLUMNS,
    ]
    keep_cols.extend([c for c in optional if c in market.columns])

    merged = pred.merge(
        market[keep_cols],
        on=["date_str", "home_team", "away_team"],
        how="inner",
    )
    if "actual_total" not in merged.columns and "total_runs" in merged.columns:
        merged["actual_total"] = merged["total_runs"]
    elif "actual_total" in merged.columns and "total_runs" in merged.columns:
        merged["actual_total"] = merged["actual_total"].fillna(merged["total_runs"])
    return merged


def merge_predictions_with_lines(pred_df: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    return _merge_predictions_with_market_rows(pred_df, lines)


def merge_predictions_with_snapshots(pred_df: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    return _merge_predictions_with_market_rows(pred_df, snapshots)


def probability_over_line(mean_total: np.ndarray, sigma: np.ndarray, line: np.ndarray) -> np.ndarray:
    safe_sigma = np.clip(np.asarray(sigma, dtype=float), 1e-6, None)
    return 1.0 - stats.norm.cdf(np.asarray(line, dtype=float), loc=np.asarray(mean_total, dtype=float), scale=safe_sigma)


def probability_over_current_line(
    mean_total: float,
    base_sigma: float,
    line: float,
    high_tail_prob: float | None = None,
    high_tail_cfg: dict | None = None,
    low_tail_prob: float | None = None,
    low_tail_cfg: dict | None = None,
) -> float:
    sigma = adjusted_sigma_for_line(
        mean_total,
        base_sigma,
        line,
        high_tail_prob,
        high_tail_cfg,
        low_tail_prob=low_tail_prob,
        low_tail_cfg=low_tail_cfg,
    )
    return float(1 - stats.norm.cdf(line, loc=mean_total, scale=sigma))


def _bucket_label(value: float, bins: list[float]) -> str:
    value = float(value)
    for lo, hi in zip(bins[:-1], bins[1:]):
        if value <= hi:
            return f"{lo:.1f}-{hi:.1f}"
    return f"{bins[-2]:.1f}+"


def _build_adjusted_total(predicted_total: np.ndarray, market_line: np.ndarray, alpha: float) -> np.ndarray:
    return np.asarray(predicted_total, dtype=float) + float(alpha) * (
        np.asarray(market_line, dtype=float) - np.asarray(predicted_total, dtype=float)
    )


def _score_alpha(frame: pd.DataFrame, alpha: float) -> tuple[float, float, float]:
    adjusted = _build_adjusted_total(frame["predicted_total"], frame["close_total_line"], alpha)
    mae = float(mean_absolute_error(frame["actual_total"], adjusted))
    probs = np.clip(
        probability_over_line(adjusted, frame["prediction_std"], frame["close_total_line"]),
        1e-6,
        1 - 1e-6,
    )
    actual_over = (frame["actual_total"] > frame["close_total_line"]).astype(int)
    brier = float(brier_score_loss(actual_over, probs))
    win_rate = float(((probs >= 0.5) == actual_over).mean())
    score = brier + 0.015 * mae
    return score, mae, win_rate


def fit_market_shrinkage(
    matched_lines: pd.DataFrame,
    gap_buckets: list[float] | None = None,
    sigma_buckets: list[float] | None = None,
    alpha_grid: list[float] | None = None,
    min_bucket_samples: int = MIN_BUCKET_SAMPLES,
) -> dict[str, Any]:
    """
    Fit a bounded lookup-table shrinkage policy from historical model-vs-market
    disagreements.
    """
    if matched_lines.empty:
        return {"enabled": False, "reason": "no_matched_lines"}

    gap_buckets = gap_buckets or DEFAULT_GAP_BUCKETS
    sigma_buckets = sigma_buckets or DEFAULT_SIGMA_BUCKETS
    alpha_grid = alpha_grid or DEFAULT_ALPHA_GRID

    df = matched_lines.copy()
    df = df.dropna(subset=["predicted_total", "close_total_line", "actual_total", "prediction_std"]).copy()
    if len(df) < 300:
        return {"enabled": False, "reason": "insufficient_samples", "samples": int(len(df))}

    df["abs_model_market_gap"] = (df["predicted_total"] - df["close_total_line"]).abs()
    df["gap_bucket"] = df["abs_model_market_gap"].apply(lambda x: _bucket_label(x, gap_buckets))
    df["sigma_bucket"] = df["prediction_std"].apply(lambda x: _bucket_label(x, sigma_buckets))

    global_candidates = []
    for alpha in alpha_grid:
        score, mae, win_rate = _score_alpha(df, alpha)
        global_candidates.append({"alpha": float(alpha), "score": score, "mae": mae, "win_rate": win_rate})
    best_global = min(global_candidates, key=lambda row: (row["score"], row["alpha"]))

    bucket_rows = []
    bucket_map: dict[str, dict[str, Any]] = {}
    grouped = df.groupby(["gap_bucket", "sigma_bucket"], observed=True)
    for (gap_bucket, sigma_bucket), sub in grouped:
        if len(sub) < min_bucket_samples:
            continue
        candidates = []
        for alpha in alpha_grid:
            score, mae, win_rate = _score_alpha(sub, alpha)
            candidates.append({"alpha": float(alpha), "score": score, "mae": mae, "win_rate": win_rate})
        best_local = min(candidates, key=lambda row: (row["score"], row["alpha"]))
        improvement = best_global["score"] - best_local["score"]
        chosen_alpha = best_local["alpha"] if improvement > 0.0005 else best_global["alpha"]
        key = f"{gap_bucket}|{sigma_bucket}"
        bucket_map[key] = {
            "alpha": float(chosen_alpha),
            "samples": int(len(sub)),
            "local_alpha": float(best_local["alpha"]),
            "local_score": float(best_local["score"]),
            "global_score": float(best_global["score"]),
            "improvement": float(improvement),
        }
        bucket_rows.append({"bucket": key, **bucket_map[key]})

    fitted = {
        "enabled": True,
        "method": "bucketed_market_shrinkage",
        "samples": int(len(df)),
        "global_alpha": float(best_global["alpha"]),
        "global_score": float(best_global["score"]),
        "global_mae": float(best_global["mae"]),
        "global_win_rate": float(best_global["win_rate"]),
        "gap_buckets": [float(x) for x in gap_buckets],
        "sigma_buckets": [float(x) for x in sigma_buckets],
        "alpha_grid": [float(x) for x in alpha_grid],
        "min_bucket_samples": int(min_bucket_samples),
        "bucket_map": bucket_map,
        "bucket_rows": bucket_rows,
    }
    return fitted


def build_market_adjustment_features(
    predicted_total: float,
    market_line: float,
    prediction_std: float | None = None,
    num_books: int | None = None,
) -> dict[str, float]:
    gap = float(predicted_total) - float(market_line)
    return {
        "predicted_total": float(predicted_total),
        "market_line": float(market_line),
        "model_market_gap": gap,
        "abs_model_market_gap": abs(gap),
        "prediction_std": float(prediction_std) if prediction_std is not None else 0.0,
        "num_books": float(num_books) if num_books is not None else 0.0,
    }


def apply_market_shrinkage(
    predicted_total: float,
    market_line: float | None,
    cfg: dict[str, Any] | None = None,
    prediction_std: float | None = None,
    num_books: int | None = None,
    learned_cfg: dict[str, Any] | None = None,
) -> dict[str, float | bool | str | None]:
    cfg = cfg or {}
    learned_cfg = learned_cfg or {}
    enabled = bool(cfg.get("enabled", False) and cfg.get("use_for_post_model_shrinkage", False))

    if market_line is None:
        return {
            "enabled": enabled,
            "used_market_line": False,
            "adjusted_total": float(predicted_total),
            "shrink_fraction": 0.0,
            "market_line": None,
            "bucket_key": None,
        }

    if not enabled:
        return {
            "enabled": enabled,
            "used_market_line": False,
            "adjusted_total": float(predicted_total),
            "shrink_fraction": 0.0,
            "market_line": float(market_line),
            "bucket_key": None,
        }

    min_books = int(cfg.get("min_books", 1))
    has_num_books = num_books is not None and not pd.isna(num_books)
    if has_num_books and int(num_books) < min_books:
        return {
            "enabled": True,
            "used_market_line": False,
            "adjusted_total": float(predicted_total),
            "shrink_fraction": 0.0,
            "market_line": float(market_line),
            "bucket_key": None,
        }

    gap = float(predicted_total) - float(market_line)
    deadband = float(cfg.get("shrink_deadband_runs", 0.25))
    if abs(gap) <= deadband:
        return {
            "enabled": True,
            "used_market_line": True,
            "adjusted_total": float(predicted_total),
            "shrink_fraction": 0.0,
            "market_line": float(market_line),
            "bucket_key": None,
        }

    max_shrink = float(cfg.get("max_shrink_fraction", 0.35))
    learned_enabled = bool(learned_cfg.get("enabled"))
    alpha = float(learned_cfg.get("global_alpha", 0.0)) if learned_enabled else 0.0
    bucket_key = None
    if learned_enabled:
        gap_bins = learned_cfg.get("gap_buckets", DEFAULT_GAP_BUCKETS)
        sigma_bins = learned_cfg.get("sigma_buckets", DEFAULT_SIGMA_BUCKETS)
        gap_bucket = _bucket_label(abs(gap), gap_bins)
        sigma_bucket = _bucket_label(float(prediction_std or 0.0), sigma_bins)
        bucket_key = f"{gap_bucket}|{sigma_bucket}"
        alpha = float(learned_cfg.get("bucket_map", {}).get(bucket_key, {}).get("alpha", alpha))

    alpha = max(0.0, min(max_shrink, alpha))
    adjusted_total = float(predicted_total) + alpha * (float(market_line) - float(predicted_total))
    return {
        "enabled": True,
        "used_market_line": True,
        "adjusted_total": adjusted_total,
        "shrink_fraction": alpha,
        "market_line": float(market_line),
        "bucket_key": bucket_key,
    }


def ensure_market_probability_columns(
    frame: pd.DataFrame,
    high_tail_cfg: dict | None = None,
    low_tail_cfg: dict | None = None,
) -> pd.DataFrame:
    df = _normalize_market_columns(frame)
    if "base_p_over" in df.columns and df["base_p_over"].notna().all():
        df["base_p_over"] = clip_probabilities(df["base_p_over"])
        return df

    probs = []
    for row in df.itertuples(index=False):
        probs.append(
            probability_over_current_line(
                mean_total=float(getattr(row, "predicted_total")),
                base_sigma=float(getattr(row, "prediction_std")),
                line=float(getattr(row, "close_total_line")),
                high_tail_prob=getattr(row, "high_tail_prob_9p5", None),
                high_tail_cfg=high_tail_cfg,
                low_tail_prob=getattr(row, "low_tail_prob_7p5", None),
                low_tail_cfg=low_tail_cfg,
            )
        )
    df["base_p_over"] = clip_probabilities(probs)
    return df


def build_market_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_market_columns(frame)
    feats = pd.DataFrame(index=df.index)

    feats["predicted_total"] = _numeric_series(df, "predicted_total")
    feats["prediction_std"] = _numeric_series(df, "prediction_std")
    feats["market_line"] = _numeric_series(df, "close_total_line")
    feats["model_market_gap"] = feats["predicted_total"] - feats["market_line"]
    feats["abs_model_market_gap"] = feats["model_market_gap"].abs()
    feats["gap_per_sigma"] = feats["model_market_gap"] / np.clip(feats["prediction_std"], 1e-6, None)
    feats["num_books"] = _numeric_series(df, "num_books").fillna(0.0)
    feats["base_p_over"] = clip_probabilities(_numeric_series(df, "base_p_over", default=0.5))
    feats["base_edge_prob"] = feats["base_p_over"] - 0.5
    feats["high_tail_prob_9p5"] = _numeric_series(df, "high_tail_prob_9p5").fillna(0.0)
    feats["low_tail_prob_7p5"] = _numeric_series(df, "low_tail_prob_7p5").fillna(0.0)

    hours_to_first_pitch = pd.Series(0.0, index=df.index, dtype=float)
    if "snapshot_ts" in df.columns and "commence_time" in df.columns:
        delta_hours = (
            pd.to_datetime(df["commence_time"], errors="coerce", utc=True)
            - pd.to_datetime(df["snapshot_ts"], errors="coerce", utc=True)
        ).dt.total_seconds() / 3600.0
        hours_to_first_pitch = pd.to_numeric(delta_hours, errors="coerce").fillna(0.0).clip(lower=0.0, upper=48.0)
    feats["hours_to_first_pitch"] = hours_to_first_pitch

    available_lines = []
    for col in BOOK_LINE_COLUMNS:
        key = col.replace("_line", "")
        series = _numeric_series(df, col, default=np.nan)
        feats[f"{key}_available"] = series.notna().astype(float)
        feats[f"{key}_delta_consensus"] = (series - feats["market_line"]).fillna(0.0)
        feats[f"{key}_delta_model"] = (series - feats["predicted_total"]).fillna(0.0)
        available_lines.append(series)

    if available_lines:
        book_frame = pd.concat(available_lines, axis=1)
        feats["market_line_std"] = book_frame.std(axis=1, skipna=True).fillna(0.0)
        feats["market_line_range"] = (book_frame.max(axis=1, skipna=True) - book_frame.min(axis=1, skipna=True)).fillna(0.0)
        offsets = book_frame.sub(feats["market_line"], axis=0)
        feats["books_above_consensus"] = offsets.gt(0).sum(axis=1).astype(float)
        feats["books_below_consensus"] = offsets.lt(0).sum(axis=1).astype(float)
    else:
        feats["market_line_std"] = 0.0
        feats["market_line_range"] = 0.0
        feats["books_above_consensus"] = 0.0
        feats["books_below_consensus"] = 0.0

    return feats.fillna(0.0)


def build_market_edge_model(early_stopping_rounds: int | None = None):
    params = MARKET_EDGE_PARAMS.copy()
    if early_stopping_rounds is not None:
        params["early_stopping_rounds"] = early_stopping_rounds
    return xgb.XGBClassifier(**params)


def apply_market_probability_calibration(probabilities, cfg: dict | None):
    probs = clip_probabilities(probabilities)
    if not cfg or not cfg.get("enabled"):
        return probs

    xs = np.asarray(cfg.get("x", []), dtype=float)
    ys = np.asarray(cfg.get("y", []), dtype=float)
    if len(xs) == 0 or len(xs) != len(ys):
        return probs

    mapped = np.interp(probs, xs, ys, left=ys[0], right=ys[-1])
    alpha = float(cfg.get("alpha", 1.0))
    return clip_probabilities(probs + alpha * (mapped - probs))


def fit_market_edge_model(
    matched_market_rows: pd.DataFrame,
    high_tail_cfg: dict | None = None,
    low_tail_cfg: dict | None = None,
    min_samples: int = MARKET_EDGE_MIN_SAMPLES,
) -> tuple[xgb.XGBClassifier | None, dict[str, Any]]:
    if matched_market_rows.empty:
        return None, {"enabled": False, "reason": "no_market_rows"}

    df = matched_market_rows.copy()
    if "snapshot_ts" in df.columns and "commence_time" in df.columns:
        snap_ts = pd.to_datetime(df["snapshot_ts"], errors="coerce", utc=True)
        commence_ts = pd.to_datetime(df["commence_time"], errors="coerce", utc=True)
        pregame_mask = snap_ts.isna() | commence_ts.isna() | (snap_ts <= commence_ts)
        df = df.loc[pregame_mask].copy()
    df = df.dropna(subset=["predicted_total", "close_total_line", "actual_total", "prediction_std"]).copy()
    if len(df) < min_samples:
        return None, {"enabled": False, "reason": "insufficient_samples", "samples": int(len(df))}

    df = ensure_market_probability_columns(df, high_tail_cfg=high_tail_cfg, low_tail_cfg=low_tail_cfg)
    sort_col = "snapshot_ts" if "snapshot_ts" in df.columns and df["snapshot_ts"].notna().any() else "date"
    df = df.sort_values(sort_col).reset_index(drop=True)

    X = build_market_feature_frame(df)
    target = (df["actual_total"] > df["close_total_line"]).astype(int)
    tscv = TimeSeriesSplit(n_splits=5)
    oof_probs = pd.Series(np.nan, index=df.index, dtype=float)

    for train_idx, val_idx in tscv.split(X):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = target.iloc[train_idx]
        y_val = target.iloc[val_idx]

        fold_model = build_market_edge_model(early_stopping_rounds=20)
        fold_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        oof_probs.iloc[val_idx] = fold_model.predict_proba(X_val)[:, 1]

    valid = oof_probs.notna()
    if int(valid.sum()) < 500:
        return None, {"enabled": False, "reason": "insufficient_oof_samples", "samples": int(valid.sum())}

    raw_probs = clip_probabilities(oof_probs.loc[valid].to_numpy(dtype=float))
    actual = target.loc[valid].to_numpy(dtype=int)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic_probs = clip_probabilities(calibrator.fit_transform(raw_probs, actual))
    raw_brier = float(brier_score_loss(actual, raw_probs))

    candidates = []
    for alpha in np.linspace(0.0, 1.0, 21):
        blended = clip_probabilities(raw_probs + alpha * (isotonic_probs - raw_probs))
        brier = float(brier_score_loss(actual, blended))
        ll = float(log_loss(actual, blended, labels=[0, 1]))
        candidates.append((alpha, blended, brier, ll))

    best_alpha, calibrated_probs, best_brier, best_logloss = min(
        candidates,
        key=lambda item: (item[2], item[3], item[0]),
    )

    final_model = build_market_edge_model(early_stopping_rounds=20)
    split_idx = max(int(len(X) * 0.8), 1)
    split_idx = min(split_idx, len(X) - 1) if len(X) > 1 else 1
    if len(X) > 1 and split_idx < len(X):
        final_model.fit(
            X.iloc[:split_idx],
            target.iloc[:split_idx],
            eval_set=[(X.iloc[split_idx:], target.iloc[split_idx:])],
            verbose=False,
        )
    else:
        final_model.fit(X, target, verbose=False)

    cfg = {
        "enabled": True,
        "file": MARKET_EDGE_MODEL_FILE,
        "method": "snapshot_market_edge_classifier",
        "features": X.columns.tolist(),
        "alpha": float(best_alpha),
        "x": [float(v) for v in calibrator.X_thresholds_],
        "y": [float(v) for v in calibrator.y_thresholds_],
        "samples": int(len(df)),
        "positive_rate": float(target.mean()),
        "oof_rate_pred_before": float(raw_probs.mean()),
        "oof_rate_pred_after": float(calibrated_probs.mean()),
        "oof_rate_actual": float(actual.mean()),
        "oof_brier_before": raw_brier,
        "oof_brier_after": best_brier,
        "oof_logloss_after": best_logloss,
        "best_iteration": getattr(final_model, "best_iteration", None),
    }
    return final_model, cfg


def predict_market_edge_probs(market_model, market_cfg: dict | None, feature_frame: pd.DataFrame) -> np.ndarray:
    if market_model is None or not market_cfg or not market_cfg.get("enabled"):
        return np.full(len(feature_frame), np.nan, dtype=float)
    X = feature_frame.reindex(columns=market_cfg.get("features", feature_frame.columns.tolist()), fill_value=0.0)
    probs = market_model.predict_proba(X)[:, 1]
    return apply_market_probability_calibration(probs, market_cfg)


def implied_total_from_probability(market_line: float, sigma: float, p_over: float) -> float:
    p = float(np.clip(p_over, 1e-6, 1 - 1e-6))
    return float(market_line + float(max(sigma, 1e-6)) * stats.norm.ppf(p))


def apply_fallback_guardrails(
    adjusted_total: float,
    market_line: float,
    adjusted_p_over: float,
    cfg: dict[str, Any] | None = None,
) -> tuple[float, float]:
    cfg = cfg or {}
    max_abs_edge = float(cfg.get("fallback_max_abs_edge_runs", SHRINKAGE_FALLBACK_MAX_ABS_EDGE))
    prob_shrink = float(cfg.get("fallback_probability_shrink", SHRINKAGE_FALLBACK_PROB_SHRINK))
    confidence_cap = float(cfg.get("fallback_confidence_cap", SHRINKAGE_FALLBACK_CONFIDENCE_CAP))

    edge = float(adjusted_total) - float(market_line)
    edge = float(np.clip(edge, -max_abs_edge, max_abs_edge))
    guarded_total = float(market_line) + edge

    guarded_p = 0.5 + prob_shrink * (float(adjusted_p_over) - 0.5)
    low = 1.0 - confidence_cap
    high = confidence_cap
    guarded_p = float(np.clip(guarded_p, low, high))
    return guarded_total, guarded_p


def apply_market_context(
    predicted_total: float,
    market_line: float | None,
    cfg: dict[str, Any] | None = None,
    prediction_std: float | None = None,
    num_books: int | None = None,
    market_features: dict[str, Any] | None = None,
    high_tail_prob: float | None = None,
    high_tail_cfg: dict | None = None,
    low_tail_prob: float | None = None,
    low_tail_cfg: dict | None = None,
    market_model=None,
    market_model_cfg: dict[str, Any] | None = None,
    learned_shrink_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = cfg or {}
    market_model_cfg = market_model_cfg or {}
    enabled = bool(cfg.get("enabled", False) and cfg.get("use_for_post_model_shrinkage", False))

    base_sigma = float(prediction_std) if prediction_std is not None and np.isfinite(prediction_std) else 4.5
    base_p_over = None
    if market_line is not None:
        base_p_over = probability_over_current_line(
            predicted_total,
            base_sigma,
            market_line,
            high_tail_prob=high_tail_prob,
            high_tail_cfg=high_tail_cfg,
            low_tail_prob=low_tail_prob,
            low_tail_cfg=low_tail_cfg,
        )

    default_result = {
        "enabled": enabled,
        "used_market_line": False,
        "method": "none",
        "adjusted_total": float(predicted_total),
        "p_over": float(base_p_over) if base_p_over is not None else None,
        "market_line": float(market_line) if market_line is not None else None,
        "shrink_fraction": 0.0,
        "bucket_key": None,
    }

    if market_line is None:
        return default_result

    min_books = int(cfg.get("min_books", 1))
    if num_books is not None and not pd.isna(num_books) and int(num_books) < min_books:
        return default_result

    if enabled and market_model is not None and market_model_cfg.get("enabled"):
        row = {
            "predicted_total": float(predicted_total),
            "close_total_line": float(market_line),
            "prediction_std": base_sigma,
            "num_books": float(num_books) if num_books is not None and not pd.isna(num_books) else 0.0,
            "base_p_over": float(base_p_over) if base_p_over is not None else 0.5,
            "high_tail_prob_9p5": float(high_tail_prob) if high_tail_prob is not None else 0.0,
            "low_tail_prob_7p5": float(low_tail_prob) if low_tail_prob is not None else 0.0,
        }
        if market_features:
            row.update(market_features)

        market_row = ensure_market_probability_columns(pd.DataFrame([row]), high_tail_cfg=high_tail_cfg, low_tail_cfg=low_tail_cfg)
        feature_frame = build_market_feature_frame(market_row)
        market_prob = float(predict_market_edge_probs(market_model, market_model_cfg, feature_frame)[0])

        sigma_for_line = adjusted_sigma_for_line(
            predicted_total,
            base_sigma,
            market_line,
            high_tail_prob=high_tail_prob,
            high_tail_cfg=high_tail_cfg,
            low_tail_prob=low_tail_prob,
            low_tail_cfg=low_tail_cfg,
        )
        adjusted_total = implied_total_from_probability(market_line, sigma_for_line, market_prob)
        return {
            "enabled": True,
            "used_market_line": True,
            "method": "edge_model",
            "adjusted_total": adjusted_total,
            "p_over": market_prob,
            "market_line": float(market_line),
            "shrink_fraction": 0.0,
            "bucket_key": None,
        }

    shrink = apply_market_shrinkage(
        predicted_total=predicted_total,
        market_line=market_line,
        cfg=cfg,
        prediction_std=base_sigma,
        num_books=num_books,
        learned_cfg=learned_shrink_cfg or {},
    )
    adjusted_total = float(shrink.get("adjusted_total", predicted_total))
    adjusted_p_over = probability_over_current_line(
        adjusted_total,
        base_sigma,
        market_line,
        high_tail_prob=high_tail_prob,
        high_tail_cfg=high_tail_cfg,
        low_tail_prob=low_tail_prob,
        low_tail_cfg=low_tail_cfg,
    )
    guarded_total, guarded_p_over = apply_fallback_guardrails(
        adjusted_total=adjusted_total,
        market_line=float(market_line),
        adjusted_p_over=adjusted_p_over,
        cfg=cfg,
    )
    return {
        "enabled": bool(shrink.get("enabled", enabled)),
        "used_market_line": bool(shrink.get("used_market_line", False)),
        "method": "shrinkage_guarded" if enabled else "base_model",
        "adjusted_total": guarded_total,
        "p_over": guarded_p_over,
        "market_line": float(market_line),
        "shrink_fraction": float(shrink.get("shrink_fraction", 0.0)),
        "bucket_key": shrink.get("bucket_key"),
    }
