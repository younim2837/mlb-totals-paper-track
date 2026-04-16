"""
Shared model loading and runtime inference helpers.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb

from modeling_utils import (
    apply_high_tail_calibration,
    apply_low_tail_calibration,
    apply_total_calibration,
    build_high_tail_features,
    build_low_tail_features,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")


def load_model(model_dir: str = MODEL_DIR):
    with open(os.path.join(model_dir, "model_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    model_family = meta.get("model_family", "single_total")

    if model_family == "team_split":
        models = {}
        for side, side_meta in meta.get("side_models", {}).items():
            side_model = xgb.XGBRegressor()
            side_model.load_model(os.path.join(model_dir, side_meta["file"]))
            models[side] = side_model
        return models, meta

    model = xgb.XGBRegressor()
    model.load_model(os.path.join(model_dir, "total_runs_xgb.json"))
    return model, meta


def _load_optional_model(meta: dict, meta_key: str, model_dir: str, model_cls):
    cfg = meta.get(meta_key) or {}
    model_file = cfg.get("file")
    if not (cfg.get("enabled") and model_file):
        return None, cfg

    path = os.path.join(model_dir, model_file)
    if not os.path.exists(path):
        return None, cfg

    model = model_cls()
    model.load_model(path)
    return model, cfg


def load_uncertainty_model(meta, model_dir: str = MODEL_DIR):
    return _load_optional_model(meta, "uncertainty_model", model_dir, xgb.XGBRegressor)


def load_high_tail_model(meta, model_dir: str = MODEL_DIR):
    return _load_optional_model(meta, "high_tail_model", model_dir, xgb.XGBClassifier)


def load_low_tail_model(meta, model_dir: str = MODEL_DIR):
    return _load_optional_model(meta, "low_tail_model", model_dir, xgb.XGBClassifier)


def load_market_edge_model(meta, model_dir: str = MODEL_DIR):
    return _load_optional_model(meta, "market_edge_model", model_dir, xgb.XGBClassifier)


def load_model_bundle(model_dir: str = MODEL_DIR):
    model, meta = load_model(model_dir=model_dir)
    uncertainty_model, uncertainty_cfg = load_uncertainty_model(meta, model_dir=model_dir)
    high_tail_model, high_tail_cfg = load_high_tail_model(meta, model_dir=model_dir)
    low_tail_model, low_tail_cfg = load_low_tail_model(meta, model_dir=model_dir)
    market_edge_model, market_edge_cfg = load_market_edge_model(meta, model_dir=model_dir)
    return {
        "model": model,
        "meta": meta,
        "uncertainty_model": uncertainty_model,
        "uncertainty_cfg": uncertainty_cfg,
        "high_tail_model": high_tail_model,
        "high_tail_cfg": high_tail_cfg,
        "low_tail_model": low_tail_model,
        "low_tail_cfg": low_tail_cfg,
        "market_edge_model": market_edge_model,
        "market_edge_cfg": market_edge_cfg,
    }


def load_historical_data(data_dir: str = DATA_DIR):
    path = os.path.join(data_dir, "mlb_games_raw.tsv")
    return pd.read_csv(path, sep="\t", parse_dates=["date"])


def feature_bucket_name(feature: str) -> str:
    if feature.startswith("dc_") or "_dc_" in feature or feature.startswith(("home_dc_", "away_dc_")):
        return "DC priors"
    if "lineup_" in feature:
        return "Lineups"
    if feature.startswith("league_avg_total_runs_"):
        return "Environment"
    if feature in {"park_factor", "temp_f", "wind_mph", "precip_mm", "humidity_pct", "dew_point_f",
                   "first_pitch_local_hour", "is_night_game", "is_dome"}:
        return "Environment"
    if feature.startswith("ump_"):
        return "Umpire"
    if "pitcher_" in feature or feature.startswith("combined_pitcher_"):
        return "Starters"
    if "bullpen_" in feature:
        return "Bullpen"
    if "rolling_ops" in feature or "rolling_bb_pct" in feature or "rolling_k_pct" in feature:
        return "Recent batting"
    if feature.startswith(("home_avg_", "away_avg_", "combined_scoring_", "combined_allowed_",
                           "home_offense_vs_", "away_offense_vs_", "combined_season_",
                           "home_days_rest", "away_days_rest", "h2h_avg_")):
        return "Team form"
    if "elo" in feature:
        return "Elo"
    return "Other"


def summarize_model_drivers(model, X: pd.DataFrame, feature_order: list[str]) -> dict:
    dmat = xgb.DMatrix(X[feature_order], feature_names=feature_order)
    contribs = model.get_booster().predict(dmat, pred_contribs=True)[0]
    feature_contribs = list(zip(feature_order, contribs[:-1]))
    bias = float(contribs[-1])

    bucket_totals = defaultdict(float)
    for feat, val in feature_contribs:
        bucket_totals[feature_bucket_name(feat)] += float(val)

    top_buckets = [
        (bucket, value)
        for bucket, value in sorted(bucket_totals.items(), key=lambda item: abs(item[1]), reverse=True)
        if abs(value) >= 0.05
    ][:3]
    top_features = [
        (feat, float(val))
        for feat, val in sorted(feature_contribs, key=lambda item: abs(item[1]), reverse=True)
        if abs(val) >= 0.05
    ][:5]

    return {
        "bias": bias,
        "bucket_totals": dict(bucket_totals),
        "feature_contribs": [(feat, float(val)) for feat, val in feature_contribs],
        "top_buckets": top_buckets,
        "top_features": top_features,
    }


def summarize_team_split_drivers(models: dict, X: pd.DataFrame, meta: dict) -> dict:
    combined_buckets = defaultdict(float)
    combined_features = []
    side_summaries = {}

    for side, model in models.items():
        summary = summarize_model_drivers(model, X, meta["features"])
        side_summaries[side] = summary
        for bucket, value in summary["bucket_totals"].items():
            combined_buckets[bucket] += float(value)
        for feat, value in summary["feature_contribs"]:
            combined_features.append((f"{side}:{feat}", float(value)))

    top_buckets = [
        (bucket, value)
        for bucket, value in sorted(combined_buckets.items(), key=lambda item: abs(item[1]), reverse=True)
        if abs(value) >= 0.05
    ][:3]
    top_features = [
        (feat, value)
        for feat, value in sorted(combined_features, key=lambda item: abs(item[1]), reverse=True)
        if abs(value) >= 0.05
    ][:5]

    return {
        "top_buckets": top_buckets,
        "top_features": top_features,
        "side_summaries": side_summaries,
    }


def _fill_missing_features(X: pd.DataFrame, meta: dict, month: int | None = None) -> pd.DataFrame:
    """Fill NaN features with training-set medians to avoid XGBoost default-branch bias.

    When *month* is provided and monthly medians are available, those are used
    first (reducing early-season bias from bullpen/weather features), falling
    back to global medians for any feature not covered.
    """
    global_medians = meta.get("feature_medians") or {}
    monthly_medians = (meta.get("monthly_feature_medians") or {}).get(str(month), {}) if month else {}
    if not global_medians and not monthly_medians:
        return X
    null_mask = X.isnull()
    if not null_mask.any().any():
        return X
    fill_map = {}
    for col in X.columns:
        if not null_mask[col].any():
            continue
        val = monthly_medians.get(col) if monthly_medians else None
        if val is None:
            val = global_medians.get(col)
        if val is not None:
            fill_map[col] = val
    if fill_map:
        X = X.fillna(fill_map)
    return X


def predict_point_outputs(model, meta: dict, feature_source: pd.DataFrame):
    feature_order = meta["features"]
    X = feature_source.reindex(columns=feature_order)
    # Determine month for month-specific median fill.
    month = None
    if "date" in feature_source.columns:
        dates = pd.to_datetime(feature_source["date"], errors="coerce")
        months = dates.dt.month.dropna().unique()
        if len(months) == 1:
            month = int(months[0])
    X = _fill_missing_features(X, meta, month=month)
    model_family = meta.get("model_family", "single_total")

    if model_family == "team_split":
        side_predictions = {}
        side_components = {}
        total_prediction = np.zeros(len(X), dtype=float)
        baseline_total = np.zeros(len(X), dtype=float)

        for side, side_meta in meta.get("side_models", {}).items():
            raw_component = model[side].predict(X)
            prediction_mode = side_meta.get("prediction_mode", "direct")
            base_feature = side_meta.get("base_feature")
            base = np.zeros(len(X), dtype=float)
            if prediction_mode == "dc_residual" and base_feature in feature_source.columns:
                base = feature_source[base_feature].to_numpy(dtype=float)
            side_total = raw_component + base if prediction_mode == "dc_residual" else raw_component
            side_predictions[side] = side_total
            side_components[side] = raw_component
            total_prediction += side_total
            baseline_total += base

        raw_total_prediction = total_prediction.copy()
        calibrated_total = apply_total_calibration(raw_total_prediction, meta.get("total_calibration"))
        if np.any(total_prediction != 0):
            scale = np.divide(
                calibrated_total,
                raw_total_prediction,
                out=np.ones_like(calibrated_total),
                where=np.abs(raw_total_prediction) > 1e-9,
            )
            for side in side_predictions:
                side_predictions[side] = side_predictions[side] * scale
        driver_summary = summarize_team_split_drivers(model, X, meta)
        return X, calibrated_total, {
            "model_family": model_family,
            "side_predictions": side_predictions,
            "side_components": side_components,
            "driver_summary": driver_summary,
            "baseline_total": baseline_total,
            "raw_total": raw_total_prediction,
            "calibrated_total": calibrated_total,
        }

    xgb_pred = model.predict(X)
    prediction_mode = meta.get("prediction_mode", "direct")
    base_feature = meta.get("base_feature")
    baseline_total = np.zeros(len(X), dtype=float)
    if prediction_mode == "dc_residual" and base_feature in feature_source.columns:
        baseline_total = feature_source[base_feature].to_numpy(dtype=float)
        total_prediction = xgb_pred + baseline_total
    else:
        total_prediction = xgb_pred

    raw_total_prediction = np.asarray(total_prediction, dtype=float)
    calibrated_total = apply_total_calibration(raw_total_prediction, meta.get("total_calibration"))
    driver_summary = summarize_model_drivers(model, X, feature_order)
    return X, calibrated_total, {
        "model_family": model_family,
        "side_predictions": {},
        "side_components": {},
        "driver_summary": driver_summary,
        "baseline_total": baseline_total,
        "prediction_mode": prediction_mode,
        "xgb_component": xgb_pred,
        "raw_total": raw_total_prediction,
        "calibrated_total": calibrated_total,
    }


def get_side_residual_distribution(meta: dict) -> dict:
    cfg = dict(meta.get("side_residual_distribution") or {})
    if not cfg:
        return {
            "enabled": False,
            "source": "missing",
            "samples": 0,
            "rho": 0.0,
        }

    rho = float(cfg.get("rho", 0.0))
    cfg["rho"] = float(np.clip(rho, -0.999, 0.999))
    cfg.setdefault("enabled", True)
    cfg.setdefault("source", "metadata")
    cfg.setdefault("samples", 0)
    return cfg


def compute_residual_std(model, meta, data_dir: str = DATA_DIR) -> float:
    path = os.path.join(data_dir, "mlb_model_data.tsv")
    df = pd.read_csv(path, sep="\t", parse_dates=["date"])
    y = df["total_runs"]
    _, preds, _ = predict_point_outputs(model, meta, df)
    residuals = y - preds
    return float(residuals.std())


def estimate_prediction_std(
    feature_row: pd.DataFrame,
    point_prediction: float,
    uncertainty_model,
    uncertainty_cfg: dict | None,
    fallback_std: float,
) -> float:
    if uncertainty_model is None or not uncertainty_cfg:
        return float(fallback_std)

    unc_features = uncertainty_cfg.get("features") or []
    if not unc_features:
        return float(fallback_std)

    unc_row = feature_row.copy()
    unc_row["point_prediction"] = float(point_prediction)
    unc_row = unc_row.reindex(columns=unc_features, fill_value=0.0)

    raw_abs = float(np.clip(uncertainty_model.predict(unc_row)[0], 0.1, None))
    sigma = raw_abs * float(uncertainty_cfg.get("mae_to_sigma_scale", 1.2533))
    min_sigma = float(uncertainty_cfg.get("min_sigma", fallback_std))
    max_sigma = float(uncertainty_cfg.get("max_sigma", fallback_std))
    return float(np.clip(sigma, min_sigma, max_sigma))


def predict_high_tail_prob(
    feature_row: pd.DataFrame,
    point_prediction: float,
    high_tail_model,
    high_tail_cfg: dict | None,
):
    if high_tail_model is None or not high_tail_cfg or not high_tail_cfg.get("enabled"):
        return None

    tail_features = build_high_tail_features(feature_row, [point_prediction])
    tail_features = tail_features.reindex(
        columns=high_tail_cfg.get("features", tail_features.columns.tolist()),
        fill_value=0.0,
    )
    raw_prob = high_tail_model.predict_proba(tail_features)[0, 1]
    return float(apply_high_tail_calibration([raw_prob], high_tail_cfg)[0])


def predict_low_tail_prob(
    feature_row: pd.DataFrame,
    point_prediction: float,
    low_tail_model,
    low_tail_cfg: dict | None,
):
    if low_tail_model is None or not low_tail_cfg or not low_tail_cfg.get("enabled"):
        return None

    tail_features = build_low_tail_features(feature_row, [point_prediction])
    tail_features = tail_features.reindex(
        columns=low_tail_cfg.get("features", tail_features.columns.tolist()),
        fill_value=0.0,
    )
    raw_prob = low_tail_model.predict_proba(tail_features)[0, 1]
    return float(apply_low_tail_calibration([raw_prob], low_tail_cfg)[0])
