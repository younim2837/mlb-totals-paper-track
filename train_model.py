"""
MLB Team-Split Runs Prediction Model
Trains separate XGBoost regressors for home and away runs, then sums them to
produce total-run predictions.

Outputs:
  - models/home_runs_xgb.json
  - models/away_runs_xgb.json
  - models/total_runs_uncertainty_xgb.json
  - models/model_meta.json
"""

import json
import os
from math import pi, sqrt

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from market_adjustment import (
    MARKET_EDGE_MODEL_FILE,
    apply_market_context,
    fit_market_edge_model,
    fit_market_shrinkage,
    load_historical_lines,
    load_historical_line_snapshots,
    merge_predictions_with_lines,
    merge_predictions_with_snapshots,
    probability_over_current_line,
)
from modeling_utils import (
    apply_high_tail_calibration,
    apply_low_tail_calibration,
    apply_total_calibration,
    build_high_tail_features,
    build_low_tail_features,
    clip_probabilities,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

EXCLUDE_COLS = [
    "game_id", "date", "home_team", "away_team", "venue", "venue_id",
    "home_pitcher", "away_pitcher", "total_runs", "home_score", "away_score",
    "doubleheader", "game_num", "hp_umpire", "ump_games_called",
]
EXCLUDE_SUBSTRINGS = [
    "_bullpen_used_",
    "_bullpen_heavy_arms_",
    "_bullpen_app_count_",
    "_bullpen_arms_yesterday",
    "_bullpen_b2b_arms",
    "_bullpen_fatigue_score",
]

TOTAL_TARGET = "total_runs"
MODEL_FAMILY = "team_split"
UNCERTAINTY_MODEL_FILE = "total_runs_uncertainty_xgb.json"
UNCERTAINTY_POINT_FEATURE = "point_prediction"
UNCERTAINTY_MIN_SIGMA = 2.5
UNCERTAINTY_MAX_SIGMA = 6.0
HIGH_TAIL_MODEL_FILE = "total_runs_high_tail_xgb.json"
HIGH_TAIL_TARGET_LINE = 9.5
HIGH_TAIL_POINT_FEATURE = "point_prediction"
HIGH_TAIL_MAX_SIGMA_MULTIPLIER = 1.8
LOW_TAIL_MODEL_FILE = "total_runs_low_tail_xgb.json"
LOW_TAIL_TARGET_LINE = 7.5
LOW_TAIL_POINT_FEATURE = "point_prediction"
LOW_TAIL_MAX_SIGMA_MULTIPLIER = 1.8
CALIBRATION_MIN_SAMPLES = 500
CALIBRATION_MIN_STD_RETENTION = 0.85
TAIL_CALIBRATION_MIN_STD_RETENTION = 0.60
TAIL_QUANTILE_LEVELS = [0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98]
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
TRAIN_START_YEAR = 2021
BACKTEST_YEAR = 2025

SIDE_SPECS = {
    "home": {
        "target": "home_score",
        "base_feature": "dc_lambda_home",
        "model_file": "home_runs_xgb.json",
    },
    "away": {
        "target": "away_score",
        "base_feature": "dc_lambda_away",
        "model_file": "away_runs_xgb.json",
    },
}

POINT_MODEL_PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
}

UNCERTAINTY_MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_alpha": 0.2,
    "reg_lambda": 2.0,
    "random_state": 42,
}

HIGH_TAIL_MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_alpha": 0.2,
    "reg_lambda": 2.0,
    "random_state": 42,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
}


def load_data():
    path = os.path.join(DATA_DIR, "mlb_model_data.tsv")
    df = pd.read_csv(path, sep="\t", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def build_point_model(early_stopping_rounds=None):
    params = POINT_MODEL_PARAMS.copy()
    if early_stopping_rounds is not None:
        params["early_stopping_rounds"] = early_stopping_rounds
    return xgb.XGBRegressor(**params)


def build_uncertainty_model(early_stopping_rounds=None):
    params = UNCERTAINTY_MODEL_PARAMS.copy()
    if early_stopping_rounds is not None:
        params["early_stopping_rounds"] = early_stopping_rounds
    return xgb.XGBRegressor(**params)


def build_high_tail_model(early_stopping_rounds=None):
    params = HIGH_TAIL_MODEL_PARAMS.copy()
    if early_stopping_rounds is not None:
        params["early_stopping_rounds"] = early_stopping_rounds
    return xgb.XGBClassifier(**params)


def get_feature_cols(df):
    feature_cols = []
    ignored_non_numeric = []

    for c in df.columns:
        if c in EXCLUDE_COLS or any(substr in c for substr in EXCLUDE_SUBSTRINGS):
            continue
        if pd.api.types.is_bool_dtype(df[c]) or pd.api.types.is_numeric_dtype(df[c]):
            feature_cols.append(c)
        else:
            ignored_non_numeric.append(c)

    if ignored_non_numeric:
        print(f"  Ignoring non-numeric columns: {ignored_non_numeric}")

    return feature_cols


def train_test_split_by_time(df, train_start_year=TRAIN_START_YEAR, test_year=BACKTEST_YEAR):
    train = df[(df["date"].dt.year >= train_start_year) & (df["date"].dt.year < test_year)].copy()
    test = df[df["date"].dt.year == test_year].copy()
    return train, test


def get_side_prediction_mode(df: pd.DataFrame, base_feature: str):
    if base_feature in df.columns:
        return "dc_residual"
    return "direct"


def build_side_target(df: pd.DataFrame, target_col: str, prediction_mode: str, base_feature: str):
    if prediction_mode == "dc_residual":
        return df[target_col] - df[base_feature]
    return df[target_col]


def predict_side_values(model, X, prediction_mode, base=None):
    preds = model.predict(X)
    if prediction_mode == "dc_residual" and base is not None:
        preds = preds + np.asarray(base)
    return preds


def predict_team_split(models: dict, X: pd.DataFrame, df: pd.DataFrame, side_meta: dict):
    outputs = {}
    total = np.zeros(len(X), dtype=float)

    for side, model in models.items():
        meta = side_meta[side]
        base = None
        if meta["prediction_mode"] == "dc_residual" and meta["base_feature"] in df.columns:
            base = df[meta["base_feature"]].to_numpy()
        preds = predict_side_values(model, X, meta["prediction_mode"], base=base)
        outputs[side] = preds
        total += preds

    return outputs, total


def apply_total_calibration_to_team_split(side_predictions: dict, total_predictions, calibration_cfg: dict | None):
    raw_total = np.asarray(total_predictions, dtype=float)
    calibrated_total = np.asarray(apply_total_calibration(raw_total, calibration_cfg), dtype=float)
    scale = np.divide(
        calibrated_total,
        raw_total,
        out=np.ones_like(calibrated_total),
        where=np.abs(raw_total) > 1e-9,
    )
    calibrated_sides = {
        side: np.asarray(preds, dtype=float) * scale
        for side, preds in side_predictions.items()
    }
    return calibrated_sides, calibrated_total


def fit_total_calibration(train: pd.DataFrame, feature_cols: list[str], side_meta: dict, point_oof_raw=None):
    if point_oof_raw is None:
        point_oof_raw = build_point_oof_artifacts(train, feature_cols, side_meta)["oof_total_raw"]
    oof_total = pd.Series(point_oof_raw, index=train.index, dtype=float)

    valid = oof_total.notna()
    if int(valid.sum()) < CALIBRATION_MIN_SAMPLES:
        return {
            "enabled": False,
            "method": "none",
            "samples": int(valid.sum()),
        }

    raw = oof_total.loc[valid].to_numpy(dtype=float)
    actual = train.loc[valid, TOTAL_TARGET].to_numpy(dtype=float)

    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=max(actual.min(), 0.0))
    isotonic_full = calibrator.fit_transform(raw, actual)
    before_mae = mean_absolute_error(actual, raw)
    raw_std = float(np.std(raw))

    candidates = []
    for alpha in np.linspace(0.0, 1.0, 21):
        blended = raw + alpha * (isotonic_full - raw)
        blended_std = float(np.std(blended))
        if raw_std > 0 and blended_std < raw_std * CALIBRATION_MIN_STD_RETENTION:
            continue
        candidates.append((alpha, blended, blended_std, mean_absolute_error(actual, blended)))

    if candidates:
        best_alpha, calibrated, calibrated_std, after_mae = min(
            candidates,
            key=lambda item: (item[3], item[0]),
        )
    else:
        best_alpha = 0.0
        calibrated = raw
        calibrated_std = raw_std
        after_mae = before_mae

    base_blended = raw + best_alpha * (isotonic_full - raw)
    actual_upper_tail = float(np.mean(actual > 9.5))
    actual_lower_tail = float(np.mean(actual < 7.5))
    actual_std = float(np.std(actual))
    tail_x = np.quantile(base_blended, TAIL_QUANTILE_LEVELS)
    tail_y = np.quantile(actual, TAIL_QUANTILE_LEVELS)

    tail_candidates = []
    for tail_alpha in np.linspace(0.0, 0.60, 13):
        tail_mapped = np.interp(base_blended, tail_x, tail_y, left=tail_y[0], right=tail_y[-1])
        stretched = base_blended + tail_alpha * (tail_mapped - base_blended)
        score = (
            mean_absolute_error(actual, stretched) +
            1.50 * abs(float(np.mean(stretched > 9.5)) - actual_upper_tail) +
            0.60 * abs(float(np.mean(stretched < 7.5)) - actual_lower_tail) +
            0.08 * abs(float(np.std(stretched)) - actual_std)
        )
        tail_candidates.append((tail_alpha, stretched, score))

    best_tail_alpha, tail_calibrated, tail_score = min(
        tail_candidates,
        key=lambda item: (item[2], item[0]),
    )

    return {
        "enabled": True,
        "method": "hybrid_total",
        "samples": int(valid.sum()),
        "alpha": float(best_alpha),
        "x": [float(v) for v in calibrator.X_thresholds_],
        "y": [float(v) for v in calibrator.y_thresholds_],
        "oof_mae_before": float(before_mae),
        "oof_mae_after": float(after_mae),
        "oof_std_before": raw_std,
        "oof_std_after": float(calibrated_std),
        "tail_alpha": float(best_tail_alpha),
        "tail_x": [float(v) for v in tail_x],
        "tail_y": [float(v) for v in tail_y],
        "tail_oof_mae": float(mean_absolute_error(actual, tail_calibrated)),
        "tail_oof_std": float(np.std(tail_calibrated)),
        "tail_oof_upper_rate": float(np.mean(tail_calibrated > 9.5)),
        "tail_oof_lower_rate": float(np.mean(tail_calibrated < 7.5)),
        "tail_actual_upper_rate": actual_upper_tail,
        "tail_actual_lower_rate": actual_lower_tail,
        "tail_objective": float(tail_score),
    }


def _fill_oof_missing_features(X: pd.DataFrame, fold_train: pd.DataFrame,
                               feature_cols: list[str], month: pd.Series | None = None) -> pd.DataFrame:
    """Fill NaN features with fold-train medians, matching inference-time behaviour."""
    medians = fold_train[feature_cols].median()
    fill_map = {}
    null_mask = X.isnull()
    if not null_mask.any().any():
        return X
    for col in X.columns:
        if not null_mask[col].any():
            continue
        val = medians.get(col)
        if pd.notna(val):
            fill_map[col] = float(val)
    if fill_map:
        X = X.fillna(fill_map)
    return X


def build_point_oof_artifacts(train: pd.DataFrame, feature_cols: list[str], side_meta: dict) -> dict:
    """
    Train the fold point models once and reuse their predictions across the
    calibration, uncertainty, and tail-model layers.
    """
    X_train = train[feature_cols]
    tscv = TimeSeriesSplit(n_splits=5)
    oof_total = pd.Series(np.nan, index=train.index, dtype=float)
    oof_sides = {
        side: pd.Series(np.nan, index=train.index, dtype=float)
        for side in SIDE_SPECS
    }
    folds = []

    for train_idx, val_idx in tscv.split(X_train):
        fold_train = train.iloc[train_idx]
        fold_val = train.iloc[val_idx]
        fold_models = {}

        for side, spec in SIDE_SPECS.items():
            y_tr = build_side_target(
                fold_train,
                spec["target"],
                side_meta[side]["prediction_mode"],
                spec["base_feature"],
            )
            y_val = build_side_target(
                fold_val,
                spec["target"],
                side_meta[side]["prediction_mode"],
                spec["base_feature"],
            )
            model = build_point_model()
            model.fit(
                fold_train[feature_cols],
                y_tr,
                eval_set=[(fold_val[feature_cols], y_val)],
                verbose=False,
            )
            fold_models[side] = model

        # Apply median fill before predicting, matching inference-time pipeline.
        fold_train_X_filled = _fill_oof_missing_features(
            fold_train[feature_cols].copy(), fold_train, feature_cols,
        )
        fold_val_X_filled = _fill_oof_missing_features(
            fold_val[feature_cols].copy(), fold_train, feature_cols,
        )
        fold_train_side_raw, fold_train_point_raw = predict_team_split(
            fold_models,
            fold_train_X_filled,
            fold_train,
            side_meta,
        )
        fold_val_side_raw, fold_val_point_raw = predict_team_split(
            fold_models,
            fold_val_X_filled,
            fold_val,
            side_meta,
        )
        oof_total.iloc[val_idx] = fold_val_point_raw
        for side in SIDE_SPECS:
            oof_sides[side].iloc[val_idx] = np.asarray(fold_val_side_raw[side], dtype=float)
        folds.append(
            {
                "train_idx": train_idx,
                "val_idx": val_idx,
                "train_side_raw": {
                    side: np.asarray(preds, dtype=float)
                    for side, preds in fold_train_side_raw.items()
                },
                "val_side_raw": {
                    side: np.asarray(preds, dtype=float)
                    for side, preds in fold_val_side_raw.items()
                },
                "train_point_raw": np.asarray(fold_train_point_raw, dtype=float),
                "val_point_raw": np.asarray(fold_val_point_raw, dtype=float),
            }
        )

    return {
        "oof_total_raw": oof_total,
        "oof_side_raw": oof_sides,
        "folds": folds,
    }


def estimate_side_residual_distribution(
    train: pd.DataFrame,
    point_oof_artifacts: dict,
    calibration_cfg: dict | None = None,
) -> dict:
    oof_total = pd.Series(point_oof_artifacts["oof_total_raw"], index=train.index, dtype=float)
    oof_side_raw = {
        side: pd.Series(point_oof_artifacts["oof_side_raw"][side], index=train.index, dtype=float)
        for side in SIDE_SPECS
    }
    valid = oof_total.notna()
    for side in SIDE_SPECS:
        valid &= oof_side_raw[side].notna()

    samples = int(valid.sum())
    if samples < 2:
        return {
            "enabled": False,
            "samples": samples,
            "source": "oof",
        }

    side_preds, calibrated_total = apply_total_calibration_to_team_split(
        {side: oof_side_raw[side].loc[valid].to_numpy(dtype=float) for side in SIDE_SPECS},
        oof_total.loc[valid].to_numpy(dtype=float),
        calibration_cfg,
    )
    actual_home = train.loc[valid, SIDE_SPECS["home"]["target"]].to_numpy(dtype=float)
    actual_away = train.loc[valid, SIDE_SPECS["away"]["target"]].to_numpy(dtype=float)
    residual_home = actual_home - side_preds["home"]
    residual_away = actual_away - side_preds["away"]

    home_sigma = float(np.std(residual_home, ddof=1))
    away_sigma = float(np.std(residual_away, ddof=1))
    covariance = float(np.cov(residual_home, residual_away, ddof=1)[0, 1])
    denom = home_sigma * away_sigma
    rho = covariance / denom if denom > 0 else 0.0
    rho = float(np.clip(rho, -0.999, 0.999))

    return {
        "enabled": True,
        "source": "oof",
        "samples": samples,
        "calibration_applied": bool(calibration_cfg and calibration_cfg.get("enabled")),
        "home_sigma": home_sigma,
        "away_sigma": away_sigma,
        "home_bias": float(np.mean(residual_home)),
        "away_bias": float(np.mean(residual_away)),
        "home_mae": float(np.mean(np.abs(residual_home))),
        "away_mae": float(np.mean(np.abs(residual_away))),
        "covariance": covariance,
        "rho": rho,
        "margin_sigma": float(np.std(residual_home - residual_away, ddof=1)),
        "total_sigma": float(np.std((actual_home + actual_away) - calibrated_total, ddof=1)),
    }


def build_uncertainty_features(X: pd.DataFrame, point_predictions) -> pd.DataFrame:
    unc = X.copy()
    unc[UNCERTAINTY_POINT_FEATURE] = np.asarray(point_predictions)
    return unc


def evaluate(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    if label:
        print(f"  {label} MAE: {mae:.2f}")
        print(f"  {label} RMSE: {rmse:.2f}")
        print(f"  {label} R²: {r2:.3f}")
    else:
        for line in [7.5, 8.5, 9.5]:
            pred_over = y_pred > line
            actual_over = y_true > line
            correct = (pred_over == actual_over).mean()
            print(f"  Over/Under {line} accuracy: {correct:.1%}")
        print(f"  MAE: {mae:.2f} runs")
        print(f"  RMSE: {rmse:.2f} runs")
        print(f"  R²: {r2:.3f}")
        print(f"  Mean predicted: {y_pred.mean():.2f}, Mean actual: {y_true.mean():.2f}")

    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def evaluate_market_strategy(
    test_matched: pd.DataFrame,
    shrink_cfg: dict | None,
    market_edge_model=None,
    market_edge_cfg: dict | None = None,
    high_tail_cfg: dict | None = None,
    low_tail_cfg: dict | None = None,
) -> dict:
    if test_matched.empty:
        return {
            "matched_games": 0,
            "mae_before": None,
            "mae_after": None,
            "brier_before": None,
            "brier_after": None,
            "win_rate_before": None,
            "win_rate_after": None,
            "method": None,
        }

    df = test_matched.copy()
    adjusted = []
    probs_before = []
    probs_after = []

    for row in df.itertuples(index=False):
        probs_before.append(
            probability_over_current_line(
                mean_total=row.predicted_total,
                base_sigma=row.prediction_std,
                line=row.close_total_line,
                high_tail_prob=getattr(row, "high_tail_prob_9p5", None),
                high_tail_cfg=high_tail_cfg,
                low_tail_prob=getattr(row, "low_tail_prob_7p5", None),
                low_tail_cfg=low_tail_cfg,
            )
        )
        result = apply_market_context(
            predicted_total=row.predicted_total,
            market_line=row.close_total_line,
            cfg=MARKET_SHRINK_BASE_CFG,
            prediction_std=row.prediction_std,
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
            learned_shrink_cfg=shrink_cfg,
        )
        adjusted.append(float(result["adjusted_total"]))
        probs_after.append(float(result["p_over"]))

    df["market_adjusted_total"] = adjusted
    actual_over = (df["actual_total"] > df["close_total_line"]).astype(int)
    probs_before = np.clip(np.asarray(probs_before, dtype=float), 1e-6, 1 - 1e-6)
    probs_after = np.clip(np.asarray(probs_after, dtype=float), 1e-6, 1 - 1e-6)

    return {
        "matched_games": int(len(df)),
        "mae_before": float(mean_absolute_error(df["actual_total"], df["predicted_total"])),
        "mae_after": float(mean_absolute_error(df["actual_total"], df["market_adjusted_total"])),
        "brier_before": float(brier_score_loss(actual_over, np.clip(probs_before, 1e-6, 1 - 1e-6))),
        "brier_after": float(brier_score_loss(actual_over, np.clip(probs_after, 1e-6, 1 - 1e-6))),
        "win_rate_before": float((((probs_before >= 0.5).astype(int)) == actual_over).mean()),
        "win_rate_after": float((((probs_after >= 0.5).astype(int)) == actual_over).mean()),
        "method": (
            "edge_model"
            if market_edge_cfg and market_edge_cfg.get("enabled")
            else ("shrinkage" if shrink_cfg and shrink_cfg.get("enabled") else "base_model")
        ),
    }


def train_side_model(train_df: pd.DataFrame, feature_cols: list[str], side: str):
    spec = SIDE_SPECS[side]
    prediction_mode = get_side_prediction_mode(train_df, spec["base_feature"])
    X_train = train_df[feature_cols]
    y_train = build_side_target(train_df, spec["target"], prediction_mode, spec["base_feature"])
    split_idx = max(int(len(X_train) * 0.8), 1)
    split_idx = min(split_idx, len(X_train) - 1) if len(X_train) > 1 else 1

    model = build_point_model(early_stopping_rounds=30)
    if len(X_train) > 1 and split_idx < len(X_train):
        model.fit(
            X_train.iloc[:split_idx],
            y_train.iloc[:split_idx],
            eval_set=[(X_train.iloc[split_idx:], y_train.iloc[split_idx:])],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train, verbose=False)

    meta = {
        "target": spec["target"],
        "base_feature": spec["base_feature"],
        "prediction_mode": prediction_mode,
        "file": spec["model_file"],
        "best_iteration": getattr(model, "best_iteration", None),
    }
    return model, meta


def cross_validate_team_split(train_df: pd.DataFrame, feature_cols: list[str]):
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    side_meta = {
        side: {
            "prediction_mode": get_side_prediction_mode(train_df, spec["base_feature"]),
            "base_feature": spec["base_feature"],
        }
        for side, spec in SIDE_SPECS.items()
    }

    X_train = train_df[feature_cols]
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
        fold_train = train_df.iloc[train_idx]
        fold_val = train_df.iloc[val_idx]
        fold_models = {}

        for side, spec in SIDE_SPECS.items():
            y_tr = build_side_target(
                fold_train,
                spec["target"],
                side_meta[side]["prediction_mode"],
                spec["base_feature"],
            )
            y_val = build_side_target(
                fold_val,
                spec["target"],
                side_meta[side]["prediction_mode"],
                spec["base_feature"],
            )
            model = build_point_model()
            model.fit(
                fold_train[feature_cols],
                y_tr,
                eval_set=[(fold_val[feature_cols], y_val)],
                verbose=False,
            )
            fold_models[side] = model

        _, total_pred = predict_team_split(fold_models, fold_val[feature_cols], fold_val, side_meta)
        mae = mean_absolute_error(fold_val[TOTAL_TARGET], total_pred)
        scores.append(mae)
        print(f"  Fold {fold}: Total-run MAE = {mae:.2f}")

    print(f"  Avg CV MAE: {np.mean(scores):.2f} (+/- {np.std(scores):.2f})")
    return scores


def train_uncertainty_model(
    train: pd.DataFrame,
    feature_cols: list[str],
    side_meta: dict,
    calibration_cfg: dict | None = None,
    point_oof_raw=None,
):
    X_train = train[feature_cols]
    if point_oof_raw is None:
        point_oof_raw = build_point_oof_artifacts(train, feature_cols, side_meta)["oof_total_raw"]
    oof_total = pd.Series(
        apply_total_calibration(point_oof_raw, calibration_cfg),
        index=train.index,
        dtype=float,
    )

    valid = oof_total.notna()
    unc_target = (train.loc[valid, TOTAL_TARGET] - oof_total.loc[valid]).abs().clip(lower=0.1)
    unc_X = build_uncertainty_features(X_train.loc[valid], oof_total.loc[valid])

    split_idx = max(int(len(unc_X) * 0.8), 1)
    split_idx = min(split_idx, len(unc_X) - 1) if len(unc_X) > 1 else 1
    unc_model = build_uncertainty_model(early_stopping_rounds=20)
    if len(unc_X) > 1 and split_idx < len(unc_X):
        unc_model.fit(
            unc_X.iloc[:split_idx],
            unc_target.iloc[:split_idx],
            eval_set=[(unc_X.iloc[split_idx:], unc_target.iloc[split_idx:])],
            verbose=False,
        )
    else:
        unc_model.fit(unc_X, unc_target, verbose=False)

    raw_abs = np.clip(unc_model.predict(unc_X), 0.1, None)
    mean_raw = float(np.mean(raw_abs))
    mean_abs = float(np.mean(unc_target))
    mae_to_sigma_scale = sqrt(pi / 2) * (mean_abs / mean_raw if mean_raw > 0 else 1.0)

    return unc_model, {
        "enabled": True,
        "file": UNCERTAINTY_MODEL_FILE,
        "features": feature_cols + [UNCERTAINTY_POINT_FEATURE],
        "target": "absolute_error",
        "mae_to_sigma_scale": mae_to_sigma_scale,
        "min_sigma": UNCERTAINTY_MIN_SIGMA,
        "max_sigma": UNCERTAINTY_MAX_SIGMA,
        "best_iteration": getattr(unc_model, "best_iteration", None),
    }


def predict_sigmas(uncertainty_model, uncertainty_cfg, X: pd.DataFrame, point_predictions) -> np.ndarray:
    unc_X = build_uncertainty_features(X, point_predictions)
    raw_abs = np.clip(uncertainty_model.predict(unc_X), 0.1, None)
    sigmas = raw_abs * float(uncertainty_cfg["mae_to_sigma_scale"])
    return np.clip(
        sigmas,
        float(uncertainty_cfg["min_sigma"]),
        float(uncertainty_cfg["max_sigma"]),
    )


def train_high_tail_model(
    train: pd.DataFrame,
    feature_cols: list[str],
    side_meta: dict,
    calibration_cfg: dict | None = None,
    point_oof_artifacts: dict | None = None,
    full_point_predictions=None,
):
    X_train = train[feature_cols]
    target = (train[TOTAL_TARGET] > HIGH_TAIL_TARGET_LINE).astype(int)
    oof_probs = pd.Series(np.nan, index=train.index, dtype=float)
    point_oof_artifacts = point_oof_artifacts or build_point_oof_artifacts(train, feature_cols, side_meta)

    for fold in point_oof_artifacts["folds"]:
        train_idx = fold["train_idx"]
        val_idx = fold["val_idx"]
        fold_train = train.iloc[train_idx]
        fold_val = train.iloc[val_idx]

        fold_train_point = apply_total_calibration(fold["train_point_raw"], calibration_cfg)
        fold_val_point = apply_total_calibration(fold["val_point_raw"], calibration_cfg)

        tail_model = build_high_tail_model()
        tail_model.fit(
            build_high_tail_features(fold_train[feature_cols], fold_train_point),
            target.iloc[train_idx],
            eval_set=[(
                build_high_tail_features(fold_val[feature_cols], fold_val_point),
                target.iloc[val_idx],
            )],
            verbose=False,
        )
        fold_probs = tail_model.predict_proba(
            build_high_tail_features(fold_val[feature_cols], fold_val_point)
        )[:, 1]
        oof_probs.iloc[val_idx] = fold_probs

    valid = oof_probs.notna()
    if int(valid.sum()) < CALIBRATION_MIN_SAMPLES:
        return None, {
            "enabled": False,
            "file": HIGH_TAIL_MODEL_FILE,
            "line": HIGH_TAIL_TARGET_LINE,
            "samples": int(valid.sum()),
        }

    raw_probs = clip_probabilities(oof_probs.loc[valid].to_numpy(dtype=float))
    actual = target.loc[valid].to_numpy(dtype=int)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic_probs = clip_probabilities(calibrator.fit_transform(raw_probs, actual))
    raw_brier = float(brier_score_loss(actual, raw_probs))
    raw_std = float(np.std(raw_probs))

    candidates = []
    for alpha in np.linspace(0.0, 1.0, 21):
        blended = clip_probabilities(raw_probs + alpha * (isotonic_probs - raw_probs))
        blended_std = float(np.std(blended))
        if raw_std > 0 and blended_std < raw_std * TAIL_CALIBRATION_MIN_STD_RETENTION:
            continue
        brier = float(brier_score_loss(actual, blended))
        ll = float(log_loss(actual, blended, labels=[0, 1]))
        candidates.append((alpha, blended, brier, ll, blended_std))

    if candidates:
        best_alpha, calibrated_probs, best_brier, best_logloss, calibrated_std = min(
            candidates,
            key=lambda item: (item[2], item[3], item[0]),
        )
    else:
        best_alpha = 0.0
        calibrated_probs = raw_probs
        best_brier = raw_brier
        best_logloss = float(log_loss(actual, raw_probs, labels=[0, 1]))
        calibrated_std = raw_std

    if full_point_predictions is None:
        full_point_predictions = apply_total_calibration(
            build_point_oof_artifacts(train, feature_cols, side_meta)["oof_total_raw"],
            calibration_cfg,
        )
    full_point = np.asarray(full_point_predictions, dtype=float)
    final_model = build_high_tail_model(early_stopping_rounds=20)
    tail_X = build_high_tail_features(train[feature_cols], full_point)
    split_idx = max(int(len(tail_X) * 0.8), 1)
    split_idx = min(split_idx, len(tail_X) - 1) if len(tail_X) > 1 else 1
    if len(tail_X) > 1 and split_idx < len(tail_X):
        final_model.fit(
            tail_X.iloc[:split_idx],
            target.iloc[:split_idx],
            eval_set=[(tail_X.iloc[split_idx:], target.iloc[split_idx:])],
            verbose=False,
        )
    else:
        final_model.fit(tail_X, target, verbose=False)

    cfg = {
        "enabled": True,
        "file": HIGH_TAIL_MODEL_FILE,
        "line": HIGH_TAIL_TARGET_LINE,
        "features": feature_cols + [HIGH_TAIL_POINT_FEATURE],
        "alpha": float(best_alpha),
        "x": [float(v) for v in calibrator.X_thresholds_],
        "y": [float(v) for v in calibrator.y_thresholds_],
        "sigma_cap_multiplier": HIGH_TAIL_MAX_SIGMA_MULTIPLIER,
        "samples": int(valid.sum()),
        "oof_rate_pred_before": float(raw_probs.mean()),
        "oof_rate_pred_after": float(calibrated_probs.mean()),
        "oof_rate_actual": float(actual.mean()),
        "oof_brier_before": raw_brier,
        "oof_brier_after": best_brier,
        "oof_logloss_after": best_logloss,
        "oof_std_before": raw_std,
        "oof_std_after": calibrated_std,
        "best_iteration": getattr(final_model, "best_iteration", None),
    }
    return final_model, cfg


def predict_high_tail_probs(tail_model, tail_cfg, X: pd.DataFrame, point_predictions) -> np.ndarray:
    if tail_model is None or not tail_cfg or not tail_cfg.get("enabled"):
        return np.full(len(X), np.nan, dtype=float)
    tail_X = build_high_tail_features(X, point_predictions)
    tail_X = tail_X.reindex(columns=tail_cfg.get("features", tail_X.columns.tolist()), fill_value=0.0)
    probs = tail_model.predict_proba(tail_X)[:, 1]
    return apply_high_tail_calibration(probs, tail_cfg)


def train_low_tail_model(
    train: pd.DataFrame,
    feature_cols: list[str],
    side_meta: dict,
    calibration_cfg: dict | None = None,
    point_oof_artifacts: dict | None = None,
    full_point_predictions=None,
):
    X_train = train[feature_cols]
    target = (train[TOTAL_TARGET] < LOW_TAIL_TARGET_LINE).astype(int)
    oof_probs = pd.Series(np.nan, index=train.index, dtype=float)
    point_oof_artifacts = point_oof_artifacts or build_point_oof_artifacts(train, feature_cols, side_meta)

    for fold in point_oof_artifacts["folds"]:
        train_idx = fold["train_idx"]
        val_idx = fold["val_idx"]
        fold_train = train.iloc[train_idx]
        fold_val = train.iloc[val_idx]

        fold_train_point = apply_total_calibration(fold["train_point_raw"], calibration_cfg)
        fold_val_point = apply_total_calibration(fold["val_point_raw"], calibration_cfg)

        tail_model = build_high_tail_model()
        tail_model.fit(
            build_low_tail_features(fold_train[feature_cols], fold_train_point),
            target.iloc[train_idx],
            eval_set=[(
                build_low_tail_features(fold_val[feature_cols], fold_val_point),
                target.iloc[val_idx],
            )],
            verbose=False,
        )
        fold_probs = tail_model.predict_proba(
            build_low_tail_features(fold_val[feature_cols], fold_val_point)
        )[:, 1]
        oof_probs.iloc[val_idx] = fold_probs

    valid = oof_probs.notna()
    if int(valid.sum()) < CALIBRATION_MIN_SAMPLES:
        return None, {
            "enabled": False,
            "file": LOW_TAIL_MODEL_FILE,
            "line": LOW_TAIL_TARGET_LINE,
            "samples": int(valid.sum()),
        }

    raw_probs = clip_probabilities(oof_probs.loc[valid].to_numpy(dtype=float))
    actual = target.loc[valid].to_numpy(dtype=int)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic_probs = clip_probabilities(calibrator.fit_transform(raw_probs, actual))
    raw_brier = float(brier_score_loss(actual, raw_probs))
    raw_std = float(np.std(raw_probs))

    candidates = []
    for alpha in np.linspace(0.0, 1.0, 21):
        blended = clip_probabilities(raw_probs + alpha * (isotonic_probs - raw_probs))
        blended_std = float(np.std(blended))
        if raw_std > 0 and blended_std < raw_std * TAIL_CALIBRATION_MIN_STD_RETENTION:
            continue
        brier = float(brier_score_loss(actual, blended))
        ll = float(log_loss(actual, blended, labels=[0, 1]))
        candidates.append((alpha, blended, brier, ll, blended_std))

    if candidates:
        best_alpha, calibrated_probs, best_brier, best_logloss, calibrated_std = min(
            candidates,
            key=lambda item: (item[2], item[3], item[0]),
        )
    else:
        best_alpha = 0.0
        calibrated_probs = raw_probs
        best_brier = raw_brier
        best_logloss = float(log_loss(actual, raw_probs, labels=[0, 1]))
        calibrated_std = raw_std

    if full_point_predictions is None:
        full_point_predictions = apply_total_calibration(
            build_point_oof_artifacts(train, feature_cols, side_meta)["oof_total_raw"],
            calibration_cfg,
        )
    full_point = np.asarray(full_point_predictions, dtype=float)
    final_model = build_high_tail_model(early_stopping_rounds=20)
    tail_X = build_low_tail_features(train[feature_cols], full_point)
    split_idx = max(int(len(tail_X) * 0.8), 1)
    split_idx = min(split_idx, len(tail_X) - 1) if len(tail_X) > 1 else 1
    if len(tail_X) > 1 and split_idx < len(tail_X):
        final_model.fit(
            tail_X.iloc[:split_idx],
            target.iloc[:split_idx],
            eval_set=[(tail_X.iloc[split_idx:], target.iloc[split_idx:])],
            verbose=False,
        )
    else:
        final_model.fit(tail_X, target, verbose=False)

    cfg = {
        "enabled": True,
        "file": LOW_TAIL_MODEL_FILE,
        "line": LOW_TAIL_TARGET_LINE,
        "features": feature_cols + [LOW_TAIL_POINT_FEATURE],
        "alpha": float(best_alpha),
        "x": [float(v) for v in calibrator.X_thresholds_],
        "y": [float(v) for v in calibrator.y_thresholds_],
        "sigma_cap_multiplier": LOW_TAIL_MAX_SIGMA_MULTIPLIER,
        "samples": int(valid.sum()),
        "oof_rate_pred_before": float(raw_probs.mean()),
        "oof_rate_pred_after": float(calibrated_probs.mean()),
        "oof_rate_actual": float(actual.mean()),
        "oof_brier_before": raw_brier,
        "oof_brier_after": best_brier,
        "oof_logloss_after": best_logloss,
        "oof_std_before": raw_std,
        "oof_std_after": calibrated_std,
        "best_iteration": getattr(final_model, "best_iteration", None),
    }
    return final_model, cfg


def predict_low_tail_probs(tail_model, tail_cfg, X: pd.DataFrame, point_predictions) -> np.ndarray:
    if tail_model is None or not tail_cfg or not tail_cfg.get("enabled"):
        return np.full(len(X), np.nan, dtype=float)
    tail_X = build_low_tail_features(X, point_predictions)
    tail_X = tail_X.reindex(columns=tail_cfg.get("features", tail_X.columns.tolist()), fill_value=0.0)
    probs = tail_model.predict_proba(tail_X)[:, 1]
    return apply_low_tail_calibration(probs, tail_cfg)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading data...")
    df = load_data()
    feature_cols = get_feature_cols(df)
    print(f"  {len(df)} games, {len(feature_cols)} features")
    print(f"  Features: {feature_cols}")

    train, test = train_test_split_by_time(df, train_start_year=TRAIN_START_YEAR, test_year=BACKTEST_YEAR)
    print(f"\nTrain: {len(train)} games ({train['date'].dt.year.min()}-{train['date'].dt.year.max()})")
    print(f"Test:  {len(test)} games ({test['date'].dt.year.min()}-{test['date'].dt.year.max()})")

    print("\n--- Cross-Validation (on training data) ---")
    cross_validate_team_split(train, feature_cols)

    print("\n--- Training Final Team Models ---")
    side_models = {}
    side_meta = {}
    for side in ["home", "away"]:
        model, meta = train_side_model(train, feature_cols, side)
        side_models[side] = model
        side_meta[side] = meta
        print(
            f"  {side.title()} model: mode={meta['prediction_mode']} "
            f"best_iteration={meta['best_iteration']}"
        )

    print("\n--- Fitting Total Calibration ---")
    point_oof_artifacts = build_point_oof_artifacts(train, feature_cols, side_meta)
    calibration_cfg = fit_total_calibration(
        train,
        feature_cols,
        side_meta,
        point_oof_raw=point_oof_artifacts["oof_total_raw"],
    )
    if calibration_cfg.get("enabled"):
        print(
            f"  Total calibration fit on {calibration_cfg['samples']} OOF games "
            f"(alpha={calibration_cfg['alpha']:.2f}, "
            f"MAE {calibration_cfg['oof_mae_before']:.3f} -> {calibration_cfg['oof_mae_after']:.3f}, "
            f"std {calibration_cfg['oof_std_before']:.3f} -> {calibration_cfg['oof_std_after']:.3f}, "
            f"tail alpha={calibration_cfg['tail_alpha']:.2f})"
        )
    else:
        print("  Calibration disabled - insufficient OOF samples")

    print("\n--- Estimating Side Residual Distribution ---")
    side_residual_distribution = estimate_side_residual_distribution(
        train,
        point_oof_artifacts,
        calibration_cfg=calibration_cfg,
    )
    if side_residual_distribution.get("enabled"):
        print(
            f"  OOF side residuals on {side_residual_distribution['samples']} games: "
            f"rho={side_residual_distribution['rho']:+.3f}, "
            f"home sigma={side_residual_distribution['home_sigma']:.3f}, "
            f"away sigma={side_residual_distribution['away_sigma']:.3f}, "
            f"margin sigma={side_residual_distribution['margin_sigma']:.3f}"
        )
    else:
        print("  Side residual distribution unavailable - insufficient OOF samples")

    print("\n--- Test Set Performance (2025 season) ---")
    test_X = test[feature_cols]
    side_preds, total_pred = predict_team_split(side_models, test_X, test, side_meta)
    total_pred = apply_total_calibration(total_pred, calibration_cfg)
    _, train_total_pred = predict_team_split(side_models, train[feature_cols], train, side_meta)
    train_total_pred = apply_total_calibration(train_total_pred, calibration_cfg)
    total_metrics = evaluate(test[TOTAL_TARGET].values, total_pred)
    side_metrics = {
        side: evaluate(test[SIDE_SPECS[side]["target"]].values, side_preds[side], label=f"{side.title()} runs")
        for side in ["home", "away"]
    }

    print("\n--- Training Uncertainty Model ---")
    uncertainty_model, uncertainty_cfg = train_uncertainty_model(
        train,
        feature_cols,
        side_meta,
        calibration_cfg=calibration_cfg,
        point_oof_raw=point_oof_artifacts["oof_total_raw"],
    )
    test_sigmas = predict_sigmas(uncertainty_model, uncertainty_cfg, test_X, total_pred)
    abs_err_test = np.abs(test[TOTAL_TARGET].values - total_pred)
    within_1sigma = float((abs_err_test <= test_sigmas).mean())
    within_2sigma = float((abs_err_test <= 2 * test_sigmas).mean())
    print(f"  Mean predicted sigma: {test_sigmas.mean():.2f}")
    print(f"  Coverage within 1 sigma: {within_1sigma:.1%}  (target ~68%)")
    print(f"  Coverage within 2 sigma: {within_2sigma:.1%}  (target ~95%)")

    print("\n--- Training High-Tail Model (>9.5 runs) ---")
    high_tail_model, high_tail_cfg = train_high_tail_model(
        train,
        feature_cols,
        side_meta,
        calibration_cfg=calibration_cfg,
        point_oof_artifacts=point_oof_artifacts,
        full_point_predictions=train_total_pred,
    )
    if high_tail_cfg.get("enabled"):
        test_tail_probs = predict_high_tail_probs(high_tail_model, high_tail_cfg, test_X, total_pred)
        test_tail_actual = (test[TOTAL_TARGET] > HIGH_TAIL_TARGET_LINE).astype(int).to_numpy()
        print(
            f"  Tail model fit on {high_tail_cfg['samples']} OOF games "
            f"(alpha={high_tail_cfg['alpha']:.2f}, "
            f"Brier {high_tail_cfg['oof_brier_before']:.4f} -> {high_tail_cfg['oof_brier_after']:.4f})"
        )
        print(
            f"  Test tail rate at {HIGH_TAIL_TARGET_LINE:.1f}: "
            f"pred {test_tail_probs.mean():.1%} vs actual {test_tail_actual.mean():.1%}"
        )
    else:
        test_tail_probs = np.full(len(test_X), np.nan, dtype=float)
        print("  High-tail model disabled - insufficient OOF samples")

    print("\n--- Training Low-Tail Model (<7.5 runs) ---")
    low_tail_model, low_tail_cfg = train_low_tail_model(
        train,
        feature_cols,
        side_meta,
        calibration_cfg=calibration_cfg,
        point_oof_artifacts=point_oof_artifacts,
        full_point_predictions=train_total_pred,
    )
    if low_tail_cfg.get("enabled"):
        test_low_tail_probs = predict_low_tail_probs(low_tail_model, low_tail_cfg, test_X, total_pred)
        test_low_tail_actual = (test[TOTAL_TARGET] < LOW_TAIL_TARGET_LINE).astype(int).to_numpy()
        print(
            f"  Low-tail model fit on {low_tail_cfg['samples']} OOF games "
            f"(alpha={low_tail_cfg['alpha']:.2f}, "
            f"Brier {low_tail_cfg['oof_brier_before']:.4f} -> {low_tail_cfg['oof_brier_after']:.4f})"
        )
        print(
            f"  Test tail rate at {LOW_TAIL_TARGET_LINE:.1f}: "
            f"pred {test_low_tail_probs.mean():.1%} vs actual {test_low_tail_actual.mean():.1%}"
        )
    else:
        test_low_tail_probs = np.full(len(test_X), np.nan, dtype=float)
        print("  Low-tail model disabled - insufficient OOF samples")

    print("\n--- Training Market-Aware Layers ---")
    historical_lines = load_historical_lines()
    historical_snapshots = load_historical_line_snapshots()
    train_sigmas = predict_sigmas(uncertainty_model, uncertainty_cfg, train[feature_cols], train_total_pred)
    train_tail_probs = predict_high_tail_probs(high_tail_model, high_tail_cfg, train[feature_cols], train_total_pred)
    train_low_tail_probs = predict_low_tail_probs(low_tail_model, low_tail_cfg, train[feature_cols], train_total_pred)

    train_market_frame = train[["date", "away_team", "home_team", TOTAL_TARGET]].copy()
    train_market_frame["predicted_total"] = train_total_pred
    train_market_frame["prediction_std"] = train_sigmas
    train_market_frame["high_tail_prob_9p5"] = train_tail_probs
    train_market_frame["low_tail_prob_7p5"] = train_low_tail_probs
    train_market_frame = train_market_frame.rename(columns={TOTAL_TARGET: "total_runs"})

    test_market_frame = test[["date", "away_team", "home_team", TOTAL_TARGET]].copy()
    test_market_frame["predicted_total"] = total_pred
    test_market_frame["prediction_std"] = test_sigmas
    test_market_frame["high_tail_prob_9p5"] = test_tail_probs
    test_market_frame["low_tail_prob_7p5"] = test_low_tail_probs
    test_market_frame = test_market_frame.rename(columns={TOTAL_TARGET: "total_runs"})

    train_market_matched = merge_predictions_with_lines(train_market_frame, historical_lines)
    market_shrink_cfg = fit_market_shrinkage(train_market_matched)
    if market_shrink_cfg.get("enabled"):
        print(
            f"  Market shrink fit on {market_shrink_cfg['samples']} matched train games "
            f"(global alpha={market_shrink_cfg['global_alpha']:.2f}, "
            f"Brier-score objective={market_shrink_cfg['global_score']:.4f})"
        )
    else:
        print("  Market shrinkage disabled - insufficient matched historical lines")

    train_snapshot_matched = merge_predictions_with_snapshots(train_market_frame, historical_snapshots)

    # Augment with Kalshi historical lines so the edge model is calibrated for
    # the Kalshi-line-as-market-reference inference path (no sportsbook key needed).
    kalshi_lines_path = os.path.join(DATA_DIR, "kalshi_historical_lines.tsv")
    if os.path.exists(kalshi_lines_path):
        kalshi_hist = pd.read_csv(kalshi_lines_path, sep="\t", parse_dates=["date"])
        # Pick the 10am price for each game; prefer strikes close to 8.5 (typical MLB total)
        kalshi_hist = kalshi_hist[kalshi_hist["has_10am_price"].astype(str).str.lower() == "true"].copy()
        kalshi_hist["_strike_dist"] = (kalshi_hist["strike"] - 8.5).abs()
        kalshi_anchor = (
            kalshi_hist.sort_values("_strike_dist")
            .groupby(["date", "away_team", "home_team"], as_index=False)
            .first()
            .drop(columns=["_strike_dist"])
        )
        kalshi_anchor = kalshi_anchor.rename(columns={"strike": "close_total_line"})
        kalshi_anchor["consensus_total_line"] = kalshi_anchor["close_total_line"]
        kalshi_anchor["line_source"] = "kalshi"
        kalshi_anchor["num_books"] = 1

        # Build a market frame for all years in df (including 2026 which is outside train/test)
        all_years_frame = df[["date", "away_team", "home_team", TOTAL_TARGET]].copy()
        all_years_frame = all_years_frame.rename(columns={TOTAL_TARGET: "total_runs"})
        _, all_preds_raw = predict_team_split(side_models, df[feature_cols], df, side_meta)
        all_preds = apply_total_calibration(all_preds_raw, calibration_cfg)
        all_years_frame["predicted_total"] = all_preds
        all_sigmas = predict_sigmas(uncertainty_model, uncertainty_cfg, df[feature_cols], all_preds)
        all_years_frame["prediction_std"] = all_sigmas
        all_tail_probs = predict_high_tail_probs(high_tail_model, high_tail_cfg, df[feature_cols], all_preds)
        all_years_frame["high_tail_prob_9p5"] = all_tail_probs
        all_low_tail_probs = predict_low_tail_probs(low_tail_model, low_tail_cfg, df[feature_cols], all_preds)
        all_years_frame["low_tail_prob_7p5"] = all_low_tail_probs

        kalshi_matched = merge_predictions_with_lines(all_years_frame, kalshi_anchor)
        if not kalshi_matched.empty:
            # Add snapshot timestamp columns so fit_market_edge_model doesn't skip rows
            kalshi_matched["snapshot_ts"] = pd.NaT
            kalshi_matched["commence_time"] = pd.NaT
            train_snapshot_matched = pd.concat(
                [train_snapshot_matched, kalshi_matched], ignore_index=True, sort=False
            )
            print(f"  Augmented edge model training with {len(kalshi_matched)} Kalshi-matched rows")

    market_edge_model, market_edge_cfg = fit_market_edge_model(
        train_snapshot_matched,
        high_tail_cfg=high_tail_cfg,
        low_tail_cfg=low_tail_cfg,
    )
    if market_edge_cfg.get("enabled"):
        print(
            f"  Market edge model fit on {market_edge_cfg['samples']} snapshot rows "
            f"(Brier {market_edge_cfg['oof_brier_before']:.4f} -> {market_edge_cfg['oof_brier_after']:.4f})"
        )
    else:
        print("  Market edge model disabled - insufficient matched snapshot rows")

    test_market_matched = merge_predictions_with_lines(test_market_frame, historical_lines)
    market_layer_metrics = evaluate_market_strategy(
        test_market_matched,
        market_shrink_cfg,
        market_edge_model=market_edge_model,
        market_edge_cfg=market_edge_cfg,
        high_tail_cfg=high_tail_cfg,
        low_tail_cfg=low_tail_cfg,
    )
    if market_layer_metrics["matched_games"] > 0:
        print(
            f"  2025 matched lines ({market_layer_metrics['method']}): {market_layer_metrics['matched_games']} | "
            f"MAE {market_layer_metrics['mae_before']:.3f} -> {market_layer_metrics['mae_after']:.3f} | "
            f"Brier {market_layer_metrics['brier_before']:.4f} -> {market_layer_metrics['brier_after']:.4f} | "
            f"win {market_layer_metrics['win_rate_before']:.1%} -> {market_layer_metrics['win_rate_after']:.1%}"
        )

    print("\n--- Top 10 Feature Importances ---")
    for side in ["home", "away"]:
        print(f"  {side.title()} model:")
        importance = pd.Series(side_models[side].feature_importances_, index=feature_cols).sort_values(ascending=False)
        for feat, imp in importance.head(10).items():
            print(f"    {feat}: {imp:.3f}")

    for side in ["home", "away"]:
        model_path = os.path.join(MODEL_DIR, SIDE_SPECS[side]["model_file"])
        side_models[side].save_model(model_path)
        print(f"\n{side.title()} model saved to {model_path}")

    uncertainty_path = os.path.join(MODEL_DIR, UNCERTAINTY_MODEL_FILE)
    uncertainty_model.save_model(uncertainty_path)
    print(f"Uncertainty model saved to {uncertainty_path}")

    if high_tail_model is not None and high_tail_cfg.get("enabled"):
        high_tail_path = os.path.join(MODEL_DIR, HIGH_TAIL_MODEL_FILE)
        high_tail_model.save_model(high_tail_path)
        print(f"High-tail model saved to {high_tail_path}")
    if low_tail_model is not None and low_tail_cfg.get("enabled"):
        low_tail_path = os.path.join(MODEL_DIR, LOW_TAIL_MODEL_FILE)
        low_tail_model.save_model(low_tail_path)
        print(f"Low-tail model saved to {low_tail_path}")
    if market_edge_model is not None and market_edge_cfg.get("enabled"):
        market_edge_path = os.path.join(MODEL_DIR, MARKET_EDGE_MODEL_FILE)
        market_edge_model.save_model(market_edge_path)
        print(f"Market edge model saved to {market_edge_path}")

    # Save training-set feature medians for NaN-fill at inference time.
    # Features that are rarely null in training but often missing at prediction
    # time (e.g. lineup, bullpen fatigue) cause XGBoost to follow its learned
    # default branch, which can introduce a systematic scoring bias.
    # Global medians are used as fallback; month-specific medians reduce
    # early-season bias (April bullpen/weather stats differ from mid-season).
    train_medians = train[feature_cols].median()
    feature_medians = {
        col: round(float(val), 6)
        for col, val in train_medians.items()
        if pd.notna(val)
    }
    monthly_medians = {}
    for month in sorted(train["date"].dt.month.unique()):
        month_data = train[train["date"].dt.month == month]
        if len(month_data) < 50:
            continue
        month_med = month_data[feature_cols].median()
        monthly_medians[str(int(month))] = {
            col: round(float(val), 6)
            for col, val in month_med.items()
            if pd.notna(val)
        }

    meta = {
        "model_family": MODEL_FAMILY,
        "features": feature_cols,
        "feature_medians": feature_medians,
        "monthly_feature_medians": monthly_medians,
        "target": TOTAL_TARGET,
        "side_models": side_meta,
        "side_residual_distribution": side_residual_distribution,
        "total_calibration": calibration_cfg,
        "train_years": f"{TRAIN_START_YEAR}-{BACKTEST_YEAR - 1}",
        "test_year": str(BACKTEST_YEAR),
        "metrics": total_metrics,
        "side_metrics": side_metrics,
        "uncertainty_model": uncertainty_cfg,
        "uncertainty_metrics": {
            "mean_sigma": float(np.mean(test_sigmas)),
            "within_1sigma": within_1sigma,
            "within_2sigma": within_2sigma,
        },
        "market_shrinkage": market_shrink_cfg,
        "market_shrinkage_metrics": market_layer_metrics,
        "market_edge_model": market_edge_cfg,
        "market_edge_metrics": market_layer_metrics,
        "high_tail_model": high_tail_cfg,
        "high_tail_metrics": {
            "line": HIGH_TAIL_TARGET_LINE,
            "pred_rate": float(np.nanmean(test_tail_probs)) if len(test_tail_probs) else None,
            "actual_rate": float(np.mean(test[TOTAL_TARGET].values > HIGH_TAIL_TARGET_LINE)),
            "brier": float(brier_score_loss(test[TOTAL_TARGET].values > HIGH_TAIL_TARGET_LINE, clip_probabilities(test_tail_probs)))
            if np.isfinite(test_tail_probs).any() else None,
        },
        "low_tail_model": low_tail_cfg,
        "low_tail_metrics": {
            "line": LOW_TAIL_TARGET_LINE,
            "pred_rate": float(np.nanmean(test_low_tail_probs)) if len(test_low_tail_probs) else None,
            "actual_rate": float(np.mean(test[TOTAL_TARGET].values < LOW_TAIL_TARGET_LINE)),
            "brier": float(brier_score_loss(test[TOTAL_TARGET].values < LOW_TAIL_TARGET_LINE, clip_probabilities(test_low_tail_probs)))
            if np.isfinite(test_low_tail_probs).any() else None,
        },
    }
    meta_path = os.path.join(MODEL_DIR, "model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")

    print("\n--- Prediction Distribution ---")
    print(f"  Predicted total range: {total_pred.min():.1f} - {total_pred.max():.1f}")
    print(f"  Actual total range:    {test[TOTAL_TARGET].min()} - {test[TOTAL_TARGET].max()}")
    print(f"  Predicted total std:   {total_pred.std():.2f}")
    print(f"  Actual total std:      {test[TOTAL_TARGET].std():.2f}")


if __name__ == "__main__":
    main()
