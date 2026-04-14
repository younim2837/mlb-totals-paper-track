"""
Shared modeling utilities used by training, backtesting, and live prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def apply_total_calibration(point_predictions, calibration_cfg: dict | None):
    preds = np.asarray(point_predictions, dtype=float)
    if not calibration_cfg or not calibration_cfg.get("enabled"):
        return preds

    xs = np.asarray(calibration_cfg.get("x"), dtype=float)
    ys = np.asarray(calibration_cfg.get("y"), dtype=float)
    if len(xs) == 0 or len(xs) != len(ys):
        return preds

    mapped = np.interp(preds, xs, ys, left=ys[0], right=ys[-1])
    alpha = float(calibration_cfg.get("alpha", 1.0))
    preds = preds + alpha * (mapped - preds)

    tail_xs = np.asarray(calibration_cfg.get("tail_x", []), dtype=float)
    tail_ys = np.asarray(calibration_cfg.get("tail_y", []), dtype=float)
    if len(tail_xs) > 1 and len(tail_xs) == len(tail_ys):
        tail_mapped = np.interp(preds, tail_xs, tail_ys, left=tail_ys[0], right=tail_ys[-1])
        tail_alpha = float(calibration_cfg.get("tail_alpha", 0.0))
        preds = preds + tail_alpha * (tail_mapped - preds)

    return preds


def build_high_tail_features(
    X: pd.DataFrame,
    point_predictions,
    point_feature: str = "point_prediction",
) -> pd.DataFrame:
    tail_X = X.copy()
    tail_X[point_feature] = np.asarray(point_predictions)
    return tail_X


def build_low_tail_features(
    X: pd.DataFrame,
    point_predictions,
    point_feature: str = "point_prediction",
) -> pd.DataFrame:
    tail_X = X.copy()
    tail_X[point_feature] = np.asarray(point_predictions)
    return tail_X


def clip_probabilities(probs):
    return np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)


def _apply_probability_calibration(probabilities, cfg: dict | None):
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


def apply_high_tail_calibration(probabilities, cfg: dict | None):
    return _apply_probability_calibration(probabilities, cfg)


def apply_low_tail_calibration(probabilities, cfg: dict | None):
    return _apply_probability_calibration(probabilities, cfg)


def adjusted_sigma_for_line(
    mean_total: float,
    base_sigma: float,
    line: float,
    high_tail_prob: float | None,
    high_tail_cfg: dict | None,
    low_tail_prob: float | None = None,
    low_tail_cfg: dict | None = None,
):
    sigma = max(float(base_sigma), 1e-6)

    if high_tail_prob is not None and high_tail_cfg and high_tail_cfg.get("enabled"):
        tail_line = float(high_tail_cfg.get("line", 9.5))
        if float(line) >= tail_line:
            normal_tail_prob = 1 - stats.norm.cdf(tail_line, loc=mean_total, scale=sigma)
            target_tail_prob = max(normal_tail_prob, float(high_tail_prob))
            if target_tail_prob > normal_tail_prob + 1e-6:
                q = stats.norm.ppf(1 - np.clip(target_tail_prob, 1e-6, 1 - 1e-6))
                if np.isfinite(q) and abs(q) >= 1e-6:
                    sigma_target = (tail_line - mean_total) / q
                    if np.isfinite(sigma_target) and sigma_target > 0:
                        sigma_cap = sigma * float(high_tail_cfg.get("sigma_cap_multiplier", 1.8))
                        sigma = float(np.clip(max(sigma, sigma_target), sigma, sigma_cap))

    if low_tail_prob is not None and low_tail_cfg and low_tail_cfg.get("enabled"):
        tail_line = float(low_tail_cfg.get("line", 7.5))
        if float(line) <= tail_line:
            normal_tail_prob = stats.norm.cdf(tail_line, loc=mean_total, scale=sigma)
            target_tail_prob = max(normal_tail_prob, float(low_tail_prob))
            if target_tail_prob > normal_tail_prob + 1e-6:
                q = stats.norm.ppf(np.clip(target_tail_prob, 1e-6, 1 - 1e-6))
                if np.isfinite(q) and abs(q) >= 1e-6:
                    sigma_target = (tail_line - mean_total) / q
                    if np.isfinite(sigma_target) and sigma_target > 0:
                        sigma_cap = sigma * float(low_tail_cfg.get("sigma_cap_multiplier", 1.8))
                        sigma = float(np.clip(max(sigma, sigma_target), sigma, sigma_cap))

    return sigma


def probability_over_line(
    mean_total: float,
    base_sigma: float,
    line: float,
    high_tail_prob: float | None = None,
    high_tail_cfg: dict | None = None,
    low_tail_prob: float | None = None,
    low_tail_cfg: dict | None = None,
):
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
