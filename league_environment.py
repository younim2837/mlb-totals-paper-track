"""
League-level scoring environment helpers.

These features help the model adapt to season-wide run environment shifts that
team-level rolling stats do not fully capture, especially early in a season.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


LEAGUE_ENV_FEATURES = [
    "league_avg_total_runs_7d",
    "league_avg_total_runs_30d",
    "league_avg_total_runs_season",
]


def _prepare_daily_totals(frame: pd.DataFrame, date_col: str, total_col: str) -> pd.DataFrame:
    work = frame[[date_col, total_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col, total_col]).copy()
    work[date_col] = work[date_col].dt.normalize()

    daily = (
        work.groupby(date_col)[total_col]
        .agg(total_runs_sum="sum", game_count="count")
        .sort_index()
    )
    daily["season"] = daily.index.year
    return daily


def _safe_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    denom = denom.replace(0, np.nan)
    return numer / denom


def add_league_environment_features(frame: pd.DataFrame, date_col: str = "date", total_col: str = "total_runs") -> pd.DataFrame:
    df = frame.copy()
    if date_col not in df.columns or total_col not in df.columns:
        for col in LEAGUE_ENV_FEATURES:
            df[col] = np.nan
        return df

    daily = _prepare_daily_totals(df, date_col, total_col)

    daily["league_avg_total_runs_7d"] = _safe_ratio(
        daily["total_runs_sum"].rolling(7, min_periods=1).sum().shift(1),
        daily["game_count"].rolling(7, min_periods=1).sum().shift(1),
    )
    daily["league_avg_total_runs_30d"] = _safe_ratio(
        daily["total_runs_sum"].rolling(30, min_periods=1).sum().shift(1),
        daily["game_count"].rolling(30, min_periods=1).sum().shift(1),
    )
    daily["league_avg_total_runs_season"] = (
        daily.groupby("season", group_keys=False).apply(
            lambda g: _safe_ratio(
                g["total_runs_sum"].cumsum().shift(1),
                g["game_count"].cumsum().shift(1),
            )
        )
    )

    merge_cols = LEAGUE_ENV_FEATURES.copy()
    merged = df.copy()
    merged[date_col] = pd.to_datetime(merged[date_col], errors="coerce")
    merged["_merge_date"] = merged[date_col].dt.normalize()
    merged = merged.merge(
        daily[merge_cols],
        left_on="_merge_date",
        right_index=True,
        how="left",
    )
    merged = merged.drop(columns=["_merge_date"])
    return merged


def build_current_league_environment(frame: pd.DataFrame, predict_date) -> dict[str, float]:
    if frame.empty:
        return {col: np.nan for col in LEAGUE_ENV_FEATURES}

    predict_date = pd.to_datetime(predict_date, errors="coerce")
    if pd.isna(predict_date):
        return {col: np.nan for col in LEAGUE_ENV_FEATURES}

    hist = frame.copy()
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist = hist[hist["date"] < predict_date.normalize()].copy()
    hist = hist.dropna(subset=["date", "total_runs"])
    if hist.empty:
        return {col: np.nan for col in LEAGUE_ENV_FEATURES}

    daily = _prepare_daily_totals(hist, "date", "total_runs")
    last_date = daily.index.max()
    latest = add_league_environment_features(hist, date_col="date", total_col="total_runs")
    latest = latest[latest["date"].dt.normalize() == last_date].copy()
    if latest.empty:
        return {col: np.nan for col in LEAGUE_ENV_FEATURES}

    row = latest.iloc[0]
    return {col: float(row[col]) if pd.notna(row[col]) else np.nan for col in LEAGUE_ENV_FEATURES}
