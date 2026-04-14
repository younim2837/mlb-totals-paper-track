"""
Helpers for collecting and transforming bullpen usage into pregame fatigue features.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd
import requests

MLB_GAME_FEED = "https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"


def _to_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _ip_to_decimal(ip_str: Any) -> float:
    if ip_str is None:
        return 0.0
    try:
        ip = float(ip_str)
        whole = int(ip)
        partial = round(ip - whole, 1)
        return whole + (partial / 0.3) * (1 / 3)
    except (TypeError, ValueError):
        return 0.0


def _extract_team_relief_rows(game_id: int, game_date: str, season: int, side: str, team_box: dict) -> list[dict]:
    team = team_box.get("team", {}) or {}
    players = team_box.get("players", {}) or {}
    pitcher_ids = team_box.get("pitchers", []) or []

    rows: list[dict] = []
    starter_seen = False

    for raw_pid in pitcher_ids:
        pid = _to_int(raw_pid)
        if not pid:
            continue
        player = players.get(f"ID{pid}", {}) or {}
        stat = (player.get("stats", {}) or {}).get("pitching", {}) or {}
        batters_faced = _to_int(stat.get("battersFaced"))
        if batters_faced <= 0:
            continue

        if not starter_seen:
            starter_seen = True
            continue

        rows.append({
            "game_id": game_id,
            "date": game_date,
            "season": season,
            "team_id": _to_int(team.get("id")),
            "team_name": team.get("name"),
            "is_home": 1 if side == "home" else 0,
            "pitcher_id": pid,
            "pitcher_name": (player.get("person") or {}).get("fullName", ""),
            "innings_pitched_dec": _ip_to_decimal(stat.get("inningsPitched")),
            "earnedRuns": _to_int(stat.get("earnedRuns")),
            "numberOfPitches": _to_int(stat.get("numberOfPitches")),
            "battersFaced": batters_faced,
        })
    return rows


def extract_game_bullpen_rows(payload: dict) -> list[dict]:
    game_id = _to_int(payload.get("gamePk"))
    game_data = payload.get("gameData", {}) or {}
    box = payload.get("liveData", {}).get("boxscore", {}) or {}
    teams = box.get("teams", {}) or {}
    game_date = ((game_data.get("datetime") or {}).get("originalDate")) or ""
    season = _to_int((game_data.get("game") or {}).get("season"))

    rows: list[dict] = []
    rows.extend(_extract_team_relief_rows(game_id, game_date, season, "home", teams.get("home", {}) or {}))
    rows.extend(_extract_team_relief_rows(game_id, game_date, season, "away", teams.get("away", {}) or {}))
    return rows


def fetch_game_bullpen_rows(game_id: int, timeout: int = 20) -> list[dict]:
    try:
        r = requests.get(MLB_GAME_FEED.format(game_id=game_id), timeout=timeout)
        if r.status_code != 200:
            return []
        return extract_game_bullpen_rows(r.json())
    except Exception:
        return []


def fetch_many_game_bullpen_rows(game_ids: list[int], max_workers: int = 12, timeout: int = 20) -> list[dict]:
    rows: list[dict] = []
    unique_ids = [int(gid) for gid in dict.fromkeys(game_ids) if gid]
    if not unique_ids:
        return rows

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(fetch_game_bullpen_rows, gid, timeout): gid
            for gid in unique_ids
        }
        for fut in as_completed(futures):
            result = fut.result() or []
            if result:
                rows.extend(result)
    return rows


def build_pregame_bullpen_features(logs: pd.DataFrame) -> pd.DataFrame:
    """
    Convert reliever appearance logs into pregame bullpen workload features
    keyed by (team_id, date).
    """
    if logs.empty:
        return pd.DataFrame(columns=["team_id", "date"])

    logs = logs.copy()
    logs["date"] = pd.to_datetime(logs["date"]).dt.normalize()

    daily = (
        logs.groupby(["team_id", "date"], as_index=False)
        .agg(
            bullpen_app_count=("pitcher_id", "count"),
            bullpen_unique_arms=("pitcher_id", "nunique"),
            bullpen_used_ip=("innings_pitched_dec", "sum"),
            bullpen_used_pitches=("numberOfPitches", "sum"),
            bullpen_used_bf=("battersFaced", "sum"),
            bullpen_used_er=("earnedRuns", "sum"),
            bullpen_heavy_arms=("numberOfPitches", lambda s: int((s >= 25).sum())),
        )
    )

    league_bp_era = 4.50
    total_ip = logs["innings_pitched_dec"].sum()
    if total_ip > 0:
        league_bp_era = float((logs["earnedRuns"].sum() / total_ip) * 9.0)

    all_rows = []
    for team_id, grp in daily.groupby("team_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        idx = pd.date_range(grp["date"].min(), grp["date"].max() + pd.Timedelta(days=1), freq="D")
        team_daily = grp.set_index("date").reindex(idx).fillna(0.0)
        team_daily.index.name = "date"
        team_daily["team_id"] = int(team_id)

        for base_col in [
            "bullpen_app_count",
            "bullpen_unique_arms",
            "bullpen_used_ip",
            "bullpen_used_pitches",
            "bullpen_used_bf",
            "bullpen_used_er",
            "bullpen_heavy_arms",
        ]:
            shifted = team_daily[base_col].shift(1).fillna(0.0)
            team_daily[f"{base_col}_1d"] = shifted
            team_daily[f"{base_col}_3d"] = shifted.rolling(3, min_periods=1).sum()
            team_daily[f"{base_col}_5d"] = shifted.rolling(5, min_periods=1).sum()

        player_daily = (
            logs[logs["team_id"] == team_id][["date", "pitcher_id"]]
            .drop_duplicates()
            .assign(appeared=1)
        )
        player_wide = (
            player_daily.pivot(index="date", columns="pitcher_id", values="appeared")
            .reindex(idx)
            .fillna(0.0)
        )
        prev1 = player_wide.shift(1).fillna(0.0)
        prev2 = player_wide.shift(2).fillna(0.0)
        team_daily["bullpen_arms_yesterday"] = prev1.sum(axis=1)
        team_daily["bullpen_b2b_arms"] = ((prev1 > 0) & (prev2 > 0)).sum(axis=1)

        team_logs = logs[logs["team_id"] == team_id].copy()
        team_logs = team_logs.sort_values(["pitcher_id", "date", "game_id"]).reset_index(drop=True)
        rel_grp = team_logs.groupby("pitcher_id")
        team_logs["prior_apps"] = rel_grp.cumcount()
        for src, dest in [
            ("innings_pitched_dec", "prior_ip"),
            ("earnedRuns", "prior_er"),
            ("numberOfPitches", "prior_pitches"),
            ("battersFaced", "prior_bf"),
        ]:
            team_logs[dest] = rel_grp[src].transform(lambda s: s.cumsum().shift(1)).fillna(0.0)

        safe_prior_ip = team_logs["prior_ip"].replace(0.0, float("nan"))
        safe_prior_apps = team_logs["prior_apps"].replace(0, float("nan"))
        prior_era = (team_logs["prior_er"] / safe_prior_ip) * 9.0
        prior_avg_pitches = team_logs["prior_pitches"] / safe_prior_apps
        sample_weight = (team_logs["prior_ip"] / 12.0).clip(lower=0.0, upper=1.0)
        role_score = (
            (team_logs["prior_apps"] / 15.0).clip(lower=0.0, upper=1.5) +
            (prior_avg_pitches / 18.0).clip(lower=0.0, upper=1.5)
        )
        quality_score = ((league_bp_era - prior_era) / 2.0).clip(lower=-2.0, upper=2.0)
        team_logs["tier_score"] = (role_score + sample_weight * quality_score).fillna(0.0)
        team_logs.loc[team_logs["prior_apps"] < 3, "tier_score"] = 0.0

        tier_wide = (
            team_logs[["date", "pitcher_id", "tier_score"]]
            .drop_duplicates(subset=["date", "pitcher_id"], keep="last")
            .pivot(index="date", columns="pitcher_id", values="tier_score")
            .reindex(idx)
            .ffill()
            .fillna(0.0)
        )
        pitch_wide = (
            team_logs.pivot_table(index="date", columns="pitcher_id", values="numberOfPitches", aggfunc="sum")
            .reindex(idx)
            .fillna(0.0)
        )
        prev3_pitch = pitch_wide.shift(1).rolling(3, min_periods=1).sum().fillna(0.0)

        top2_used_yesterday = []
        top2_b2b = []
        top4_used_yesterday = []
        top4_b2b = []
        top4_available_score = []
        top4_burned_score = []
        top4_pitches_3d = []
        top2_quality_score = []
        top4_quality_score = []

        for current_date in idx:
            scores = tier_wide.loc[current_date]
            eligible = scores[scores > 0].sort_values(ascending=False)
            top2_ids = list(eligible.head(2).index)
            top4_ids = list(eligible.head(4).index)

            top2_scores = eligible.head(2)
            top4_scores = eligible.head(4)

            y1 = prev1.loc[current_date] if current_date in prev1.index else pd.Series(dtype=float)
            y2 = prev2.loc[current_date] if current_date in prev2.index else pd.Series(dtype=float)
            p3 = prev3_pitch.loc[current_date] if current_date in prev3_pitch.index else pd.Series(dtype=float)

            top2_used_yesterday.append(float((y1.reindex(top2_ids).fillna(0.0) > 0).sum()))
            top2_b2b.append(float(((y1.reindex(top2_ids).fillna(0.0) > 0) & (y2.reindex(top2_ids).fillna(0.0) > 0)).sum()))
            top4_used_yesterday.append(float((y1.reindex(top4_ids).fillna(0.0) > 0).sum()))
            top4_b2b.append(float(((y1.reindex(top4_ids).fillna(0.0) > 0) & (y2.reindex(top4_ids).fillna(0.0) > 0)).sum()))
            top4_available_score.append(float(top4_scores[y1.reindex(top4_ids).fillna(0.0) <= 0].sum()))
            top4_burned_score.append(float(top4_scores[y1.reindex(top4_ids).fillna(0.0) > 0].sum()))
            top4_pitches_3d.append(float(p3.reindex(top4_ids).fillna(0.0).sum()))
            top2_quality_score.append(float(top2_scores.sum()))
            top4_quality_score.append(float(top4_scores.sum()))

        team_daily["bullpen_top2_quality_score"] = top2_quality_score
        team_daily["bullpen_top4_quality_score"] = top4_quality_score
        team_daily["bullpen_top2_used_yesterday"] = top2_used_yesterday
        team_daily["bullpen_top2_b2b"] = top2_b2b
        team_daily["bullpen_top4_used_yesterday"] = top4_used_yesterday
        team_daily["bullpen_top4_b2b"] = top4_b2b
        team_daily["bullpen_top4_available_score"] = top4_available_score
        team_daily["bullpen_top4_burned_score"] = top4_burned_score
        team_daily["bullpen_top4_pitches_3d"] = top4_pitches_3d

        team_daily["bullpen_fatigue_score"] = (
            1.5 * team_daily["bullpen_used_pitches_1d"] +
            1.0 * team_daily["bullpen_used_pitches_3d"] +
            0.5 * team_daily["bullpen_used_pitches_5d"] +
            10.0 * team_daily["bullpen_b2b_arms"]
        )
        all_rows.append(team_daily.reset_index())

    result = pd.concat(all_rows, ignore_index=True)
    result = result.rename(columns={"index": "date"})
    return result.sort_values(["team_id", "date"]).reset_index(drop=True)


def get_pregame_bullpen_feature_cols(columns: list[str]) -> list[str]:
    preferred = {
        "bullpen_used_ip_1d",
        "bullpen_used_ip_3d",
        "bullpen_used_pitches_1d",
        "bullpen_used_pitches_3d",
        "bullpen_heavy_arms_1d",
        "bullpen_heavy_arms_3d",
        "bullpen_arms_yesterday",
        "bullpen_b2b_arms",
        "bullpen_fatigue_score",
        "bullpen_top2_quality_score",
        "bullpen_top4_quality_score",
        "bullpen_top2_used_yesterday",
        "bullpen_top2_b2b",
        "bullpen_top4_used_yesterday",
        "bullpen_top4_b2b",
        "bullpen_top4_available_score",
        "bullpen_top4_burned_score",
        "bullpen_top4_pitches_3d",
    }
    return [col for col in columns if col in preferred]
