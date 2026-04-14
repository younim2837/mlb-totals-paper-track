"""
Helpers for extracting same-day starting lineup features from MLB game feeds.

The historical collector and live predictor both use this module so the model
sees the same feature family in training and inference.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import requests

MLB_GAME_FEED = "https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"

# Empirical-Bayes priors to keep tiny early-season samples from dominating.
PRIOR_PA = 120.0
LEAGUE_OBP = 0.315
LEAGUE_SLG = 0.400
LEAGUE_BB_PCT = 0.082
LEAGUE_K_PCT = 0.225


def _to_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _rate(num: float, den: float) -> float:
    if den <= 0:
        return np.nan
    return float(num) / float(den)


def _shrink_rate(value: float, sample_pa: float, prior_value: float, prior_pa: float = PRIOR_PA) -> float:
    if np.isnan(value):
        value = prior_value
    weight = float(max(sample_pa, 0.0))
    regressed = (weight * value + prior_pa * prior_value) / (weight + prior_pa)
    return float(regressed)


def _pregame_hitter_snapshot(season_bat: dict, game_bat: dict) -> dict:
    """
    Reconstruct pregame batting skill by subtracting current-game stats from the
    season totals contained in the live feed.
    """
    ab = max(0, _to_int(season_bat.get("atBats")) - _to_int(game_bat.get("atBats")))
    hits = max(0, _to_int(season_bat.get("hits")) - _to_int(game_bat.get("hits")))
    doubles = max(0, _to_int(season_bat.get("doubles")) - _to_int(game_bat.get("doubles")))
    triples = max(0, _to_int(season_bat.get("triples")) - _to_int(game_bat.get("triples")))
    homers = max(0, _to_int(season_bat.get("homeRuns")) - _to_int(game_bat.get("homeRuns")))
    walks = max(0, _to_int(season_bat.get("baseOnBalls")) - _to_int(game_bat.get("baseOnBalls")))
    hbp = max(0, _to_int(season_bat.get("hitByPitch")) - _to_int(game_bat.get("hitByPitch")))
    sf = max(0, _to_int(season_bat.get("sacFlies")) - _to_int(game_bat.get("sacFlies")))
    pa = max(0, _to_int(season_bat.get("plateAppearances")) - _to_int(game_bat.get("plateAppearances")))
    so = max(0, _to_int(season_bat.get("strikeOuts")) - _to_int(game_bat.get("strikeOuts")))
    tb = max(0, _to_int(season_bat.get("totalBases")) - _to_int(game_bat.get("totalBases")))
    gp = max(0, _to_int(season_bat.get("gamesPlayed")) - _to_int(game_bat.get("gamesPlayed")))

    raw_obp = _rate(hits + walks + hbp, ab + walks + hbp + sf)
    raw_slg = _rate(tb, ab)
    raw_bb_pct = _rate(walks, pa)
    raw_k_pct = _rate(so, pa)

    obp = float(np.clip(_shrink_rate(raw_obp, pa, LEAGUE_OBP), 0.240, 0.460))
    slg = float(np.clip(_shrink_rate(raw_slg, pa, LEAGUE_SLG), 0.280, 0.650))
    ops = float(np.clip(obp + slg, 0.560, 1.020))
    bb_pct = float(np.clip(_shrink_rate(raw_bb_pct, pa, LEAGUE_BB_PCT), 0.040, 0.180))
    k_pct = float(np.clip(_shrink_rate(raw_k_pct, pa, LEAGUE_K_PCT), 0.100, 0.380))

    return {
        "pre_pa": pa,
        "pre_gp": gp,
        "pre_ops": ops,
        "pre_obp": obp,
        "pre_slg": slg,
        "pre_bb_pct": bb_pct,
        "pre_k_pct": k_pct,
    }


def _platoon_advantage_count(bat_sides: list[str], opp_pitch_hand: str | None) -> int:
    if not opp_pitch_hand:
        return 0

    count = 0
    for side in bat_sides:
        if side == "S":
            count += 1
        elif opp_pitch_hand == "R" and side == "L":
            count += 1
        elif opp_pitch_hand == "L" and side == "R":
            count += 1
    return count


def _lineup_summary(
    game_id: int,
    game_date: str,
    season: int,
    team_box: dict,
    game_data_players: dict,
    side: str,
    opp_pitch_hand: str | None,
) -> dict | None:
    team = team_box.get("team", {}) or {}
    order = team_box.get("battingOrder", []) or []
    starter_ids: list[int] = []
    for raw_pid in order:
        pid = _to_int(raw_pid)
        if pid and pid not in starter_ids:
            starter_ids.append(pid)
        if len(starter_ids) >= 9:
            break

    if not starter_ids:
        return None

    ops_vals = []
    obp_vals = []
    slg_vals = []
    bb_vals = []
    k_vals = []
    pa_vals = []
    top3_ops = []
    mid3_ops = []
    bot3_ops = []
    bat_sides = []
    hitters_with_sample = 0
    low_pa_hitters = 0

    for idx, pid in enumerate(starter_ids[:9], start=1):
        player_box = (team_box.get("players", {}) or {}).get(f"ID{pid}", {}) or {}
        gd_player = game_data_players.get(f"ID{pid}", {}) or {}
        snap = _pregame_hitter_snapshot(
            player_box.get("seasonStats", {}).get("batting", {}) or {},
            player_box.get("stats", {}).get("batting", {}) or {},
        )

        bat_code = (gd_player.get("batSide") or {}).get("code")
        if bat_code:
            bat_sides.append(bat_code)

        if snap["pre_pa"] >= 20:
            hitters_with_sample += 1
        else:
            low_pa_hitters += 1

        ops_vals.append(snap["pre_ops"])
        obp_vals.append(snap["pre_obp"])
        slg_vals.append(snap["pre_slg"])
        bb_vals.append(snap["pre_bb_pct"])
        k_vals.append(snap["pre_k_pct"])
        pa_vals.append(float(snap["pre_pa"]))

        if idx <= 3:
            top3_ops.append(snap["pre_ops"])
        elif idx <= 6:
            mid3_ops.append(snap["pre_ops"])
        else:
            bot3_ops.append(snap["pre_ops"])

    def _nanmean(values: list[float]) -> float:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0 or np.isnan(arr).all():
            return np.nan
        return float(np.nanmean(arr))

    def _nanstd(values: list[float]) -> float:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0 or np.isnan(arr).all():
            return np.nan
        return float(np.nanstd(arr))

    return {
        "game_id": game_id,
        "date": game_date,
        "season": season,
        "team_id": _to_int(team.get("id")),
        "team_name": team.get("name"),
        "is_home": 1 if side == "home" else 0,
        "opp_pitch_hand": opp_pitch_hand or "",
        "lineup_confirmed": 1 if len(starter_ids) >= 8 else 0,
        "lineup_batters_known": len(starter_ids),
        "lineup_batters_with_sample": hitters_with_sample,
        "lineup_low_pa_batters": low_pa_hitters,
        "lineup_platoon_adv_batters": _platoon_advantage_count(bat_sides, opp_pitch_hand),
        "lineup_avg_pre_pa": _nanmean(pa_vals),
        "lineup_avg_ops": _nanmean(ops_vals),
        "lineup_avg_obp": _nanmean(obp_vals),
        "lineup_avg_slg": _nanmean(slg_vals),
        "lineup_avg_bb_pct": _nanmean(bb_vals),
        "lineup_avg_k_pct": _nanmean(k_vals),
        "lineup_top3_avg_ops": _nanmean(top3_ops),
        "lineup_mid3_avg_ops": _nanmean(mid3_ops),
        "lineup_bottom3_avg_ops": _nanmean(bot3_ops),
        "lineup_std_ops": _nanstd(ops_vals),
    }


def extract_game_lineup_features(payload: dict) -> list[dict]:
    game_id = _to_int(payload.get("gamePk"))
    game_data = payload.get("gameData", {}) or {}
    box = payload.get("liveData", {}).get("boxscore", {}) or {}
    teams = box.get("teams", {}) or {}
    game_date = ((game_data.get("datetime") or {}).get("originalDate")) or ""
    season = _to_int((game_data.get("game") or {}).get("season"))
    gd_players = game_data.get("players", {}) or {}

    rows = []
    away_pitchers = (teams.get("away", {}) or {}).get("pitchers", []) or []
    home_pitchers = (teams.get("home", {}) or {}).get("pitchers", []) or []
    away_starter = gd_players.get(f"ID{away_pitchers[0]}", {}) if away_pitchers else {}
    home_starter = gd_players.get(f"ID{home_pitchers[0]}", {}) if home_pitchers else {}
    away_hand = ((away_starter or {}).get("pitchHand") or {}).get("code")
    home_hand = ((home_starter or {}).get("pitchHand") or {}).get("code")

    home_row = _lineup_summary(
        game_id=game_id,
        game_date=game_date,
        season=season,
        team_box=teams.get("home", {}) or {},
        game_data_players=gd_players,
        side="home",
        opp_pitch_hand=away_hand,
    )
    away_row = _lineup_summary(
        game_id=game_id,
        game_date=game_date,
        season=season,
        team_box=teams.get("away", {}) or {},
        game_data_players=gd_players,
        side="away",
        opp_pitch_hand=home_hand,
    )
    if home_row:
        rows.append(home_row)
    if away_row:
        rows.append(away_row)
    return rows


def fetch_game_lineup_features(game_id: int, timeout: int = 20) -> list[dict]:
    try:
        r = requests.get(MLB_GAME_FEED.format(game_id=game_id), timeout=timeout)
        if r.status_code != 200:
            return []
        return extract_game_lineup_features(r.json())
    except Exception:
        return []


def fetch_many_game_lineup_features(
    game_ids: list[int],
    max_workers: int = 12,
    timeout: int = 20,
) -> list[dict]:
    rows: list[dict] = []
    unique_ids = [int(gid) for gid in dict.fromkeys(game_ids) if gid]
    if not unique_ids:
        return rows

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(fetch_game_lineup_features, gid, timeout): gid
            for gid in unique_ids
        }
        for fut in as_completed(futures):
            result = fut.result() or []
            if result:
                rows.extend(result)
    return rows
