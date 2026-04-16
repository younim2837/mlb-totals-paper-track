"""
MLB Total Runs Predictor
Fetches today's games, computes features from recent history,
and predicts total runs with over/under confidence percentages.
"""

import statsapi
import requests
import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta
import json
import os
import sys
from bullpen_usage import build_pregame_bullpen_features, get_pregame_bullpen_feature_cols
from league_environment import LEAGUE_ENV_FEATURES, build_current_league_environment
from modeling_utils import (
    adjusted_sigma_for_line,
    probability_over_line,
)
from model_runtime import (
    compute_residual_std,
    estimate_prediction_std,
    get_side_residual_distribution,
    load_model_bundle,
    load_historical_data,
    predict_high_tail_prob,
    predict_low_tail_prob,
    predict_point_outputs,
)
from prediction_betting import (
    add_edge_to_prediction,
    add_kalshi_metrics,
    add_team_side_metrics,
    apply_market_adjustment_to_prediction,
    apply_overrides,
    get_kalshi_betting_thresholds,
    kalshi_filter_reason,
    normalize_allowed_market_adjustment_methods,
    suppress_kalshi_bet,
)
from prediction_reporting import display_predictions, export_daily_prediction_reports
from venue_metadata import DOME_VENUE_IDS, VENUE_COORDS, compute_local_time_features

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

try:
    from collect_kalshi_lines import fetch_kalshi_lines, implied_to_american
    _KALSHI_AVAILABLE = True
except ImportError:
    _KALSHI_AVAILABLE = False

try:
    from lineup_features import fetch_many_game_lineup_features
    _LINEUPS_AVAILABLE = True
except ImportError:
    _LINEUPS_AVAILABLE = False

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

WINDOWS = [10, 30]


def load_model_config() -> dict:
    """
    Load model_config.yaml. Returns a config dict with defaults if file
    is missing or yaml is not installed.
    """
    defaults = {
        "elo": {"k_factor": 8, "home_advantage": 0.15, "offseason_carry": 0.75},
        "feature_weights": {
            "elo_ratings": 1.0, "recent_form": 1.0, "season_form": 1.0,
            "pitcher": 1.0, "bullpen": 1.0, "park_factor": 1.0,
            "weather": 1.0, "umpire": 1.0, "h2h": 1.0,
        },
        "betting": {
            "min_edge_runs": 0.5,
            "min_confidence_pct": 55,
            "max_bets_per_day": 5,
            "allowed_market_adjustment_methods": ["edge_model"],
        },
        "bankroll": {
            "total": 10000,
            "kelly_fraction": 0.10,
            "max_bet_pct": 0,
            "min_bet": 1,
            "round_to": 1,
        },
        "overrides": {},
        "display": {
            "show_elo_ratings": True, "show_umpire": True, "show_weather": True,
            "min_kalshi_confidence": 40, "kalshi_max_line_diff": 2.5,
        },
        "market_lines": {
            "enabled": False,
            "use_for_display": True,
            "use_for_post_model_shrinkage": False,
            "min_books": 2,
            "max_shrink_fraction": 0.35,
            "shrink_deadband_runs": 0.25,
        },
    }

    config_path = os.path.join(os.path.dirname(__file__), "model_config.yaml")
    bankroll_override = os.environ.get("PAPER_BANKROLL_OVERRIDE")
    if bankroll_override:
        try:
            defaults.setdefault("bankroll", {})["total"] = float(bankroll_override)
        except ValueError:
            print(f"  Warning: invalid PAPER_BANKROLL_OVERRIDE={bankroll_override!r} - ignoring")
    if not _YAML_AVAILABLE or not os.path.exists(config_path):
        return defaults

    try:
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f) or {}
        # Deep merge: user values override defaults
        for section, values in user_cfg.items():
            if section in defaults and isinstance(values, dict):
                defaults[section].update(values)
            else:
                defaults[section] = values
        return defaults
    except Exception as e:
        print(f"  Warning: could not load model_config.yaml ({e}) — using defaults")
        return defaults


def _ip_str_to_decimal(ip_str) -> float:
    """Convert baseball IP string (e.g. '6.2' = 6 2/3 innings) to decimal."""
    if pd.isna(ip_str):
        return np.nan
    try:
        ip = float(ip_str)
        full = int(ip)
        partial = round(ip - full, 1)
        return full + (partial / 0.3) * (1 / 3)
    except (ValueError, TypeError):
        return np.nan


def build_team_rolling_stats(games_df: pd.DataFrame, predict_date=None) -> dict:
    """
    Build a lookup of rolling stats per team from historical game data.
    Includes bullpen ERA (10g, 30g, season) derived by subtracting starter
    earned runs / IP from team totals, and days_rest since last game.
    Returns: {team_name: {stat_name: value}}
    """
    if predict_date is None:
        predict_date = pd.Timestamp.today().normalize()
    else:
        predict_date = pd.Timestamp(predict_date).normalize()
    # ── Build per-team game log ──────────────────────────────────────────
    rows = []
    for _, g in games_df.iterrows():
        rows.append({
            "date": g["date"], "game_id": g["game_id"],
            "team": g["home_team"],
            "runs_scored": g["home_score"], "runs_allowed": g["away_score"],
            "starter_name": g.get("home_pitcher"),
        })
        rows.append({
            "date": g["date"], "game_id": g["game_id"],
            "team": g["away_team"],
            "runs_scored": g["away_score"], "runs_allowed": g["home_score"],
            "starter_name": g.get("away_pitcher"),
        })

    log = pd.DataFrame(rows).sort_values(["team", "date"]).reset_index(drop=True)

    # ── Compute bullpen stats per game (starter subtraction) ─────────────
    logs_path = os.path.join(DATA_DIR, "pitcher_game_logs.tsv")
    ids_path = os.path.join(DATA_DIR, "pitcher_ids.tsv")
    has_bullpen = False

    if os.path.exists(logs_path) and os.path.exists(ids_path):
        pitcher_logs = pd.read_csv(logs_path, sep="\t")
        pitcher_logs["date"] = pd.to_datetime(pitcher_logs["date"], format="mixed")
        pitcher_logs["earnedRuns"] = pd.to_numeric(pitcher_logs["earnedRuns"], errors="coerce")
        pitcher_logs["ip_dec"] = pitcher_logs["inningsPitched"].apply(_ip_str_to_decimal)

        ids_df = pd.read_csv(ids_path, sep="\t")
        name_to_id = dict(zip(ids_df["name"], ids_df["player_id"]))

        # Map starter name -> player_id, then look up their ER and IP for that date
        log["starter_pid"] = log["starter_name"].map(name_to_id)

        starter_lookup = pitcher_logs[["player_id", "date", "earnedRuns", "ip_dec"]].rename(
            columns={"player_id": "starter_pid", "earnedRuns": "starter_er", "ip_dec": "starter_ip"}
        )
        log["date"] = pd.to_datetime(log["date"])
        starter_lookup["date"] = pd.to_datetime(starter_lookup["date"])

        log = log.merge(starter_lookup, on=["starter_pid", "date"], how="left")

        # Bullpen contribution = team totals minus starter
        log["bullpen_runs"] = (log["runs_allowed"] - log["starter_er"]).clip(lower=0)
        log["bullpen_ip"] = (9.0 - log["starter_ip"]).clip(lower=0.1)

        matched = log["bullpen_runs"].notna().sum()
        print(f"  Bullpen stats matched for {matched}/{len(log)} team-games ({matched/len(log):.1%})")
        has_bullpen = matched > 0
    else:
        print("  Pitcher logs not found -- bullpen ERA will be NaN")
        log["bullpen_runs"] = np.nan
        log["bullpen_ip"] = np.nan

    # ── Compute rolling stats per team ───────────────────────────────────
    current_year = datetime.now().year
    team_stats = {}

    for team, grp in log.groupby("team"):
        s = {}

        for w in WINDOWS:
            window_data = grp.tail(w)
            s[f"avg_runs_scored_{w}g"] = window_data["runs_scored"].mean()
            s[f"avg_runs_allowed_{w}g"] = window_data["runs_allowed"].mean()
            s[f"avg_total_runs_{w}g"] = (window_data["runs_scored"] + window_data["runs_allowed"]).mean()
            s[f"win_pct_{w}g"] = (window_data["runs_scored"] > window_data["runs_allowed"]).mean()

            # Rolling bullpen ERA: (sum bullpen_runs / sum bullpen_ip) * 9
            if has_bullpen:
                bp_data = window_data.dropna(subset=["bullpen_runs", "bullpen_ip"])
                if len(bp_data) >= 5:
                    bp_era = (bp_data["bullpen_runs"].sum() / bp_data["bullpen_ip"].sum()) * 9
                    s[f"bullpen_era_{w}g"] = bp_era
                else:
                    s[f"bullpen_era_{w}g"] = np.nan

        # Season stats (current year)
        season_data = grp[grp["date"].dt.year == current_year]
        recent = grp.tail(30)

        if len(season_data) >= 5:
            s["season_avg_scored"] = season_data["runs_scored"].mean()
            s["season_avg_allowed"] = season_data["runs_allowed"].mean()
        else:
            s["season_avg_scored"] = recent["runs_scored"].mean()
            s["season_avg_allowed"] = recent["runs_allowed"].mean()

        # Season bullpen ERA
        if has_bullpen:
            season_bp = season_data.dropna(subset=["bullpen_runs", "bullpen_ip"])
            if len(season_bp) >= 5:
                s["bullpen_era_season"] = (season_bp["bullpen_runs"].sum() / season_bp["bullpen_ip"].sum()) * 9
            else:
                # Fall back to last 30 games if season is too short
                bp_30 = recent.dropna(subset=["bullpen_runs", "bullpen_ip"])
                if len(bp_30) >= 5:
                    s["bullpen_era_season"] = (bp_30["bullpen_runs"].sum() / bp_30["bullpen_ip"].sum()) * 9
                else:
                    s["bullpen_era_season"] = np.nan

        # Days since last game (rest days heading into today's game)
        last_game_date = grp["date"].max()
        days_rest = (predict_date - last_game_date).days
        s["days_rest"] = min(max(int(days_rest), 0), 10)

        team_stats[team] = s

    return team_stats


def build_venue_park_factor(games_df: pd.DataFrame) -> dict:
    """Compute park factors from historical data."""
    league_avg = games_df["total_runs"].mean()
    venue_avg = games_df.groupby("venue_id")["total_runs"].mean()

    park_factors = {}
    for vid in venue_avg.index:
        park_factors[vid] = venue_avg[vid] / league_avg

    return park_factors


def load_pitcher_rolling_stats() -> tuple[pd.DataFrame, dict]:
    """
    Load pitcher game logs and name->id mapping.
    Returns (pitcher_stats_df, name_to_id dict) or (empty df, {}) if not available.
    """
    logs_path = os.path.join(DATA_DIR, "pitcher_game_logs.tsv")
    ids_path = os.path.join(DATA_DIR, "pitcher_ids.tsv")

    if not os.path.exists(logs_path) or not os.path.exists(ids_path):
        return pd.DataFrame(), {}

    logs = pd.read_csv(logs_path, sep="\t")
    logs["date"] = pd.to_datetime(logs["date"], format="mixed")
    ids_df = pd.read_csv(ids_path, sep="\t")
    pitcher_ids = dict(zip(ids_df["name"], ids_df["player_id"]))

    # Compute rolling 3-start stats (same logic as build_features.py)
    logs["earnedRuns"] = pd.to_numeric(logs["earnedRuns"], errors="coerce")
    logs["hits"] = pd.to_numeric(logs["hits"], errors="coerce")
    logs["baseOnBalls"] = pd.to_numeric(logs["baseOnBalls"], errors="coerce")
    logs["strikeOuts"] = pd.to_numeric(logs["strikeOuts"], errors="coerce")
    logs["homeRuns"] = pd.to_numeric(logs["homeRuns"], errors="coerce")
    logs["numberOfPitches"] = pd.to_numeric(logs["numberOfPitches"], errors="coerce")
    logs["battersFaced"] = pd.to_numeric(logs["battersFaced"], errors="coerce")

    def ip_to_dec(ip):
        if pd.isna(ip):
            return np.nan
        try:
            ip = float(ip)
            full = int(ip)
            partial = round(ip - full, 1)
            return full + (partial / 0.3) * (1 / 3)
        except (ValueError, TypeError):
            return np.nan

    logs["ip_dec"] = logs["inningsPitched"].apply(ip_to_dec)
    logs = logs.sort_values(["player_id", "date"]).reset_index(drop=True)

    W = 3
    grp = logs.groupby("player_id")
    logs["rolling_er"] = grp["earnedRuns"].transform(lambda x: x.shift(1).rolling(W, min_periods=1).sum())
    logs["rolling_ip"] = grp["ip_dec"].transform(lambda x: x.shift(1).rolling(W, min_periods=1).sum())
    logs["rolling_bb"] = grp["baseOnBalls"].transform(lambda x: x.shift(1).rolling(W, min_periods=1).sum())
    logs["rolling_h"] = grp["hits"].transform(lambda x: x.shift(1).rolling(W, min_periods=1).sum())
    logs["rolling_k"] = grp["strikeOuts"].transform(lambda x: x.shift(1).rolling(W, min_periods=1).sum())
    logs["rolling_hr"] = grp["homeRuns"].transform(lambda x: x.shift(1).rolling(W, min_periods=1).sum())

    safe_ip = logs["rolling_ip"].replace(0, np.nan)
    logs["pitcher_era_3g"] = (logs["rolling_er"] / safe_ip) * 9
    logs["pitcher_whip_3g"] = (logs["rolling_bb"] + logs["rolling_h"]) / safe_ip
    logs["pitcher_k9_3g"] = (logs["rolling_k"] / safe_ip) * 9
    logs["pitcher_hr9_3g"] = (logs["rolling_hr"] / safe_ip) * 9
    logs["pitcher_ip_3g"] = logs["rolling_ip"] / W
    logs["pitcher_avg_pitches_3g"] = grp["numberOfPitches"].transform(
        lambda x: x.shift(1).rolling(W, min_periods=1).mean()
    )
    logs["pitcher_prev_pitches"] = grp["numberOfPitches"].shift(1)
    logs["pitcher_avg_bf_3g"] = grp["battersFaced"].transform(
        lambda x: x.shift(1).rolling(W, min_periods=1).mean()
    )
    logs["pitcher_prev_bf"] = grp["battersFaced"].shift(1)

    stat_cols = [
        "pitcher_era_3g",
        "pitcher_whip_3g",
        "pitcher_k9_3g",
        "pitcher_hr9_3g",
        "pitcher_ip_3g",
        "pitcher_avg_pitches_3g",
        "pitcher_prev_pitches",
        "pitcher_avg_bf_3g",
        "pitcher_prev_bf",
    ]
    return logs[["player_id", "date"] + stat_cols], pitcher_ids


def get_pitcher_stats(pitcher_name: str, game_date, pitcher_stats_df: pd.DataFrame, pitcher_ids: dict) -> dict:
    """Get the most recent rolling stats for a pitcher before game_date."""
    pid = pitcher_ids.get(pitcher_name)
    if pid is None or pitcher_stats_df.empty:
        return {}
    prior = pitcher_stats_df[(pitcher_stats_df["player_id"] == pid) &
                              (pitcher_stats_df["date"] < game_date)]
    if prior.empty:
        return {}
    latest = prior.iloc[-1]
    stats = latest[[
        "pitcher_era_3g",
        "pitcher_whip_3g",
        "pitcher_k9_3g",
        "pitcher_hr9_3g",
        "pitcher_ip_3g",
        "pitcher_avg_pitches_3g",
        "pitcher_prev_pitches",
        "pitcher_avg_bf_3g",
        "pitcher_prev_bf",
    ]].to_dict()
    days_rest = max((pd.Timestamp(game_date) - pd.Timestamp(latest["date"])).days, 0)
    stats["pitcher_days_rest"] = float(min(days_rest, 14))
    stats["pitcher_short_rest"] = 1.0 if days_rest <= 4 else 0.0
    stats["pitcher_long_rest"] = 1.0 if days_rest >= 7 else 0.0
    stats["pitcher_offseason_gap"] = 1.0 if days_rest >= 20 else 0.0
    avg_pitches = float(stats.get("pitcher_avg_pitches_3g") or 0.0)
    avg_ip = float(stats.get("pitcher_ip_3g") or 0.0)
    avg_bf = float(stats.get("pitcher_avg_bf_3g") or 0.0)
    stats["pitcher_short_leash_score"] = (
        max(min((65.0 - avg_pitches) / 25.0, 1.0), 0.0) +
        max(min((4.8 - avg_ip) / 2.0, 1.0), 0.0) +
        max(min((18.0 - avg_bf) / 8.0, 1.0), 0.0)
    ) / 3.0
    stats["pitcher_workhorse_score"] = (
        max(min((avg_pitches - 90.0) / 20.0, 1.0), 0.0) +
        max(min((avg_ip - 5.8) / 1.8, 1.0), 0.0) +
        max(min((avg_bf - 22.0) / 6.0, 1.0), 0.0)
    ) / 3.0
    stats["pitcher_opener_flag"] = 1.0 if (avg_ip <= 3.5 and avg_pitches <= 55.0 and avg_bf <= 15.0) else 0.0
    stats["pitcher_bulk_flag"] = 1.0 if (avg_ip >= 4.5 and avg_pitches >= 65.0 and avg_bf >= 18.0) else 0.0
    return stats


def load_current_elo_ratings(games_df: pd.DataFrame, predict_date: str) -> dict:
    """
    Load or recompute current Elo ratings for all teams as of predict_date.
    Returns {team_id: {"off_elo": float, "def_elo": float}}.
    """
    import json
    snapshot_path = os.path.join(DATA_DIR, "team_elo_current.json")

    # Use pre-built snapshot if it exists and game data is up to date
    if os.path.exists(snapshot_path):
        with open(snapshot_path) as f:
            raw = json.load(f)
        # Keys are strings in JSON — convert to int
        return {int(k): v for k, v in raw.items()}

    # Fallback: recompute on the fly (slower, ~2s)
    from team_elo import get_current_ratings
    return get_current_ratings(games_df)


def load_current_dc_params(games_df: pd.DataFrame, predict_date: str, dc_cfg: dict | None = None) -> dict | None:
    """
    Load or fit current Dixon-Coles team priors for predict_date.
    """
    dc_cfg = dc_cfg or {}
    from dixon_coles import load_or_fit as dc_load_or_fit

    return dc_load_or_fit(
        games_df,
        predict_date,
        cache_path=os.path.join(DATA_DIR, "dc_params_current.json"),
        halflife=float(dc_cfg.get("time_decay_halflife", 60)),
        window_days=int(dc_cfg.get("training_window", 365)),
        min_games=int(dc_cfg.get("min_games_for_fit", 100)),
        l2_penalty=float(dc_cfg.get("l2_penalty", 0.05)),
    )


def build_team_batting_stats(predict_date: str) -> dict:
    """
    Load team batting logs and compute rolling OPS/BB%/K% per team
    as of predict_date. Returns {team_id: {stat: value}}.
    """
    path = os.path.join(DATA_DIR, "team_batting_logs.tsv")
    if not os.path.exists(path):
        return {}

    logs = pd.read_csv(path, sep="\t")
    logs["date"] = pd.to_datetime(logs["date"])
    cutoff = pd.Timestamp(predict_date)
    logs = logs[logs["date"] < cutoff].copy()

    stats = {}
    for team_id, grp in logs.groupby("team_id"):
        grp = grp.sort_values("date")
        s = {}
        for w in [10, 30]:
            tail = grp.tail(w)
            s[f"rolling_ops_{w}g"]    = tail["ops"].mean()
            s[f"rolling_bb_pct_{w}g"] = tail["bb_pct"].mean()
            s[f"rolling_k_pct_{w}g"]  = tail["k_pct"].mean()
        stats[int(team_id)] = s

    return stats


def build_bullpen_fatigue_stats(predict_date: str) -> dict:
    """
    Load bullpen appearance logs and compute pregame workload features for the
    target date. Returns {team_id: {feature: value}}.
    """
    path = os.path.join(DATA_DIR, "bullpen_appearance_logs.tsv")
    if not os.path.exists(path):
        return {}

    logs = pd.read_csv(path, sep="\t", parse_dates=["date"])
    if logs.empty:
        return {}

    feats = build_pregame_bullpen_features(logs)
    day = pd.Timestamp(predict_date).normalize()
    feats = feats[feats["date"] == day].copy()
    feature_cols = get_pregame_bullpen_feature_cols(list(feats.columns))
    return {
        int(row["team_id"]): {c: row[c] for c in feature_cols}
        for _, row in feats.iterrows()
    }


def build_umpire_stats(predict_date: str) -> dict:
    """
    Load umpire game log and compute each umpire's historical avg total runs
    in games they've called before predict_date (min 20 games).
    Returns {hp_umpire_name: avg_total_runs}.
    """
    path = os.path.join(DATA_DIR, "umpire_game_log.tsv")
    if not os.path.exists(path):
        return {}

    ump = pd.read_csv(path, sep="\t")
    ump["date"] = pd.to_datetime(ump["date"])
    ump = ump[(ump["date"] < pd.Timestamp(predict_date)) & ump["total_runs"].notna()]

    ump_stats = {}
    for name, grp in ump.groupby("hp_umpire"):
        if len(grp) >= 20:
            ump_stats[name] = round(grp["total_runs"].mean(), 3)

    return ump_stats


def fetch_todays_umpires(todays_games: list, game_date: str) -> dict:
    """
    Fetch the scheduled HP umpire for each of today's games.
    Uses schedule endpoint with officials hydration.
    Returns {game_id: hp_umpire_name}.
    """
    try:
        r = requests.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={
                "date": game_date,
                "sportId": 1,
                "hydrate": "officials",
                "gameType": "R",
            },
            timeout=15,
        )
        if r.status_code != 200:
            return {}

        umpires = {}
        for date_entry in r.json().get("dates", []):
            for game in date_entry.get("games", []):
                for official in game.get("officials", []):
                    if official.get("officialType") == "Home Plate":
                        umpires[game["gamePk"]] = official["official"]["fullName"]
                        break
        return umpires
    except Exception:
        return {}


def fetch_todays_lineup_features(todays_games: list) -> dict:
    """
    Fetch same-day starting lineup features for today's games.
    Returns {(game_id, team_id): feature_dict}.
    """
    if not _LINEUPS_AVAILABLE or not todays_games:
        return {}

    rows = fetch_many_game_lineup_features(
        [g["game_id"] for g in todays_games],
        max_workers=min(12, max(1, len(todays_games))),
        timeout=20,
    )
    return {
        (int(r["game_id"]), int(r["team_id"])): r
        for r in rows
    }


def fetch_forecast_weather(venue_ids: list, game_date: str) -> dict:
    """
    Fetch weather forecast from Open-Meteo (free, no API key) for each venue.
    Returns daily venue weather including humidity, dew point, and sunrise/sunset.

    Uses daily max temp, mean humidity/dew point, max wind speed, total
    precipitation, and sunrise/sunset for the game date.
    """
    results = {}
    date_dt = pd.Timestamp(game_date)
    # Open-Meteo forecast covers up to 16 days out; archive covers historical
    days_ahead = (date_dt - pd.Timestamp.today()).days

    for vid in venue_ids:
        if vid not in VENUE_COORDS:
            continue
        lat, lon = VENUE_COORDS[vid]

        # Choose endpoint: archive for past dates, forecast for upcoming
        if days_ahead < 0:
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": lat, "longitude": lon,
                "start_date": game_date, "end_date": game_date,
                "daily": (
                    "temperature_2m_max,windspeed_10m_max,precipitation_sum,"
                    "relative_humidity_2m_mean,dew_point_2m_mean,sunrise,sunset"
                ),
                "temperature_unit": "fahrenheit",
                "windspeed_unit": "mph",
                "timezone": "auto",
            }
        else:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat, "longitude": lon,
                "daily": (
                    "temperature_2m_max,windspeed_10m_max,precipitation_sum,"
                    "relative_humidity_2m_mean,dew_point_2m_mean,sunrise,sunset"
                ),
                "temperature_unit": "fahrenheit",
                "windspeed_unit": "mph",
                "timezone": "auto",
                "forecast_days": min(days_ahead + 2, 16),
            }

        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code != 200:
                continue
            data = r.json().get("daily", {})
            times = data.get("time", [])
            if game_date not in times:
                continue
            idx = times.index(game_date)
            results[vid] = {
                "temp_f":    data["temperature_2m_max"][idx],
                "wind_mph":  data["windspeed_10m_max"][idx],
                "precip_mm": data["precipitation_sum"][idx],
                "humidity_pct": data["relative_humidity_2m_mean"][idx],
                "dew_point_f": data["dew_point_2m_mean"][idx],
                "sunrise": data["sunrise"][idx],
                "sunset": data["sunset"][idx],
            }
        except Exception:
            continue

    return results


def compute_h2h_stats(games_df: pd.DataFrame, todays_games: list, game_date: str) -> dict:
    """
    For each of today's matchups, compute average total runs from the last 10
    prior games between those two teams (in either direction).
    Returns {(home_team, away_team): {"h2h_avg_total_runs": float}}
    """
    games_df = games_df.copy()
    games_df["date"] = pd.to_datetime(games_df["date"])
    cutoff = pd.Timestamp(game_date)

    h2h = {}
    for game in todays_games:
        home = game["home_name"]
        away = game["away_name"]

        mask = (
            ((games_df["home_team"] == home) & (games_df["away_team"] == away)) |
            ((games_df["home_team"] == away) & (games_df["away_team"] == home))
        ) & (games_df["date"] < cutoff)

        prior = games_df[mask].sort_values("date").tail(10)
        if len(prior) >= 3:
            h2h[(home, away)] = {"h2h_avg_total_runs": prior["total_runs"].mean()}

    return h2h


def fetch_todays_games(target_date=None, include_all_games=False):
    """Fetch MLB games for the target date, defaulting to pregame-only."""
    if target_date is None:
        target_date = datetime.now().strftime("%Y-%m-%d")

    games = statsapi.schedule(date=target_date, sportId=1)
    regular = [g for g in games if g["game_type"] == "R"]

    if include_all_games:
        regular = sorted(
            regular,
            key=lambda g: pd.to_datetime(g.get("game_datetime"), utc=True, errors="coerce"),
        )
        return regular, []

    now_utc = pd.Timestamp.now(tz="UTC")
    pregame_statuses = {
        "scheduled",
        "pre-game",
        "warmup",
        "delayed start",
    }

    upcoming = []
    skipped = []
    for game in regular:
        status = str(game.get("status", "")).strip().lower()
        game_time = pd.to_datetime(game.get("game_datetime"), utc=True, errors="coerce")
        is_future = pd.notna(game_time) and game_time > now_utc
        is_pregame = status in pregame_statuses

        if is_future or is_pregame:
            upcoming.append(game)
        else:
            skipped.append(game)

    return upcoming, skipped


def fetch_live_lines(api_key):
    """
    Fetch today's MLB totals from The Odds API.
    Returns dict keyed by (away_team, home_team) -> consensus line.
    """
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "totals",
        "oddsFormat": "american",
        "bookmakers": "draftkings,fanduel,betmgm,pinnacle,caesars",
    }
    try:
        r = requests.get("https://api.the-odds-api.com/v4/sports/baseball_mlb/odds",
                         params=params, timeout=15)
        if r.status_code != 200:
            print(f"  Odds API error: {r.status_code}")
            return {}
        remaining = r.headers.get("x-requests-remaining", "?")
        print(f"  Odds API credits remaining: {remaining}")
    except Exception as e:
        print(f"  Odds API request failed: {e}")
        return {}

    lines = {}
    for game in r.json():
        home = game["home_team"]
        away = game["away_team"]
        game_time = game.get("commence_time")
        book_lines = []
        book_odds  = []
        book_line_map = {}
        for bm in game.get("bookmakers", []):
            for market in bm.get("markets", []):
                if market["key"] == "totals":
                    for outcome in market["outcomes"]:
                        if outcome["name"] == "Over":
                            book_lines.append(outcome["point"])
                            book_odds.append(outcome.get("price", -110))
                            book_line_map[f"{bm['key']}_line"] = outcome["point"]
                            break
        if book_lines:
            mid = len(book_lines) // 2
            consensus_line = sorted(book_lines)[mid]
            # Pair each line with its odds, pick the median line's odds
            paired = sorted(zip(book_lines, book_odds))
            consensus_odds = paired[mid][1]
            lines[(away, home)] = {
                "line": consensus_line,
                "odds": consensus_odds,  # American odds (e.g. -110)
                "num_books": len(book_lines),
                "commence_time": game_time,
                **book_line_map,
            }

    return lines


def predict_game(game, team_stats, park_factors, model, meta, residual_std,
                 pitcher_stats_df=None, pitcher_ids=None, weather_by_venue=None,
                 h2h_by_matchup=None, team_batting_stats=None,
                 umpire_stats=None, todays_umpires=None,
                 elo_ratings=None, dc_params=None,
                 league_environment=None,
                 todays_lineups=None, bullpen_fatigue_stats=None,
                 uncertainty_model=None, uncertainty_cfg=None,
                 high_tail_model=None, high_tail_cfg=None,
                 low_tail_model=None, low_tail_cfg=None):
    """Generate prediction for a single game."""
    home = game["home_name"]
    away = game["away_name"]
    venue_id = game["venue_id"]
    game_date = pd.to_datetime(game.get("game_datetime"), utc=True, errors="coerce")
    if pd.isna(game_date):
        game_date = pd.Timestamp.now(tz="UTC")
    game_date = game_date.tz_convert(None)

    if home not in team_stats or away not in team_stats:
        return None

    hs = team_stats[home]
    aws = team_stats[away]

    features = {}

    # Team IDs
    features["home_team_id"] = game["home_id"]
    features["away_team_id"] = game["away_id"]

    # Home team features
    for key in hs:
        features[f"home_{key}"] = hs[key]

    # Away team features
    for key in aws:
        features[f"away_{key}"] = aws[key]

    # Combined features
    for w in WINDOWS:
        features[f"combined_scoring_{w}g"] = hs[f"avg_runs_scored_{w}g"] + aws[f"avg_runs_scored_{w}g"]
        features[f"combined_allowed_{w}g"] = hs[f"avg_runs_allowed_{w}g"] + aws[f"avg_runs_allowed_{w}g"]
        features[f"home_offense_vs_away_pitching_{w}g"] = hs[f"avg_runs_scored_{w}g"] - aws[f"avg_runs_allowed_{w}g"]
        features[f"away_offense_vs_home_pitching_{w}g"] = aws[f"avg_runs_scored_{w}g"] - hs[f"avg_runs_allowed_{w}g"]

    features["combined_season_scoring"] = hs["season_avg_scored"] + aws["season_avg_scored"]
    features["combined_season_allowed"] = hs["season_avg_allowed"] + aws["season_avg_allowed"]

    # Combined bullpen ERA features
    for w in WINDOWS:
        h_bp = hs.get(f"bullpen_era_{w}g", np.nan)
        a_bp = aws.get(f"bullpen_era_{w}g", np.nan)
        if not np.isnan(h_bp) and not np.isnan(a_bp):
            features[f"combined_bullpen_era_{w}g"] = h_bp + a_bp

    h_bp_s = hs.get("bullpen_era_season", np.nan)
    a_bp_s = aws.get("bullpen_era_season", np.nan)
    if not np.isnan(h_bp_s) and not np.isnan(a_bp_s):
        features["combined_bullpen_era_season"] = h_bp_s + a_bp_s

    # Park factor
    features["park_factor"] = park_factors.get(venue_id, 1.0)

    if league_environment:
        for key in LEAGUE_ENV_FEATURES:
            if key in league_environment:
                features[key] = league_environment[key]

    # Rest days
    features["home_days_rest"] = hs.get("days_rest", np.nan)
    features["away_days_rest"] = aws.get("days_rest", np.nan)

    # Head-to-head features
    if h2h_by_matchup:
        h2h = h2h_by_matchup.get((home, away), {})
        if h2h:
            features["h2h_avg_total_runs"] = h2h["h2h_avg_total_runs"]

    # Elo ratings (schedule-adjusted team strength)
    if elo_ratings:
        home_id = game["home_id"]
        away_id = game["away_id"]
        h_elo = elo_ratings.get(home_id, {})
        a_elo = elo_ratings.get(away_id, {})
        if h_elo and a_elo:
            from team_elo import LEAGUE_AVG_RUNS, INIT_ELO, ELO_SCALE, HOME_ADVANTAGE
            h_off = h_elo["off_elo"]
            h_def = h_elo["def_elo"]
            a_off = a_elo["off_elo"]
            a_def = a_elo["def_elo"]
            features["home_off_elo"] = h_off
            features["home_def_elo"] = h_def
            features["away_off_elo"] = a_off
            features["away_def_elo"] = a_def
            exp_home = LEAGUE_AVG_RUNS + HOME_ADVANTAGE + (h_off - INIT_ELO) * ELO_SCALE - (a_def - INIT_ELO) * ELO_SCALE
            exp_away = LEAGUE_AVG_RUNS + (a_off - INIT_ELO) * ELO_SCALE - (h_def - INIT_ELO) * ELO_SCALE
            features["elo_expected_total"] = round(exp_home + exp_away, 3)
            features["elo_offense_sum"]    = round(h_off + a_off, 2)
            features["elo_matchup_score"]  = round((h_off + a_off) - (h_def + a_def), 2)
            features["elo_home_edge"]      = round((h_off - a_def) - (a_off - h_def), 2)

    # Dixon-Coles priors
    if dc_params:
        from dixon_coles import predict_game as dc_predict_game

        dc_pred = dc_predict_game(dc_params, home, away)
        if dc_pred:
            features["home_dc_attack"] = dc_pred["home_attack"]
            features["home_dc_defense"] = dc_pred["home_defense"]
            features["away_dc_attack"] = dc_pred["away_attack"]
            features["away_dc_defense"] = dc_pred["away_defense"]
            features["dc_lambda_home"] = dc_pred["lambda_home"]
            features["dc_lambda_away"] = dc_pred["lambda_away"]
            features["dc_expected_total"] = dc_pred["expected_total"]
            features["dc_home_edge"] = dc_pred["home_edge"]
            features["dc_fit_n_games"] = dc_params.get("n_games", 0)
            features["_dc_expected_total"] = dc_pred["expected_total"]
            features["_dc_lambda_home"] = dc_pred["lambda_home"]
            features["_dc_lambda_away"] = dc_pred["lambda_away"]

    # Dome flag — set before weather so we can suppress irrelevant readings
    is_dome = 1 if venue_id in DOME_VENUE_IDS else 0
    features["is_dome"] = is_dome

    # Weather features (skip for dome/retractable venues)
    if weather_by_venue and not is_dome:
        wx = weather_by_venue.get(venue_id, {})
        if wx:
            features["temp_f"] = wx["temp_f"]
            features["wind_mph"] = wx["wind_mph"]
            features["precip_mm"] = wx["precip_mm"]
            features["humidity_pct"] = wx.get("humidity_pct")
            features["dew_point_f"] = wx.get("dew_point_f")

    time_features = compute_local_time_features(
        game_datetime=game.get("game_datetime"),
        venue_id=venue_id,
        sunrise=(weather_by_venue or {}).get(venue_id, {}).get("sunrise"),
        sunset=(weather_by_venue or {}).get(venue_id, {}).get("sunset"),
    )
    features["first_pitch_local_hour"] = time_features["first_pitch_local_hour"]
    features["is_night_game"] = time_features["is_night_game"]

    # Team batting quality features (OPS, BB%, K%)
    if team_batting_stats:
        home_id = game["home_id"]
        away_id = game["away_id"]
        hb = team_batting_stats.get(home_id, {})
        ab = team_batting_stats.get(away_id, {})
        for k, v in hb.items():
            features[f"home_{k}"] = v
        for k, v in ab.items():
            features[f"away_{k}"] = v

    # Same-day lineup strength features
    if todays_lineups:
        home_lineup = todays_lineups.get((game["game_id"], game["home_id"]), {})
        away_lineup = todays_lineups.get((game["game_id"], game["away_id"]), {})
        for k, v in home_lineup.items():
            if k.startswith("lineup_"):
                features[f"home_{k}"] = v
        for k, v in away_lineup.items():
            if k.startswith("lineup_"):
                features[f"away_{k}"] = v

    # Lineup deltas tell the model whether today's nine is stronger or weaker
    # than the team's recent baseline, which is more stable than raw lineup OPS.
    for side in ["home", "away"]:
        lineup_ops = features.get(f"{side}_lineup_avg_ops")
        rolling_ops = features.get(f"{side}_rolling_ops_30g")
        lineup_bb = features.get(f"{side}_lineup_avg_bb_pct")
        rolling_bb = features.get(f"{side}_rolling_bb_pct_30g")
        lineup_k = features.get(f"{side}_lineup_avg_k_pct")
        rolling_k = features.get(f"{side}_rolling_k_pct_30g")
        top3_ops = features.get(f"{side}_lineup_top3_avg_ops")
        bot3_ops = features.get(f"{side}_lineup_bottom3_avg_ops")

        if lineup_ops is not None and rolling_ops is not None:
            features[f"{side}_lineup_delta_ops_30g"] = lineup_ops - rolling_ops
        if lineup_bb is not None and rolling_bb is not None:
            features[f"{side}_lineup_delta_bb_pct_30g"] = lineup_bb - rolling_bb
        if lineup_k is not None and rolling_k is not None:
            features[f"{side}_lineup_delta_k_pct_30g"] = lineup_k - rolling_k
        if top3_ops is not None and rolling_ops is not None:
            features[f"{side}_lineup_top3_delta_ops_30g"] = top3_ops - rolling_ops
        if top3_ops is not None and bot3_ops is not None:
            features[f"{side}_lineup_depth_gap_ops"] = top3_ops - bot3_ops

    # Bullpen workload / availability features based on prior relief usage.
    if bullpen_fatigue_stats:
        for side, team_id in [("home", game["home_id"]), ("away", game["away_id"])]:
            usage = bullpen_fatigue_stats.get(team_id, {})
            for k, v in usage.items():
                features[f"{side}_{k}"] = v

    # Umpire scoring factor
    if todays_umpires and umpire_stats:
        game_pk = game.get("game_id")
        hp_ump = todays_umpires.get(game_pk)
        if hp_ump:
            features["_hp_umpire"] = hp_ump  # carry through for display
            if hp_ump in umpire_stats:
                features["ump_avg_total_runs"] = umpire_stats[hp_ump]
                features["_ump_avg_display"]   = umpire_stats[hp_ump]

    # Pitcher stats (if available)
    if pitcher_stats_df is not None and not pitcher_stats_df.empty and pitcher_ids:
        home_p_name = game.get("home_probable_pitcher", "")
        away_p_name = game.get("away_probable_pitcher", "")
        home_p_stats = get_pitcher_stats(home_p_name, game_date, pitcher_stats_df, pitcher_ids)
        away_p_stats = get_pitcher_stats(away_p_name, game_date, pitcher_stats_df, pitcher_ids)
        for k, v in home_p_stats.items():
            features[f"home_{k}"] = v
        for k, v in away_p_stats.items():
            features[f"away_{k}"] = v
        if home_p_stats and away_p_stats:
            features["combined_pitcher_era"] = home_p_stats.get("pitcher_era_3g", np.nan) + away_p_stats.get("pitcher_era_3g", np.nan)
            features["combined_pitcher_whip"] = home_p_stats.get("pitcher_whip_3g", np.nan) + away_p_stats.get("pitcher_whip_3g", np.nan)
            features["combined_pitcher_k9"] = home_p_stats.get("pitcher_k9_3g", np.nan) + away_p_stats.get("pitcher_k9_3g", np.nan)
            features["combined_pitcher_avg_pitches_3g"] = (
                home_p_stats.get("pitcher_avg_pitches_3g", np.nan) +
                away_p_stats.get("pitcher_avg_pitches_3g", np.nan)
            )
            features["combined_pitcher_prev_pitches"] = (
                home_p_stats.get("pitcher_prev_pitches", np.nan) +
                away_p_stats.get("pitcher_prev_pitches", np.nan)
            )
            features["combined_pitcher_avg_bf_3g"] = (
                home_p_stats.get("pitcher_avg_bf_3g", np.nan) +
                away_p_stats.get("pitcher_avg_bf_3g", np.nan)
            )
            features["combined_pitcher_short_rest_flags"] = (
                home_p_stats.get("pitcher_short_rest", 0.0) +
                away_p_stats.get("pitcher_short_rest", 0.0)
            )
            features["combined_pitcher_offseason_gap_flags"] = (
                home_p_stats.get("pitcher_offseason_gap", 0.0) +
                away_p_stats.get("pitcher_offseason_gap", 0.0)
            )
            features["combined_pitcher_short_leash_score"] = (
                home_p_stats.get("pitcher_short_leash_score", 0.0) +
                away_p_stats.get("pitcher_short_leash_score", 0.0)
            )
            features["combined_pitcher_workhorse_score"] = (
                home_p_stats.get("pitcher_workhorse_score", 0.0) +
                away_p_stats.get("pitcher_workhorse_score", 0.0)
            )
            features["combined_pitcher_opener_flags"] = (
                home_p_stats.get("pitcher_opener_flag", 0.0) +
                away_p_stats.get("pitcher_opener_flag", 0.0)
            )
            features["combined_pitcher_bulk_flags"] = (
                home_p_stats.get("pitcher_bulk_flag", 0.0) +
                away_p_stats.get("pitcher_bulk_flag", 0.0)
            )

    feature_row = pd.DataFrame([features])
    X, point_predictions, point_meta = predict_point_outputs(model, meta, feature_row)
    raw_predicted_total = float(point_predictions[0])
    driver_summary = point_meta["driver_summary"]
    model_family = point_meta["model_family"]
    baseline_total = float(point_meta["baseline_total"][0]) if len(point_meta["baseline_total"]) else 0.0
    raw_total_before_calibration = float(point_meta.get("raw_total", [raw_predicted_total])[0])
    side_predictions = {
        side: float(values[0]) for side, values in point_meta.get("side_predictions", {}).items()
    }
    side_components = {
        side: float(values[0]) for side, values in point_meta.get("side_components", {}).items()
    }
    prediction_mode = point_meta.get("prediction_mode", meta.get("prediction_mode", "direct"))
    prediction_std = estimate_prediction_std(
        X,
        raw_predicted_total,
        uncertainty_model,
        uncertainty_cfg,
        residual_std,
    )
    high_tail_prob = predict_high_tail_prob(
        X,
        raw_predicted_total,
        high_tail_model,
        high_tail_cfg,
    )
    low_tail_prob = predict_low_tail_prob(
        X,
        raw_predicted_total,
        low_tail_model,
        low_tail_cfg,
    )
    predicted_total = round(raw_predicted_total, 1)

    # Confidence intervals using residual distribution
    # We model the actual total as Normal(predicted, residual_std)
    result = {
        "game_id": game.get("game_id"),
        "commence_time": game.get("game_datetime"),
        "home_team": home,
        "away_team": away,
        "venue": game["venue_name"],
        "home_pitcher": game.get("home_probable_pitcher", "TBD"),
        "away_pitcher": game.get("away_probable_pitcher", "TBD"),
        "predicted_total": predicted_total,
        "prediction_std": round(prediction_std, 2),
        "high_tail_prob_9p5": round(high_tail_prob * 100, 1) if high_tail_prob is not None else None,
        "high_tail_sigma": round(
            adjusted_sigma_for_line(
                raw_predicted_total,
                prediction_std,
                9.5,
                high_tail_prob,
                high_tail_cfg,
                low_tail_prob=low_tail_prob,
                low_tail_cfg=low_tail_cfg,
            ),
            2,
        ) if high_tail_prob is not None else None,
        "low_tail_prob_7p5": round(low_tail_prob * 100, 1) if low_tail_prob is not None else None,
        "low_tail_sigma": round(
            adjusted_sigma_for_line(
                raw_predicted_total,
                prediction_std,
                7.5,
                high_tail_prob,
                high_tail_cfg,
                low_tail_prob=low_tail_prob,
                low_tail_cfg=low_tail_cfg,
            ),
            2,
        ) if low_tail_prob is not None else None,
        "model_family": model_family,
        "calibration_adjustment": round(raw_predicted_total - raw_total_before_calibration, 3),
        "xgb_prediction_mode": prediction_mode,
        "xgb_residual": round(sum(side_components.values()), 3) if side_components else round(float(point_meta.get("xgb_component", [0.0])[0]), 3),
        "xgb_home_component": round(side_components.get("home", 0.0), 3) if side_components else None,
        "xgb_away_component": round(side_components.get("away", 0.0), 3) if side_components else None,
        "xgb_top_buckets": driver_summary["top_buckets"],
        "xgb_top_features": driver_summary["top_features"],
        "predicted_home_runs": round(side_predictions.get("home"), 2) if "home" in side_predictions else None,
        "predicted_away_runs": round(side_predictions.get("away"), 2) if "away" in side_predictions else None,
        "dc_baseline_total": round(baseline_total, 3) if baseline_total else None,
        # Carry through display values
        "home_off_elo":       features.get("home_off_elo"),
        "home_def_elo":       features.get("home_def_elo"),
        "away_off_elo":       features.get("away_off_elo"),
        "away_def_elo":       features.get("away_def_elo"),
        "home_dc_attack":     features.get("home_dc_attack"),
        "home_dc_defense":    features.get("home_dc_defense"),
        "away_dc_attack":     features.get("away_dc_attack"),
        "away_dc_defense":    features.get("away_dc_defense"),
        "dc_expected_total":  features.get("_dc_expected_total"),
        "dc_lambda_home":     features.get("_dc_lambda_home"),
        "dc_lambda_away":     features.get("_dc_lambda_away"),
        "temp_f":             features.get("temp_f"),
        "wind_mph":           features.get("wind_mph"),
        "precip_mm":          features.get("precip_mm"),
        "humidity_pct":       features.get("humidity_pct"),
        "dew_point_f":        features.get("dew_point_f"),
        "first_pitch_local_hour": features.get("first_pitch_local_hour"),
        "is_night_game":      features.get("is_night_game"),
        "home_lineup_avg_ops": features.get("home_lineup_avg_ops"),
        "away_lineup_avg_ops": features.get("away_lineup_avg_ops"),
        "home_lineup_delta_ops_30g": features.get("home_lineup_delta_ops_30g"),
        "away_lineup_delta_ops_30g": features.get("away_lineup_delta_ops_30g"),
        "home_lineup_confirmed": features.get("home_lineup_confirmed"),
        "away_lineup_confirmed": features.get("away_lineup_confirmed"),
        "home_lineup_platoon_adv_batters": features.get("home_lineup_platoon_adv_batters"),
        "away_lineup_platoon_adv_batters": features.get("away_lineup_platoon_adv_batters"),
        "home_pitcher_days_rest": features.get("home_pitcher_days_rest"),
        "away_pitcher_days_rest": features.get("away_pitcher_days_rest"),
        "home_pitcher_avg_pitches_3g": features.get("home_pitcher_avg_pitches_3g"),
        "away_pitcher_avg_pitches_3g": features.get("away_pitcher_avg_pitches_3g"),
        "home_pitcher_prev_pitches": features.get("home_pitcher_prev_pitches"),
        "away_pitcher_prev_pitches": features.get("away_pitcher_prev_pitches"),
        "home_pitcher_avg_bf_3g": features.get("home_pitcher_avg_bf_3g"),
        "away_pitcher_avg_bf_3g": features.get("away_pitcher_avg_bf_3g"),
        "home_pitcher_short_leash_score": features.get("home_pitcher_short_leash_score"),
        "away_pitcher_short_leash_score": features.get("away_pitcher_short_leash_score"),
        "home_pitcher_opener_flag": features.get("home_pitcher_opener_flag"),
        "away_pitcher_opener_flag": features.get("away_pitcher_opener_flag"),
        "home_bullpen_used_pitches_3d": features.get("home_bullpen_used_pitches_3d"),
        "away_bullpen_used_pitches_3d": features.get("away_bullpen_used_pitches_3d"),
        "home_bullpen_b2b_arms": features.get("home_bullpen_b2b_arms"),
        "away_bullpen_b2b_arms": features.get("away_bullpen_b2b_arms"),
        "home_bullpen_top4_available_score": features.get("home_bullpen_top4_available_score"),
        "away_bullpen_top4_available_score": features.get("away_bullpen_top4_available_score"),
        "home_bullpen_top4_burned_score": features.get("home_bullpen_top4_burned_score"),
        "away_bullpen_top4_burned_score": features.get("away_bullpen_top4_burned_score"),
        "home_bullpen_top2_used_yesterday": features.get("home_bullpen_top2_used_yesterday"),
        "away_bullpen_top2_used_yesterday": features.get("away_bullpen_top2_used_yesterday"),
        "home_bullpen_top4_b2b": features.get("home_bullpen_top4_b2b"),
        "away_bullpen_top4_b2b": features.get("away_bullpen_top4_b2b"),
        "hp_umpire":          features.get("_hp_umpire"),
        "ump_avg_total_runs": features.get("_ump_avg_display"),
        "is_dome":            features.get("is_dome", 0),
    }

    result = add_team_side_metrics(
        result,
        side_distribution=get_side_residual_distribution(meta),
    )

    # Over/under probabilities for common lines
    for line in [6.5, 7.5, 8.5, 9.5, 10.5, 11.5]:
        p_over = probability_over_line(
            predicted_total,
            prediction_std,
            line,
            high_tail_prob=high_tail_prob,
            high_tail_cfg=high_tail_cfg,
            low_tail_prob=low_tail_prob,
            low_tail_cfg=low_tail_cfg,
        )
        result[f"over_{line}"] = round(p_over * 100, 1)
        result[f"under_{line}"] = round((1 - p_over) * 100, 1)

    return result



def main():
    parser = argparse.ArgumentParser(description="Predict MLB total runs for a target date.")
    parser.add_argument("date", nargs="?", help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument(
        "--all-games",
        action="store_true",
        help="Include live and final games too; default is upcoming/pregame only.",
    )
    args = parser.parse_args()
    target_date = args.date

    print("Loading config...")
    cfg = load_model_config()
    bet_cfg  = cfg["betting"]
    disp_cfg = cfg["display"]
    bankroll_cfg = cfg.get("bankroll", {})
    overrides = cfg.get("overrides") or {}
    if overrides:
        print(f"  Manual overrides active for: {list(overrides.keys())}")

    print("Loading model...")
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

    print("Loading historical data...")
    games_df = load_historical_data()

    print("Computing residual std for confidence intervals...")
    residual_std = compute_residual_std(model, meta)
    print(f"  Residual std: {residual_std:.2f} runs")
    if uncertainty_model is not None:
        print("  Dynamic uncertainty model loaded")
    else:
        print("  Dynamic uncertainty model unavailable - using global sigma")
    if high_tail_model is not None:
        print("  High-tail probability model loaded")
    if low_tail_model is not None:
        print("  Low-tail probability model loaded")

    date_str = target_date or datetime.now().strftime("%Y-%m-%d")

    print("Building team rolling stats...")
    team_stats = build_team_rolling_stats(games_df, predict_date=date_str)
    park_factors = build_venue_park_factor(games_df)
    league_environment = build_current_league_environment(games_df, pd.Timestamp(date_str))

    print("Loading pitcher stats...")
    pitcher_stats_df, pitcher_ids = load_pitcher_rolling_stats()
    if pitcher_stats_df.empty:
        print("  Pitcher stats not available - predictions will use team stats only")
    else:
        print(f"  Loaded stats for {pitcher_stats_df['player_id'].nunique()} pitchers")
    print(f"Fetching games for {date_str}...")
    todays_games, skipped_games = fetch_todays_games(target_date, include_all_games=args.all_games)
    if args.all_games:
        print(f"  Found {len(todays_games)} regular season games (--all-games)")
    else:
        print(f"  Found {len(todays_games)} upcoming regular season games")
        if skipped_games:
            live_count = sum(
                1 for g in skipped_games
                if str(g.get("status", "")).strip().lower() not in {"final", "game over"}
            )
            final_count = len(skipped_games) - live_count
            print(f"  Skipped {len(skipped_games)} games already underway/final "
                  f"({live_count} live, {final_count} final)")

    # Fetch live lines if API key is available
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    api_key = config.get("odds_api_key") or os.environ.get("ODDS_API_KEY")

    live_lines = {}
    if api_key:
        print("Fetching live lines from The Odds API...")
        live_lines = fetch_live_lines(api_key)
        print(f"  Got lines for {len(live_lines)} games")
    else:
        print("No Odds API key found - run without lines (add key to config.json)")

    # Fetch Kalshi prediction market lines (always free, no key needed)
    kalshi_lines = {}
    if _KALSHI_AVAILABLE and todays_games:
        print("Fetching Kalshi prediction market lines...")
        kalshi_lines = fetch_kalshi_lines(date_str)
        upcoming_keys = {(g["away_name"], g["home_name"]) for g in todays_games}
        matched_kalshi = sum(1 for key in kalshi_lines if key in upcoming_keys)
        print(f"  Kalshi lines found for {matched_kalshi}/{len(todays_games)} games")
        missing_kalshi = sorted(upcoming_keys - set(kalshi_lines))
        if missing_kalshi:
            preview = ", ".join(f"{away} @ {home}" for away, home in missing_kalshi[:6])
            if len(missing_kalshi) > 6:
                preview += ", ..."
            print(f"  Kalshi matches missing for: {preview}")

    # Note: Kalshi is shown as supplementary data only — not substituted as the line.
    # Use The Odds API (or config.json key) for real sportsbook lines.

    # Head-to-head stats
    h2h_by_matchup = {}
    if todays_games:
        h2h_by_matchup = compute_h2h_stats(games_df, todays_games, date_str)
        print(f"  H2H history found for {len(h2h_by_matchup)}/{len(todays_games)} matchups")

    # Fetch weather forecasts (outdoor venues only)
    weather_by_venue = {}
    if todays_games:
        outdoor_venue_ids = list({g["venue_id"] for g in todays_games
                                  if g["venue_id"] not in DOME_VENUE_IDS})
        dome_count = len({g["venue_id"] for g in todays_games}) - len(outdoor_venue_ids)
        print(f"Fetching weather for {len(outdoor_venue_ids)} outdoor venues "
              f"({dome_count} dome/retractable skipped)...")
        weather_by_venue = fetch_forecast_weather(outdoor_venue_ids, date_str)
        covered = sum(1 for vid in outdoor_venue_ids if vid in weather_by_venue)
        print(f"  Weather fetched for {covered}/{len(outdoor_venue_ids)} venues")

    todays_lineups = {}
    if todays_games:
        print("Fetching confirmed lineups...")
        todays_lineups = fetch_todays_lineup_features(todays_games)
        confirmed_games = len({gid for gid, _ in todays_lineups.keys()})
        print(f"  Lineups found for {confirmed_games}/{len(todays_games)} games")

    # Team batting quality stats (OPS, BB%, K%)
    print("Loading team batting stats...")
    team_batting_stats = build_team_batting_stats(date_str)
    if team_batting_stats:
        print(f"  Batting stats loaded for {len(team_batting_stats)} teams")
    else:
        print("  Team batting stats not available (run collect_team_batting.py)")

    print("Loading bullpen fatigue stats...")
    bullpen_fatigue_stats = build_bullpen_fatigue_stats(date_str)
    if bullpen_fatigue_stats:
        print(f"  Bullpen fatigue loaded for {len(bullpen_fatigue_stats)} teams")
    else:
        print("  Bullpen fatigue stats not available (run collect_bullpen_usage.py)")

    feature_names = set(meta.get("features", []))
    needs_elo = any("elo" in f for f in feature_names)
    needs_dc_features = any(
        f.startswith("home_dc_") or f.startswith("away_dc_") or f.startswith("dc_")
        for f in feature_names
    )

    # Elo ratings (legacy fallback for older models)
    elo_ratings = {}
    if needs_elo:
        print("Loading Elo ratings...")
        elo_ratings = load_current_elo_ratings(games_df, date_str)
        print(f"  Elo ratings loaded for {len(elo_ratings)} teams")

    # Umpire stats
    print("Loading umpire stats...")
    umpire_stats = build_umpire_stats(date_str)
    todays_umpires = {}
    if umpire_stats and todays_games:
        print(f"  Umpire history for {len(umpire_stats)} umpires")
        todays_umpires = fetch_todays_umpires(todays_games, date_str)
        matched_umps = sum(1 for g in todays_games
                           if todays_umpires.get(g["game_id"]) in umpire_stats)
        print(f"  Today's umpires matched: {matched_umps}/{len(todays_games)}")
    else:
        print("  Umpire data not available (run collect_umpires.py)")

    # Dixon-Coles priors / optional blend
    dc_params = None
    dc_cfg = cfg.get("dixon_coles", {})
    use_dc = needs_dc_features or dc_cfg.get("enabled", True)
    if use_dc:
        print("Loading Dixon-Coles priors...")
        try:
            dc_params = load_current_dc_params(games_df, date_str, dc_cfg=dc_cfg)
            if dc_params:
                print(f"  DC fit on {dc_params['n_games']} games  "
                      f"(mu={dc_params['mu']:.3f}  home_adv={dc_params['home_adv']:.3f})")
            else:
                print("  DC fit failed - using XGBoost only")
        except Exception as e:
            print(f"  DC error: {e} - using XGBoost only")

    print("Generating predictions...\n")
    predictions = []
    for game in todays_games:
        pred = predict_game(game, team_stats, park_factors, model, meta, residual_std,
                            pitcher_stats_df=pitcher_stats_df, pitcher_ids=pitcher_ids,
                            weather_by_venue=weather_by_venue,
                            h2h_by_matchup=h2h_by_matchup,
                            team_batting_stats=team_batting_stats,
                            umpire_stats=umpire_stats,
                            todays_umpires=todays_umpires,
                            elo_ratings=elo_ratings,
                            dc_params=dc_params,
                            league_environment=league_environment,
                            todays_lineups=todays_lineups,
                            bullpen_fatigue_stats=bullpen_fatigue_stats,
                            uncertainty_model=uncertainty_model,
                            uncertainty_cfg=uncertainty_cfg,
                            high_tail_model=high_tail_model,
                            high_tail_cfg=high_tail_cfg,
                            low_tail_model=low_tail_model,
                            low_tail_cfg=low_tail_cfg)
        if pred:
            key = (game["away_name"], game["home_name"])
            if key in live_lines:
                line_data = live_lines[key]
                pred["posted_line"] = line_data["line"]
                pred["posted_odds"] = line_data.get("odds", -110)
                pred["market_num_books"] = line_data.get("num_books")
                for col in ["commence_time", "pinnacle_line", "draftkings_line", "fanduel_line", "betmgm_line", "caesars_line"]:
                    if col in line_data:
                        pred[col] = line_data[col]
            # Attach Kalshi market detail if available
            if key in kalshi_lines:
                kd = kalshi_lines[key]
                pred["kalshi_line"]      = kd["kalshi_line"]
                pred["kalshi_anchor_strike"] = kd.get("anchor_strike")
                pred["kalshi_over_pct"]  = kd["implied_over_pct"]
                pred["kalshi_yes_ask"]   = kd["yes_ask"]
                # Use Kalshi line as market reference for edge model when no
                # sportsbook line is available. Kalshi strikes track sportsbook
                # totals closely enough to keep edge-model features in range.
                if "posted_line" not in pred:
                    pred["posted_line"] = float(kd["kalshi_line"])
                    pred["posted_odds"] = -110  # standard vig placeholder
            pred["_bankroll_cfg"] = bankroll_cfg
            pred["_high_tail_cfg"] = high_tail_cfg
            pred["_low_tail_cfg"] = low_tail_cfg
            predictions.append(pred)

    # Dixon-Coles blend: mix DC Poisson prediction with XGBoost prediction
    if dc_params and dc_cfg.get("enabled", True):
        from dixon_coles import predict_game as dc_predict
        dc_weight = float(dc_cfg.get("dc_weight", 0.0))
        xgb_weight = 1.0 - dc_weight

        for pred in predictions:
            if dc_weight <= 0:
                continue
            dc_pred = dc_predict(dc_params, pred["home_team"], pred["away_team"])
            if dc_pred is None:
                continue

            # Blend expected totals
            blended_total = round(
                xgb_weight * pred["predicted_total"] +
                dc_weight  * dc_pred["expected_total"],
                1
            )
            pred["predicted_total"]  = blended_total
            pred["dc_expected_total"] = dc_pred["expected_total"]
            pred["dc_lambda_home"]    = dc_pred["lambda_home"]
            pred["dc_lambda_away"]    = dc_pred["lambda_away"]

            # Blend P(over X) at each standard line
            for line in [6.5, 7.5, 8.5, 9.5, 10.5, 11.5]:
                xgb_p = pred[f"over_{line}"] / 100.0
                dc_p  = dc_pred["p_over"].get(line, xgb_p)
                blended_p = xgb_weight * xgb_p + dc_weight * dc_p
                pred[f"over_{line}"]  = round(blended_p * 100, 1)
                pred[f"under_{line}"] = round((1 - blended_p) * 100, 1)

            # Re-apply edge calc if we have a posted line
            if "posted_line" in pred:
                posted = pred["posted_line"]
                sigma = float(pred.get("prediction_std", residual_std))
                xgb_p_over = probability_over_line(
                    pred["predicted_total"],
                    sigma,
                    posted,
                    high_tail_prob=(float(pred["high_tail_prob_9p5"]) / 100.0) if pred.get("high_tail_prob_9p5") is not None else None,
                    high_tail_cfg=pred.get("_high_tail_cfg"),
                    low_tail_prob=(float(pred["low_tail_prob_7p5"]) / 100.0) if pred.get("low_tail_prob_7p5") is not None else None,
                    low_tail_cfg=pred.get("_low_tail_cfg"),
                )
                dc_p_over = dc_pred["p_over"].get(
                    posted if posted in dc_pred["p_over"] else
                    min(dc_pred["p_over"], key=lambda x: abs(x - posted)),
                    xgb_p_over
                )
                blended_p_over = xgb_weight * xgb_p_over + dc_weight * dc_p_over
                pred["p_over_line"]  = round(blended_p_over * 100, 1)
                pred["p_under_line"] = round((1 - blended_p_over) * 100, 1)
                pred["edge"] = round(pred["predicted_total"] - posted, 2)

    # Apply manual overrides from config
    if overrides:
        predictions = apply_overrides(predictions, overrides, residual_std)

    market_cfg = cfg.get("market_lines", {})
    learned_market_cfg = meta.get("market_shrinkage") or {}
    for pred in predictions:
        if "posted_line" in pred:
            apply_market_adjustment_to_prediction(
                pred,
                residual_std=residual_std,
                market_cfg=market_cfg,
                learned_cfg=learned_market_cfg,
                market_edge_model=market_edge_model,
                market_edge_cfg=market_edge_cfg,
            )

    # Apply config betting thresholds (re-evaluate signals with user's min_edge/min_confidence)
    min_edge  = bet_cfg.get("min_edge_runs", 0.5)
    min_conf  = bet_cfg.get("min_confidence_pct", 55)
    max_bets  = bet_cfg.get("max_bets_per_day", 5)
    kalshi_thresholds = get_kalshi_betting_thresholds(bet_cfg)
    allowed_market_methods = normalize_allowed_market_adjustment_methods(bet_cfg)
    for pred in predictions:
        if "posted_line" in pred:
            edge = abs(pred.get("edge", 0))
            p_over  = pred.get("p_over_line", 0)
            p_under = pred.get("p_under_line", 0)
            pred.pop("bet_block_reason", None)
            market_method = pred.get("market_adjustment_method")
            method_allowed = (
                allowed_market_methods is None or
                market_method in allowed_market_methods
            )
            if not method_allowed:
                pred["bet_signal"] = "NO EDGE"
                pred["bet_confidence"] = max(p_over, p_under)
                pred["bet_block_reason"] = "market_adjustment_method_not_allowed"
            elif edge >= min_edge and p_over >= min_conf:
                pred["bet_signal"]      = "OVER"
                pred["bet_confidence"]  = p_over
            elif edge >= min_edge and p_under >= min_conf:
                pred["bet_signal"]      = "UNDER"
                pred["bet_confidence"]  = p_under
            else:
                pred["bet_signal"]      = "NO EDGE"
                pred["bet_confidence"]  = max(p_over, p_under)

    # Attach model-vs-Kalshi metrics after all prediction adjustments are final.
    for pred in predictions:
        if pred.get("kalshi_line") is not None:
            pred = add_kalshi_metrics(pred, residual_std)

    # Kalshi display filter from config
    kalshi_max_diff = disp_cfg.get("kalshi_max_line_diff", 2.5)
    for pred in predictions:
        if "kalshi_line" in pred:
            pred.pop("kalshi_bet_block_reason", None)
            reason = kalshi_filter_reason(
                pred,
                max_line_diff=kalshi_max_diff,
                min_edge_pct=kalshi_thresholds["min_kalshi_edge_pct"],
                min_confidence_pct=kalshi_thresholds["min_kalshi_confidence_pct"],
            )
            if reason == "line_diff_too_large":
                del pred["kalshi_line"]
                del pred["kalshi_over_pct"]
                del pred["kalshi_yes_ask"]
                pred.pop("kalshi_model_over_pct", None)
                pred.pop("kalshi_fair_price_pct", None)
                pred.pop("kalshi_edge_pct", None)
                pred.pop("kalshi_side", None)
                pred.pop("kalshi_side_model_prob", None)
                pred.pop("kalshi_side_market_prob", None)
                pred.pop("kalshi_kelly", None)
            elif reason is not None:
                suppress_kalshi_bet(pred, reason)

    line_source = "The Odds API" if api_key and live_lines else ("Kalshi" if kalshi_lines else "none")
    display_predictions(predictions, has_lines=bool(live_lines), line_source=line_source,
                        cfg=cfg, max_bets=max_bets)
    board_path, picks_path = export_daily_prediction_reports(
        predictions,
        target_date=date_str,
        include_all_games=args.all_games,
    )
    print(f"\nSaved board export: predictions/{os.path.basename(board_path)}")
    print(f"Saved picks export: predictions/{os.path.basename(picks_path)}")


if __name__ == "__main__":
    main()

