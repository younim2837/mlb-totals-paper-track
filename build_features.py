"""
MLB Feature Engineering Pipeline
Takes raw game data and builds rolling features for total runs prediction.
Outputs: data/mlb_model_data.tsv
"""

import pandas as pd
import numpy as np
import os
from bullpen_usage import build_pregame_bullpen_features, get_pregame_bullpen_feature_cols
from league_environment import LEAGUE_ENV_FEATURES, add_league_environment_features
from venue_metadata import DOME_VENUE_IDS, compute_local_time_features

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Rolling windows (games)
WINDOWS = [10, 30]

# Venue IDs for indoor / retractable-roof stadiums.
# Weather features are meaningless for these games — we null them out
# so the model doesn't learn spurious correlations.
def load_dixon_coles_config() -> dict:
    """
    Load the Dixon-Coles config block so historical priors match live settings.
    """
    defaults = {
        "time_decay_halflife": 60,
        "training_window": 365,
        "min_games_for_fit": 100,
        "l2_penalty": 0.05,
    }
    config_path = os.path.join(os.path.dirname(__file__), "model_config.yaml")
    if not _YAML_AVAILABLE or not os.path.exists(config_path):
        return defaults

    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        defaults.update(cfg.get("dixon_coles", {}) or {})
    except Exception:
        pass
    return defaults


def load_raw_games() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "mlb_games_raw.tsv")
    df = pd.read_csv(path, sep="\t")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "game_datetime" in df.columns:
        df["game_datetime"] = pd.to_datetime(df["game_datetime"], errors="coerce", utc=True)
    else:
        df["game_datetime"] = pd.NaT
    df = df.sort_values(["date", "game_datetime", "game_id"]).reset_index(drop=True)

    # Historical raw files may already contain duplicated completed games.
    # Drop them here so one bad row cannot cascade into many-to-many merges.
    before = len(df)
    df = df.drop_duplicates(subset=["game_id"], keep="first").reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"  Dropped {removed} duplicate raw game rows by game_id")
    return df


def build_team_game_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape: each row = one team's performance in one game.
    This makes it easy to compute rolling stats per team.
    """
    # Home team rows
    home = df.copy()
    home["team"] = home["home_team"]
    home["team_id"] = home["home_team_id"]
    home["opponent"] = home["away_team"]
    home["runs_scored"] = home["home_score"]
    home["runs_allowed"] = home["away_score"]
    home["is_home"] = 1

    # Away team rows
    away = df.copy()
    away["team"] = away["away_team"]
    away["team_id"] = away["away_team_id"]
    away["opponent"] = away["home_team"]
    away["runs_scored"] = away["away_score"]
    away["runs_allowed"] = away["home_score"]
    away["is_home"] = 0

    cols = ["game_id", "date", "team", "team_id", "opponent", "venue", "venue_id",
            "runs_scored", "runs_allowed", "is_home", "total_runs"]

    log = pd.concat([home[cols], away[cols]], ignore_index=True)
    log = log.sort_values(["team", "date", "game_id"]).reset_index(drop=True)
    return log


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


def compute_rest_days(team_log: pd.DataFrame) -> pd.DataFrame:
    """
    Add days_rest column: number of days since each team's previous game.
    First game of each team has NaN. Capped at 10 (longer breaks add no signal).
    """
    team_log = team_log.copy()
    team_log = team_log.sort_values(["team", "date"]).reset_index(drop=True)
    team_log["days_rest"] = (
        team_log.groupby("team")["date"]
        .transform(lambda x: x.diff().dt.days)
        .clip(upper=10)
    )
    return team_log


def add_bullpen_stats_to_game_log(games: pd.DataFrame,
                                  team_log: pd.DataFrame) -> pd.DataFrame:
    """
    For each team-game, compute what the bullpen gave up by subtracting
    the starter's earned runs and innings from the team totals.

    Adds: bullpen_runs, bullpen_ip, bullpen_era_game to team_log.
    """
    logs_path = os.path.join(DATA_DIR, "pitcher_game_logs.tsv")
    ids_path = os.path.join(DATA_DIR, "pitcher_ids.tsv")

    if not os.path.exists(logs_path) or not os.path.exists(ids_path):
        print("  Pitcher game logs not found — skipping bullpen stats")
        team_log["bullpen_runs"] = np.nan
        team_log["bullpen_ip"] = np.nan
        return team_log

    pitcher_logs = pd.read_csv(logs_path, sep="\t")
    pitcher_logs["date"] = pd.to_datetime(pitcher_logs["date"], format="mixed")
    pitcher_logs["earnedRuns"] = pd.to_numeric(pitcher_logs["earnedRuns"], errors="coerce")
    pitcher_logs["ip_dec"] = pitcher_logs["inningsPitched"].apply(_ip_str_to_decimal)

    ids_df = pd.read_csv(ids_path, sep="\t")
    name_to_id = dict(zip(ids_df["name"], ids_df["player_id"]))

    # Build starter lookup: for each game, map starter name → (ER, IP)
    # Home starters
    home_starter = games[["game_id", "date", "home_pitcher"]].copy()
    home_starter["starter_pid"] = home_starter["home_pitcher"].map(name_to_id)
    home_starter = home_starter.merge(
        pitcher_logs[["player_id", "date", "earnedRuns", "ip_dec"]].rename(
            columns={"player_id": "starter_pid", "earnedRuns": "starter_er", "ip_dec": "starter_ip"}
        ),
        on=["starter_pid", "date"],
        how="left",
    )
    home_starter["team"] = games["home_team"]
    home_starter = home_starter[["game_id", "team", "starter_er", "starter_ip"]]

    # Away starters
    away_starter = games[["game_id", "date", "away_pitcher"]].copy()
    away_starter["starter_pid"] = away_starter["away_pitcher"].map(name_to_id)
    away_starter = away_starter.merge(
        pitcher_logs[["player_id", "date", "earnedRuns", "ip_dec"]].rename(
            columns={"player_id": "starter_pid", "earnedRuns": "starter_er", "ip_dec": "starter_ip"}
        ),
        on=["starter_pid", "date"],
        how="left",
    )
    away_starter["team"] = games["away_team"]
    away_starter = away_starter[["game_id", "team", "starter_er", "starter_ip"]]

    # Combine and merge into team log
    starter_stats = pd.concat([home_starter, away_starter], ignore_index=True)
    starter_stats = starter_stats.drop_duplicates(subset=["game_id", "team"])

    team_log = team_log.merge(starter_stats, on=["game_id", "team"], how="left")

    # Bullpen contribution:
    #   bullpen_runs = runs_allowed - starter_er  (floored at 0)
    #   bullpen_ip = estimated_game_innings - starter_ip (floored at 0.1)
    team_log["bullpen_runs"] = (team_log["runs_allowed"] - team_log["starter_er"]).clip(lower=0)
    team_log["bullpen_ip"] = (9.0 - team_log["starter_ip"]).clip(lower=0.1)

    # Drop intermediate columns
    team_log.drop(columns=["starter_er", "starter_ip"], inplace=True)

    matched = team_log["bullpen_runs"].notna().sum()
    print(f"  Bullpen stats computed for {matched}/{len(team_log)} team-games "
          f"({matched/len(team_log):.1%})")

    return team_log


def add_rolling_features(team_log: pd.DataFrame) -> pd.DataFrame:
    """Add rolling averages per team for runs scored and runs allowed."""
    team_log = team_log.copy()
    team_log["_season"] = team_log["date"].dt.year

    for window in WINDOWS:
        grp = team_log.groupby(["team", "_season"])

        # Rolling average runs scored (offensive strength)
        team_log[f"avg_runs_scored_{window}g"] = (
            grp["runs_scored"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=5).mean())
        )

        # Rolling average runs allowed (defensive strength / pitching)
        team_log[f"avg_runs_allowed_{window}g"] = (
            grp["runs_allowed"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=5).mean())
        )

        # Rolling average total runs in games (is this team in high/low scoring games?)
        team_log[f"avg_total_runs_{window}g"] = (
            grp["total_runs"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=5).mean())
        )

        # Rolling win pct (momentum indicator)
        team_log[f"win_pct_{window}g"] = (
            grp.apply(
                lambda g: (g["runs_scored"] > g["runs_allowed"]).shift(1).rolling(window, min_periods=5).mean(),
                include_groups=False,
            )
            .reset_index(level=[0, 1], drop=True)
        )

    # Rolling bullpen ERA: (bullpen_runs / bullpen_ip) * 9 over last N games
    if "bullpen_runs" in team_log.columns and team_log["bullpen_runs"].notna().any():
        for window in WINDOWS:
            grp = team_log.groupby(["team", "_season"])
            rolling_bp_runs = grp["bullpen_runs"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=5).sum()
            )
            rolling_bp_ip = grp["bullpen_ip"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=5).sum()
            )
            team_log[f"bullpen_era_{window}g"] = (
                rolling_bp_runs / rolling_bp_ip.replace(0, np.nan)
            ) * 9

        # Season-long bullpen ERA
        grp_season = team_log.groupby(["team", team_log["date"].dt.year])
        season_bp_runs = grp_season["bullpen_runs"].transform(
            lambda x: x.shift(1).expanding(min_periods=5).sum()
        )
        season_bp_ip = grp_season["bullpen_ip"].transform(
            lambda x: x.shift(1).expanding(min_periods=5).sum()
        )
        team_log["bullpen_era_season"] = (
            season_bp_runs / season_bp_ip.replace(0, np.nan)
        ) * 9

    # Season-long averages (shift to avoid leakage)
    grp = team_log.groupby(["team", "_season"])
    team_log["season_avg_scored"] = (
        grp["runs_scored"]
        .transform(lambda x: x.shift(1).expanding(min_periods=5).mean())
    )
    team_log["season_avg_allowed"] = (
        grp["runs_allowed"]
        .transform(lambda x: x.shift(1).expanding(min_periods=5).mean())
    )

    team_log.drop(columns=["_season"], inplace=True)
    return team_log


def build_matchup_features(games: pd.DataFrame, team_log: pd.DataFrame) -> pd.DataFrame:
    """
    For each game, combine the rolling features of both teams
    into a single row suitable for model training. Uses vectorized merges.
    """
    feature_cols = [c for c in team_log.columns if any(
        c.startswith(p) for p in ["avg_", "win_pct_", "season_avg_", "bullpen_era_"]
    )]
    if "days_rest" in team_log.columns:
        feature_cols.append("days_rest")

    keep_cols = ["game_id", "team"] + feature_cols
    team_feats = team_log[keep_cols].drop_duplicates(subset=["game_id", "team"]).copy()

    # Merge home team features
    home_feats = team_feats.rename(columns={c: f"home_{c}" for c in feature_cols})
    result = games.merge(
        home_feats, left_on=["game_id", "home_team"], right_on=["game_id", "team"], how="inner"
    ).drop(columns=["team"])

    # Merge away team features
    away_feats = team_feats.rename(columns={c: f"away_{c}" for c in feature_cols})
    result = result.merge(
        away_feats, left_on=["game_id", "away_team"], right_on=["game_id", "team"], how="inner"
    ).drop(columns=["team"])

    # Combined / interaction features (vectorized)
    for w in WINDOWS:
        result[f"combined_scoring_{w}g"] = (
            result[f"home_avg_runs_scored_{w}g"] + result[f"away_avg_runs_scored_{w}g"]
        )
        result[f"combined_allowed_{w}g"] = (
            result[f"home_avg_runs_allowed_{w}g"] + result[f"away_avg_runs_allowed_{w}g"]
        )
        result[f"home_offense_vs_away_pitching_{w}g"] = (
            result[f"home_avg_runs_scored_{w}g"] - result[f"away_avg_runs_allowed_{w}g"]
        )
        result[f"away_offense_vs_home_pitching_{w}g"] = (
            result[f"away_avg_runs_scored_{w}g"] - result[f"home_avg_runs_allowed_{w}g"]
        )

    result["combined_season_scoring"] = (
        result["home_season_avg_scored"] + result["away_season_avg_scored"]
    )
    result["combined_season_allowed"] = (
        result["home_season_avg_allowed"] + result["away_season_avg_allowed"]
    )

    # Combined bullpen ERA features (if available)
    for w in WINDOWS:
        col_h = f"home_bullpen_era_{w}g"
        col_a = f"away_bullpen_era_{w}g"
        if col_h in result.columns and col_a in result.columns:
            result[f"combined_bullpen_era_{w}g"] = result[col_h] + result[col_a]

    if "home_bullpen_era_season" in result.columns and "away_bullpen_era_season" in result.columns:
        result["combined_bullpen_era_season"] = (
            result["home_bullpen_era_season"] + result["away_bullpen_era_season"]
        )

    result["date"] = pd.to_datetime(result["date"])
    return result


def build_pitcher_rolling_stats(logs_path: str) -> pd.DataFrame:
    """
    From per-start pitcher game logs, compute rolling stats over the last 3 starts.
    Returns a DataFrame indexed by (player_id, date) with rolling features.
    Uses shift(1) so stats reflect only *prior* starts (no leakage).
    """
    logs = pd.read_csv(logs_path, sep="\t", parse_dates=["date"])

    # Convert innings pitched string (e.g. "6.1") to decimal
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
    logs["earnedRuns"] = pd.to_numeric(logs["earnedRuns"], errors="coerce")
    logs["hits"] = pd.to_numeric(logs["hits"], errors="coerce")
    logs["baseOnBalls"] = pd.to_numeric(logs["baseOnBalls"], errors="coerce")
    logs["strikeOuts"] = pd.to_numeric(logs["strikeOuts"], errors="coerce")
    logs["homeRuns"] = pd.to_numeric(logs["homeRuns"], errors="coerce")
    logs["numberOfPitches"] = pd.to_numeric(logs["numberOfPitches"], errors="coerce")
    logs["battersFaced"] = pd.to_numeric(logs["battersFaced"], errors="coerce")

    logs = logs.sort_values(["player_id", "date"]).reset_index(drop=True)

    W = 3  # rolling window: last 3 starts
    grp = logs.groupby("player_id")

    # ERA over last 3 starts: (ER / IP) * 9
    logs["rolling_er"] = grp["earnedRuns"].transform(lambda x: x.shift(1).rolling(W, min_periods=1).sum())
    logs["rolling_ip"] = grp["ip_dec"].transform(lambda x: x.shift(1).rolling(W, min_periods=1).sum())
    logs["pitcher_era_3g"] = (logs["rolling_er"] / logs["rolling_ip"].replace(0, np.nan)) * 9

    # WHIP over last 3 starts
    logs["rolling_bb"] = grp["baseOnBalls"].transform(lambda x: x.shift(1).rolling(W, min_periods=1).sum())
    logs["rolling_h"] = grp["hits"].transform(lambda x: x.shift(1).rolling(W, min_periods=1).sum())
    logs["pitcher_whip_3g"] = (logs["rolling_bb"] + logs["rolling_h"]) / logs["rolling_ip"].replace(0, np.nan)

    # K/9 over last 3 starts
    logs["rolling_k"] = grp["strikeOuts"].transform(lambda x: x.shift(1).rolling(W, min_periods=1).sum())
    logs["pitcher_k9_3g"] = (logs["rolling_k"] / logs["rolling_ip"].replace(0, np.nan)) * 9

    # HR/9 over last 3 starts
    logs["rolling_hr"] = grp["homeRuns"].transform(lambda x: x.shift(1).rolling(W, min_periods=1).sum())
    logs["pitcher_hr9_3g"] = (logs["rolling_hr"] / logs["rolling_ip"].replace(0, np.nan)) * 9

    # Avg IP per start (indicates how deep a starter goes / bullpen usage)
    logs["pitcher_ip_3g"] = logs["rolling_ip"] / W

    # Same-day starter context: recent workload and likely leash depth.
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
    return logs[["player_id", "date"] + stat_cols]


def merge_pitcher_stats(matchups: pd.DataFrame, pitcher_stats: pd.DataFrame,
                        pitcher_ids: dict) -> pd.DataFrame:
    """
    For each game, look up rolling stats for home and away starters.
    Uses the most recent stats available before the game date.
    """
    matchups = matchups.copy()
    matchups["_matchup_row_id"] = np.arange(len(matchups), dtype=int)

    # Get unique (pitcher_name, game_date) pairs
    pitcher_stats = pitcher_stats.sort_values(["player_id", "date"]).copy()

    def get_pitcher_stats_for_game(pitcher_name, game_date):
        pid = pitcher_ids.get(pitcher_name)
        if pid is None:
            return {}
        prior = pitcher_stats[(pitcher_stats["player_id"] == pid) &
                               (pitcher_stats["date"] < game_date)]
        if prior.empty:
            return {}
        latest = prior.iloc[-1]
        return latest[[
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

    # Ensure pitcher_stats date is datetime
    pitcher_stats["date"] = pd.to_datetime(pitcher_stats["date"], format="mixed")

    def add_role_features(df: pd.DataFrame, side: str) -> pd.DataFrame:
        ip_col = f"{side}_pitcher_ip_3g"
        pitches_col = f"{side}_pitcher_avg_pitches_3g"
        bf_col = f"{side}_pitcher_avg_bf_3g"

        ip = df[ip_col]
        pitches = df[pitches_col]
        bf = df[bf_col]

        short_leash = (
            ((65.0 - pitches) / 25.0).clip(lower=0.0, upper=1.0).fillna(0.0) +
            ((4.8 - ip) / 2.0).clip(lower=0.0, upper=1.0).fillna(0.0) +
            ((18.0 - bf) / 8.0).clip(lower=0.0, upper=1.0).fillna(0.0)
        ) / 3.0
        workhorse = (
            ((pitches - 90.0) / 20.0).clip(lower=0.0, upper=1.0).fillna(0.0) +
            ((ip - 5.8) / 1.8).clip(lower=0.0, upper=1.0).fillna(0.0) +
            ((bf - 22.0) / 6.0).clip(lower=0.0, upper=1.0).fillna(0.0)
        ) / 3.0

        df[f"{side}_pitcher_short_leash_score"] = short_leash
        df[f"{side}_pitcher_workhorse_score"] = workhorse
        df[f"{side}_pitcher_opener_flag"] = (
            (ip <= 3.5) &
            (pitches <= 55.0) &
            (bf <= 15.0)
        ).astype(float)
        df[f"{side}_pitcher_bulk_flag"] = (
            (ip >= 4.5) &
            (pitches >= 65.0) &
            (bf >= 18.0)
        ).astype(float)
        return df

    # Vectorized: merge on player_id + most recent start before game date using
    # merge_asof. Join back by a stable row id so duplicated game_ids in raw
    # history cannot fan out into many-to-many explosions.
    for side, pitcher_col in [("home", "home_pitcher"), ("away", "away_pitcher")]:
        # Map name -> player_id, coerce to Int64 so it matches pitcher_stats.player_id
        matchups[f"{side}_pitcher_id"] = pd.to_numeric(
            matchups[pitcher_col].map(pitcher_ids), errors="coerce"
        ).astype("Int64")

        side_stats = pitcher_stats.copy()
        side_stats["player_id"] = side_stats["player_id"].astype("Int64")
        side_stats = side_stats.rename(columns={c: f"{side}_{c}" for c in stat_cols})
        side_stats = side_stats.rename(columns={"player_id": f"{side}_pitcher_id", "date": f"{side}_pitcher_last_start"})

        left = matchups[["_matchup_row_id", "game_id", "date", f"{side}_pitcher_id"]].sort_values("date").copy()
        left[f"{side}_pitcher_id"] = left[f"{side}_pitcher_id"].astype("Int64")

        # For each game, get the most recent start before game date
        merged = pd.merge_asof(
            left,
            side_stats.sort_values(f"{side}_pitcher_last_start"),
            left_on="date",
            right_on=f"{side}_pitcher_last_start",
            by=f"{side}_pitcher_id",
            direction="backward",
        )
        raw_days_rest = (merged["date"] - merged[f"{side}_pitcher_last_start"]).dt.days.clip(lower=0)
        merged[f"{side}_pitcher_days_rest"] = raw_days_rest.clip(upper=14)
        merged[f"{side}_pitcher_short_rest"] = (
            merged[f"{side}_pitcher_days_rest"] <= 4
        ).astype(float)
        merged[f"{side}_pitcher_long_rest"] = (
            raw_days_rest >= 7
        ).astype(float)
        merged[f"{side}_pitcher_offseason_gap"] = (raw_days_rest >= 20).astype(float)

        # Join back to matchups
        matchups = matchups.merge(
            merged[["_matchup_row_id"] + [f"{side}_{c}" for c in stat_cols] + [
                f"{side}_pitcher_days_rest",
                f"{side}_pitcher_short_rest",
                f"{side}_pitcher_long_rest",
                f"{side}_pitcher_offseason_gap",
            ]],
            on="_matchup_row_id",
            how="left",
        )
        matchups.drop(columns=[f"{side}_pitcher_id"], inplace=True)
        matchups = add_role_features(matchups, side)

    # Combined pitcher metrics
    matchups["combined_pitcher_era"] = matchups["home_pitcher_era_3g"] + matchups["away_pitcher_era_3g"]
    matchups["combined_pitcher_whip"] = matchups["home_pitcher_whip_3g"] + matchups["away_pitcher_whip_3g"]
    matchups["combined_pitcher_k9"] = matchups["home_pitcher_k9_3g"] + matchups["away_pitcher_k9_3g"]
    matchups["combined_pitcher_avg_pitches_3g"] = (
        matchups["home_pitcher_avg_pitches_3g"] + matchups["away_pitcher_avg_pitches_3g"]
    )
    matchups["combined_pitcher_prev_pitches"] = (
        matchups["home_pitcher_prev_pitches"] + matchups["away_pitcher_prev_pitches"]
    )
    matchups["combined_pitcher_avg_bf_3g"] = (
        matchups["home_pitcher_avg_bf_3g"] + matchups["away_pitcher_avg_bf_3g"]
    )
    matchups["combined_pitcher_short_rest_flags"] = (
        matchups["home_pitcher_short_rest"] + matchups["away_pitcher_short_rest"]
    )
    matchups["combined_pitcher_offseason_gap_flags"] = (
        matchups["home_pitcher_offseason_gap"] + matchups["away_pitcher_offseason_gap"]
    )
    matchups["combined_pitcher_short_leash_score"] = (
        matchups["home_pitcher_short_leash_score"] + matchups["away_pitcher_short_leash_score"]
    )
    matchups["combined_pitcher_workhorse_score"] = (
        matchups["home_pitcher_workhorse_score"] + matchups["away_pitcher_workhorse_score"]
    )
    matchups["combined_pitcher_opener_flags"] = (
        matchups["home_pitcher_opener_flag"] + matchups["away_pitcher_opener_flag"]
    )
    matchups["combined_pitcher_bulk_flags"] = (
        matchups["home_pitcher_bulk_flag"] + matchups["away_pitcher_bulk_flag"]
    )
    matchups.drop(columns=["_matchup_row_id"], inplace=True)

    return matchups


def add_head_to_head_features(games: pd.DataFrame) -> pd.DataFrame:
    """
    For each game, compute rolling total runs from prior matchups between the same
    two teams (regardless of which was home/away in those past games).
    Uses last 10 H2H games, min 3 to produce a value. Shift(1) prevents leakage.
    """
    games = games.sort_values("date").reset_index(drop=True).copy()

    # Normalize pair so "A vs B" and "B vs A" hash to the same key
    games["h2h_key"] = [
        tuple(sorted([h, a]))
        for h, a in zip(games["home_team"], games["away_team"])
    ]

    games["h2h_avg_total_runs"] = (
        games.groupby("h2h_key")["total_runs"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    )

    games.drop(columns=["h2h_key"], inplace=True)
    matched = games["h2h_avg_total_runs"].notna().sum()
    print(f"  H2H features: {matched}/{len(games)} games have history ({matched/len(games):.1%})")
    return games


def add_venue_park_factor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a simple park factor: average total runs at each venue
    relative to league average. Uses only past data to avoid leakage.
    """
    df = df.sort_values("date").copy()

    # Expanding mean of total_runs at each venue (shifted to avoid leakage)
    df["venue_game_num"] = df.groupby("venue_id").cumcount()
    venue_running_total = df.groupby("venue_id")["total_runs"].cumsum() - df["total_runs"]
    df["venue_avg_runs"] = venue_running_total / df["venue_game_num"].replace(0, np.nan)

    # League average total runs (expanding, shifted)
    df["league_game_num"] = range(len(df))
    league_running_total = df["total_runs"].cumsum() - df["total_runs"]
    df["league_avg_runs"] = league_running_total / df["league_game_num"].replace(0, np.nan)

    # Park factor = venue avg / league avg
    df["park_factor"] = df["venue_avg_runs"] / df["league_avg_runs"]

    # Clean up temp columns
    df.drop(columns=["venue_game_num", "league_game_num", "venue_avg_runs", "league_avg_runs"],
            inplace=True)

    return df


def add_league_environment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add league-level scoring environment features using only prior days.
    These help the model adapt to season-wide run-environment shifts.
    """
    df = add_league_environment_features(df, date_col="date", total_col="total_runs")
    matched = df["league_avg_total_runs_7d"].notna().sum()
    print(f"  League environment features built for {matched}/{len(df)} games ({matched/len(df):.1%})")
    return df


def merge_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge daily weather (temp, wind, precip) into the matchup DataFrame.
    Joins on (date, venue_id). Missing weather rows are left as NaN.
    Dome/retractable-roof venues have weather nulled out — it's irrelevant there.
    Also adds is_dome binary feature.
    """
    # Mark dome games first (before merging weather)
    df["is_dome"] = df["venue_id"].isin(DOME_VENUE_IDS).astype(int)
    dome_count = df["is_dome"].sum()
    print(f"  Dome/retractable games: {dome_count}/{len(df)} ({dome_count/len(df):.1%})")

    weather_path = os.path.join(DATA_DIR, "weather_historical.tsv")
    if not os.path.exists(weather_path):
        print("  Weather data not found — skipping (run collect_weather.py first)")
        return df

    weather = pd.read_csv(weather_path, sep="\t", parse_dates=["date"])
    weather_cols = [
        "date", "venue_id", "temp_f", "wind_mph", "precip_mm",
        "humidity_pct", "dew_point_f", "sunrise", "sunset",
    ]
    weather = weather[[c for c in weather_cols if c in weather.columns]]

    before = len(df)
    df = df.merge(weather, on=["date", "venue_id"], how="left")

    # Null out weather for dome/retractable venues — those readings are meaningless
    dome_mask = df["is_dome"] == 1
    nullable_weather_cols = [c for c in ["temp_f", "wind_mph", "precip_mm", "humidity_pct", "dew_point_f"] if c in df.columns]
    df.loc[dome_mask, nullable_weather_cols] = np.nan

    outdoor_matched = df.loc[~dome_mask, "temp_f"].notna().sum()
    outdoor_total = (~dome_mask).sum()
    print(f"  Weather matched for {outdoor_matched}/{outdoor_total} outdoor games "
          f"({outdoor_matched/outdoor_total:.1%})")
    return df


def add_game_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add first-pitch local hour and a day/night flag.
    Uses venue-local conversion plus sunrise/sunset when weather metadata exists.
    """
    df = df.copy()
    if "game_datetime" not in df.columns:
        df["first_pitch_local_hour"] = np.nan
        df["is_night_game"] = np.nan
        return df

    feature_rows = df.apply(
        lambda row: pd.Series(
            compute_local_time_features(
                game_datetime=row.get("game_datetime"),
                venue_id=row.get("venue_id"),
                sunrise=row.get("sunrise"),
                sunset=row.get("sunset"),
            )
        ),
        axis=1,
    )
    df["first_pitch_local_hour"] = feature_rows["first_pitch_local_hour"]
    df["is_night_game"] = feature_rows["is_night_game"]
    matched = df["first_pitch_local_hour"].notna().sum()
    print(f"  Local first-pitch time features built for {matched}/{len(df)} games ({matched/len(df):.1%})")
    return df


def merge_dixon_coles_features(matchups: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """
    Merge leakage-safe pregame Dixon-Coles priors into the matchup DataFrame.
    If the historical DC file is missing or stale, build it on demand.
    """
    path = os.path.join(DATA_DIR, "dc_ratings_history.tsv")
    dc_cfg = load_dixon_coles_config()

    from dixon_coles import load_or_build_history

    dc = load_or_build_history(
        games,
        history_path=path,
        halflife=float(dc_cfg.get("time_decay_halflife", 60)),
        window_days=int(dc_cfg.get("training_window", 365)),
        min_games=int(dc_cfg.get("min_games_for_fit", 100)),
        l2_penalty=float(dc_cfg.get("l2_penalty", 0.05)),
    )
    dc_cols = [
        "game_id",
        "home_dc_attack",
        "home_dc_defense",
        "away_dc_attack",
        "away_dc_defense",
        "dc_lambda_home",
        "dc_lambda_away",
        "dc_expected_total",
        "dc_home_edge",
        "dc_fit_n_games",
    ]

    before = len(matchups)
    matchups = matchups.merge(dc[dc_cols], on="game_id", how="left")
    matched = matchups["dc_expected_total"].notna().sum()
    print(f"  Dixon-Coles features merged for {matched}/{before} games ({matched/before:.1%})")
    return matchups


def merge_team_batting_features(matchups: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling offensive quality features per team: OPS, BB%, K%.
    Uses per-game team batting logs (data/team_batting_logs.tsv).
    Computes rolling 10g and 30g windows with shift(1) to prevent leakage.
    """
    path = os.path.join(DATA_DIR, "team_batting_logs.tsv")
    if not os.path.exists(path):
        print("  team_batting_logs.tsv not found — skipping (run collect_team_batting.py)")
        return matchups

    logs = pd.read_csv(path, sep="\t")
    logs["date"] = pd.to_datetime(logs["date"])
    logs = logs.sort_values(["team_id", "date"]).reset_index(drop=True)
    logs["_season"] = logs["date"].dt.year

    grp = logs.groupby(["team_id", "_season"])

    for w in [10, 30]:
        logs[f"rolling_ops_{w}g"] = grp["ops"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=5).mean()
        )
        logs[f"rolling_bb_pct_{w}g"] = grp["bb_pct"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=5).mean()
        )
        logs[f"rolling_k_pct_{w}g"] = grp["k_pct"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=5).mean()
        )
    logs.drop(columns=["_season"], inplace=True)

    bat_cols = [f"rolling_ops_{w}g" for w in [10, 30]] + \
               [f"rolling_bb_pct_{w}g" for w in [10, 30]] + \
               [f"rolling_k_pct_{w}g" for w in [10, 30]]

    keep = logs[["team_id", "date", "game_id"] + bat_cols].copy()

    # Merge home team batting
    home_bat = keep.rename(columns={c: f"home_{c}" for c in bat_cols})
    matchups = pd.merge_asof(
        matchups.sort_values("date"),
        home_bat.rename(columns={"team_id": "home_team_id", "date": "bat_date"}
                        ).sort_values("bat_date"),
        left_on="date", right_on="bat_date",
        by="home_team_id", direction="backward",
    ).drop(columns=["bat_date", "game_id_y"], errors="ignore")
    if "game_id_x" in matchups.columns:
        matchups = matchups.rename(columns={"game_id_x": "game_id"})

    # Merge away team batting
    away_bat = keep.rename(columns={c: f"away_{c}" for c in bat_cols})
    matchups = pd.merge_asof(
        matchups.sort_values("date"),
        away_bat.rename(columns={"team_id": "away_team_id", "date": "bat_date"}
                        ).sort_values("bat_date"),
        left_on="date", right_on="bat_date",
        by="away_team_id", direction="backward",
    ).drop(columns=["bat_date", "game_id_y"], errors="ignore")
    if "game_id_x" in matchups.columns:
        matchups = matchups.rename(columns={"game_id_x": "game_id"})

    matched = matchups["home_rolling_ops_10g"].notna().sum()
    print(f"  Team batting features merged for {matched}/{len(matchups)} games "
          f"({matched/len(matchups):.1%})")
    return matchups


def merge_lineup_features(matchups: pd.DataFrame) -> pd.DataFrame:
    """
    Merge same-day starting lineup strength features for both teams.
    These rows are pregame snapshots reconstructed from the MLB game feed.
    """
    path = os.path.join(DATA_DIR, "team_lineup_features.tsv")
    if not os.path.exists(path):
        print("  team_lineup_features.tsv not found - skipping (run collect_team_lineups.py)")
        return matchups

    lineup = pd.read_csv(path, sep="\t", parse_dates=["date"])
    if lineup.empty:
        print("  team_lineup_features.tsv is empty - skipping")
        return matchups

    feature_cols = [
        c for c in lineup.columns
        if c.startswith("lineup_")
    ]
    keep = lineup[["game_id", "team_id", "is_home"] + feature_cols].copy()

    home = keep[keep["is_home"] == 1].drop(columns=["is_home"])
    home = home.rename(columns={"team_id": "home_team_id", **{c: f"home_{c}" for c in feature_cols}})
    away = keep[keep["is_home"] == 0].drop(columns=["is_home"])
    away = away.rename(columns={"team_id": "away_team_id", **{c: f"away_{c}" for c in feature_cols}})

    matchups = matchups.merge(home, on=["game_id", "home_team_id"], how="left")
    matchups = matchups.merge(away, on=["game_id", "away_team_id"], how="left")

    matched = matchups["home_lineup_avg_ops"].notna().sum()
    print(f"  Lineup features merged for {matched}/{len(matchups)} games "
          f"({matched/len(matchups):.1%})")

    # Today's lineup matters most relative to what the team usually looks like.
    for side in ["home", "away"]:
        if f"{side}_lineup_avg_ops" in matchups.columns and f"{side}_rolling_ops_30g" in matchups.columns:
            matchups[f"{side}_lineup_delta_ops_30g"] = (
                matchups[f"{side}_lineup_avg_ops"] - matchups[f"{side}_rolling_ops_30g"]
            )
        if f"{side}_lineup_avg_bb_pct" in matchups.columns and f"{side}_rolling_bb_pct_30g" in matchups.columns:
            matchups[f"{side}_lineup_delta_bb_pct_30g"] = (
                matchups[f"{side}_lineup_avg_bb_pct"] - matchups[f"{side}_rolling_bb_pct_30g"]
            )
        if f"{side}_lineup_avg_k_pct" in matchups.columns and f"{side}_rolling_k_pct_30g" in matchups.columns:
            matchups[f"{side}_lineup_delta_k_pct_30g"] = (
                matchups[f"{side}_lineup_avg_k_pct"] - matchups[f"{side}_rolling_k_pct_30g"]
            )
        if f"{side}_lineup_top3_avg_ops" in matchups.columns and f"{side}_rolling_ops_30g" in matchups.columns:
            matchups[f"{side}_lineup_top3_delta_ops_30g"] = (
                matchups[f"{side}_lineup_top3_avg_ops"] - matchups[f"{side}_rolling_ops_30g"]
            )
        if f"{side}_lineup_top3_avg_ops" in matchups.columns and f"{side}_lineup_bottom3_avg_ops" in matchups.columns:
            matchups[f"{side}_lineup_depth_gap_ops"] = (
                matchups[f"{side}_lineup_top3_avg_ops"] - matchups[f"{side}_lineup_bottom3_avg_ops"]
            )

    return matchups


def merge_bullpen_fatigue_features(matchups: pd.DataFrame) -> pd.DataFrame:
    """
    Merge pregame bullpen workload / availability features built from relief
    appearance logs.
    """
    path = os.path.join(DATA_DIR, "bullpen_appearance_logs.tsv")
    if not os.path.exists(path):
        print("  bullpen_appearance_logs.tsv not found - skipping (run collect_bullpen_usage.py)")
        return matchups

    logs = pd.read_csv(path, sep="\t", parse_dates=["date"])
    if logs.empty:
        print("  bullpen_appearance_logs.tsv is empty - skipping")
        return matchups

    feats = build_pregame_bullpen_features(logs)
    feature_cols = get_pregame_bullpen_feature_cols(list(feats.columns))
    keep = feats[["team_id", "date"] + feature_cols].copy()

    home = keep.rename(columns={"team_id": "home_team_id", **{c: f"home_{c}" for c in feature_cols}})
    away = keep.rename(columns={"team_id": "away_team_id", **{c: f"away_{c}" for c in feature_cols}})

    matchups = matchups.merge(home, on=["home_team_id", "date"], how="left")
    matchups = matchups.merge(away, on=["away_team_id", "date"], how="left")

    matched = matchups["home_bullpen_used_pitches_3d"].notna().sum()
    print(f"  Bullpen fatigue features merged for {matched}/{len(matchups)} games "
          f"({matched/len(matchups):.1%})")
    return matchups


def merge_umpire_features(matchups: pd.DataFrame) -> pd.DataFrame:
    """
    Add HP umpire scoring factor as a feature.
    ump_avg_total_runs = expanding mean of total_runs in all prior games this
    umpire has called (min 20 games for reliability).
    """
    path = os.path.join(DATA_DIR, "umpire_game_log.tsv")
    if not os.path.exists(path):
        print("  umpire_game_log.tsv not found — skipping (run collect_umpires.py)")
        return matchups

    ump = pd.read_csv(path, sep="\t")
    ump["date"] = pd.to_datetime(ump["date"])
    ump = ump.dropna(subset=["total_runs"]).sort_values("date").reset_index(drop=True)

    # Expanding mean per umpire with shift(1) — no leakage
    ump["ump_avg_total_runs"] = (
        ump.groupby("hp_umpire")["total_runs"]
        .transform(lambda x: x.shift(1).expanding(min_periods=20).mean())
    )
    ump["ump_games_called"] = (
        ump.groupby("hp_umpire")["total_runs"]
        .transform(lambda x: x.shift(1).expanding().count())
    )

    ump_feats = ump[["game_id", "hp_umpire", "ump_avg_total_runs", "ump_games_called"]].copy()

    before = len(matchups)
    matchups = matchups.merge(ump_feats, on="game_id", how="left")
    matched = matchups["ump_avg_total_runs"].notna().sum()
    unique_umps = matchups["hp_umpire"].nunique()
    print(f"  Umpire features merged for {matched}/{before} games "
          f"({matched/before:.1%}, {unique_umps} unique umpires)")
    return matchups


def main():
    print("Loading raw game data...")
    games = load_raw_games()
    print(f"  {len(games)} games loaded")

    print("Building team game log...")
    team_log = build_team_game_log(games)
    print(f"  {len(team_log)} team-game rows")

    print("Computing rest days...")
    team_log = compute_rest_days(team_log)

    print("Computing bullpen stats per game...")
    team_log = add_bullpen_stats_to_game_log(games, team_log)

    print("Computing rolling features...")
    team_log = add_rolling_features(team_log)

    print("Adding head-to-head features...")
    games = add_head_to_head_features(games)

    print("Building matchup features...")
    matchups = build_matchup_features(games, team_log)
    print(f"  {len(matchups)} matchup rows (before dropping NaN)")

    print("Adding park factor...")
    matchups = add_venue_park_factor(matchups)

    print("Adding league scoring environment...")
    matchups = add_league_environment(matchups)

    print("Merging weather data...")
    matchups = merge_weather_data(matchups)

    print("Adding first-pitch local time features...")
    matchups = add_game_time_features(matchups)

    print("Merging Dixon-Coles priors...")
    matchups = merge_dixon_coles_features(matchups, games)

    print("Merging team batting features (OPS, K%, BB%)...")
    matchups = merge_team_batting_features(matchups)

    print("Merging lineup features...")
    matchups = merge_lineup_features(matchups)

    print("Merging bullpen fatigue features...")
    matchups = merge_bullpen_fatigue_features(matchups)

    print("Merging umpire features...")
    matchups = merge_umpire_features(matchups)

    # Add pitcher stats if available
    logs_path = os.path.join(DATA_DIR, "pitcher_game_logs.tsv")
    ids_path = os.path.join(DATA_DIR, "pitcher_ids.tsv")
    if os.path.exists(logs_path) and os.path.exists(ids_path):
        print("Adding pitcher rolling stats...")
        pitcher_stats = build_pitcher_rolling_stats(logs_path)
        ids_df = pd.read_csv(ids_path, sep="\t")
        pitcher_ids = dict(zip(ids_df["name"], ids_df["player_id"]))
        matchups = merge_pitcher_stats(matchups, pitcher_stats, pitcher_ids)
        print(f"  Pitcher stats merged. Columns with pitcher data: "
              f"{sum(1 for c in matchups.columns if 'pitcher' in c)}")
    else:
        print("Pitcher stats not found — skipping (run collect_pitcher_stats.py first)")

    # Drop rows where rolling features aren't available yet (start of seasons)
    feature_cols = [c for c in matchups.columns if any(
        c.startswith(p) for p in ["home_avg", "away_avg", "home_win", "away_win",
                                   "home_season", "away_season", "combined_", "park_",
                                   "home_pitcher_", "away_pitcher_",
                                   "home_bullpen", "away_bullpen",
                                   "home_bullpen_used_", "away_bullpen_used_",
                                   "home_bullpen_app_", "away_bullpen_app_",
                                   "home_bullpen_unique_", "away_bullpen_unique_",
                                   "home_bullpen_heavy_", "away_bullpen_heavy_",
                                   "home_bullpen_arms_", "away_bullpen_arms_",
                                   "home_bullpen_b2b_", "away_bullpen_b2b_",
                                   "home_bullpen_fatigue_", "away_bullpen_fatigue_",
                                   "home_rolling_", "away_rolling_",
                                   "home_dc_", "away_dc_", "dc_",
                                   "home_lineup_", "away_lineup_"]
    )] + [c for c in [*LEAGUE_ENV_FEATURES,
                      "temp_f", "wind_mph", "precip_mm", "humidity_pct", "dew_point_f",
                      "first_pitch_local_hour", "is_night_game", "is_dome",
                      "home_days_rest", "away_days_rest",
                      "h2h_avg_total_runs", "ump_avg_total_runs"] if c in matchups.columns]
    # Only require core team features for row inclusion (pitcher/bullpen/weather/rest/h2h/batting/ump may be NaN)
    core_feature_cols = [c for c in feature_cols
                         if "pitcher" not in c and "bullpen" not in c
                         and "rolling_ops" not in c and "rolling_bb_pct" not in c
                         and "rolling_k_pct" not in c and "ump_" not in c
                         and "lineup_" not in c
                         and c not in (*LEAGUE_ENV_FEATURES,
                                       "temp_f", "wind_mph", "precip_mm", "humidity_pct",
                                       "dew_point_f", "first_pitch_local_hour", "is_night_game",
                                       "is_dome", "home_days_rest", "away_days_rest",
                                       "h2h_avg_total_runs", "hp_umpire",
                                       "ump_games_called")]
    matchups_clean = matchups.dropna(subset=core_feature_cols)
    print(f"  {len(matchups_clean)} rows after dropping NaN (early-season games removed)")

    # Summary
    print(f"\nDataset shape: {matchups_clean.shape}")
    print(f"Date range: {matchups_clean['date'].min()} to {matchups_clean['date'].max()}")
    print(f"Avg total runs: {matchups_clean['total_runs'].mean():.2f}")
    print(f"\nFeature columns ({len(feature_cols)}):")
    for c in sorted(feature_cols):
        print(f"  {c}")

    out_path = os.path.join(DATA_DIR, "mlb_model_data.tsv")
    matchups_clean.to_csv(out_path, index=False, sep="\t")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
