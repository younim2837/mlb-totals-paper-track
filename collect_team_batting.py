"""
MLB Team Batting Stats Collection
Pulls per-game batting stats (OBP, SLG, OPS, K%, BB%) for every team.
Outputs: data/team_batting_logs.tsv  (one row per team per game)

Run modes:
    python collect_team_batting.py                  # backfill all seasons
    python collect_team_batting.py --update-current # refresh current season only
"""

import requests
import pandas as pd
import time
import os
import sys
from datetime import datetime

MLB_API = "https://statsapi.mlb.com/api/v1"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]


def get_team_ids_from_games() -> list:
    """Extract unique (team_id, team_name) from our existing game data."""
    df = pd.read_csv(os.path.join(DATA_DIR, "mlb_games_raw.tsv"), sep="\t")
    home = df[["home_team_id", "home_team"]].rename(
        columns={"home_team_id": "team_id", "home_team": "team_name"}
    )
    away = df[["away_team_id", "away_team"]].rename(
        columns={"away_team_id": "team_id", "away_team": "team_name"}
    )
    teams = pd.concat([home, away]).drop_duplicates(subset=["team_id"])
    return list(zip(teams["team_id"].astype(int), teams["team_name"]))


def fetch_team_batting_log(team_id: int, season: int) -> list:
    """
    Fetch per-game batting stats for one team in one season.
    Returns list of dicts, one per game played.
    """
    try:
        r = requests.get(
            f"{MLB_API}/teams/{team_id}/stats",
            params={
                "stats": "gameLog",
                "group": "hitting",
                "season": season,
                "sportId": 1,
                "gameType": "R",
            },
            timeout=15,
        )
        if r.status_code != 200:
            return []

        stats_list = r.json().get("stats", [])
        if not stats_list:
            return []

        records = []
        for stat_group in stats_list:
            for s in stat_group.get("splits", []):
                stat = s.get("stat", {})
                date_str = s.get("date")
                game_pk  = s.get("game", {}).get("gamePk")
                if not date_str or not game_pk:
                    continue

                ab  = int(stat.get("atBats", 0) or 0)
                pa  = int(stat.get("plateAppearances", 0) or 0)
                so  = int(stat.get("strikeOuts", 0) or 0)
                bb  = int(stat.get("baseOnBalls", 0) or 0)

                # Use pre-computed per-game rates from API (most accurate)
                try:
                    obp = float(stat.get("obp") or 0)
                    slg = float(stat.get("slg") or 0)
                    ops = float(stat.get("ops") or 0)
                except (TypeError, ValueError):
                    obp = slg = ops = 0.0

                if ab == 0 or pa == 0:
                    continue

                records.append({
                    "team_id": team_id,
                    "game_id": game_pk,
                    "date": date_str,
                    "season": season,
                    "ab": ab,
                    "pa": pa,
                    "so": so,
                    "bb": bb,
                    "obp": obp,
                    "slg": slg,
                    "ops": ops,
                    "k_pct": round(so / pa, 4) if pa > 0 else None,
                    "bb_pct": round(bb / pa, 4) if pa > 0 else None,
                })
        return records
    except Exception as e:
        return []


def update_current_season(out_path: str):
    """Re-fetch current season for all known teams. Used for daily updates."""
    current_year = datetime.now().year
    print(f"Updating {current_year} team batting logs...")

    teams = get_team_ids_from_games()
    print(f"  {len(teams)} teams found")

    existing = pd.read_csv(out_path, sep="\t") if os.path.exists(out_path) else pd.DataFrame()

    new_records = []
    for i, (team_id, team_name) in enumerate(teams):
        records = fetch_team_batting_log(team_id, current_year)
        new_records.extend(records)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(teams)} teams updated...")
        time.sleep(0.2)

    if not new_records:
        print(f"  No {current_year} games found yet.")
        return

    new_df = pd.DataFrame(new_records)
    new_df["date"] = pd.to_datetime(new_df["date"])

    if not existing.empty:
        existing["date"] = pd.to_datetime(existing["date"])
        prior = existing[existing["date"].dt.year != current_year]
        combined = pd.concat([prior, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.sort_values(["team_id", "date"]).reset_index(drop=True)
    combined = combined.drop_duplicates(subset=["team_id", "game_id"])
    combined.to_csv(out_path, sep="\t", index=False)
    print(f"  Done. {len(new_df)} game-logs for {current_year}. Total: {len(combined)}.")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "team_batting_logs.tsv")

    if "--update-current" in sys.argv:
        update_current_season(out_path)
        return

    # Full backfill
    teams = get_team_ids_from_games()
    print(f"Fetching batting logs for {len(teams)} teams x {len(SEASONS)} seasons...")

    existing = pd.read_csv(out_path, sep="\t") if os.path.exists(out_path) else pd.DataFrame()

    all_records = []
    total = len(teams) * len(SEASONS)
    done = 0

    for team_id, team_name in teams:
        for season in SEASONS:
            records = fetch_team_batting_log(team_id, season)
            all_records.extend(records)
            done += 1
            time.sleep(0.15)

        if done % (len(SEASONS) * 5) == 0:
            print(f"  {done}/{total} team-seasons fetched ({len(all_records)} records so far)...")

    if not all_records:
        print("No records fetched.")
        return

    new_df = pd.DataFrame(all_records)
    new_df["date"] = pd.to_datetime(new_df["date"])

    if not existing.empty:
        existing["date"] = pd.to_datetime(existing["date"])
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.drop_duplicates(subset=["team_id", "game_id"])
    combined = combined.sort_values(["team_id", "date"]).reset_index(drop=True)
    combined.to_csv(out_path, sep="\t", index=False)

    print(f"\nSaved {len(combined)} team-game records to {out_path}")
    print(f"  Unique teams: {combined['team_id'].nunique()}")
    print(f"  Date range: {combined['date'].min()} to {combined['date'].max()}")
    print(f"  Seasons: {sorted(combined['season'].unique())}")


if __name__ == "__main__":
    main()
