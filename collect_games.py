"""
MLB Game Data Collection Pipeline
Pulls regular-season game results from the MLB Stats API.
Outputs: data/mlb_games_raw.tsv

Run modes:
    python collect_games.py               # full backfill of all known seasons
    python collect_games.py --since DATE  # incremental: only fetch from DATE forward
"""

import statsapi
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# MLB regular season approximate date ranges — add new seasons here each year
SEASON_DATES = {
    2021: ("2021-04-01", "2021-10-03"),
    2022: ("2022-04-07", "2022-10-05"),
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-28", "2024-09-29"),
    2025: ("2025-03-27", "2025-09-28"),
    2026: ("2026-03-26", "2026-09-27"),
}

def all_seasons():
    return sorted(SEASON_DATES.keys())


def fetch_season_games(season: int) -> list[dict]:
    """Fetch all regular-season games for a given season."""
    start_date, end_date = SEASON_DATES[season]

    # For current/future season, cap at today's date
    today = datetime.now().strftime("%Y-%m-%d")
    if end_date > today:
        end_date = today
        print(f"  (capping at today: {today})")

    print(f"Fetching {season} season: {start_date} to {end_date}")

    all_games = []

    # Pull in 30-day chunks to avoid API timeouts
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while current <= end:
        chunk_end = min(current + timedelta(days=29), end)
        chunk_start_str = current.strftime("%Y-%m-%d")
        chunk_end_str = chunk_end.strftime("%Y-%m-%d")

        try:
            games = statsapi.schedule(
                start_date=chunk_start_str,
                end_date=chunk_end_str,
                sportId=1,  # MLB
            )
            all_games.extend(games)
            print(f"  {chunk_start_str} to {chunk_end_str}: {len(games)} games")
        except Exception as e:
            print(f"  ERROR {chunk_start_str} to {chunk_end_str}: {e}")

        current = chunk_end + timedelta(days=1)
        time.sleep(0.5)  # Be polite to the API

    return all_games


def process_games(raw_games: list[dict]) -> pd.DataFrame:
    """Clean raw game data into a structured DataFrame."""
    records = []
    for g in raw_games:
        # Only include completed regular-season games
        if g["game_type"] != "R":
            continue
        if "Final" not in g["status"] and "Completed" not in g["status"]:
            continue

        records.append({
            "game_id": g["game_id"],
            "date": g["game_date"],
            "game_datetime": g.get("game_datetime"),
            "away_team": g["away_name"],
            "away_team_id": g["away_id"],
            "home_team": g["home_name"],
            "home_team_id": g["home_id"],
            "away_score": g["away_score"],
            "home_score": g["home_score"],
            "total_runs": g["away_score"] + g["home_score"],
            "venue": g["venue_name"],
            "venue_id": g["venue_id"],
            "away_pitcher": g["away_probable_pitcher"],
            "home_pitcher": g["home_probable_pitcher"],
            "doubleheader": g["doubleheader"],
            "game_num": g["game_num"],
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["game_datetime"] = pd.to_datetime(df["game_datetime"], errors="coerce", utc=True)
    df = df.sort_values(["date", "game_datetime", "game_id"]).reset_index(drop=True)

    # The Stats API can occasionally surface the same completed game on
    # multiple dates (for example around schedule changes). Keep a single row
    # per game_id so downstream feature merges stay one-to-one.
    before = len(df)
    df = df.drop_duplicates(subset=["game_id"], keep="first").reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"  Removed {removed} duplicate completed game rows by game_id")
    return df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "mlb_games_raw.tsv")

    # ── Incremental mode: --since YYYY-MM-DD ─────────────────────────────────
    since_date = None
    if "--since" in sys.argv:
        idx = sys.argv.index("--since")
        since_date = sys.argv[idx + 1]

    if since_date is None and os.path.exists(out_path):
        # Auto-detect: if TSV exists, only fetch from day after last game
        existing = pd.read_csv(out_path, sep="\t", parse_dates=["date"])
        if "game_datetime" not in existing.columns:
            print("Existing game file is missing game_datetime — running full backfill to add it.")
        else:
            last = existing["date"].max()
            since_date = (last + timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"Existing data through {last.date()} — fetching from {since_date} forward")

    if since_date:
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        if since_date > today:
            print(f"Data already up to date (last game: {since_date}, today: {today})")
            return
        end_fetch = today  # include today's completed games
        print(f"Incremental fetch: {since_date} to {end_fetch}")
        try:
            raw = statsapi.schedule(start_date=since_date, end_date=end_fetch, sportId=1)
        except Exception as e:
            print(f"  ERROR fetching games: {e}")
            return

        new_df = process_games(raw)
        if new_df.empty:
            print("  No new completed games found.")
            return

        existing = pd.read_csv(out_path, sep="\t", parse_dates=["date"])
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["game_id"]).sort_values(["date", "game_id"])
        combined.to_csv(out_path, index=False, sep="\t")
        print(f"  Added {len(new_df)} games. Total: {len(combined)} games through "
              f"{combined['date'].max().date()}")
        return

    # ── Full backfill (first-time setup) ─────────────────────────────────────
    all_games_raw = []
    for season in all_seasons():
        games = fetch_season_games(season)
        all_games_raw.extend(games)
        print(f"  -> {season} total raw entries: {len(games)}")
        print()

    print(f"Total raw game entries: {len(all_games_raw)}")

    df = process_games(all_games_raw)
    print(f"Final cleaned games: {len(df)}")
    print(f"Seasons covered: {df['date'].dt.year.unique()}")
    print(f"Average total runs per game: {df['total_runs'].mean():.2f}")
    print()

    print("Games per season:")
    print(df.groupby(df["date"].dt.year).agg(
        games=("game_id", "count"),
        avg_total_runs=("total_runs", "mean"),
    ).to_string())

    df.to_csv(out_path, index=False, sep="\t")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
