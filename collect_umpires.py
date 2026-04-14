"""
MLB Umpire Data Collection
Pulls the home plate umpire for every historical game.
Outputs: data/umpire_game_log.tsv  (game_id, date, hp_umpire, total_runs)

The umpire's historical avg total runs per game called is used as a feature.
Umpires with wide strike zones suppress walks and scoring; tight zones inflate them.

Run modes:
    python collect_umpires.py              # backfill all seasons
    python collect_umpires.py --update     # fetch current season only
"""

import requests
import pandas as pd
import time
import os
import sys
from datetime import datetime, timedelta

MLB_API = "https://statsapi.mlb.com/api/v1"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

SEASON_DATES = {
    2021: ("2021-04-01", "2021-10-03"),
    2022: ("2022-04-07", "2022-10-05"),
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-28", "2024-09-29"),
    2025: ("2025-03-27", "2025-09-28"),
    2026: ("2026-03-26", "2026-09-27"),
}


def fetch_officials_for_range(start_date: str, end_date: str) -> list:
    """
    Fetch HP umpire for all completed games in a date range.
    Uses the schedule endpoint with officials hydration — one call per chunk,
    much faster than per-game boxscore calls.
    """
    try:
        r = requests.get(
            f"{MLB_API}/schedule",
            params={
                "startDate": start_date,
                "endDate": end_date,
                "sportId": 1,
                "hydrate": "officials",
                "gameType": "R",
            },
            timeout=30,
        )
        if r.status_code != 200:
            return []

        records = []
        for date_entry in r.json().get("dates", []):
            date_str = date_entry["date"]
            for game in date_entry.get("games", []):
                status = game.get("status", {}).get("detailedState", "")
                if "Final" not in status and "Completed" not in status:
                    continue

                officials = game.get("officials", [])
                hp_ump = None
                for official in officials:
                    if official.get("officialType") == "Home Plate":
                        hp_ump = official.get("official", {}).get("fullName")
                        break

                if hp_ump:
                    records.append({
                        "game_id": game["gamePk"],
                        "date": date_str,
                        "hp_umpire": hp_ump,
                    })
        return records
    except Exception as e:
        print(f"  Error for {start_date} to {end_date}: {e}")
        return []


def merge_total_runs(records: list) -> pd.DataFrame:
    """Join umpire records with total_runs from our game data."""
    games_path = os.path.join(DATA_DIR, "mlb_games_raw.tsv")
    games = pd.read_csv(games_path, sep="\t", usecols=["game_id", "total_runs"])
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.merge(games, on="game_id", how="left")
    return df


def fetch_for_season(season: int, existing_ids: set) -> list:
    """Fetch umpire data for a full season in 30-day chunks."""
    start_str, end_str = SEASON_DATES[season]
    today = datetime.now().strftime("%Y-%m-%d")
    if end_str > today:
        end_str = today

    current = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")

    all_records = []
    while current <= end:
        chunk_end = min(current + timedelta(days=29), end)
        chunk_s = current.strftime("%Y-%m-%d")
        chunk_e = chunk_end.strftime("%Y-%m-%d")

        records = fetch_officials_for_range(chunk_s, chunk_e)
        new = [r for r in records if r["game_id"] not in existing_ids]
        all_records.extend(new)
        existing_ids.update(r["game_id"] for r in new)

        print(f"  {season} {chunk_s} to {chunk_e}: {len(records)} games "
              f"({len(new)} new)")
        current = chunk_end + timedelta(days=1)
        time.sleep(0.3)

    return all_records


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "umpire_game_log.tsv")

    existing = pd.read_csv(out_path, sep="\t") if os.path.exists(out_path) else pd.DataFrame()
    existing_ids = set(existing["game_id"].tolist()) if not existing.empty else set()
    print(f"Existing umpire records: {len(existing_ids)}")

    if "--update" in sys.argv:
        # Only fetch current season
        current_year = datetime.now().year
        seasons_to_fetch = [current_year]
        # Remove existing current-year records so we re-fetch cleanly
        if not existing.empty:
            existing = existing[pd.to_datetime(existing["date"]).dt.year != current_year]
            existing_ids = set(existing["game_id"].tolist())
        print(f"Updating {current_year} only...")
    else:
        seasons_to_fetch = list(SEASON_DATES.keys())

    all_new = []
    for season in seasons_to_fetch:
        print(f"\nFetching {season}...")
        records = fetch_for_season(season, existing_ids)
        all_new.extend(records)

    if not all_new:
        print("No new records fetched.")
        return

    # Merge total_runs in
    new_df = merge_total_runs(all_new)

    if not existing.empty:
        existing["date"] = pd.to_datetime(existing["date"])
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.drop_duplicates(subset=["game_id"])
    combined = combined.sort_values("date").reset_index(drop=True)
    combined.to_csv(out_path, sep="\t", index=False)

    print(f"\nSaved {len(combined)} umpire records to {out_path}")
    matched = combined["total_runs"].notna().sum()
    print(f"  total_runs matched: {matched}/{len(combined)} ({matched/len(combined):.1%})")
    print(f"  Unique umpires: {combined['hp_umpire'].nunique()}")
    print(f"  Date range: {combined['date'].min()} to {combined['date'].max()}")

    # Print top/bottom umpires by avg total runs
    ump_stats = (
        combined.dropna(subset=["total_runs"])
        .groupby("hp_umpire")
        .agg(games=("total_runs", "count"), avg_runs=("total_runs", "mean"))
        .query("games >= 30")
        .sort_values("avg_runs")
    )
    print(f"\n  Top 5 low-scoring umpires (>= 30 games called):")
    for name, row in ump_stats.head(5).iterrows():
        print(f"    {name}: {row['avg_runs']:.2f} avg runs ({int(row['games'])} games)")
    print(f"\n  Top 5 high-scoring umpires:")
    for name, row in ump_stats.tail(5).iterrows():
        print(f"    {name}: {row['avg_runs']:.2f} avg runs ({int(row['games'])} games)")


if __name__ == "__main__":
    main()
