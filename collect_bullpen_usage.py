"""
Collect MLB bullpen appearance logs from game feeds.

Outputs: data/bullpen_appearance_logs.tsv

Run modes:
    python collect_bullpen_usage.py
    python collect_bullpen_usage.py --update-current
    python collect_bullpen_usage.py --rebuild
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import pandas as pd

from bullpen_usage import fetch_many_game_bullpen_rows

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_GAMES_PATH = os.path.join(DATA_DIR, "mlb_games_raw.tsv")
OUT_PATH = os.path.join(DATA_DIR, "bullpen_appearance_logs.tsv")


def load_raw_games() -> pd.DataFrame:
    return pd.read_csv(RAW_GAMES_PATH, sep="\t", parse_dates=["date"]).sort_values(["date", "game_id"])


def load_existing() -> pd.DataFrame:
    if not os.path.exists(OUT_PATH):
        return pd.DataFrame()
    return pd.read_csv(OUT_PATH, sep="\t", parse_dates=["date"]).sort_values(["date", "game_id", "team_id"])


def refresh_games(target_games: pd.DataFrame, max_workers: int = 12) -> pd.DataFrame:
    if target_games.empty:
        return pd.DataFrame()
    rows = fetch_many_game_bullpen_rows(target_games["game_id"].astype(int).tolist(), max_workers=max_workers)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "game_id", "team_id", "pitcher_id"]).reset_index(drop=True)


def combine_and_save(existing: pd.DataFrame, fresh: pd.DataFrame):
    if existing.empty:
        combined = fresh.copy()
    elif fresh.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, fresh], ignore_index=True)

    if combined.empty:
        print("No bullpen logs available to save.")
        return

    combined = combined.drop_duplicates(subset=["game_id", "team_id", "pitcher_id"], keep="last")
    combined = combined.sort_values(["date", "game_id", "team_id", "pitcher_id"]).reset_index(drop=True)
    combined.to_csv(OUT_PATH, sep="\t", index=False)
    print(f"Saved {len(combined)} bullpen rows to {OUT_PATH}")
    print(f"  Unique games: {combined['game_id'].nunique()}")
    print(f"  Date range : {combined['date'].min().date()} to {combined['date'].max().date()}")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    games = load_raw_games()
    rebuild_all = "--rebuild" in sys.argv
    existing = pd.DataFrame() if rebuild_all else load_existing()

    current_year = datetime.now().year
    update_current_only = "--update-current" in sys.argv

    if update_current_only:
        target = games[games["date"].dt.year == current_year].copy()
        mode_label = f"{current_year} season"
    else:
        target = games.copy()
        mode_label = "full history"
        if rebuild_all:
            mode_label += " (rebuild)"

    done_ids = set(existing["game_id"].astype(int)) if not existing.empty else set()
    if update_current_only and not existing.empty:
        done_ids = set(
            existing.loc[existing["date"].dt.year == current_year, "game_id"].astype(int)
        )

    missing = target[~target["game_id"].astype(int).isin(done_ids)].copy()
    print(f"Collecting bullpen logs for {mode_label}...")
    print(f"  Games in scope : {len(target)}")
    print(f"  Missing games  : {len(missing)}")

    if missing.empty:
        print("  Bullpen logs already current.")
        return

    fresh = refresh_games(missing, max_workers=12)
    if fresh.empty:
        print("  No bullpen rows fetched.")
        return

    print(f"  Fetched bullpen rows: {len(fresh)}")
    combine_and_save(existing, fresh)


if __name__ == "__main__":
    main()
