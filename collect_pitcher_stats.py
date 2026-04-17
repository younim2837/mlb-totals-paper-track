"""
MLB Pitcher Stats Collection
Pulls per-start game logs for every starting pitcher in our game dataset.
Outputs: data/pitcher_game_logs.tsv  (one row per start)
         data/pitcher_ids.tsv        (name -> player_id lookup)

Run modes:
    python collect_pitcher_stats.py                      # backfill new pitchers only
    python collect_pitcher_stats.py --update-current     # refresh current season for all known pitchers
"""

import statsapi
import pandas as pd
import time
import os
import sys
import json
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]

# Stat fields we care about from each start
STAT_FIELDS = [
    "inningsPitched", "earnedRuns", "hits", "homeRuns",
    "baseOnBalls", "strikeOuts", "numberOfPitches", "battersFaced",
    "era", "whip",
]


def load_pitcher_names() -> set:
    """Extract all unique probable pitcher names from our game data."""
    df = pd.read_csv(os.path.join(DATA_DIR, "mlb_games_raw.tsv"), sep="\t")
    names = set()
    for col in ["home_pitcher", "away_pitcher"]:
        names.update(df[col].dropna().unique())
    # Remove blanks / TBD entries
    names = {n for n in names if n and n.strip() and n.lower() not in ("tbd", "", "unknown")}
    return names


def build_full_player_index(seasons: list) -> dict:
    """
    Fetch all MLB players across seasons in bulk.
    Returns: {fullName: player_id}
    The bulk endpoint is reliable where name-by-name lookup fails.
    """
    name_to_id = {}
    for season in seasons:
        try:
            data = statsapi.get("sports_players", {"season": season, "sportId": 1, "gameType": "R"})
            for p in data.get("people", []):
                name_to_id[p["fullName"]] = p["id"]
            print(f"  Season {season}: {len(data.get('people', []))} players indexed")
        except Exception as e:
            print(f"  Season {season}: ERROR - {e}")
        time.sleep(0.5)
    return name_to_id


def lookup_player_ids(names: set, cache_path: str, seasons: list) -> dict:
    """
    Match pitcher names to player IDs using the full seasonal player index.
    Caches results to avoid re-fetching on reruns.
    Returns: {name: player_id}
    """
    # Load existing cache
    if os.path.exists(cache_path):
        existing = pd.read_csv(cache_path, sep="\t")
        name_to_id = dict(zip(existing["name"], existing["player_id"]))
        print(f"  Loaded {len(name_to_id)} cached IDs")
    else:
        name_to_id = {}

    missing = names - set(name_to_id.keys())
    if not missing:
        print("  All pitcher IDs already cached.")
        return name_to_id

    print(f"  Building full player index for {len(missing)} unresolved names...")
    full_index = build_full_player_index(seasons)
    print(f"  Total players in index: {len(full_index)}")

    found, failed = [], []
    for name in sorted(missing):
        if name in full_index:
            name_to_id[name] = full_index[name]
            found.append(name)
        else:
            failed.append(name)

    print(f"  Matched {len(found)} new pitchers")
    if failed:
        print(f"  Still unmatched ({len(failed)}): {failed[:10]}")

    # Save updated cache
    rows = [{"name": n, "player_id": pid} for n, pid in name_to_id.items()]
    pd.DataFrame(rows).to_csv(cache_path, sep="\t", index=False)
    print(f"  Saved {len(name_to_id)} IDs to cache")

    return name_to_id


def fetch_pitcher_game_logs(player_id: int, seasons: list) -> list:
    """
    Fetch per-start game logs for a pitcher across multiple seasons.
    Returns list of dicts with date + stats.
    """
    records = []
    for season in seasons:
        try:
            data = statsapi.get("person", {
                "personId": player_id,
                "hydrate": f"stats(group=pitching,type=gameLog,season={season})",
            })
            stats_list = data["people"][0].get("stats", [])
            if not stats_list:
                continue

            splits = stats_list[0].get("splits", [])
            for s in splits:
                # Only include starts (not relief appearances)
                if not s["stat"].get("gamesStarted", 0):
                    continue
                record = {
                    "player_id": player_id,
                    "date": s.get("date"),
                    "season": season,
                    "opponent": s.get("opponent", {}).get("name", ""),
                    "is_home": 1 if s.get("isHome") else 0,
                }
                for field in STAT_FIELDS:
                    record[field] = s["stat"].get(field)
                records.append(record)
        except Exception:
            pass

        time.sleep(0.2)

    return records


def compute_innings_float(ip_str) -> float:
    """Convert '6.2' (6 innings, 2 outs) to decimal innings (6.667)."""
    if ip_str is None:
        return None
    try:
        ip = float(ip_str)
        full = int(ip)
        partial = round(ip - full, 1)
        return full + (partial / 0.3) * (1 / 3)
    except (ValueError, TypeError):
        return None


def parse_pitcher_log_dates(series: pd.Series) -> pd.Series:
    """
    Parse stored pitcher log dates robustly across mixed string formats.
    """
    try:
        return pd.to_datetime(series, format="mixed", errors="coerce")
    except TypeError:
        return pd.to_datetime(series, errors="coerce")


def update_current_season(logs_path, id_cache_path):
    """
    Re-fetch current season game logs for all known pitchers.
    Used for daily updates — picks up new starts without re-fetching all history.
    """
    current_year = datetime.now().year
    print(f"Updating {current_year} season logs for all known pitchers...")

    if not os.path.exists(id_cache_path):
        print("  No pitcher ID cache found — run full backfill first.")
        return

    ids_df = pd.read_csv(id_cache_path, sep="\t")
    all_ids = sorted(ids_df["player_id"].unique())
    print(f"  Refreshing {len(all_ids)} pitchers for {current_year}...")

    existing_logs = pd.read_csv(logs_path, sep="\t") if os.path.exists(logs_path) else pd.DataFrame()

    new_records = []
    for i, player_id in enumerate(all_ids):
        records = fetch_pitcher_game_logs(player_id, [current_year])
        new_records.extend(records)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(all_ids)} pitchers updated...")
            time.sleep(0.5)

    if not new_records:
        print(f"  No {current_year} starts found yet.")
        return

    new_df = pd.DataFrame(new_records)
    new_df["innings_pitched_dec"] = new_df["inningsPitched"].apply(compute_innings_float)
    new_df["date"] = parse_pitcher_log_dates(new_df["date"])

    # Drop existing current-season rows, replace with fresh fetch
    if not existing_logs.empty:
        existing_logs["date"] = parse_pitcher_log_dates(existing_logs["date"])
        prior = existing_logs[
            existing_logs["date"].isna() | (existing_logs["date"].dt.year != current_year)
        ]
        combined = pd.concat([prior, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.sort_values(["player_id", "date"]).reset_index(drop=True)
    combined.to_csv(logs_path, sep="\t", index=False)
    print(f"  Done. {len(new_df)} starts in {current_year}. Total log: {len(combined)} starts.")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    id_cache_path = os.path.join(DATA_DIR, "pitcher_ids.tsv")
    logs_path = os.path.join(DATA_DIR, "pitcher_game_logs.tsv")

    # ── Current-season refresh mode ───────────────────────────────────────────
    if "--update-current" in sys.argv:
        update_current_season(logs_path, id_cache_path)
        return

    # ── Full backfill: new pitchers only ─────────────────────────────────────
    print("Loading pitcher names from game data...")
    names = load_pitcher_names()
    print(f"  Found {len(names)} unique pitcher names")

    print("\nLooking up player IDs...")
    name_to_id = lookup_player_ids(names, id_cache_path, SEASONS)

    if os.path.exists(logs_path):
        existing_logs = pd.read_csv(logs_path, sep="\t")
        existing_ids = set(existing_logs["player_id"].unique())
        print(f"\nFound {len(existing_ids)} pitchers already in logs, skipping them")
    else:
        existing_logs = pd.DataFrame()
        existing_ids = set()

    unique_ids = set(name_to_id.values()) - existing_ids
    print(f"Fetching game logs for {len(unique_ids)} pitchers...")

    all_records = []
    for i, player_id in enumerate(sorted(unique_ids)):
        records = fetch_pitcher_game_logs(player_id, SEASONS)
        all_records.extend(records)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(unique_ids)} pitchers done ({len(all_records)} starts so far)")
            time.sleep(1)

    new_df = pd.DataFrame(all_records)
    if not new_df.empty:
        new_df["innings_pitched_dec"] = new_df["inningsPitched"].apply(compute_innings_float)
        new_df["date"] = parse_pitcher_log_dates(new_df["date"])
        combined = pd.concat([existing_logs, new_df], ignore_index=True) if not existing_logs.empty else new_df
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
        combined = combined.sort_values(["player_id", "date"]).reset_index(drop=True)
        combined.to_csv(logs_path, sep="\t", index=False)
        print(f"\nSaved {len(combined)} total starts to {logs_path}")
        print(f"  Unique pitchers: {combined['player_id'].nunique()}")
        print(f"  Date range: {combined['date'].min()} to {combined['date'].max()}")
    else:
        print("No new records fetched.")


if __name__ == "__main__":
    main()
