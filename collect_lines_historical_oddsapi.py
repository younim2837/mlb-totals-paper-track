"""
MLB Historical Totals Backfill via The Odds API

This script prepares a historical sportsbook totals dataset using The Odds API's
historical snapshots endpoint. It is designed to be safe to add before an API
key exists:

1. Estimate usage credits for a given year range / snapshot plan.
2. Build a snapshot schedule from the local MLB game history.
3. Fetch historical totals snapshots once an Odds API key is available.
4. Collapse multiple snapshots into a best-available pregame line per matchup.

Outputs:
  - data/lines_historical_oddsapi_snapshots.tsv
  - data/lines_historical_oddsapi.tsv

Example estimation:
  python collect_lines_historical_oddsapi.py --years 2022 2023 2024 2025 --estimate-only

Example backfill later:
  python collect_lines_historical_oddsapi.py --years 2022 2023 2024 2025 --fetch
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")
RAW_GAMES_FILE = os.path.join(DATA_DIR, "mlb_games_raw.tsv")
SNAPSHOT_OUT = os.path.join(DATA_DIR, "lines_historical_oddsapi_snapshots.tsv")
FINAL_OUT = os.path.join(DATA_DIR, "lines_historical_oddsapi.tsv")

ODDS_API_HOST = "https://api.the-odds-api.com"
HISTORICAL_ODDS_ENDPOINT = "/v4/historical/sports/baseball_mlb/odds"

DEFAULT_BOOKMAKERS = ["pinnacle", "draftkings", "fanduel", "betmgm", "caesars"]
DEFAULT_SNAPSHOT_HOURS = [9, 12, 15, 18, 21]

HISTORICAL_COST_MULTIPLIER = 10


@dataclass
class SnapshotPlan:
    requested_ts: pd.Timestamp
    requested_ts_iso: str
    game_date: pd.Timestamp
    local_label: str


def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_api_key() -> str | None:
    config = load_config()
    return config.get("odds_api_key") or os.environ.get("ODDS_API_KEY")


def parse_years(values: list[str]) -> list[int]:
    years = []
    for value in values:
        if "-" in value:
            start, end = value.split("-", 1)
            years.extend(range(int(start), int(end) + 1))
        else:
            years.append(int(value))
    return sorted(set(years))


def load_local_game_dates(years: list[int]) -> pd.DataFrame:
    if not os.path.exists(RAW_GAMES_FILE):
        raise FileNotFoundError(f"Missing raw games file: {RAW_GAMES_FILE}")

    df = pd.read_csv(RAW_GAMES_FILE, sep="\t", parse_dates=["date"])
    df = df[df["date"].dt.year.isin(years)].copy()
    if df.empty:
        return pd.DataFrame(columns=["season", "game_date", "games"])

    summary = (
        df.assign(game_date=df["date"].dt.normalize(), season=df["date"].dt.year)
          .groupby(["season", "game_date"], as_index=False)
          .size()
          .rename(columns={"size": "games"})
          .sort_values(["season", "game_date"])
          .reset_index(drop=True)
    )
    return summary


def build_snapshot_plan(
    date_summary: pd.DataFrame,
    snapshot_hours_local: list[int],
    timezone_name: str,
) -> list[SnapshotPlan]:
    tz = ZoneInfo(timezone_name)
    plans: list[SnapshotPlan] = []

    for row in date_summary.itertuples(index=False):
        game_date = pd.Timestamp(row.game_date).normalize()
        for hour in snapshot_hours_local:
            local_dt = datetime(
                game_date.year,
                game_date.month,
                game_date.day,
                int(hour),
                0,
                0,
                tzinfo=tz,
            )
            utc_ts = pd.Timestamp(local_dt.astimezone(ZoneInfo("UTC")))
            plans.append(
                SnapshotPlan(
                    requested_ts=utc_ts,
                    requested_ts_iso=utc_ts.isoformat().replace("+00:00", "Z"),
                    game_date=game_date,
                    local_label=local_dt.strftime("%Y-%m-%d %H:%M %Z"),
                )
            )

    return plans


def estimate_credits(snapshot_count: int, markets: int = 1, bookmaker_groups: int = 1) -> int:
    return int(snapshot_count * HISTORICAL_COST_MULTIPLIER * markets * bookmaker_groups)


def fetch_historical_snapshot(
    api_key: str,
    requested_ts_iso: str,
    bookmakers: list[str],
    markets: str = "totals",
    odds_format: str = "american",
) -> tuple[pd.DataFrame, dict]:
    url = f"{ODDS_API_HOST}{HISTORICAL_ODDS_ENDPOINT}"
    params = {
        "apiKey": api_key,
        "markets": markets,
        "bookmakers": ",".join(bookmakers),
        "oddsFormat": odds_format,
        "dateFormat": "iso",
        "date": requested_ts_iso,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    snapshot_ts = payload.get("timestamp")
    previous_ts = payload.get("previous_timestamp")
    events = payload.get("data") or []

    rows = []
    for event in events:
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        commence_time = event.get("commence_time")
        event_id = event.get("id")

        lines_by_book = {}
        for bookmaker in event.get("bookmakers", []):
            book_key = bookmaker.get("key")
            total_line = None
            last_update = bookmaker.get("last_update")

            for market in bookmaker.get("markets", []):
                if market.get("key") != "totals":
                    continue
                for outcome in market.get("outcomes", []):
                    if outcome.get("name") == "Over":
                        total_line = outcome.get("point")
                        break
                if total_line is not None:
                    break

            if total_line is not None:
                lines_by_book[book_key] = {
                    "line": float(total_line),
                    "last_update": last_update,
                }

        if not lines_by_book:
            continue

        line_values = sorted(item["line"] for item in lines_by_book.values())
        consensus = float(np.median(line_values))

        row = {
            "requested_snapshot_ts": requested_ts_iso,
            "snapshot_ts": snapshot_ts,
            "previous_snapshot_ts": previous_ts,
            "event_id": event_id,
            "commence_time": commence_time,
            "home_team": home_team,
            "away_team": away_team,
            "consensus_total_line": consensus,
            "num_books": int(len(lines_by_book)),
        }
        for book_key, item in lines_by_book.items():
            row[f"{book_key}_line"] = item["line"]
            row[f"{book_key}_last_update"] = item["last_update"]
        rows.append(row)

    meta = {
        "requested_snapshot_ts": requested_ts_iso,
        "snapshot_ts": snapshot_ts,
        "previous_snapshot_ts": previous_ts,
        "events": len(events),
        "rows": len(rows),
        "x_requests_last": response.headers.get("x-requests-last"),
        "x_requests_used": response.headers.get("x-requests-used"),
        "x_requests_remaining": response.headers.get("x-requests-remaining"),
    }
    return pd.DataFrame(rows), meta


def load_existing_snapshots() -> pd.DataFrame:
    if not os.path.exists(SNAPSHOT_OUT):
        return pd.DataFrame()
    return pd.read_csv(SNAPSHOT_OUT, sep="\t", parse_dates=["requested_snapshot_ts", "snapshot_ts", "previous_snapshot_ts", "commence_time"])


def persist_snapshots(snapshot_df: pd.DataFrame):
    snapshot_df = snapshot_df.copy()
    snapshot_df = snapshot_df.sort_values(["requested_snapshot_ts", "away_team", "home_team"]).reset_index(drop=True)
    snapshot_df.to_csv(SNAPSHOT_OUT, sep="\t", index=False)


def finalize_snapshots(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    if snapshot_df.empty:
        return snapshot_df.copy()

    df = snapshot_df.copy()
    for col in ["requested_snapshot_ts", "snapshot_ts", "previous_snapshot_ts", "commence_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    df["date"] = df["commence_time"].dt.tz_convert(None).dt.normalize()
    df["pregame"] = (df["snapshot_ts"] <= df["commence_time"]) | df["commence_time"].isna()
    df["pregame_rank"] = np.where(df["pregame"], 0, 1)
    df["snapshot_delta_seconds"] = (df["commence_time"] - df["snapshot_ts"]).dt.total_seconds()
    df["snapshot_delta_seconds"] = df["snapshot_delta_seconds"].fillna(10**12)
    df.loc[~df["pregame"], "snapshot_delta_seconds"] = 10**12

    sort_cols = ["date", "away_team", "home_team", "pregame_rank", "snapshot_delta_seconds", "num_books"]
    ascending = [True, True, True, True, True, False]
    df = df.sort_values(sort_cols, ascending=ascending)
    final = df.groupby(["date", "away_team", "home_team"], as_index=False).first()

    keep = [
        "date",
        "away_team",
        "home_team",
        "commence_time",
        "snapshot_ts",
        "requested_snapshot_ts",
        "consensus_total_line",
        "num_books",
    ]
    extra = [c for c in final.columns if c.endswith("_line")]
    final = final[keep + extra].copy()
    final = final.rename(columns={"consensus_total_line": "close_total_line"})

    if os.path.exists(RAW_GAMES_FILE):
        raw = pd.read_csv(RAW_GAMES_FILE, sep="\t", parse_dates=["date"])
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
        raw["actual_total"] = raw["away_score"] + raw["home_score"]
        raw["season"] = raw["date"].dt.year
        actuals = (
            raw[["date", "away_team", "home_team", "actual_total", "season"]]
            .dropna(subset=["date", "away_team", "home_team"])
            .drop_duplicates(subset=["date", "away_team", "home_team"], keep="last")
        )
        final = final.merge(
            actuals,
            on=["date", "away_team", "home_team"],
            how="left",
        )
    final.to_csv(FINAL_OUT, sep="\t", index=False)
    return final


def print_estimate(date_summary: pd.DataFrame, snapshot_hours_local: list[int], timezone_name: str, bookmakers: list[str]):
    snapshot_count = int(len(date_summary) * len(snapshot_hours_local))
    bookmaker_groups = max(1, (len(bookmakers) + 9) // 10)
    credits = estimate_credits(snapshot_count, markets=1, bookmaker_groups=bookmaker_groups)

    print("Historical Odds API backfill estimate")
    print(f"  Seasons: {int(date_summary['season'].min())}–{int(date_summary['season'].max())}")
    print(f"  Slate dates: {len(date_summary)}")
    print(f"  Snapshot hours ({timezone_name}): {', '.join(str(h) for h in snapshot_hours_local)}")
    print(f"  Snapshots per day: {len(snapshot_hours_local)}")
    print(f"  Total snapshot calls: {snapshot_count}")
    print(f"  Bookmakers: {', '.join(bookmakers)}")
    print(f"  Bookmaker groups billed: {bookmaker_groups}")
    print(f"  Estimated credits: {credits}")
    print(f"  Snapshot output: {SNAPSHOT_OUT}")
    print(f"  Final output: {FINAL_OUT}")


def main():
    parser = argparse.ArgumentParser(description="Backfill MLB historical totals using The Odds API.")
    parser.add_argument("--years", nargs="+", default=["2022-2025"], help="Years or ranges, e.g. 2022 2023 2024 2025 or 2022-2025")
    parser.add_argument("--timezone", default="America/New_York", help="Local timezone for snapshot hours")
    parser.add_argument(
        "--snapshot-hours-local",
        default="9,12,15,18,21",
        help="Comma-separated local hours to query each slate date",
    )
    parser.add_argument(
        "--bookmakers",
        default=",".join(DEFAULT_BOOKMAKERS),
        help="Comma-separated bookmaker keys; up to 10 counts as one billed region",
    )
    parser.add_argument("--estimate-only", action="store_true", help="Only estimate credits; do not fetch")
    parser.add_argument("--fetch", action="store_true", help="Fetch snapshots using ODDS_API_KEY / config.json")
    parser.add_argument("--resume", action="store_true", help="Skip requested timestamps already present in snapshot output")
    args = parser.parse_args()

    years = parse_years(args.years)
    snapshot_hours_local = [int(x.strip()) for x in args.snapshot_hours_local.split(",") if x.strip()]
    bookmakers = [x.strip() for x in args.bookmakers.split(",") if x.strip()]

    os.makedirs(DATA_DIR, exist_ok=True)

    date_summary = load_local_game_dates(years)
    if date_summary.empty:
        print("No local MLB game dates found for the requested years.")
        return

    print_estimate(date_summary, snapshot_hours_local, args.timezone, bookmakers)
    if args.estimate_only and not args.fetch:
        return

    api_key = load_api_key()
    if not api_key:
        print("No Odds API key found. Set ODDS_API_KEY or add odds_api_key to config.json.")
        return

    plans = build_snapshot_plan(date_summary, snapshot_hours_local, args.timezone)
    existing = load_existing_snapshots()
    existing_requested = set()
    if args.resume and not existing.empty and "requested_snapshot_ts" in existing.columns:
        existing_requested = {
            pd.Timestamp(ts).isoformat().replace("+00:00", "Z")
            for ts in pd.to_datetime(existing["requested_snapshot_ts"], errors="coerce", utc=True).dropna().unique()
        }

    fetched_frames = [existing] if not existing.empty else []
    metas = []
    total_calls = 0

    for idx, plan in enumerate(plans, start=1):
        if plan.requested_ts_iso in existing_requested:
            print(f"[{idx}/{len(plans)}] Skip existing {plan.local_label}")
            continue

        print(f"[{idx}/{len(plans)}] Fetch {plan.local_label} -> {plan.requested_ts_iso}")
        try:
            frame, meta = fetch_historical_snapshot(
                api_key=api_key,
                requested_ts_iso=plan.requested_ts_iso,
                bookmakers=bookmakers,
            )
        except Exception as exc:
            print(f"  Failed: {exc}")
            continue

        if not frame.empty:
            fetched_frames.append(frame)
        metas.append(meta)
        total_calls += 1

        if fetched_frames:
            combined = pd.concat(fetched_frames, ignore_index=True)
            persist_snapshots(combined)

        remaining = meta.get("x_requests_remaining")
        last_cost = meta.get("x_requests_last")
        print(
            f"  Rows: {meta.get('rows', 0)} | snapshot {meta.get('snapshot_ts')} "
            f"| cost {last_cost} | remaining {remaining}"
        )

    if fetched_frames:
        combined = pd.concat(fetched_frames, ignore_index=True)
        persist_snapshots(combined)
        final = finalize_snapshots(combined)
        print(f"\nSaved {len(combined)} snapshot rows to {SNAPSHOT_OUT}")
        print(f"Saved {len(final)} deduped game rows to {FINAL_OUT}")
    else:
        print("\nNo snapshot rows were collected.")

    if metas:
        used_values = [int(m["x_requests_used"]) for m in metas if str(m.get("x_requests_used", "")).isdigit()]
        if used_values:
            print(f"Requests used after run: {max(used_values)}")
        print(f"Snapshot calls attempted this run: {total_calls}")


if __name__ == "__main__":
    main()
