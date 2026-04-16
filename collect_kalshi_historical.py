"""
Collect historical Kalshi MLB total-run lines for the current season.

Fetches the 10 AM Pacific (17:00 UTC) pre-game snapshot for every settled
MLB totals event.  This matches the time the daily bot runs, so the prices
reflect exactly what the bot would have seen.

Usage:
    python collect_kalshi_historical.py                # full 2026 season
    python collect_kalshi_historical.py --season 2026
    python collect_kalshi_historical.py --date 2026-04-13   # single date

Output: data/kalshi_historical_lines.tsv
    date | away_team | home_team | strike | yes_price | has_10am_price
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data"
OUT_PATH    = DATA_DIR / "kalshi_historical_lines.tsv"

KALSHI_API       = "https://api.elections.kalshi.com/trade-api/v2"
MLB_TOTAL_SERIES = "KXMLBTOTAL"

KALSHI_TEAM_MAP = {
    "Arizona":        "Arizona Diamondbacks",
    "Arizona D":      "Arizona Diamondbacks",
    "Atlanta":        "Atlanta Braves",
    "Baltimore":      "Baltimore Orioles",
    "Boston":         "Boston Red Sox",
    "Chicago C":      "Chicago Cubs",
    "Chicago W":      "Chicago White Sox",
    "Chicago WS":     "Chicago White Sox",
    "Cincinnati":     "Cincinnati Reds",
    "Cleveland":      "Cleveland Guardians",
    "Colorado":       "Colorado Rockies",
    "Detroit":        "Detroit Tigers",
    "Houston":        "Houston Astros",
    "Kansas City":    "Kansas City Royals",
    "KC":             "Kansas City Royals",
    "Los Angeles A":  "Los Angeles Angels",
    "Los Angeles D":  "Los Angeles Dodgers",
    "LA Angels":      "Los Angeles Angels",
    "LA Dodgers":     "Los Angeles Dodgers",
    "Miami":          "Miami Marlins",
    "Milwaukee":      "Milwaukee Brewers",
    "Minnesota":      "Minnesota Twins",
    "New York M":     "New York Mets",
    "New York Y":     "New York Yankees",
    "NY Mets":        "New York Mets",
    "NY Yankees":     "New York Yankees",
    "Oakland":        "Athletics",
    "A's":            "Athletics",
    "Athletics":      "Athletics",
    "Philadelphia":   "Philadelphia Phillies",
    "Pittsburgh":     "Pittsburgh Pirates",
    "San Diego":      "San Diego Padres",
    "San Francisco":  "San Francisco Giants",
    "Seattle":        "Seattle Mariners",
    "St. Louis":      "St. Louis Cardinals",
    "St Louis":       "St. Louis Cardinals",
    "Tampa Bay":      "Tampa Bay Rays",
    "Texas":          "Texas Rangers",
    "Toronto":        "Toronto Blue Jays",
    "Washington":     "Washington Nationals",
}


def _parse_teams(title: str) -> tuple[str, str] | None:
    m = re.match(r"^(.+?)\s+vs\s+(.+?)(?:\s*:.*)?$", title, re.IGNORECASE)
    if not m:
        return None
    a = KALSHI_TEAM_MAP.get(m.group(1).strip())
    h = KALSHI_TEAM_MAP.get(m.group(2).strip())
    return (a, h) if a and h else None


def _event_date(ticker: str) -> date | None:
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", ticker)
    if not m:
        return None
    yy, mon, dd = m.groups()
    mo = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
          "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}.get(mon)
    if not mo:
        return None
    try:
        return date(2000 + int(yy), mo, int(dd))
    except ValueError:
        return None


def _api_get(url: str, params: dict, retries: int = 3) -> requests.Response | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                wait = 2 ** attempt
                print(f"  rate-limited, sleeping {wait}s…")
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            return None
        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def snapshot_unix(game_date: date) -> int:
    """10:00 AM Pacific = 17:00 UTC on the given date."""
    dt = datetime(game_date.year, game_date.month, game_date.day, 17, 0, 0,
                  tzinfo=timezone.utc)
    return int(dt.timestamp())


def load_all_settled_events() -> list[dict]:
    """Fetch every settled KXMLBTOTAL event (paginated)."""
    events: list[dict] = []
    cursor = None
    for _ in range(20):
        params: dict = {"series_ticker": MLB_TOTAL_SERIES, "status": "settled", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        r = _api_get(f"{KALSHI_API}/events", params)
        if r is None:
            break
        data = r.json()
        page = data.get("events", [])
        events.extend(page)
        cursor = data.get("cursor")
        if not cursor or not page:
            break
    return events


def collect_date(game_date: date, events: list[dict], rate: float = 0.20) -> list[dict]:
    """
    For a single game date, collect the 10-AM-PT Kalshi price at every
    available strike for every game.  Returns a list of row dicts.
    """
    max_ts = snapshot_unix(game_date)
    target_events = [
        ev for ev in events if _event_date(ev.get("event_ticker", "")) == game_date
    ]
    if not target_events:
        return []

    rows: list[dict] = []
    for ev in target_events:
        teams = _parse_teams(ev.get("title", ""))
        if not teams:
            continue
        away, home = teams

        # Fetch strike markets
        r = _api_get(
            f"{KALSHI_API}/markets",
            {"event_ticker": ev["event_ticker"], "limit": 50},
        )
        time.sleep(rate)
        if r is None:
            continue
        markets = r.json().get("markets", [])

        for mkt in markets:
            strike = mkt.get("floor_strike")
            if strike is None:
                continue
            strike = float(strike)

            # Last trade at or before 10 AM PT
            tr = _api_get(
                f"{KALSHI_API}/markets/trades",
                {"ticker": mkt["ticker"], "max_ts": max_ts, "limit": 1},
            )
            time.sleep(rate)

            api_failed = tr is None
            trades = [] if api_failed else tr.json().get("trades", [])

            if trades:
                t = trades[0]
                rows.append({
                    "date":           str(game_date),
                    "away_team":      away,
                    "home_team":      home,
                    "strike":         strike,
                    "yes_price":      float(t["yes_price_dollars"]),
                    "trade_time":     t["created_time"],
                    "has_10am_price": True,
                    "api_failed":     False,
                })
            elif api_failed:
                # API call failed after all retries — NOT the same as no trades
                rows.append({
                    "date":           str(game_date),
                    "away_team":      away,
                    "home_team":      home,
                    "strike":         strike,
                    "yes_price":      float("nan"),
                    "trade_time":     None,
                    "has_10am_price": False,
                    "api_failed":     True,
                })
            else:
                # Market existed but genuinely had no trades by 10 AM
                rows.append({
                    "date":           str(game_date),
                    "away_team":      away,
                    "home_team":      home,
                    "strike":         strike,
                    "yes_price":      float("nan"),
                    "trade_time":     None,
                    "has_10am_price": False,
                    "api_failed":     False,
                })

    return rows


def season_dates(season: int) -> list[date]:
    """All game dates in our historical game file for the given season."""
    path = DATA_DIR / "mlb_games_raw.tsv"
    if not path.exists():
        return []
    df = pd.read_csv(path, sep="\t", usecols=["date"])
    df["date"] = pd.to_datetime(df["date"])
    today = pd.Timestamp.now().normalize()
    mask = (df["date"].dt.year == season) & (df["date"] < today)
    return sorted(df.loc[mask, "date"].dt.date.unique())


def validate_coverage(df: pd.DataFrame, season: int) -> None:
    """
    Cross-check collected games against mlb_games_raw.tsv.
    Prints any games that are completely missing from the collected data,
    and any dates that have API failures.
    """
    raw_path = DATA_DIR / "mlb_games_raw.tsv"
    if not raw_path.exists():
        return

    raw = pd.read_csv(raw_path, sep="\t", usecols=["date", "away_team", "home_team"])
    raw["date"] = pd.to_datetime(raw["date"])
    today = pd.Timestamp.now().normalize()
    raw = raw[(raw["date"].dt.year == season) & (raw["date"] < today)]

    collected_keys = set(
        zip(df["date"].astype(str), df["away_team"], df["home_team"])
    )

    missing_games: list[str] = []
    for _, row in raw.drop_duplicates(["date", "away_team", "home_team"]).iterrows():
        key = (str(row["date"].date()), row["away_team"], row["home_team"])
        if key not in collected_keys:
            missing_games.append(f"  {key[0]}  {key[1]} @ {key[2]}")

    failed_rows = df[df.get("api_failed", pd.Series(False, index=df.index)).astype(bool)]

    if missing_games:
        print(f"\n[VALIDATION] {len(missing_games)} games completely missing from Kalshi data:")
        for g in missing_games:
            print(g)
    else:
        print(f"\n[VALIDATION] All {len(raw.drop_duplicates(['date','away_team','home_team']))} expected games have Kalshi data.")

    if not failed_rows.empty:
        n_failed = len(failed_rows)
        n_games_failed = failed_rows.drop_duplicates(["date","away_team","home_team"]).shape[0]
        print(f"[VALIDATION] {n_failed} strikes across {n_games_failed} games had API failures (api_failed=True).")
        print("             Re-run with --fix-missing to retry those rows.")
    else:
        print("[VALIDATION] No API failures detected in collected data.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect historical Kalshi 10-AM-PT lines")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--date",   type=str, default=None,
                        help="Collect a single date YYYY-MM-DD instead of whole season")
    parser.add_argument("--resume", action="store_true",
                        help="Skip dates already in the output file")
    parser.add_argument("--fix-missing", action="store_true",
                        help="Re-fetch only strikes where api_failed=True in the existing file")
    args = parser.parse_args()

    if args.date:
        dates = [date.fromisoformat(args.date)]
    else:
        dates = season_dates(args.season)

    if not dates:
        print("No game dates found.")
        return

    # Load existing data if resuming, fixing, or adding a single date (always merge to avoid overwrite).
    existing: pd.DataFrame = pd.DataFrame()
    if (args.resume or args.fix_missing or args.date) and OUT_PATH.exists():
        existing = pd.read_csv(OUT_PATH, sep="\t")
        # Back-fill api_failed column if missing (rows collected before this feature)
        if "api_failed" not in existing.columns:
            existing["api_failed"] = False

    if args.fix_missing and not existing.empty:
        # Only re-fetch strikes that previously failed the API call
        if "api_failed" not in existing.columns:
            print("No api_failed column in existing file — nothing to fix.")
            return
        failed = existing[existing["api_failed"].astype(bool)]
        if failed.empty:
            print("No API failures found in existing data.")
            validate_coverage(existing, args.season)
            return
        # Collect only the dates that had failures
        dates = sorted(pd.to_datetime(failed["date"]).dt.date.unique())
        # We'll drop failed rows and re-collect those dates
        existing = existing[~existing["api_failed"].astype(bool)]
        print(f"Fix-missing: {len(failed)} failed strikes across {len(dates)} dates — re-fetching")
    elif args.resume and not existing.empty:
        already = set(existing["date"].unique())
        dates = [d for d in dates if str(d) not in already]
        print(f"Resuming: {len(already)} dates already collected, {len(dates)} remaining")

    if not dates:
        print("All dates already collected.")
        return

    # Fetch all settled events once (cheaper than re-fetching per date)
    print("Fetching settled event list…")
    all_events = load_all_settled_events()
    print(f"  {len(all_events)} settled events found")

    all_rows: list[dict] = []
    for i, d in enumerate(dates, 1):
        print(f"  [{i:2d}/{len(dates)}] {d}…", end=" ", flush=True)
        rows = collect_date(d, all_events)
        games = len({(r["away_team"], r["home_team"]) for r in rows})
        priced = sum(1 for r in rows if r["has_10am_price"])
        print(f"{games} games, {priced}/{len(rows)} strikes with pre-10am price")
        all_rows.extend(rows)

    new_df = pd.DataFrame(all_rows)
    if not existing.empty:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date","away_team","home_team","strike"])
        combined = combined.sort_values(["date","away_team","home_team","strike"])
    else:
        combined = new_df.sort_values(["date","away_team","home_team","strike"])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH, sep="\t", index=False)
    print(f"\nSaved {len(combined)} rows to {OUT_PATH.relative_to(PROJECT_DIR)}")

    # Quick coverage summary
    if not combined.empty:
        priced = combined["has_10am_price"].sum()
        total  = len(combined)
        games  = combined.drop_duplicates(["date","away_team","home_team"]).shape[0]
        failed = combined.get("api_failed", pd.Series(False, index=combined.index)).astype(bool).sum()
        print(f"Coverage: {games} games, {priced}/{total} strikes with 10-AM price ({priced/total*100:.0f}%)")
        if failed:
            print(f"WARNING: {failed} strikes have api_failed=True — run with --fix-missing to retry")
        validate_coverage(combined, args.season)


if __name__ == "__main__":
    main()
