"""
MLB Historical Featured Markets Backfill via The Odds API

This script prepares a historical sportsbook dataset using The Odds API's
historical snapshots endpoint. It now supports featured MLB markets:

1. Moneyline (`h2h`)
2. Run line (`spreads`)
3. Totals (`totals`)

It is designed to be robust for long-running backfills:

1. Estimate usage credits for a given year range / snapshot plan.
2. Build a snapshot schedule from the local MLB game history.
3. Fetch historical featured-market snapshots with retry/backoff.
4. Persist request-level success/failure logs so `--resume` can recover cleanly.
5. Collapse multiple snapshots into a best-available pregame line per matchup.

Outputs:
  - data/lines_historical_oddsapi_snapshots.tsv
  - data/lines_historical_oddsapi.tsv
  - data/lines_historical_oddsapi_requests.tsv
"""

from __future__ import annotations

import argparse
import json
import os
import time
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
REQUEST_LOG_OUT = os.path.join(DATA_DIR, "lines_historical_oddsapi_requests.tsv")

ODDS_API_HOST = "https://api.the-odds-api.com"
HISTORICAL_ODDS_ENDPOINT = "/v4/historical/sports/baseball_mlb/odds"

DEFAULT_BOOKMAKERS = ["pinnacle", "draftkings", "fanduel", "betmgm", "caesars"]
DEFAULT_MARKETS = ["h2h", "spreads", "totals"]
DEFAULT_SNAPSHOT_HOURS = [9, 12, 15, 18, 21]
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_DELAY_SECONDS = 2.0
DEFAULT_REQUEST_PAUSE_SECONDS = 0.25

HISTORICAL_COST_MULTIPLIER = 10
RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
REQUEST_TIMEOUT_SECONDS = 30
PREGAME_SENTINEL_SECONDS = 10**12

EVENT_KEY_COLS = [
    "requested_snapshot_ts",
    "snapshot_ts",
    "previous_snapshot_ts",
    "event_id",
    "commence_time",
    "home_team",
    "away_team",
]
REQUEST_LOG_DATE_COLS = ["requested_snapshot_ts", "snapshot_ts", "previous_snapshot_ts"]


@dataclass
class SnapshotPlan:
    requested_ts: pd.Timestamp
    requested_ts_iso: str
    game_date: pd.Timestamp
    local_label: str
    expected_games: int


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


def parse_csv_values(raw: str) -> list[str]:
    return [value.strip() for value in str(raw).split(",") if value.strip()]


def normalize_ts_iso(value) -> str | None:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).isoformat().replace("+00:00", "Z")


def median_or_none(values) -> float | None:
    clean = [float(v) for v in values if v is not None and pd.notna(v)]
    if not clean:
        return None
    return float(np.median(clean))


def first_present(*values):
    for value in values:
        if value is not None and pd.notna(value):
            return value
    return None


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
                    expected_games=int(row.games),
                )
            )

    return plans


def estimate_credits(snapshot_count: int, markets: int = 1, bookmaker_groups: int = 1) -> int:
    return int(snapshot_count * HISTORICAL_COST_MULTIPLIER * markets * bookmaker_groups)


def _market_map(bookmaker: dict) -> dict:
    return {
        market.get("key"): market
        for market in bookmaker.get("markets", []) or []
        if market.get("key")
    }


def _parse_h2h_market(market: dict | None, home_team: str, away_team: str) -> dict:
    result = {
        "h2h_home_price": None,
        "h2h_away_price": None,
        "h2h_last_update": None,
    }
    if not market:
        return result

    result["h2h_last_update"] = market.get("last_update")
    for outcome in market.get("outcomes", []) or []:
        if outcome.get("name") == home_team:
            result["h2h_home_price"] = outcome.get("price")
        elif outcome.get("name") == away_team:
            result["h2h_away_price"] = outcome.get("price")
    return result


def _parse_spreads_market(market: dict | None, home_team: str, away_team: str) -> dict:
    result = {
        "spread_home_line": None,
        "spread_home_price": None,
        "spread_away_line": None,
        "spread_away_price": None,
        "spreads_last_update": None,
    }
    if not market:
        return result

    result["spreads_last_update"] = market.get("last_update")
    for outcome in market.get("outcomes", []) or []:
        if outcome.get("name") == home_team:
            result["spread_home_line"] = outcome.get("point")
            result["spread_home_price"] = outcome.get("price")
        elif outcome.get("name") == away_team:
            result["spread_away_line"] = outcome.get("point")
            result["spread_away_price"] = outcome.get("price")
    return result


def _parse_totals_market(market: dict | None) -> dict:
    result = {
        "total_line": None,
        "over_price": None,
        "under_price": None,
        "totals_last_update": None,
    }
    if not market:
        return result

    result["totals_last_update"] = market.get("last_update")
    for outcome in market.get("outcomes", []) or []:
        if outcome.get("name") == "Over":
            result["total_line"] = outcome.get("point")
            result["over_price"] = outcome.get("price")
        elif outcome.get("name") == "Under":
            result["under_price"] = outcome.get("price")
    return result


def _parse_event_book(bookmaker: dict, home_team: str, away_team: str) -> dict:
    markets = _market_map(bookmaker)
    parsed = {}
    parsed.update(_parse_h2h_market(markets.get("h2h"), home_team, away_team))
    parsed.update(_parse_spreads_market(markets.get("spreads"), home_team, away_team))
    parsed.update(_parse_totals_market(markets.get("totals")))
    parsed["book_last_update"] = first_present(
        parsed.get("h2h_last_update"),
        parsed.get("spreads_last_update"),
        parsed.get("totals_last_update"),
        bookmaker.get("last_update"),
    )
    return parsed


def parse_historical_payload(
    payload: dict,
    requested_ts_iso: str,
) -> tuple[pd.DataFrame, dict]:
    snapshot_ts = payload.get("timestamp")
    previous_ts = payload.get("previous_timestamp")
    events = payload.get("data")
    if not isinstance(events, list):
        raise ValueError("Historical odds payload missing list-valued data")

    rows = []
    for event in events:
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        commence_time = event.get("commence_time")
        event_id = event.get("id")

        book_values = {}
        h2h_home_prices = []
        h2h_away_prices = []
        spread_home_lines = []
        spread_home_prices = []
        spread_away_lines = []
        spread_away_prices = []
        total_lines = []
        over_prices = []
        under_prices = []
        num_books_h2h = 0
        num_books_spreads = 0
        num_books_totals = 0

        for bookmaker in event.get("bookmakers", []) or []:
            book_key = bookmaker.get("key")
            if not book_key:
                continue

            parsed = _parse_event_book(bookmaker, home_team, away_team)
            book_values[book_key] = parsed

            if parsed.get("h2h_home_price") is not None and parsed.get("h2h_away_price") is not None:
                h2h_home_prices.append(parsed["h2h_home_price"])
                h2h_away_prices.append(parsed["h2h_away_price"])
                num_books_h2h += 1
            if parsed.get("spread_home_line") is not None and parsed.get("spread_away_line") is not None:
                spread_home_lines.append(parsed["spread_home_line"])
                spread_away_lines.append(parsed["spread_away_line"])
                if parsed.get("spread_home_price") is not None:
                    spread_home_prices.append(parsed["spread_home_price"])
                if parsed.get("spread_away_price") is not None:
                    spread_away_prices.append(parsed["spread_away_price"])
                num_books_spreads += 1
            if parsed.get("total_line") is not None:
                total_lines.append(parsed["total_line"])
                if parsed.get("over_price") is not None:
                    over_prices.append(parsed["over_price"])
                if parsed.get("under_price") is not None:
                    under_prices.append(parsed["under_price"])
                num_books_totals += 1

        if not book_values:
            continue

        row = {
            "requested_snapshot_ts": requested_ts_iso,
            "snapshot_ts": snapshot_ts,
            "previous_snapshot_ts": previous_ts,
            "event_id": event_id,
            "commence_time": commence_time,
            "home_team": home_team,
            "away_team": away_team,
            "consensus_h2h_home_price": median_or_none(h2h_home_prices),
            "consensus_h2h_away_price": median_or_none(h2h_away_prices),
            "consensus_spread_home_line": median_or_none(spread_home_lines),
            "consensus_spread_home_price": median_or_none(spread_home_prices),
            "consensus_spread_away_line": median_or_none(spread_away_lines),
            "consensus_spread_away_price": median_or_none(spread_away_prices),
            "consensus_total_line": median_or_none(total_lines),
            "consensus_over_price": median_or_none(over_prices),
            "consensus_under_price": median_or_none(under_prices),
            "num_books": int(num_books_totals),
            "num_books_h2h": int(num_books_h2h),
            "num_books_spreads": int(num_books_spreads),
            "num_books_totals": int(num_books_totals),
        }
        for book_key, parsed in book_values.items():
            # Legacy totals aliases retained for existing consumers.
            row[f"{book_key}_line"] = parsed.get("total_line")
            row[f"{book_key}_last_update"] = parsed.get("book_last_update")

            row[f"{book_key}_h2h_home_price"] = parsed.get("h2h_home_price")
            row[f"{book_key}_h2h_away_price"] = parsed.get("h2h_away_price")
            row[f"{book_key}_h2h_last_update"] = parsed.get("h2h_last_update")

            row[f"{book_key}_spread_home_line"] = parsed.get("spread_home_line")
            row[f"{book_key}_spread_home_price"] = parsed.get("spread_home_price")
            row[f"{book_key}_spread_away_line"] = parsed.get("spread_away_line")
            row[f"{book_key}_spread_away_price"] = parsed.get("spread_away_price")
            row[f"{book_key}_spreads_last_update"] = parsed.get("spreads_last_update")

            row[f"{book_key}_total_line"] = parsed.get("total_line")
            row[f"{book_key}_over_price"] = parsed.get("over_price")
            row[f"{book_key}_under_price"] = parsed.get("under_price")
            row[f"{book_key}_totals_last_update"] = parsed.get("totals_last_update")
        rows.append(row)

    meta = {
        "requested_snapshot_ts": requested_ts_iso,
        "snapshot_ts": snapshot_ts,
        "previous_snapshot_ts": previous_ts,
        "events": len(events),
        "rows": len(rows),
    }
    return pd.DataFrame(rows), meta


def _should_retry_payload(meta: dict, expected_games: int) -> bool:
    if not meta.get("snapshot_ts"):
        return True
    if expected_games > 0 and int(meta.get("events", 0) or 0) == 0:
        return True
    return False


def fetch_historical_snapshot(
    api_key: str,
    requested_ts_iso: str,
    bookmakers: list[str],
    markets: list[str] | None = None,
    odds_format: str = "american",
    expected_games: int = 0,
    session: requests.Session | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
) -> tuple[pd.DataFrame, dict]:
    request_session = session or requests.Session()
    url = f"{ODDS_API_HOST}{HISTORICAL_ODDS_ENDPOINT}"
    params = {
        "apiKey": api_key,
        "markets": ",".join(markets or DEFAULT_MARKETS),
        "bookmakers": ",".join(bookmakers),
        "oddsFormat": odds_format,
        "dateFormat": "iso",
        "date": requested_ts_iso,
    }

    last_meta = {
        "requested_snapshot_ts": requested_ts_iso,
        "status": "failed",
        "attempt_count": 0,
        "events": 0,
        "rows": 0,
        "error": "unknown_error",
    }

    for attempt in range(1, int(max_retries) + 1):
        start = time.time()
        try:
            response = request_session.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            status_code = int(response.status_code)
            if status_code >= 400:
                error_text = ""
                try:
                    error_payload = response.json()
                    error_text = json.dumps(error_payload)[:500]
                except Exception:
                    error_text = (response.text or "").strip()[:500]
                if status_code in RETRYABLE_STATUS_CODES and attempt < int(max_retries):
                    retry_after = response.headers.get("Retry-After")
                    wait_seconds = float(retry_after) if str(retry_after).replace(".", "", 1).isdigit() else 0.0
                    wait_seconds = max(wait_seconds, float(retry_delay_seconds) * (2 ** (attempt - 1)))
                    time.sleep(wait_seconds)
                    continue
                raise requests.HTTPError(f"HTTP {status_code}: {error_text}", response=response)

            payload = response.json()
            frame, payload_meta = parse_historical_payload(payload, requested_ts_iso=requested_ts_iso)
            meta = {
                **payload_meta,
                "requested_snapshot_ts": requested_ts_iso,
                "status": "success",
                "attempt_count": attempt,
                "error": "",
                "duration_seconds": round(time.time() - start, 3),
                "x_requests_last": response.headers.get("x-requests-last"),
                "x_requests_used": response.headers.get("x-requests-used"),
                "x_requests_remaining": response.headers.get("x-requests-remaining"),
            }
            if _should_retry_payload(meta, expected_games=expected_games) and attempt < int(max_retries):
                time.sleep(float(retry_delay_seconds) * (2 ** (attempt - 1)))
                last_meta = meta | {"status": "retrying", "error": "empty_payload"}
                continue
            return frame, meta
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            last_meta = {
                "requested_snapshot_ts": requested_ts_iso,
                "status": "failed",
                "attempt_count": attempt,
                "events": 0,
                "rows": 0,
                "error": str(exc)[:500],
                "duration_seconds": round(time.time() - start, 3),
            }
            if attempt >= int(max_retries):
                break
            time.sleep(float(retry_delay_seconds) * (2 ** (attempt - 1)))

    return pd.DataFrame(), last_meta


def load_existing_snapshots() -> pd.DataFrame:
    if not os.path.exists(SNAPSHOT_OUT):
        return pd.DataFrame()
    return pd.read_csv(
        SNAPSHOT_OUT,
        sep="\t",
        parse_dates=["requested_snapshot_ts", "snapshot_ts", "previous_snapshot_ts", "commence_time"],
    )


def load_request_log() -> pd.DataFrame:
    if not os.path.exists(REQUEST_LOG_OUT):
        return pd.DataFrame()
    return pd.read_csv(REQUEST_LOG_OUT, sep="\t", parse_dates=REQUEST_LOG_DATE_COLS)


def persist_snapshots(snapshot_df: pd.DataFrame):
    if snapshot_df.empty:
        snapshot_df.to_csv(SNAPSHOT_OUT, sep="\t", index=False)
        return
    snapshot_df = snapshot_df.copy()
    snapshot_df = snapshot_df.sort_values(["requested_snapshot_ts", "away_team", "home_team"]).reset_index(drop=True)
    snapshot_df.to_csv(SNAPSHOT_OUT, sep="\t", index=False)


def persist_request_log(request_log_df: pd.DataFrame):
    if request_log_df.empty:
        request_log_df.to_csv(REQUEST_LOG_OUT, sep="\t", index=False)
        return
    request_log_df = request_log_df.copy()
    request_log_df = request_log_df.sort_values(["requested_snapshot_ts", "attempt_count"]).reset_index(drop=True)
    request_log_df.to_csv(REQUEST_LOG_OUT, sep="\t", index=False)


def _dedupe_snapshots(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    if snapshot_df.empty:
        return snapshot_df.copy()
    ordered = snapshot_df.sort_values(
        ["requested_snapshot_ts", "event_id", "snapshot_ts", "away_team", "home_team"]
    )
    return ordered.drop_duplicates(subset=["requested_snapshot_ts", "event_id"], keep="last").reset_index(drop=True)


def finalize_snapshots(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    if snapshot_df.empty:
        return snapshot_df.copy()

    df = _dedupe_snapshots(snapshot_df)
    for col in ["requested_snapshot_ts", "snapshot_ts", "previous_snapshot_ts", "commence_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    df["date"] = df["commence_time"].dt.tz_convert(None).dt.normalize()
    df["pregame"] = (df["snapshot_ts"] <= df["commence_time"]) | df["commence_time"].isna()
    df["pregame_rank"] = np.where(df["pregame"], 0, 1)
    df["snapshot_delta_seconds"] = (df["commence_time"] - df["snapshot_ts"]).dt.total_seconds()
    df["snapshot_delta_seconds"] = df["snapshot_delta_seconds"].fillna(PREGAME_SENTINEL_SECONDS)
    df.loc[~df["pregame"], "snapshot_delta_seconds"] = PREGAME_SENTINEL_SECONDS

    sort_cols = ["date", "away_team", "home_team", "pregame_rank", "snapshot_delta_seconds", "num_books_totals", "num_books_h2h", "num_books_spreads"]
    ascending = [True, True, True, True, True, False, False, False]
    df = df.sort_values(sort_cols, ascending=ascending)
    final = df.groupby(["date", "away_team", "home_team"], as_index=False).first()

    rename_map = {
        "consensus_total_line": "close_total_line",
        "consensus_over_price": "close_over_price",
        "consensus_under_price": "close_under_price",
        "consensus_h2h_home_price": "close_h2h_home_price",
        "consensus_h2h_away_price": "close_h2h_away_price",
        "consensus_spread_home_line": "close_spread_home_line",
        "consensus_spread_home_price": "close_spread_home_price",
        "consensus_spread_away_line": "close_spread_away_line",
        "consensus_spread_away_price": "close_spread_away_price",
    }
    final = final.rename(columns=rename_map)

    keep = [
        "date",
        "away_team",
        "home_team",
        "commence_time",
        "snapshot_ts",
        "requested_snapshot_ts",
        "close_total_line",
        "close_over_price",
        "close_under_price",
        "close_h2h_home_price",
        "close_h2h_away_price",
        "close_spread_home_line",
        "close_spread_home_price",
        "close_spread_away_line",
        "close_spread_away_price",
        "num_books",
        "num_books_h2h",
        "num_books_spreads",
        "num_books_totals",
    ]
    helper_cols = {"pregame", "pregame_rank", "snapshot_delta_seconds"}
    extra = [
        col
        for col in final.columns
        if col not in keep
        and col not in helper_cols
        and not col.startswith("consensus_")
    ]
    final = final[keep + extra].copy()

    if os.path.exists(RAW_GAMES_FILE):
        raw = pd.read_csv(RAW_GAMES_FILE, sep="\t", parse_dates=["date"])
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
        raw["actual_total"] = raw["away_score"] + raw["home_score"]
        raw["actual_margin"] = raw["home_score"] - raw["away_score"]
        raw["actual_home_win"] = (raw["actual_margin"] > 0).astype(float)
        raw["season"] = raw["date"].dt.year
        actuals = (
            raw[["date", "away_team", "home_team", "away_score", "home_score", "actual_total", "actual_margin", "actual_home_win", "season"]]
            .dropna(subset=["date", "away_team", "home_team"])
            .drop_duplicates(subset=["date", "away_team", "home_team"], keep="last")
        )
        final = final.merge(actuals, on=["date", "away_team", "home_team"], how="left")

    if "close_spread_home_line" in final.columns and "actual_margin" in final.columns:
        final["actual_home_cover"] = np.where(
            final["close_spread_home_line"].notna() & final["actual_margin"].notna(),
            (final["actual_margin"] + final["close_spread_home_line"]) > 0,
            np.nan,
        )

    final.to_csv(FINAL_OUT, sep="\t", index=False)
    return final


def print_estimate(
    date_summary: pd.DataFrame,
    snapshot_hours_local: list[int],
    timezone_name: str,
    bookmakers: list[str],
    markets: list[str],
):
    snapshot_count = int(len(date_summary) * len(snapshot_hours_local))
    bookmaker_groups = max(1, (len(bookmakers) + 9) // 10)
    credits = estimate_credits(snapshot_count, markets=len(markets), bookmaker_groups=bookmaker_groups)

    print("Historical Odds API backfill estimate")
    print(f"  Seasons: {int(date_summary['season'].min())}-{int(date_summary['season'].max())}")
    print(f"  Markets: {', '.join(markets)}")
    print(f"  Slate dates: {len(date_summary)}")
    print(f"  Snapshot hours ({timezone_name}): {', '.join(str(h) for h in snapshot_hours_local)}")
    print(f"  Snapshots per day: {len(snapshot_hours_local)}")
    print(f"  Total snapshot calls: {snapshot_count}")
    print(f"  Bookmakers: {', '.join(bookmakers)}")
    print(f"  Bookmaker groups billed: {bookmaker_groups}")
    print(f"  Estimated credits: {credits}")
    print(f"  Snapshot output: {SNAPSHOT_OUT}")
    print(f"  Final output: {FINAL_OUT}")
    print(f"  Request log: {REQUEST_LOG_OUT}")


def main():
    parser = argparse.ArgumentParser(description="Backfill MLB historical featured markets using The Odds API.")
    parser.add_argument("--years", nargs="+", default=["2021-2024"], help="Years or ranges, e.g. 2021 2022 2023 2024 or 2021-2024")
    parser.add_argument("--timezone", default="America/New_York", help="Local timezone for snapshot hours")
    parser.add_argument(
        "--snapshot-hours-local",
        default="9,12,15,18,21",
        help="Comma-separated local hours to query each slate date",
    )
    parser.add_argument(
        "--bookmakers",
        default=",".join(DEFAULT_BOOKMAKERS),
        help="Comma-separated bookmaker keys; recommended 10 or fewer per run",
    )
    parser.add_argument(
        "--markets",
        default=",".join(DEFAULT_MARKETS),
        help="Comma-separated featured market keys, e.g. h2h,spreads,totals",
    )
    parser.add_argument("--estimate-only", action="store_true", help="Only estimate credits; do not fetch")
    parser.add_argument("--fetch", action="store_true", help="Fetch snapshots using ODDS_API_KEY / config.json")
    parser.add_argument("--resume", action="store_true", help="Skip requested timestamps already marked successful in the request log")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help="Retries per requested timestamp")
    parser.add_argument("--retry-delay-seconds", type=float, default=DEFAULT_RETRY_DELAY_SECONDS, help="Base delay for exponential backoff")
    parser.add_argument("--pause-seconds", type=float, default=DEFAULT_REQUEST_PAUSE_SECONDS, help="Pause between successful API calls")
    parser.add_argument("--allow-failures", action="store_true", help="Do not exit non-zero when some timestamps still fail after retries")
    args = parser.parse_args()

    years = parse_years(args.years)
    snapshot_hours_local = [int(x.strip()) for x in args.snapshot_hours_local.split(",") if x.strip()]
    bookmakers = parse_csv_values(args.bookmakers)
    markets = parse_csv_values(args.markets)

    os.makedirs(DATA_DIR, exist_ok=True)

    date_summary = load_local_game_dates(years)
    if date_summary.empty:
        print("No local MLB game dates found for the requested years.")
        return

    print_estimate(date_summary, snapshot_hours_local, args.timezone, bookmakers, markets)
    if args.estimate_only and not args.fetch:
        return

    api_key = load_api_key()
    if not api_key:
        print("No Odds API key found. Set ODDS_API_KEY or add odds_api_key to config.json.")
        return

    plans = build_snapshot_plan(date_summary, snapshot_hours_local, args.timezone)
    existing_snapshots = load_existing_snapshots()
    request_log = load_request_log()

    successful_requested = set()
    if args.resume and not request_log.empty and "status" in request_log.columns:
        successful_requested = {
            normalize_ts_iso(ts)
            for ts in request_log.loc[request_log["status"] == "success", "requested_snapshot_ts"].dropna().unique()
        }
        successful_requested.discard(None)

    fetched_frames = [existing_snapshots] if not existing_snapshots.empty else []
    request_log_frames = [request_log] if not request_log.empty else []
    failed_count = 0
    attempted_count = 0

    with requests.Session() as session:
        for idx, plan in enumerate(plans, start=1):
            if plan.requested_ts_iso in successful_requested:
                print(f"[{idx}/{len(plans)}] Skip existing success {plan.local_label}")
                continue

            print(
                f"[{idx}/{len(plans)}] Fetch {plan.local_label} -> {plan.requested_ts_iso} "
                f"(expected games {plan.expected_games})"
            )
            frame, meta = fetch_historical_snapshot(
                api_key=api_key,
                requested_ts_iso=plan.requested_ts_iso,
                bookmakers=bookmakers,
                markets=markets,
                expected_games=plan.expected_games,
                session=session,
                max_retries=args.max_retries,
                retry_delay_seconds=args.retry_delay_seconds,
            )
            attempted_count += 1
            request_log_frames.append(pd.DataFrame([meta]))

            if meta.get("status") == "success":
                if not frame.empty:
                    fetched_frames.append(frame)
                    combined = _dedupe_snapshots(pd.concat(fetched_frames, ignore_index=True))
                    persist_snapshots(combined)
                print(
                    f"  Success | rows {meta.get('rows', 0)} | events {meta.get('events', 0)} "
                    f"| snapshot {meta.get('snapshot_ts')} | attempts {meta.get('attempt_count')} "
                    f"| cost {meta.get('x_requests_last')} | remaining {meta.get('x_requests_remaining')}"
                )
            else:
                failed_count += 1
                print(
                    f"  Failed after {meta.get('attempt_count')} attempt(s): {meta.get('error', 'unknown_error')}"
                )

            combined_log = pd.concat(request_log_frames, ignore_index=True) if request_log_frames else pd.DataFrame()
            persist_request_log(combined_log)

            if float(args.pause_seconds) > 0:
                time.sleep(float(args.pause_seconds))

    if fetched_frames:
        combined = _dedupe_snapshots(pd.concat(fetched_frames, ignore_index=True))
        persist_snapshots(combined)
        final = finalize_snapshots(combined)
        print(f"\nSaved {len(combined)} snapshot rows to {SNAPSHOT_OUT}")
        print(f"Saved {len(final)} deduped game rows to {FINAL_OUT}")
    else:
        print("\nNo snapshot rows were collected.")

    combined_log = pd.concat(request_log_frames, ignore_index=True) if request_log_frames else pd.DataFrame()
    if not combined_log.empty:
        persist_request_log(combined_log)

    used_values = [
        int(value)
        for value in combined_log.get("x_requests_used", pd.Series(dtype=object)).dropna().astype(str)
        if value.isdigit()
    ] if not combined_log.empty else []
    if used_values:
        print(f"Requests used after run: {max(used_values)}")
    print(f"Snapshot timestamps attempted this run: {attempted_count}")
    print(f"Failed timestamps this run: {failed_count}")

    if failed_count and not args.allow_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
