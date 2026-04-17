"""
Per-game pregame prediction poller.

Runs every 20 minutes via GitHub Actions. Identifies all games starting
within the target window (WINDOW_OPEN_MIN to WINDOW_CLOSE_MIN minutes from
now), runs predict_today.py ONCE for the full slate, then extracts and saves
each qualifying game's prediction. Handles simultaneous game slates (5-6
games starting at the same time) correctly with a single model run.

Also logs timing outcomes for every game — hits, misses, too-earlies —
so we can measure the scheduler's actual success rate over time.

Usage:
    python predict_pregame.py
    python predict_pregame.py --date 2026-04-17   # override date (testing)
    python predict_pregame.py --dry-run            # log timing only, no predictions
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import statsapi

PROJECT_DIR = Path(__file__).resolve().parent
PAPER_TRACKING_DIR = PROJECT_DIR / "paper_tracking"
PREDICTIONS_DIR = PROJECT_DIR / "predictions"

PACIFIC_TZ = ZoneInfo("America/Los_Angeles")
UTC = timezone.utc

# A game is "in window" when it starts between these many minutes from now.
WINDOW_OPEN_MIN = 5    # too close / possibly live
WINDOW_CLOSE_MIN = 35  # lineup not confirmed yet

LOG_COLUMNS = [
    "date",
    "game_id",
    "away_team",
    "home_team",
    "scheduled_pt",
    "polled_at_pt",
    "minutes_before",
    "status",   # hit | missed | failed | too_early | already_done
    "notes",
]


def log_path(year: str) -> Path:
    return PAPER_TRACKING_DIR / f"pregame_log_{year}.tsv"


def load_log(date_str: str) -> pd.DataFrame:
    path = log_path(date_str[:4])
    if path.exists():
        return pd.read_csv(path, sep="\t", dtype=str)
    return pd.DataFrame(columns=LOG_COLUMNS)


def save_log(df: pd.DataFrame, date_str: str) -> None:
    path = log_path(date_str[:4])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def append_entry(log: pd.DataFrame, entry: dict) -> pd.DataFrame:
    return pd.concat([log, pd.DataFrame([entry])], ignore_index=True)


def already_done(game_id: str | int, log: pd.DataFrame) -> bool:
    if log.empty:
        return False
    rows = log[log["game_id"].astype(str) == str(game_id)]
    return any(rows["status"].isin(["hit", "already_done"]))


def run_predictions(date_str: str) -> pd.DataFrame | None:
    """Run predict_today.py (pregame-only, no --all-games) and return the board."""
    print(f"  Running predict_today.py for {date_str}...")
    result = subprocess.run(
        [sys.executable, str(PROJECT_DIR / "predict_today.py"), date_str],
        capture_output=True,
        text=True,
        cwd=PROJECT_DIR,
    )
    if result.returncode != 0:
        print(f"  ERROR: predict_today.py exited {result.returncode}")
        print(result.stderr[-500:] if result.stderr else "")
        return None

    board_path = PREDICTIONS_DIR / f"{date_str}-board.tsv"
    if not board_path.exists():
        print(f"  ERROR: {board_path.name} not found after running predict_today.py")
        return None

    df = pd.read_csv(board_path, sep="\t", dtype=str)
    if df.empty:
        print(f"  WARNING: board is empty (all games may have started already)")
        return None

    return df


def save_game_prediction(board: pd.DataFrame, game_id: str | int, date_str: str) -> bool:
    """Extract a single game from the board and save as a per-game JSON."""
    row = board[board["game_id"].astype(str) == str(game_id)]
    if row.empty:
        return False
    out = PREDICTIONS_DIR / f"{date_str}-pregame-{game_id}.json"
    row.to_json(out, orient="records", indent=2)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="Override date YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true",
                        help="Log timing only, skip predictions")
    args = parser.parse_args()

    now_utc = datetime.now(UTC)
    now_pt = now_utc.astimezone(PACIFIC_TZ)
    date_str = args.date or now_pt.strftime("%Y-%m-%d")
    polled_at_pt = now_pt.strftime("%H:%M")

    print(f"Pregame poller — {now_pt.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"Target date: {date_str}  |  Window: {WINDOW_OPEN_MIN}–{WINDOW_CLOSE_MIN} min before first pitch")
    print()

    log = load_log(date_str)
    try:
        games = statsapi.schedule(date=date_str, sportId=1)
    except Exception as e:
        print(f"ERROR: could not fetch schedule: {e}")
        return

    if not games:
        print("No games today.")
        return

    # ── Classify every game ─────────────────────────────────────────────────
    in_window: list[dict] = []    # needs a fresh prediction right now
    any_changes = False

    print(f"Found {len(games)} games:")
    for g in games:
        game_id = str(g.get("game_id", ""))
        away = g.get("away_name", "?")
        home = g.get("home_name", "?")
        status = g.get("status", "")
        game_dt_str = g.get("game_datetime")

        if not game_dt_str:
            continue

        game_utc = datetime.fromisoformat(game_dt_str.replace("Z", "+00:00"))
        game_pt = game_utc.astimezone(PACIFIC_TZ).strftime("%H:%M")
        mins = (game_utc - now_utc).total_seconds() / 60.0
        matchup = f"{away} @ {home}"

        if already_done(game_id, log):
            print(f"  {matchup} ({game_pt} PT): already predicted")
            continue

        is_final = "Final" in status or "Game Over" in status or "Completed" in status

        if is_final or mins < -WINDOW_OPEN_MIN:
            # Game is over or already started past the window — log missed
            print(f"  {matchup} ({game_pt} PT): {'final' if is_final else f'live ({abs(mins):.0f} min past start)'} — MISSED")
            log = append_entry(log, {
                "date": date_str, "game_id": game_id,
                "away_team": away, "home_team": home,
                "scheduled_pt": game_pt, "polled_at_pt": polled_at_pt,
                "minutes_before": f"{mins:.0f}",
                "status": "missed",
                "notes": "final" if is_final else "already live when polled",
            })
            any_changes = True
            continue

        if mins > WINDOW_CLOSE_MIN:
            print(f"  {matchup} ({game_pt} PT): {mins:.0f} min away — too early")
            continue

        # ── In window ────────────────────────────────────────────────────────
        print(f"  {matchup} ({game_pt} PT): {mins:.1f} min to first pitch — IN WINDOW")
        in_window.append({
            "game_id": game_id, "away": away, "home": home,
            "game_pt": game_pt, "mins": mins,
        })

    # ── Run predictions once for all in-window games ─────────────────────────
    if in_window:
        print(f"\n{len(in_window)} game(s) in window — running predictions (1 model run)...")
        board = None if args.dry_run else run_predictions(date_str)

        for ginfo in in_window:
            game_id = ginfo["game_id"]
            matchup = f"{ginfo['away']} @ {ginfo['home']}"

            if args.dry_run:
                print(f"  [dry-run] {matchup}")
                status, notes = "hit", "dry-run"
            elif board is None:
                status, notes = "failed", "predict_today.py error"
            else:
                ok = save_game_prediction(board, game_id, date_str)
                status = "hit" if ok else "failed"
                notes = "" if ok else f"game_id {game_id} not in board"
                if ok:
                    print(f"  Saved pregame prediction for {matchup}")

            log = append_entry(log, {
                "date": date_str, "game_id": game_id,
                "away_team": ginfo["away"], "home_team": ginfo["home"],
                "scheduled_pt": ginfo["game_pt"], "polled_at_pt": polled_at_pt,
                "minutes_before": f"{ginfo['mins']:.1f}",
                "status": status, "notes": notes,
            })
            any_changes = True

    if any_changes:
        save_log(log, date_str)

    # ── Print running success rate ────────────────────────────────────────────
    today = log[log["date"] == date_str] if not log.empty else pd.DataFrame()
    if not today.empty:
        hits = (today["status"] == "hit").sum()
        misses = (today["status"] == "missed").sum()
        failed = (today["status"] == "failed").sum()
        decided = hits + misses + failed
        rate = hits / decided if decided else 0.0
        print(f"\nToday's hit rate: {hits}/{decided} = {rate:.0%}  "
              f"(misses={misses}, failed={failed})")

    # ── Print season-level summary ────────────────────────────────────────────
    full_log = load_log(date_str)
    if not full_log.empty:
        hits = (full_log["status"] == "hit").sum()
        misses = (full_log["status"] == "missed").sum()
        failed = (full_log["status"] == "failed").sum()
        decided = hits + misses + failed
        rate = hits / decided if decided else 0.0
        print(f"Season hit rate:  {hits}/{decided} = {rate:.0%}  "
              f"(misses={misses}, failed={failed})")


if __name__ == "__main__":
    main()
