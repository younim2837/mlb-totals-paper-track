"""
Run the daily paper-test pipeline and record a lightweight run manifest.

Usage:
    python paper_track_daily.py
    python paper_track_daily.py 2026-04-14
    python paper_track_daily.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


PROJECT_DIR = Path(__file__).resolve().parent
PREDICTIONS_DIR = PROJECT_DIR / "predictions"
PAPER_TRACKING_DIR = PROJECT_DIR / "paper_tracking"
RUNS_DIR = PAPER_TRACKING_DIR / "runs"
PACIFIC_TZ = ZoneInfo("America/Los_Angeles")


def default_target_date() -> str:
    return datetime.now(PACIFIC_TZ).strftime("%Y-%m-%d")


def run_daily_pipeline(target_date: str, quick: bool, no_update: bool, include_all_games: bool) -> int:
    cmd = [sys.executable, str(PROJECT_DIR / "run_today.py"), target_date]
    if quick:
        cmd.append("--quick")
    if no_update:
        cmd.append("--no-update")
    if include_all_games:
        cmd.append("--all-games")

    print(f"Running daily paper test for {target_date}...")
    print(" ".join(cmd))
    completed = subprocess.run(cmd, cwd=PROJECT_DIR)
    return int(completed.returncode)


def expected_outputs(target_date: str, include_all_games: bool) -> dict[str, Path]:
    base_name = f"{target_date}-all-games" if include_all_games else target_date
    return {
        "report_txt": PREDICTIONS_DIR / f"{base_name}.txt",
        "board_tsv": PREDICTIONS_DIR / f"{base_name}-board.tsv",
        "picks_tsv": PREDICTIONS_DIR / f"{base_name}-picks.tsv",
    }


def write_run_manifest(target_date: str, include_all_games: bool, return_code: int) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = expected_outputs(target_date, include_all_games)
    manifest = {
        "target_date": target_date,
        "include_all_games": include_all_games,
        "return_code": return_code,
        "ran_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "outputs": {
            name: {
                "path": str(path.relative_to(PROJECT_DIR)),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
            for name, path in outputs.items()
        },
    }
    manifest_path = RUNS_DIR / f"{target_date}.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Run the scheduled paper-test daily workflow.")
    parser.add_argument("date", nargs="?", help="Target date YYYY-MM-DD. Defaults to current Pacific date.")
    parser.add_argument("--quick", action="store_true", help="Skip pitcher refresh in the underlying daily run.")
    parser.add_argument("--no-update", action="store_true", help="Do not refresh source data before predicting.")
    parser.add_argument("--all-games", action="store_true", help="Also build the all-games report variant.")
    args = parser.parse_args()

    target_date = args.date or default_target_date()
    return_code = run_daily_pipeline(
        target_date=target_date,
        quick=args.quick,
        no_update=args.no_update,
        include_all_games=args.all_games,
    )
    manifest_path = write_run_manifest(
        target_date=target_date,
        include_all_games=args.all_games,
        return_code=return_code,
    )

    print(f"Run manifest written to {manifest_path.relative_to(PROJECT_DIR)}")

    if return_code != 0:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
