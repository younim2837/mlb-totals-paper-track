"""
MLB Betting Model - Daily Runner
================================
The single entry point for daily use. Checks data freshness, updates what's
stale, runs predictions, and saves output to predictions/YYYY-MM-DD.txt.

Usage:
    python run_today.py                   # update data + predict today
    python run_today.py 2026-04-10        # predict for a specific date
    python run_today.py --no-update       # skip data update, just predict
    python run_today.py --all-games       # include live/final games in report
    python run_today.py --quick           # update games only (skip ~5min pitcher refresh)

Data update cadence:
    Games       — fetched incrementally every run (seconds, not minutes)
    Pitchers    — refreshed once every 3 days (starters pitch every 5-6 days)
    Bullpens    — refreshed daily (same-day workload changes quickly)
"""

import argparse
import json
import os
import sys
import subprocess
import pandas as pd
from datetime import date, datetime, timedelta

# Force utf-8 on Windows console so Unicode in predictions prints cleanly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
PRED_DIR    = os.path.join(PROJECT_DIR, "predictions")


# ─────────────────────────────────────────────────────────────────────────────
# Freshness checks
# ─────────────────────────────────────────────────────────────────────────────

def games_max_date():
    path = os.path.join(DATA_DIR, "mlb_games_raw.tsv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep="\t", usecols=["date"], parse_dates=["date"])
    return df["date"].max().date()


def pitchers_last_modified():
    path = os.path.join(DATA_DIR, "pitcher_game_logs.tsv")
    if not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)
    return date.fromtimestamp(mtime)


def bullpens_last_modified():
    path = os.path.join(DATA_DIR, "bullpen_appearance_logs.tsv")
    if not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)
    return date.fromtimestamp(mtime)


def run_script(args, label):
    """Run a Python script as a subprocess, streaming output to console."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    cmd = [sys.executable] + args
    result = subprocess.run(cmd, cwd=PROJECT_DIR)
    if result.returncode != 0:
        print(f"\nWARNING: {label} exited with code {result.returncode}")
        return False
    return True


def load_model_feature_flags():
    meta_path = os.path.join(PROJECT_DIR, "models", "model_meta.json")
    if not os.path.exists(meta_path):
        return False, True

    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        features = set(meta.get("features", []))
        needs_elo = any("elo" in f for f in features)
        needs_dc = any(
            f.startswith("home_dc_") or f.startswith("away_dc_") or f.startswith("dc_")
            for f in features
        )
        return needs_elo, needs_dc
    except Exception:
        return False, True


# ─────────────────────────────────────────────────────────────────────────────
# Prediction runner with output capture (console + file simultaneously)
# ─────────────────────────────────────────────────────────────────────────────

def run_predictions(target_date, out_path, include_all_games=False):
    """
    Run predict_today.py and stream output to both console and a saved file.
    """
    os.makedirs(PRED_DIR, exist_ok=True)
    script = os.path.join(PROJECT_DIR, "predict_today.py")
    cmd = [sys.executable, script]
    if target_date:
        cmd.append(target_date)
    if include_all_games:
        cmd.append("--all-games")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        for line in proc.stdout:
            print(line, end="", flush=True)
            f.write(line)

    proc.wait()
    return proc.returncode == 0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MLB model daily runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "date", nargs="?",
        help="Target date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--no-update", action="store_true",
        help="Skip all data updates — use existing files as-is",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Update games only, skip pitcher refresh (saves ~5 min)",
    )
    parser.add_argument(
        "--all-games", action="store_true",
        help="Include live and final games too; default report is pregame only.",
    )
    args = parser.parse_args()

    target_date = args.date
    date_str    = target_date or date.today().strftime("%Y-%m-%d")
    yesterday   = date.today() - timedelta(days=1)
    needs_elo, needs_dc = load_model_feature_flags()

    print(f"\nMLB PREDICTIONS — {date_str}")
    print("=" * 60)

    # ── 1. Data freshness check ───────────────────────────────────────────────
    if not args.no_update:
        g_max  = games_max_date()
        p_last = pitchers_last_modified()
        b_last = bullpens_last_modified()
        today  = date.today()

        games_stale    = g_max is None or g_max < yesterday
        pitchers_stale = p_last is None or (today - p_last).days >= 3
        bullpens_stale = b_last is None or b_last < today

        print(f"\nData status:")
        print(f"  Games      : {'STALE' if games_stale else 'OK'}"
              f"  (through {g_max or 'none'})")
        print(f"  Pitchers   : {'STALE' if pitchers_stale else 'OK'}"
              f"  (last updated {p_last or 'never'})")
        print(f"  Bullpens   : {'STALE' if bullpens_stale else 'OK'}"
              f"  (last updated {b_last or 'never'})")

        # ── Update games (fast — only fetches since last game) ────────────────
        if games_stale:
            run_script(
                [os.path.join(PROJECT_DIR, "collect_games.py")],
                "Updating game data (incremental)...",
            )
        else:
            print(f"\n  Games already current — skipping.")

        # ── Update pitcher logs (slower — once every 3 days) ──────────────────
        if pitchers_stale and not args.quick:
            run_script(
                [os.path.join(PROJECT_DIR, "collect_pitcher_stats.py"),
                 "--update-current"],
                "Refreshing current-season pitcher stats...",
            )
        elif pitchers_stale and args.quick:
            print(f"\n  Pitcher stats stale but --quick set — skipping pitcher refresh.")
        else:
            print(f"  Pitcher stats current — skipping.")

        if bullpens_stale:
            run_script(
                [os.path.join(PROJECT_DIR, "collect_bullpen_usage.py"),
                 "--update-current"],
                "Refreshing current-season bullpen usage...",
            )
        else:
            print(f"  Bullpen usage current — skipping.")

        # Refresh the Bayesian prior artifacts required by the active model.
        if games_stale:
            if needs_elo:
                run_script(
                    [os.path.join(PROJECT_DIR, "team_elo.py")],
                    "Refreshing Elo ratings...",
                )
            if needs_dc:
                run_script(
                    [os.path.join(PROJECT_DIR, "dixon_coles.py"), "--date", date_str, "--cache-only"],
                    "Refreshing Dixon-Coles priors...",
                )

    else:
        print("\n  Skipping data update (--no-update).")

    # ── 2. Run predictions ────────────────────────────────────────────────────
    file_name = f"{date_str}-all-games.txt" if args.all_games else f"{date_str}.txt"
    out_path = os.path.join(PRED_DIR, file_name)

    print(f"\n{'='*60}")
    print(f"  PREDICTIONS")
    print(f"{'='*60}\n")

    success = run_predictions(target_date, out_path, include_all_games=args.all_games)

    # ── 3. Summary ───────────────────────────────────────────────────────────
    if success:
        print(f"\n{'='*60}")
        print(f"  Saved to: predictions/{file_name}")
        print(f"{'='*60}")
    else:
        print(f"\nWARNING: Prediction script exited with an error.")


if __name__ == "__main__":
    main()
