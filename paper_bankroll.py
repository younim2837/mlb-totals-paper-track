"""
Helpers for tracking the paper-trading bankroll across daily runs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


PROJECT_DIR = Path(__file__).resolve().parent
PAPER_TRACKING_DIR = PROJECT_DIR / "paper_tracking"


def load_starting_bankroll(default: float = 10000.0) -> float:
    config_path = PROJECT_DIR / "model_config.yaml"
    if not _YAML_AVAILABLE or not config_path.exists():
        return float(default)

    try:
        with config_path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        bankroll_cfg = cfg.get("bankroll", {}) or {}
        return float(bankroll_cfg.get("total", default) or default)
    except Exception:
        return float(default)


def resolve_paper_bankroll(target_date: str, season: int, default: float = 10000.0) -> float:
    starting_bankroll = load_starting_bankroll(default=default)
    tracker_path = PAPER_TRACKING_DIR / f"kalshi_tracker_{season}.tsv"
    if not tracker_path.exists() or tracker_path.stat().st_size == 0:
        return starting_bankroll

    try:
        tracker = pd.read_csv(tracker_path, sep="\t")
    except Exception:
        return starting_bankroll

    required_cols = {"target_date", "paper_bankroll_after_day"}
    if tracker.empty or not required_cols.issubset(tracker.columns):
        return starting_bankroll

    prior_rows = tracker[tracker["target_date"].astype(str) < str(target_date)].copy()
    if prior_rows.empty:
        return starting_bankroll

    bankroll_series = pd.to_numeric(prior_rows["paper_bankroll_after_day"], errors="coerce").dropna()
    if bankroll_series.empty:
        return starting_bankroll
    return float(bankroll_series.iloc[-1])
