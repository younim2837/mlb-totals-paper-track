"""
Betting Goal Framework Scorecard

Reads a walk-forward betting summary and line-level detail, then prints a
betting-oriented scorecard focused on market performance rather than point
forecast error alone.

Example:
    python betting_goal_framework.py --year 2025
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from backtest import roi_at_110
from train_model import DATA_DIR


EDGE_THRESHOLDS = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00]
CONFIDENCE_THRESHOLDS = [0.53, 0.55, 0.57, 0.60, 0.65]


def classify_roi(roi: float) -> str:
    if roi >= 2.0:
        return "strong"
    if roi > 0.0:
        return "positive"
    if roi >= -2.0:
        return "watch"
    return "negative"


def classify_brier(brier: float) -> str:
    if brier <= 0.245:
        return "strong"
    if brier <= 0.252:
        return "good"
    if brier <= 0.258:
        return "watch"
    return "weak"


def summarize_subset(df: pd.DataFrame) -> dict:
    total = int(len(df))
    if total == 0:
        return {
            "bets": 0,
            "wins": 0,
            "win_rate": np.nan,
            "roi": np.nan,
            "avg_edge": np.nan,
            "avg_prob": np.nan,
        }

    wins = int(df["bet_won"].sum())
    return {
        "bets": total,
        "wins": wins,
        "win_rate": float(wins / total),
        "roi": float(roi_at_110(wins, total)),
        "avg_edge": float(df["edge"].abs().mean()),
        "avg_prob": float(np.maximum(df["p_over"], df["p_under"]).mean()),
    }


def evaluate_thresholds(lines_df: pd.DataFrame) -> dict:
    edge_rows = []
    for threshold in EDGE_THRESHOLDS:
        sub = lines_df[lines_df["edge"].abs() >= threshold].copy()
        row = summarize_subset(sub)
        row["threshold"] = threshold
        edge_rows.append(row)

    confidence_rows = []
    confidence = np.maximum(lines_df["p_over"], lines_df["p_under"])
    for threshold in CONFIDENCE_THRESHOLDS:
        sub = lines_df[confidence >= threshold].copy()
        row = summarize_subset(sub)
        row["threshold"] = threshold
        confidence_rows.append(row)

    monthly_rows = []
    for month, sub in lines_df.groupby("evaluation_month"):
        row = summarize_subset(sub)
        row["month"] = str(month)
        monthly_rows.append(row)

    direction_rows = []
    for bet, sub in lines_df.groupby("bet"):
        row = summarize_subset(sub)
        row["bet"] = str(bet)
        direction_rows.append(row)

    return {
        "edge_thresholds": edge_rows,
        "confidence_thresholds": confidence_rows,
        "monthly": monthly_rows,
        "by_direction": direction_rows,
    }


def build_goal_framework(summary: dict, lines_df: pd.DataFrame) -> dict:
    betting = summary.get("betting", {})
    threshold_views = evaluate_thresholds(lines_df)

    best_conf = max(
        threshold_views["confidence_thresholds"],
        key=lambda row: (-np.inf if np.isnan(row["roi"]) else row["roi"], row["bets"]),
    )
    best_edge = max(
        threshold_views["edge_thresholds"],
        key=lambda row: (-np.inf if np.isnan(row["roi"]) else row["roi"], row["bets"]),
    )

    framework = {
        "north_star": {
            "metric": "Walk-forward ROI vs closing lines",
            "current_value": betting.get("roi"),
            "status": classify_roi(float(betting.get("roi", np.nan))),
            "target": "Sustain positive ROI out-of-sample; aim for +2%+ before scaling stake size.",
        },
        "probability_quality": {
            "metric": "Closing-line Brier score",
            "current_value": betting.get("brier"),
            "status": classify_brier(float(betting.get("brier", np.nan))),
            "target": "Keep closing-line probabilities calibrated enough that Brier stays <= 0.252.",
        },
        "volume": {
            "metric": "Matched bets",
            "current_value": betting.get("matched_games"),
            "status": "strong" if betting.get("matched_games", 0) >= 1500 else "watch",
            "target": "Maintain at least 1,500 matched bets for a full-season read.",
        },
        "stability": {
            "metric": "Worst month ROI",
            "current_value": min((row["roi"] for row in threshold_views["monthly"] if not np.isnan(row["roi"])), default=np.nan),
            "status": "watch",
            "target": "Avoid repeated deep monthly drawdowns; investigate any month worse than -8%.",
        },
        "bet_selection": {
            "best_confidence_cut": best_conf,
            "best_edge_cut": best_edge,
        },
        "next_metric_to_add": {
            "metric": "CLV / beat-the-close rate",
            "why": "Historical snapshots now exist, but we still need a defined entry-time betting rule to evaluate whether our bets beat the eventual close.",
        },
    }
    return framework


def print_scorecard(summary: dict, framework: dict, thresholds: dict) -> None:
    betting = summary["betting"]
    print("\n=== Betting Goal Framework ===")
    print(f"Year: {summary['year']}")
    print(f"North star ROI vs close: {betting['roi']:.1f}%  [{framework['north_star']['status']}]")
    print(f"Closing-line win rate: {betting['win_rate']:.1%}")
    print(f"Closing-line Brier: {betting['brier']:.4f}  [{framework['probability_quality']['status']}]")
    print(f"Matched bets: {betting['matched_games']}")
    print(f"Weighted MAE: {summary['weighted_mae']:.2f}")
    print(f"Weighted bias: {summary['weighted_bias']:+.2f}")

    print("\nBest confidence cuts:")
    for row in thresholds["confidence_thresholds"]:
        if row["bets"] == 0:
            continue
        print(
            f"  max(p_over,p_under)>={row['threshold']:.2f}: "
            f"{row['bets']} bets | win {row['win_rate']:.1%} | ROI {row['roi']:.1f}%"
        )

    print("\nBest edge cuts:")
    for row in thresholds["edge_thresholds"]:
        if row["bets"] == 0:
            continue
        print(
            f"  |edge|>={row['threshold']:.2f}: "
            f"{row['bets']} bets | win {row['win_rate']:.1%} | ROI {row['roi']:.1f}%"
        )

    print("\nMonthly stability:")
    for row in thresholds["monthly"]:
        print(
            f"  {row['month']}: {row['bets']} bets | win {row['win_rate']:.1%} | ROI {row['roi']:.1f}%"
        )


def main():
    parser = argparse.ArgumentParser(description="Score a walk-forward betting season against the betting goal framework.")
    parser.add_argument("--year", type=int, default=2025, help="Target season year")
    args = parser.parse_args()

    year = args.year
    summary_path = os.path.join(DATA_DIR, f"walk_forward_betting_summary_{year}.json")
    lines_path = os.path.join(DATA_DIR, f"walk_forward_betting_lines_{year}.tsv")
    out_path = os.path.join(DATA_DIR, f"betting_goal_scorecard_{year}.json")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    if not os.path.exists(lines_path):
        raise FileNotFoundError(f"Missing line detail file: {lines_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    lines_df = pd.read_csv(lines_path, sep="\t")

    thresholds = evaluate_thresholds(lines_df)
    framework = build_goal_framework(summary, lines_df)

    payload = {
        "summary": summary,
        "framework": framework,
        "threshold_views": thresholds,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print_scorecard(summary, framework, thresholds)
    print(f"\nScorecard JSON saved to {out_path}")


if __name__ == "__main__":
    main()
