"""
Snapshot betting failure analysis.

Reads a saved snapshot-entry backtest file and produces a compact report that
helps localize where a season went wrong.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from backtest import roi_at_110
from train_model import DATA_DIR


def summarize(sub: pd.DataFrame) -> dict:
    if sub.empty:
        return {"bets": 0, "win_rate": np.nan, "roi": np.nan, "mean_edge": np.nan, "mean_confidence": np.nan, "mean_clv": np.nan}
    wins = int(sub["bet_won_entry"].sum())
    total = int(len(sub))
    return {
        "bets": total,
        "win_rate": float(wins / total),
        "roi": float(roi_at_110(wins, total)),
        "mean_edge": float(sub["abs_edge"].mean()),
        "mean_confidence": float(sub["confidence"].mean()),
        "mean_clv": float(sub["clv_runs"].mean()),
    }


def bucketed_summary(df: pd.DataFrame, bucket_col: str) -> list[dict]:
    rows = []
    for bucket, sub in df.groupby(bucket_col, dropna=False):
        row = summarize(sub)
        row[bucket_col] = "NaN" if pd.isna(bucket) else str(bucket)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Analyze a snapshot-entry backtest file to localize betting failures.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--entry-rule", type=str, default="closest_before_minutes")
    parser.add_argument("--minutes-before", type=int, default=60)
    parser.add_argument("--min-confidence", type=float, default=0.53)
    parser.add_argument("--min-edge", type=float, default=0.50)
    args = parser.parse_args()

    suffix = f"{args.year}_{args.entry_rule}_{args.minutes_before}m_conf{args.min_confidence:.2f}_edge{args.min_edge:.2f}"
    suffix = suffix.replace(".", "p")
    entry_path = os.path.join(DATA_DIR, f"walk_forward_snapshot_entries_{suffix}.tsv")
    out_path = os.path.join(DATA_DIR, f"snapshot_failure_analysis_{suffix}.json")

    if not os.path.exists(entry_path):
        raise FileNotFoundError(entry_path)

    df = pd.read_csv(entry_path, sep="\t")

    df["line_bucket"] = pd.cut(
        pd.to_numeric(df["entry_total_line"], errors="coerce"),
        bins=[0.0, 7.5, 8.5, 9.5, 10.5, 99.0],
        labels=["<=7.5", "7.5-8.5", "8.5-9.5", "9.5-10.5", "10.5+"],
        include_lowest=True,
    )
    df["confidence_bucket"] = pd.cut(
        pd.to_numeric(df["confidence"], errors="coerce"),
        bins=[0.5, 0.55, 0.6, 0.65, 0.7, 1.01],
        labels=["0.50-0.55", "0.55-0.60", "0.60-0.65", "0.65-0.70", "0.70+"],
        include_lowest=True,
    )

    payload = {
        "overall": summarize(df),
        "by_month": bucketed_summary(df, "evaluation_month"),
        "by_bet": bucketed_summary(df, "bet"),
        "by_adjustment_method": bucketed_summary(df, "market_adjustment_method"),
        "by_line_bucket": bucketed_summary(df, "line_bucket"),
        "by_confidence_bucket": bucketed_summary(df, "confidence_bucket"),
        "top_abs_edge_examples": df.sort_values("abs_edge", ascending=False)[[
            "evaluation_month",
            "away_team",
            "home_team",
            "entry_total_line",
            "market_adjusted_total",
            "abs_edge",
            "confidence",
            "p_over",
            "bet",
            "actual_total",
            "bet_won_entry",
            "clv_runs",
            "market_adjustment_method",
        ]].head(20).to_dict(orient="records"),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("=== Snapshot Failure Analysis ===")
    print(f"Overall: {payload['overall']}")
    print("\nBy month:")
    for row in payload["by_month"]:
        print(row)
    print("\nBy bet:")
    for row in payload["by_bet"]:
        print(row)
    print("\nBy adjustment method:")
    for row in payload["by_adjustment_method"]:
        print(row)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
