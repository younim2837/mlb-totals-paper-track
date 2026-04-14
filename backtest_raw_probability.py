"""
Backtest: Raw Model Probability vs Proxy Market Strike
======================================================
Uses sportsbook closing totals as a proxy strike when historical Kalshi
markets are unavailable.

This script does not claim to reproduce real Kalshi PnL because it does not
have historical Kalshi prices. Instead, it answers a narrower question:
"How does the raw model probability behave when graded against a market-like
strike?"

By default it only keeps sportsbook closes that already end in ".5", which
matches the half-run strike style used by the live Kalshi display. An optional
remap mode can coerce non-half lines onto a nearby half-run proxy strike.

Usage:
    python backtest_raw_probability.py
    python backtest_raw_probability.py --years 2024,2025
    python backtest_raw_probability.py --thresholds 55,58,60
    python backtest_raw_probability.py --line-mode nearest_half
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

from market_adjustment import probability_over_current_line

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")


def parse_int_list(text):
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def load_tail_cfgs():
    path = os.path.join(MODEL_DIR, "model_meta.json")
    if not os.path.exists(path):
        return {}, {}

    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    return meta.get("high_tail_model") or {}, meta.get("low_tail_model") or {}


def line_type(line):
    if pd.isna(line):
        return "missing"

    frac = round(float(line) - np.floor(float(line)), 2)
    if np.isclose(frac, 0.50):
        return "half"
    if np.isclose(frac, 0.00):
        return "whole"
    if np.isclose(frac, 0.25) or np.isclose(frac, 0.75):
        return "quarter"
    return "other"


def proxy_line_from_sportsbook(line, line_mode):
    if pd.isna(line):
        return np.nan

    line = float(line)
    if line_type(line) == "half":
        return line
    if line_mode == "half_only":
        return np.nan
    if line_mode == "nearest_half":
        return np.floor(line) + 0.5
    raise ValueError(f"Unsupported line mode: {line_mode}")


def optional_float(value):
    if pd.isna(value):
        return None
    return float(value)


def compute_raw_probabilities(df, high_tail_cfg, low_tail_cfg):
    probs = [
        probability_over_current_line(
            mean_total=float(predicted_total),
            base_sigma=float(prediction_std),
            line=float(proxy_line),
            high_tail_prob=optional_float(high_tail_prob),
            high_tail_cfg=high_tail_cfg,
            low_tail_prob=optional_float(low_tail_prob),
            low_tail_cfg=low_tail_cfg,
        )
        for predicted_total, prediction_std, proxy_line, high_tail_prob, low_tail_prob in zip(
            df["predicted_total"],
            df["prediction_std"],
            df["proxy_line"],
            df.get("high_tail_prob_9p5", pd.Series(np.nan, index=df.index)),
            df.get("low_tail_prob_7p5", pd.Series(np.nan, index=df.index)),
        )
    ]
    return np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)


def load_year(year, line_mode, high_tail_cfg, low_tail_cfg):
    path = os.path.join(DATA_DIR, f"walk_forward_betting_lines_{year}.tsv")
    if not os.path.exists(path):
        return None, None

    full_df = pd.read_csv(path, sep="\t")
    line_types = full_df["close_total_line"].map(line_type)
    proxy_lines = full_df["close_total_line"].map(lambda x: proxy_line_from_sportsbook(x, line_mode))

    stats = {
        "year": year,
        "total_rows": int(len(full_df)),
        "eligible_rows": int(proxy_lines.notna().sum()),
        "dropped_rows": int(proxy_lines.isna().sum()),
        "line_counts": Counter(line_types),
    }

    df = full_df.loc[proxy_lines.notna()].copy()
    if df.empty:
        df["proxy_line"] = pd.Series(dtype=float)
        return df, stats

    df["proxy_line"] = proxy_lines.loc[df.index].astype(float)
    df["raw_p_over"] = compute_raw_probabilities(df, high_tail_cfg, low_tail_cfg)
    df["raw_p_under"] = 1.0 - df["raw_p_over"]
    df["actual_over_proxy"] = df["actual_total"] > df["proxy_line"]
    df["proxy_matches_close"] = np.isclose(df["proxy_line"], df["close_total_line"])
    return df.copy(), stats


def evaluate_threshold(df, conf_threshold, label=""):
    """Evaluate betting at a given confidence threshold."""
    conf = conf_threshold / 100.0

    over_mask = df["raw_p_over"] >= conf
    under_mask = df["raw_p_under"] >= conf

    bets = []
    if over_mask.any():
        over_bets = df.loc[over_mask].copy()
        over_bets["side"] = "OVER"
        over_bets["predicted_prob"] = over_bets["raw_p_over"]
        over_bets["bet_won"] = over_bets["actual_over_proxy"]
        bets.append(over_bets)
    if under_mask.any():
        under_bets = df.loc[under_mask].copy()
        under_bets["side"] = "UNDER"
        under_bets["predicted_prob"] = under_bets["raw_p_under"]
        under_bets["bet_won"] = ~under_bets["actual_over_proxy"]
        bets.append(under_bets)

    if not bets:
        return None

    bets = pd.concat(bets, ignore_index=True)
    bets["actual_win"] = bets["bet_won"].astype(int)

    n = len(bets)
    wins = int(bets["bet_won"].sum())
    win_rate = wins / n

    profit_110 = wins * (100 / 110) - (n - wins)
    roi_110 = profit_110 / n * 100

    roi_50c_proxy = (win_rate * 2 - 1) * 100

    n_over = int((bets["side"] == "OVER").sum())
    n_under = int((bets["side"] == "UNDER").sum())
    over_wr = bets.loc[bets["side"] == "OVER", "bet_won"].mean() if n_over > 0 else 0.0
    under_wr = bets.loc[bets["side"] == "UNDER", "bet_won"].mean() if n_under > 0 else 0.0
    brier = float(np.mean((bets["predicted_prob"] - bets["actual_win"]) ** 2))

    return {
        "threshold": conf_threshold,
        "bets": n,
        "wins": wins,
        "win_rate": win_rate,
        "roi_110": roi_110,
        "roi_50c_proxy": roi_50c_proxy,
        "n_over": n_over,
        "n_under": n_under,
        "over_wr": over_wr,
        "under_wr": under_wr,
        "brier": brier,
        "label": label,
    }


def evaluate_by_month(df, conf_threshold):
    """Break down results by month."""
    grouped_df = df.copy()
    if "evaluation_month" in grouped_df.columns:
        month_col = "evaluation_month"
    elif "date_str" in grouped_df.columns:
        grouped_df["_month"] = grouped_df["date_str"].astype(str).str[:7]
        month_col = "_month"
    elif "date" in grouped_df.columns:
        grouped_df["_month"] = grouped_df["date"].astype(str).str[:7]
        month_col = "_month"
    else:
        return []

    results = []
    for month, mdf in sorted(grouped_df.groupby(month_col)):
        result = evaluate_threshold(mdf, conf_threshold, label=str(month))
        if result:
            results.append(result)
    return results


def resolve_selection_years(loaded_years, selection_years_arg):
    loaded_years = sorted(loaded_years)
    if not loaded_years:
        return []
    if selection_years_arg == "auto":
        if len(loaded_years) == 1:
            return loaded_years
        return loaded_years[:-1]

    requested = parse_int_list(selection_years_arg)
    return [year for year in requested if year in loaded_years]


def combined_line_counts(stats_by_year):
    counts = Counter()
    for stats in stats_by_year.values():
        counts.update(stats["line_counts"])
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--years",
        default="2022,2023,2024,2025",
        help="Comma-separated years to backtest",
    )
    parser.add_argument(
        "--thresholds",
        default="53,55,58,60,65",
        help="Comma-separated confidence thresholds to test",
    )
    parser.add_argument(
        "--line-mode",
        default="half_only",
        choices=["half_only", "nearest_half"],
        help="How to convert sportsbook closes into a Kalshi-style proxy strike",
    )
    parser.add_argument(
        "--selection-years",
        default="auto",
        help="Years used to freeze the threshold. Default: all loaded years except the latest.",
    )
    parser.add_argument(
        "--min-bets-for-selection",
        type=int,
        default=200,
        help="Minimum number of bets required when selecting the frozen threshold",
    )
    args = parser.parse_args()

    years = parse_int_list(args.years)
    thresholds = parse_int_list(args.thresholds)
    high_tail_cfg, low_tail_cfg = load_tail_cfgs()

    all_data = {}
    stats_by_year = {}
    for year in years:
        df, stats = load_year(year, args.line_mode, high_tail_cfg, low_tail_cfg)
        if df is None or stats is None:
            print(f"{year}: no data found")
            continue

        stats_by_year[year] = stats
        all_data[year] = df
        print(
            f"Loaded {year}: {stats['eligible_rows']}/{stats['total_rows']} proxy-eligible rows "
            f"(dropped {stats['dropped_rows']})"
        )

    if not all_data:
        print("No data loaded.")
        return

    combined = pd.concat(all_data.values(), ignore_index=True)
    loaded_years = sorted(all_data.keys())
    selection_years = resolve_selection_years(loaded_years, args.selection_years)
    if not selection_years:
        selection_years = loaded_years
    selection_df = pd.concat([all_data[year] for year in selection_years], ignore_index=True)

    counts = combined_line_counts(stats_by_year)
    total_rows = sum(stats["total_rows"] for stats in stats_by_year.values())
    retained_rows = len(combined)

    print("\n" + "=" * 75)
    print("  RAW PROBABILITY PROXY-STRIKE BACKTEST")
    print("=" * 75)
    print(f"Proxy strike mode: {args.line_mode}")
    if args.line_mode == "half_only":
        print("Only sportsbook closes already ending in .5 are kept as Kalshi-style proxy strikes.")
    else:
        print("Non-.5 sportsbook closes are remapped to floor(close) + 0.5 as an experimental proxy.")
    print(
        f"Retained {retained_rows}/{total_rows} rows | "
        f"half={counts.get('half', 0)} whole={counts.get('whole', 0)} "
        f"quarter={counts.get('quarter', 0)} other={counts.get('other', 0)}"
    )
    print(
        f"Frozen threshold selection years: {', '.join(str(year) for year in selection_years)} "
        f"(min bets {args.min_bets_for_selection})"
    )
    print("ROI@50c is an illustrative even-money proxy, not real historical Kalshi PnL.")

    print(
        f"\n{'Conf%':>6} {'Bets':>6} {'Win%':>7} {'ROI@-110':>9} {'ROI@50c':>9} "
        f"{'OVER':>5} {'OVR W%':>7} {'UNDER':>6} {'UND W%':>7} {'Brier':>7}"
    )
    print("-" * 75)

    for thresh in thresholds:
        result = evaluate_threshold(combined, thresh, label=f"all_{thresh}")
        if result:
            print(
                f"{result['threshold']:>5}% {result['bets']:>6} {result['win_rate']:>6.1%} "
                f"{result['roi_110']:>+8.1f}% {result['roi_50c_proxy']:>+8.1f}% "
                f"{result['n_over']:>5} {result['over_wr']:>6.1%} "
                f"{result['n_under']:>6} {result['under_wr']:>6.1%} "
                f"{result['brier']:>6.4f}"
            )
        else:
            print(f"{thresh:>5}%   (no bets)")

    best_thresh = thresholds[0]
    best_roi = -999.0
    for thresh in thresholds:
        result = evaluate_threshold(selection_df, thresh)
        if result and result["bets"] >= args.min_bets_for_selection and result["roi_110"] > best_roi:
            best_roi = result["roi_110"]
            best_thresh = thresh

    print(f"\n{'=' * 75}")
    print(f"  PER-YEAR BREAKDOWN @ {best_thresh}% confidence")
    print(f"{'=' * 75}")

    print(
        f"\n{'Year':>6} {'Bets':>6} {'Win%':>7} {'ROI@-110':>9} {'ROI@50c':>9} "
        f"{'OVER':>5} {'OVR W%':>7} {'UNDER':>6} {'UND W%':>7}"
    )
    print("-" * 70)

    for year in loaded_years:
        result = evaluate_threshold(all_data[year], best_thresh, label=str(year))
        if result:
            print(
                f"{year:>6} {result['bets']:>6} {result['win_rate']:>6.1%} "
                f"{result['roi_110']:>+8.1f}% {result['roi_50c_proxy']:>+8.1f}% "
                f"{result['n_over']:>5} {result['over_wr']:>6.1%} "
                f"{result['n_under']:>6} {result['under_wr']:>6.1%}"
            )
        else:
            print(f"{year:>6}   (no bets)")

    total_result = evaluate_threshold(combined, best_thresh)
    if total_result:
        print("-" * 70)
        print(
            f"{'TOTAL':>6} {total_result['bets']:>6} {total_result['win_rate']:>6.1%} "
            f"{total_result['roi_110']:>+8.1f}% {total_result['roi_50c_proxy']:>+8.1f}% "
            f"{total_result['n_over']:>5} {total_result['over_wr']:>6.1%} "
            f"{total_result['n_under']:>6} {total_result['under_wr']:>6.1%}"
        )

    print(f"\n{'=' * 75}")
    print(f"  MONTHLY DETAIL @ {best_thresh}% confidence")
    print(f"{'=' * 75}")

    print(
        f"\n{'Month':>8} {'Bets':>5} {'Win%':>7} {'ROI@-110':>9} "
        f"{'OVER':>5} {'UNDER':>6} {'OVR W%':>7} {'UND W%':>7}"
    )
    print("-" * 60)

    for month_result in evaluate_by_month(combined, best_thresh):
        print(
            f"{month_result['label']:>8} {month_result['bets']:>5} {month_result['win_rate']:>6.1%} "
            f"{month_result['roi_110']:>+8.1f}% "
            f"{month_result['n_over']:>5} {month_result['n_under']:>6} "
            f"{month_result['over_wr']:>6.1%} {month_result['under_wr']:>6.1%}"
        )

    print(f"\n{'=' * 75}")
    print(f"  COMPARISON: Raw Probability vs Edge Model @ {best_thresh}%")
    print(f"{'=' * 75}")

    if args.line_mode != "half_only":
        print("Skipped edge-model comparison because remapped proxy strikes change the graded line.")
        return

    for year in loaded_years:
        df = all_data[year]
        raw_result = evaluate_threshold(df, best_thresh)

        if "p_over" in df.columns and "bet" in df.columns:
            edge_bets = df[df["bet"].isin(["OVER", "UNDER"])].copy()
            if len(edge_bets) > 0:
                edge_wins = int(edge_bets["bet_won"].sum())
                edge_n = len(edge_bets)
                edge_wr = edge_wins / edge_n
                edge_roi = (edge_wins * (100 / 110) - (edge_n - edge_wins)) / edge_n * 100
            else:
                edge_n, edge_wr, edge_roi = 0, 0.0, 0.0
        else:
            edge_n, edge_wr, edge_roi = 0, 0.0, 0.0

        raw_bets = raw_result["bets"] if raw_result else 0
        raw_wr = raw_result["win_rate"] if raw_result else 0.0
        raw_roi = raw_result["roi_110"] if raw_result else 0.0

        print(f"\n  {year}:")
        print(f"    {'':>20} {'Bets':>6} {'Win%':>7} {'ROI@-110':>9}")
        print(f"    {'Raw probability':>20} {raw_bets:>6} {raw_wr:>6.1%} {raw_roi:>+8.1f}%")
        print(f"    {'Edge model':>20} {edge_n:>6} {edge_wr:>6.1%} {edge_roi:>+8.1f}%")


if __name__ == "__main__":
    main()
