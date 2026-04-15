"""
Sweep less-aggressive Kalshi replay rules over the 2026 historical season.

Usage:
    python sweep_kalshi_2026.py
    python sweep_kalshi_2026.py --season 2026

Outputs:
    data/season_sim_sweep_2026.tsv
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd

from simulate_2026_season import (
    build_summary,
    load_config,
    load_features_2026,
    load_kalshi_historical,
    predict_all,
    simulate,
)
from model_runtime import load_model_bundle


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"


def max_drawdown_pct(sim_df: pd.DataFrame, starting_bankroll: float) -> float:
    bankroll_series = sim_df["bankroll_after"].dropna()
    if bankroll_series.empty:
        return 0.0
    values = pd.concat([pd.Series([starting_bankroll]), bankroll_series.reset_index(drop=True)])
    peaks = values.cummax()
    drawdowns = (values - peaks) / peaks * 100.0
    return float(drawdowns.min())


def build_bankroll_cfg(base_cfg: dict, kelly_fraction: float, max_bet_pct: float) -> dict:
    cfg = dict(base_cfg)
    cfg["kelly_fraction"] = float(kelly_fraction)
    cfg["max_bet_pct"] = float(max_bet_pct)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Sweep Kalshi replay thresholds over the 2026 historical season.")
    parser.add_argument("--season", type=int, default=2026)
    args = parser.parse_args()

    cfg = load_config()
    bankroll_cfg = dict(cfg.get("bankroll", {}) or {})
    display_cfg = cfg.get("display", {}) or {}

    print("Loading model bundle...")
    bundle = load_model_bundle()
    print("Loading feature rows...")
    feat_df = load_features_2026(args.season)
    print("Running cached predictions...")
    preds_df = predict_all(feat_df, bundle)
    print("Loading historical Kalshi lines...")
    kalshi = load_kalshi_historical()

    edge_thresholds = [0.0, 2.0, 4.0, 6.0]
    confidence_thresholds = [0.0, 52.0, 55.0, 57.5, 60.0]
    kelly_fractions = [0.05, 0.10, 0.15, 0.25]
    max_bet_pcts = [2.0, 3.0, 5.0, 0.0]

    rows = []
    total_runs = len(edge_thresholds) * len(confidence_thresholds) * len(kelly_fractions) * len(max_bet_pcts)
    index = 0
    for edge_pct, conf_pct, kelly_fraction, max_bet_pct in product(
        edge_thresholds,
        confidence_thresholds,
        kelly_fractions,
        max_bet_pcts,
    ):
        index += 1
        current_bankroll_cfg = build_bankroll_cfg(bankroll_cfg, kelly_fraction, max_bet_pct)
        sim_df = simulate(
            preds_df,
            kalshi,
            current_bankroll_cfg,
            display_cfg,
            min_kalshi_edge_pct=edge_pct,
            min_kalshi_confidence_pct=conf_pct,
        )
        summary = build_summary(sim_df, float(current_bankroll_cfg.get("total", 10000)))
        bets = sim_df[sim_df["bet_amount"] > 0].copy()
        avg_bet = float(bets["bet_amount"].mean()) if not bets.empty else 0.0
        drawdown = max_drawdown_pct(sim_df, float(current_bankroll_cfg.get("total", 10000)))
        rows.append(
            {
                "min_kalshi_edge_pct": edge_pct,
                "min_kalshi_confidence_pct": conf_pct,
                "kelly_fraction": kelly_fraction,
                "max_bet_pct": max_bet_pct,
                "bets_placed": summary["bets_placed"],
                "wins": summary["wins"],
                "losses": summary["losses"],
                "win_rate_pct": summary["win_rate"],
                "roi_pct": summary["roi_pct"],
                "total_pnl": summary["total_pnl"],
                "avg_edge_pct": summary["avg_edge_pct"],
                "avg_bet_amount": round(avg_bet, 2),
                "final_bankroll": summary["final_bankroll"],
                "bankroll_return_pct": summary["bankroll_return_pct"],
                "max_drawdown_pct": round(drawdown, 2),
            }
        )
        print(
            f"[{index:>3}/{total_runs}] edge>={edge_pct:.1f} conf>={conf_pct:.1f} "
            f"kelly={kelly_fraction:.2f} cap={max_bet_pct:.1f} -> "
            f"{summary['bets_placed']} bets, ROI {summary['roi_pct']:+.2f}%"
        )

    sweep_df = pd.DataFrame(rows).sort_values(
        by=["roi_pct", "bets_placed", "bankroll_return_pct"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    out_path = DATA_DIR / f"season_sim_sweep_{args.season}.tsv"
    sweep_df.to_csv(out_path, sep="\t", index=False)

    print(f"\nSaved sweep: {out_path.relative_to(PROJECT_DIR)}")
    print("\nTop 12 rule sets:")
    print(
        sweep_df.head(12)[
            [
                "min_kalshi_edge_pct",
                "min_kalshi_confidence_pct",
                "kelly_fraction",
                "max_bet_pct",
                "bets_placed",
                "win_rate_pct",
                "roi_pct",
                "max_drawdown_pct",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
