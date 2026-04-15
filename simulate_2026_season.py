"""
Simulate the MLB totals bot's 2026 Kalshi decisions on historical 10 AM PT lines.

For each completed 2026 game with a historical Kalshi strike ladder:
  1. Load the same feature rows used by the live model
  2. Run the shared runtime prediction stack
  3. Derive the Kalshi side / fair price / Kelly sizing using the live helper
  4. Apply the same Kalshi line-diff display filter as the live bot
  5. Settle the hypothetical bet against the actual game result

Usage:
    python simulate_2026_season.py
    python simulate_2026_season.py --season 2026
    python simulate_2026_season.py --bankroll 10000

Output:
    data/season_sim_2026.tsv
    data/season_sim_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

from model_runtime import (
    estimate_prediction_std,
    load_model_bundle,
    predict_high_tail_prob,
    predict_low_tail_prob,
    predict_point_outputs,
)
from prediction_betting import add_kalshi_metrics
from prediction_betting import get_kalshi_betting_thresholds, kalshi_filter_reason


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"


def load_config() -> dict:
    if _YAML_AVAILABLE:
        config_path = PROJECT_DIR / "model_config.yaml"
        if config_path.exists():
            with config_path.open(encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    return {}


def load_features_2026(season: int = 2026) -> pd.DataFrame:
    path = DATA_DIR / "mlb_model_data.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    df = pd.read_csv(path, sep="\t", parse_dates=["date"])
    return df[df["date"].dt.year == season].sort_values(["date", "game_id"]).reset_index(drop=True)


def load_kalshi_historical() -> dict:
    """
    Returns {(date_str, away_team, home_team): {strike: yes_price}}
    for strikes that had a valid 10 AM PT trade.
    """
    path = DATA_DIR / "kalshi_historical_lines.tsv"
    if not path.exists():
        return {}

    df = pd.read_csv(path, sep="\t")
    if "api_failed" not in df.columns:
        df["api_failed"] = False
    df = df[df["has_10am_price"] == True].dropna(subset=["yes_price"])

    result = {}
    for row in df.itertuples(index=False):
        key = (str(row.date), row.away_team, row.home_team)
        result.setdefault(key, {})[float(row.strike)] = float(row.yes_price)
    return result


def find_consensus(strike_prices: dict[float, float]) -> tuple[float, float] | None:
    """
    Pick the tradable strike whose price is closest to 50 cents, while
    preserving the exact strike/price pair that would have been buyable.
    """
    if not strike_prices:
        return None

    strikes = sorted(strike_prices)
    for lo, hi in zip(strikes, strikes[1:]):
        p_lo = strike_prices[lo]
        p_hi = strike_prices[hi]
        if p_lo >= 0.50 >= p_hi:
            anchor = lo if abs(p_lo - 0.50) <= abs(p_hi - 0.50) else hi
            return float(anchor), float(strike_prices[anchor])

    strike, yes_price = min(strike_prices.items(), key=lambda kv: abs(kv[1] - 0.50))
    return float(strike), float(yes_price)


def predict_all(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        feature_row = pd.DataFrame([row])
        _, point_predictions, _ = predict_point_outputs(bundle["model"], bundle["meta"], feature_row)
        predicted_total = float(point_predictions[0])
        sigma = estimate_prediction_std(
            feature_row,
            predicted_total,
            bundle["uncertainty_model"],
            bundle["uncertainty_cfg"],
            fallback_std=4.0,
        )
        high_tail = predict_high_tail_prob(
            feature_row,
            predicted_total,
            bundle["high_tail_model"],
            bundle["high_tail_cfg"],
        )
        low_tail = predict_low_tail_prob(
            feature_row,
            predicted_total,
            bundle["low_tail_model"],
            bundle["low_tail_cfg"],
        )
        rows.append(
            {
                "date": row["date"],
                "game_id": row.get("game_id"),
                "away_team": row["away_team"],
                "home_team": row["home_team"],
                "predicted_total": predicted_total,
                "prediction_std": sigma,
                "high_tail_prob_9p5": (high_tail * 100.0) if high_tail is not None else None,
                "low_tail_prob_7p5": (low_tail * 100.0) if low_tail is not None else None,
                "total_runs": row["total_runs"],
            }
        )
    return pd.DataFrame(rows)


def _no_bet_row(date_str, away, home, predicted, actual, reason, **extras):
    return {
        "date": date_str,
        "game_id": extras.get("game_id"),
        "away_team": away,
        "home_team": home,
        "predicted_total": round(float(predicted), 2),
        "prediction_std": extras.get("prediction_std"),
        "kalshi_line": extras.get("kalshi_line"),
        "kalshi_yes_price": extras.get("kalshi_yes_price"),
        "kalshi_side": None,
        "kalshi_side_market_prob": extras.get("kalshi_side_market_prob"),
        "kalshi_fair_price_pct": extras.get("kalshi_fair_price_pct"),
        "kalshi_edge_pct": extras.get("kalshi_edge_pct"),
        "bet_amount": 0.0,
        "bet_contracts": 0.0,
        "risked_amount": 0.0,
        "bet_pct_bankroll": 0.0,
        "actual_total": actual,
        "won": None,
        "result": "no_bet",
        "pnl_dollars": 0.0,
        "roi_pct": 0.0,
        "bankroll_before": extras.get("bankroll_before"),
        "bankroll_after": extras.get("bankroll_after"),
        "skip_reason": reason,
        "settled": False,
    }


def simulate(
    preds_df: pd.DataFrame,
    kalshi: dict,
    bankroll_cfg: dict,
    display_cfg: dict,
    min_kalshi_edge_pct: float = 0.0,
    min_kalshi_confidence_pct: float = 0.0,
) -> pd.DataFrame:
    rows = []
    running_bankroll = float(bankroll_cfg.get("total", 10000))
    max_line_diff = float(display_cfg.get("kalshi_max_line_diff", 2.5))

    for _, game in preds_df.sort_values(["date", "game_id"]).iterrows():
        date_str = str(pd.Timestamp(game["date"]).date())
        away = game["away_team"]
        home = game["home_team"]
        actual = float(game["total_runs"])
        bankroll_before = round(running_bankroll, 2)
        key = (date_str, away, home)
        strike_prices = kalshi.get(key)

        if not strike_prices:
            rows.append(
                _no_bet_row(
                    date_str,
                    away,
                    home,
                    predicted=game["predicted_total"],
                    actual=actual,
                    reason="no_kalshi",
                    game_id=game.get("game_id"),
                    prediction_std=round(float(game["prediction_std"]), 3),
                    bankroll_before=bankroll_before,
                    bankroll_after=round(running_bankroll, 2),
                )
            )
            continue

        consensus = find_consensus(strike_prices)
        if consensus is None:
            rows.append(
                _no_bet_row(
                    date_str,
                    away,
                    home,
                    predicted=game["predicted_total"],
                    actual=actual,
                    reason="no_consensus",
                    game_id=game.get("game_id"),
                    prediction_std=round(float(game["prediction_std"]), 3),
                    bankroll_before=bankroll_before,
                    bankroll_after=round(running_bankroll, 2),
                )
            )
            continue

        kalshi_line, yes_price = consensus
        pred = {
            "game_id": game.get("game_id"),
            "away_team": away,
            "home_team": home,
            "predicted_total": float(game["predicted_total"]),
            "prediction_std": float(game["prediction_std"]),
            "high_tail_prob_9p5": float(game["high_tail_prob_9p5"]) if pd.notna(game["high_tail_prob_9p5"]) else None,
            "low_tail_prob_7p5": float(game["low_tail_prob_7p5"]) if pd.notna(game["low_tail_prob_7p5"]) else None,
            "kalshi_line": float(kalshi_line),
            "kalshi_over_pct": float(yes_price) * 100.0,
            "kalshi_yes_ask": float(yes_price),
            "_bankroll_cfg": bankroll_cfg,
            "_high_tail_cfg": None,
            "_low_tail_cfg": None,
        }
        add_kalshi_metrics(pred, residual_std=float(game["prediction_std"]))
        reason = kalshi_filter_reason(
            pred,
            max_line_diff=max_line_diff,
            min_edge_pct=min_kalshi_edge_pct,
            min_confidence_pct=min_kalshi_confidence_pct,
        )
        if reason is not None:
            rows.append(
                _no_bet_row(
                    date_str,
                    away,
                    home,
                    predicted=game["predicted_total"],
                    actual=actual,
                    reason=reason,
                    game_id=game.get("game_id"),
                    prediction_std=round(float(game["prediction_std"]), 3),
                    kalshi_line=pred["kalshi_line"],
                    kalshi_yes_price=yes_price,
                    kalshi_side_market_prob=pred.get("kalshi_side_market_prob"),
                    kalshi_fair_price_pct=pred.get("kalshi_fair_price_pct"),
                    kalshi_edge_pct=pred.get("kalshi_edge_pct"),
                    bankroll_before=bankroll_before,
                    bankroll_after=round(running_bankroll, 2),
                )
            )
            continue

        kelly = pred.get("kalshi_kelly") or {}
        bet_amount = float(kelly.get("recommended_bet", 0.0) or 0.0)
        bet_pct_bankroll = (bet_amount / bankroll_before * 100.0) if bankroll_before > 0 else 0.0
        side = pred.get("kalshi_side")
        market_price = float(pred.get("kalshi_side_market_prob", 0.0) or 0.0) / 100.0
        contracts = (bet_amount / market_price) if bet_amount > 0 and market_price > 0 else 0.0

        won = None
        pnl = 0.0
        roi_pct = 0.0
        settled = False
        result = "no_bet" if bet_amount <= 0 else "pending"

        if bet_amount > 0:
            actual_over = actual > float(pred["kalshi_line"])
            won = (side == "OVER" and actual_over) or (side == "UNDER" and not actual_over)
            pnl = bet_amount * ((1.0 - market_price) / market_price) if won else -bet_amount
            running_bankroll += pnl
            roi_pct = (pnl / bet_amount * 100.0) if bet_amount > 0 else 0.0
            settled = True
            result = "win" if won else "loss"

        rows.append(
            {
                "date": date_str,
                "game_id": game.get("game_id"),
                "away_team": away,
                "home_team": home,
                "predicted_total": round(float(game["predicted_total"]), 2),
                "prediction_std": round(float(game["prediction_std"]), 3),
                "kalshi_line": float(pred["kalshi_line"]),
                "kalshi_yes_price": float(yes_price),
                "kalshi_side": side,
                "kalshi_side_market_prob": float(pred.get("kalshi_side_market_prob", 0.0) or 0.0),
                "kalshi_fair_price_pct": float(pred.get("kalshi_fair_price_pct", 0.0) or 0.0),
                "kalshi_edge_pct": float(pred.get("kalshi_edge_pct", 0.0) or 0.0),
                "bet_amount": round(bet_amount, 2),
                "bet_contracts": round(contracts, 4),
                "risked_amount": round(bet_amount, 2),
                "bet_pct_bankroll": round(bet_pct_bankroll, 3),
                "actual_total": actual,
                "won": won,
                "result": result,
                "pnl_dollars": round(pnl, 2),
                "roi_pct": round(roi_pct, 2),
                "bankroll_before": bankroll_before,
                "bankroll_after": round(running_bankroll, 2),
                "skip_reason": None if bet_amount > 0 else "kelly_below_min_bet",
                "settled": settled,
            }
        )

    return pd.DataFrame(rows)


def build_summary(sim: pd.DataFrame, starting_bankroll: float) -> dict:
    bets = sim[sim["bet_amount"] > 0].copy()
    settled = bets[bets["settled"] == True].copy()
    wins = int((settled["result"] == "win").sum())
    losses = int((settled["result"] == "loss").sum())
    total = wins + losses
    total_wagered = float(settled["risked_amount"].sum())
    total_pnl = float(settled["pnl_dollars"].sum())
    roi_pct = (total_pnl / total_wagered * 100.0) if total_wagered > 0 else 0.0
    final_bankroll = (
        float(sim["bankroll_after"].dropna().iloc[-1])
        if not sim["bankroll_after"].dropna().empty
        else starting_bankroll
    )
    avg_edge = float(bets["kalshi_edge_pct"].dropna().mean()) if not bets.empty else 0.0

    return {
        "starting_bankroll": round(float(starting_bankroll), 2),
        "final_bankroll": round(final_bankroll, 2),
        "bankroll_return_pct": round(((final_bankroll - starting_bankroll) / starting_bankroll * 100.0), 2),
        "total_games": int(len(sim)),
        "games_with_kalshi": int(sim["kalshi_line"].notna().sum()),
        "bets_placed": int(len(bets)),
        "wins": wins,
        "losses": losses,
        "win_rate": round((wins / total * 100.0), 1) if total > 0 else 0.0,
        "total_wagered": round(total_wagered, 2),
        "total_pnl": round(total_pnl, 2),
        "roi_pct": round(roi_pct, 2),
        "avg_edge_pct": round(avg_edge, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Simulate 2026 Kalshi betting using historical 10 AM PT lines.")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--bankroll", type=float, default=None)
    parser.add_argument("--kelly-fraction", type=float, default=None)
    parser.add_argument("--max-bet-pct", type=float, default=None)
    parser.add_argument("--min-bet", type=float, default=None)
    parser.add_argument("--round-to", type=float, default=None)
    parser.add_argument("--min-kalshi-edge-pct", type=float, default=None)
    parser.add_argument("--min-kalshi-confidence-pct", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config()
    bankroll_cfg = dict(cfg.get("bankroll", {}) or {})
    bet_cfg = cfg.get("betting", {}) or {}
    kalshi_thresholds = get_kalshi_betting_thresholds(bet_cfg)
    if args.bankroll is not None:
        bankroll_cfg["total"] = float(args.bankroll)
    if args.kelly_fraction is not None:
        bankroll_cfg["kelly_fraction"] = float(args.kelly_fraction)
    if args.max_bet_pct is not None:
        bankroll_cfg["max_bet_pct"] = float(args.max_bet_pct)
    if args.min_bet is not None:
        bankroll_cfg["min_bet"] = float(args.min_bet)
    if args.round_to is not None:
        bankroll_cfg["round_to"] = float(args.round_to)
    display_cfg = cfg.get("display", {}) or {}
    min_kalshi_edge_pct = (
        float(args.min_kalshi_edge_pct)
        if args.min_kalshi_edge_pct is not None
        else float(kalshi_thresholds["min_kalshi_edge_pct"])
    )
    min_kalshi_confidence_pct = (
        float(args.min_kalshi_confidence_pct)
        if args.min_kalshi_confidence_pct is not None
        else float(kalshi_thresholds["min_kalshi_confidence_pct"])
    )

    print(f"=== {args.season} Kalshi Historical Simulation ===")
    print("Loading model bundle...")
    bundle = load_model_bundle()

    print("Loading feature rows...")
    feat_df = load_features_2026(args.season)
    print(f"  {len(feat_df)} rows")

    print("Running predictions...")
    preds_df = predict_all(feat_df, bundle)
    print(f"  Predicted {len(preds_df)} games (mean total {preds_df['predicted_total'].mean():.2f})")

    print("Loading historical Kalshi lines...")
    kalshi = load_kalshi_historical()
    if not kalshi:
        print("No historical Kalshi lines found. Run collect_kalshi_historical.py first.")
        return
    matched = sum(
        1
        for _, row in preds_df.iterrows()
        if (str(pd.Timestamp(row['date']).date()), row["away_team"], row["home_team"]) in kalshi
    )
    print(f"  Matched {matched}/{len(preds_df)} games to historical Kalshi ladders")

    print("Simulating...")
    sim = simulate(
        preds_df,
        kalshi,
        bankroll_cfg,
        display_cfg,
        min_kalshi_edge_pct=min_kalshi_edge_pct,
        min_kalshi_confidence_pct=min_kalshi_confidence_pct,
    )
    summary = build_summary(sim, float(bankroll_cfg.get("total", 10000)))
    summary.update(
        {
            "kelly_fraction": round(float(bankroll_cfg.get("kelly_fraction", 0.25)), 4),
            "max_bet_pct": round(float(bankroll_cfg.get("max_bet_pct", 0.0)), 4),
            "min_bet": round(float(bankroll_cfg.get("min_bet", 1.0)), 4),
            "min_kalshi_edge_pct": round(float(min_kalshi_edge_pct), 4),
            "min_kalshi_confidence_pct": round(float(min_kalshi_confidence_pct), 4),
        }
    )

    out_sim = DATA_DIR / f"season_sim_{args.season}.tsv"
    out_sum = DATA_DIR / "season_sim_summary.json"
    sim.to_csv(out_sim, sep="\t", index=False)
    with out_sum.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Results ===")
    print(f"Games:         {summary['total_games']}")
    print(f"Kalshi games:  {summary['games_with_kalshi']}")
    print(f"Bets placed:   {summary['bets_placed']} ({summary['wins']}W / {summary['losses']}L)")
    print(f"Win rate:      {summary['win_rate']:.1f}%")
    print(f"Total wagered: ${summary['total_wagered']:,.2f}")
    print(f"Total P&L:     ${summary['total_pnl']:+,.2f}")
    print(f"ROI:           {summary['roi_pct']:+.2f}%")
    print(f"Final roll:    ${summary['final_bankroll']:,.2f} ({summary['bankroll_return_pct']:+.2f}%)")
    print("\nSaved:")
    print(f"  {out_sim.relative_to(PROJECT_DIR)}")
    print(f"  {out_sum.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
