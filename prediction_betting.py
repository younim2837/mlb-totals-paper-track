"""
Betting and market-side helpers for daily prediction output.
"""

from __future__ import annotations

from modeling_utils import (
    margin_distribution,
    probability_home_covering_spread,
    probability_home_win,
    probability_over_line,
)
from market_adjustment import apply_market_context


STANDARD_TOTAL_LINES = [6.5, 7.5, 8.5, 9.5, 10.5, 11.5]
DEFAULT_HOME_RUN_LINE = -1.5


def american_to_decimal(american: float) -> float:
    american = float(american)
    if american > 0:
        return 1 + american / 100.0
    return 1 + 100.0 / abs(american)


def kelly_size(
    p_win: float,
    american_odds: float,
    bankroll: float,
    kelly_fraction: float = 0.25,
    max_bet_pct: float = 5.0,
    min_bet: float = 1.0,
    round_to: float = 5.0,
):
    p_win = float(p_win)
    b = american_to_decimal(american_odds) - 1.0
    q = 1.0 - p_win
    full_kelly = max(0.0, (b * p_win - q) / b) if b > 0 else 0.0
    frac_kelly = full_kelly * float(kelly_fraction)
    cap_fraction = None if max_bet_pct is None or float(max_bet_pct) <= 0 else float(max_bet_pct) / 100.0
    capped = frac_kelly if cap_fraction is None else min(frac_kelly, cap_fraction)
    raw_bet = float(bankroll) * capped
    if raw_bet < float(min_bet):
        recommended = 0.0
    else:
        recommended = max(float(round_to), round(raw_bet / float(round_to)) * float(round_to))

    return {
        "full_kelly_pct": round(full_kelly * 100, 1),
        "frac_kelly_pct": round(frac_kelly * 100, 1),
        "capped_kelly_pct": round(capped * 100, 1),
        "recommended_bet": float(recommended),
        "raw_bet": float(raw_bet),
        "was_capped": cap_fraction is not None and frac_kelly > cap_fraction,
        "edge_pct": round((p_win - 1.0 / american_to_decimal(american_odds)) * 100, 1),
    }


def kalshi_kelly_size(
    p_win: float,
    market_price: float,
    bankroll: float,
    kelly_fraction: float = 0.25,
    max_bet_pct: float = 5.0,
    min_bet: float = 1.0,
    round_to: float = 1.0,
):
    p_win = float(p_win)
    market_price = float(market_price)
    if market_price <= 0.0 or market_price >= 1.0:
        return {
            "full_kelly_pct": 0.0,
            "frac_kelly_pct": 0.0,
            "capped_kelly_pct": 0.0,
            "recommended_bet": 0.0,
            "raw_bet": 0.0,
            "was_capped": False,
            "edge_pct": 0.0,
        }

    b = (1.0 - market_price) / market_price
    q = 1.0 - p_win
    full_kelly = max(0.0, (b * p_win - q) / b) if b > 0 else 0.0
    frac_kelly = full_kelly * float(kelly_fraction)
    cap_fraction = None if max_bet_pct is None or float(max_bet_pct) <= 0 else float(max_bet_pct) / 100.0
    capped = frac_kelly if cap_fraction is None else min(frac_kelly, cap_fraction)
    raw_bet = float(bankroll) * capped
    if raw_bet < float(min_bet):
        recommended = 0.0
    else:
        recommended = max(float(round_to), round(raw_bet / float(round_to)) * float(round_to))

    return {
        "full_kelly_pct": round(full_kelly * 100, 1),
        "frac_kelly_pct": round(frac_kelly * 100, 1),
        "capped_kelly_pct": round(capped * 100, 1),
        "recommended_bet": float(recommended),
        "raw_bet": float(raw_bet),
        "bankroll_used": float(bankroll),
        "recommended_bet_pct_bankroll": (float(recommended) / float(bankroll) * 100.0) if float(bankroll) > 0 else 0.0,
        "raw_bet_pct_bankroll": (float(raw_bet) / float(bankroll) * 100.0) if float(bankroll) > 0 else 0.0,
        "was_capped": cap_fraction is not None and frac_kelly > cap_fraction,
        "edge_pct": round((p_win - market_price) * 100, 1),
    }


def get_kalshi_betting_thresholds(bet_cfg: dict | None) -> dict:
    cfg = bet_cfg or {}
    return {
        "min_kalshi_edge_pct": float(cfg.get("min_kalshi_edge_pct", 0.0) or 0.0),
        "min_kalshi_confidence_pct": float(cfg.get("min_kalshi_confidence_pct", 0.0) or 0.0),
    }


def add_kalshi_metrics(pred: dict, residual_std: float) -> dict:
    if "kalshi_line" not in pred:
        return pred

    line = float(pred["kalshi_line"])
    market_p = float(pred.get("kalshi_over_pct", 0.0)) / 100.0
    sigma = float(pred.get("prediction_std", residual_std))

    # Use the pre-edge-model prediction for Kalshi probability.
    # The edge model adjusts predicted_total using the market line as a reference,
    # then computing P(over/under that same line) from the adjusted total creates
    # a circular dependency that amplifies disagreement into inflated edges.
    # market_total_before_shrink is set by apply_market_adjustment_to_prediction;
    # fall back to predicted_total in the simulation path (no edge model runs there).
    raw_total = float(pred.get("market_total_before_shrink") or pred["predicted_total"])

    model_p = probability_over_line(
        raw_total,
        sigma,
        line,
        high_tail_prob=(float(pred["high_tail_prob_9p5"]) / 100.0) if pred.get("high_tail_prob_9p5") is not None else None,
        high_tail_cfg=pred.get("_high_tail_cfg"),
        low_tail_prob=(float(pred["low_tail_prob_7p5"]) / 100.0) if pred.get("low_tail_prob_7p5") is not None else None,
        low_tail_cfg=pred.get("_low_tail_cfg"),
    )
    fair_price = round(float(model_p) * 100, 1)
    edge = round((float(model_p) - market_p) * 100, 1)
    side = "OVER" if edge >= 0 else "UNDER"
    side_p = float(model_p) if side == "OVER" else (1 - float(model_p))
    market_side_p = market_p if side == "OVER" else (1 - market_p)
    market_side_price = market_p if side == "OVER" else (1 - market_p)

    pred["kalshi_model_over_pct"] = round(float(model_p) * 100, 1)
    pred["kalshi_fair_price_pct"] = fair_price
    pred["kalshi_edge_pct"] = edge
    pred["kalshi_side"] = side
    pred["kalshi_side_model_prob"] = round(side_p * 100, 1)
    pred["kalshi_side_market_prob"] = round(market_side_p * 100, 1)

    bankroll_cfg = pred.get("_bankroll_cfg")
    if bankroll_cfg and bankroll_cfg.get("total"):
        pred["kalshi_kelly"] = kalshi_kelly_size(
            p_win=side_p,
            market_price=market_side_price,
            bankroll=float(bankroll_cfg["total"]),
            kelly_fraction=float(bankroll_cfg.get("kelly_fraction", 0.25)),
            max_bet_pct=float(bankroll_cfg.get("max_bet_pct", 5.0)),
            min_bet=float(bankroll_cfg.get("min_bet", 1)),
            round_to=float(bankroll_cfg.get("round_to", 5)),
        )
    else:
        pred["kalshi_kelly"] = None

    return pred


def kalshi_filter_reason(
    pred: dict,
    *,
    max_line_diff: float | None = None,
    min_edge_pct: float = 0.0,
    min_confidence_pct: float = 0.0,
) -> str | None:
    if pred.get("kalshi_line") is None:
        return "no_kalshi"

    if max_line_diff is not None:
        diff = abs(float(pred["kalshi_line"]) - float(pred["predicted_total"]))
        if diff > float(max_line_diff):
            return "line_diff_too_large"

    edge_pct = abs(float(pred.get("kalshi_edge_pct", 0.0) or 0.0))
    if edge_pct < float(min_edge_pct):
        return "edge_below_threshold"

    confidence_pct = float(pred.get("kalshi_side_model_prob", 0.0) or 0.0)
    if confidence_pct < float(min_confidence_pct):
        return "confidence_below_threshold"

    kelly = pred.get("kalshi_kelly") or {}
    if float(kelly.get("recommended_bet", 0.0) or 0.0) <= 0.0:
        return "kelly_below_min_bet"

    return None


def suppress_kalshi_bet(pred: dict, reason: str) -> dict:
    pred["kalshi_bet_block_reason"] = reason
    kelly = pred.get("kalshi_kelly")
    if kelly:
        muted = dict(kelly)
        muted["recommended_bet"] = 0.0
        pred["kalshi_kelly"] = muted
    return pred


def add_team_side_metrics(
    pred: dict,
    side_distribution: dict | None,
    home_run_line: float = DEFAULT_HOME_RUN_LINE,
) -> dict:
    home_runs = pred.get("predicted_home_runs")
    away_runs = pred.get("predicted_away_runs")
    total_sigma = pred.get("prediction_std")
    if home_runs is None or away_runs is None or total_sigma is None:
        return pred

    dist = margin_distribution(
        mean_home_runs=float(home_runs),
        mean_away_runs=float(away_runs),
        total_sigma=float(total_sigma),
        side_distribution=side_distribution,
    )
    home_win_prob = probability_home_win(
        mean_home_runs=float(home_runs),
        mean_away_runs=float(away_runs),
        total_sigma=float(total_sigma),
        side_distribution=side_distribution,
    )
    home_cover_prob = probability_home_covering_spread(
        mean_home_runs=float(home_runs),
        mean_away_runs=float(away_runs),
        total_sigma=float(total_sigma),
        home_spread_line=float(home_run_line),
        side_distribution=side_distribution,
    )

    pred["predicted_margin"] = round(float(dist["mean_margin"]), 2)
    pred["margin_sigma"] = round(float(dist["margin_sigma"]), 2)
    pred["side_sigma_home"] = round(float(dist["sigma_home"]), 2)
    pred["side_sigma_away"] = round(float(dist["sigma_away"]), 2)
    pred["side_rho"] = round(float(dist["rho"]), 3)
    pred["home_win_pct"] = round(float(home_win_prob) * 100, 1)
    pred["away_win_pct"] = round((1.0 - float(home_win_prob)) * 100, 1)
    pred["home_cover_minus_1p5_pct"] = round(float(home_cover_prob) * 100, 1)
    pred["away_cover_plus_1p5_pct"] = round((1.0 - float(home_cover_prob)) * 100, 1)
    return pred


def add_edge_to_prediction(pred, posted_line, residual_std, odds=-110):
    sigma = float(pred.get("prediction_std", residual_std))
    p_over = probability_over_line(
        pred["predicted_total"],
        sigma,
        posted_line,
        high_tail_prob=(float(pred["high_tail_prob_9p5"]) / 100.0) if pred.get("high_tail_prob_9p5") is not None else None,
        high_tail_cfg=pred.get("_high_tail_cfg"),
        low_tail_prob=(float(pred["low_tail_prob_7p5"]) / 100.0) if pred.get("low_tail_prob_7p5") is not None else None,
        low_tail_cfg=pred.get("_low_tail_cfg"),
    )
    p_under = 1 - p_over
    edge = pred["predicted_total"] - posted_line

    pred["posted_line"] = posted_line
    pred["posted_odds"] = odds
    pred["edge"] = round(edge, 2)
    pred["p_over_line"] = round(p_over * 100, 1)
    pred["p_under_line"] = round(p_under * 100, 1)
    pred["bet_signal"] = (
        "OVER" if p_over >= 0.55 else
        "UNDER" if p_under >= 0.55 else
        "NO EDGE"
    )
    pred["bet_confidence"] = round(max(p_over, p_under) * 100, 1)
    return pred


def normalize_allowed_market_adjustment_methods(bet_cfg: dict | None) -> set[str] | None:
    raw_methods = (bet_cfg or {}).get("allowed_market_adjustment_methods")
    if raw_methods is None:
        return None
    if isinstance(raw_methods, str):
        raw_methods = [raw_methods]
    if not isinstance(raw_methods, (list, tuple, set)):
        return None
    methods = {str(method).strip() for method in raw_methods if str(method).strip()}
    return methods or None


def apply_market_adjustment_to_prediction(
    pred: dict,
    residual_std: float,
    market_cfg: dict | None,
    learned_cfg: dict | None,
    market_edge_model=None,
    market_edge_cfg: dict | None = None,
) -> dict:
    if "posted_line" not in pred:
        return pred

    base_total = float(pred["predicted_total"])
    market_result = apply_market_context(
        predicted_total=base_total,
        market_line=float(pred["posted_line"]),
        cfg=market_cfg or {},
        prediction_std=float(pred.get("prediction_std", residual_std)),
        num_books=pred.get("market_num_books"),
        market_features={
            col: pred.get(col)
            for col in ["commence_time", "snapshot_ts", "pinnacle_line", "draftkings_line", "fanduel_line", "betmgm_line", "caesars_line"]
            if pred.get(col) is not None
        },
        high_tail_prob=(float(pred["high_tail_prob_9p5"]) / 100.0) if pred.get("high_tail_prob_9p5") is not None else None,
        high_tail_cfg=pred.get("_high_tail_cfg"),
        low_tail_prob=(float(pred["low_tail_prob_7p5"]) / 100.0) if pred.get("low_tail_prob_7p5") is not None else None,
        low_tail_cfg=pred.get("_low_tail_cfg"),
        market_model=market_edge_model,
        market_model_cfg=market_edge_cfg or {},
        learned_shrink_cfg=learned_cfg or {},
    )
    pred["market_total_before_shrink"] = round(base_total, 3)
    pred["market_shrink_fraction"] = round(float(market_result.get("shrink_fraction", 0.0)), 3)
    pred["market_shrink_bucket"] = market_result.get("bucket_key")
    pred["market_adjustment_method"] = market_result.get("method")
    adjusted_total = float(market_result.get("adjusted_total", base_total))
    pred["market_shrink_delta"] = round(adjusted_total - base_total, 3)
    pred["predicted_total"] = round(adjusted_total, 1)

    return add_edge_to_prediction(
        pred,
        posted_line=pred["posted_line"],
        residual_std=residual_std,
        odds=pred.get("posted_odds", -110),
    )


def apply_overrides(predictions: list, overrides: dict, residual_std: float) -> list:
    if not overrides:
        return predictions

    for pred in predictions:
        adj = 0.0
        for team_key, team_adj in overrides.items():
            if team_key in (pred["home_team"], pred["away_team"]):
                adj += float(team_adj.get("offense_adj", 0))
                adj -= float(team_adj.get("defense_adj", 0))

        if adj != 0.0:
            old = pred["predicted_total"]
            pred["predicted_total"] = round(old + adj, 1)
            pred["override_adj"] = round(adj, 1)
            sigma = float(pred.get("prediction_std", residual_std))

            for line in STANDARD_TOTAL_LINES:
                p_over = probability_over_line(
                    pred["predicted_total"],
                    sigma,
                    line,
                    high_tail_prob=(float(pred["high_tail_prob_9p5"]) / 100.0) if pred.get("high_tail_prob_9p5") is not None else None,
                    high_tail_cfg=pred.get("_high_tail_cfg"),
                    low_tail_prob=(float(pred["low_tail_prob_7p5"]) / 100.0) if pred.get("low_tail_prob_7p5") is not None else None,
                    low_tail_cfg=pred.get("_low_tail_cfg"),
                )
                pred[f"over_{line}"] = round(p_over * 100, 1)
                pred[f"under_{line}"] = round((1 - p_over) * 100, 1)

            if "posted_line" in pred:
                p_over = probability_over_line(
                    pred["predicted_total"],
                    sigma,
                    pred["posted_line"],
                    high_tail_prob=(float(pred["high_tail_prob_9p5"]) / 100.0) if pred.get("high_tail_prob_9p5") is not None else None,
                    high_tail_cfg=pred.get("_high_tail_cfg"),
                    low_tail_prob=(float(pred["low_tail_prob_7p5"]) / 100.0) if pred.get("low_tail_prob_7p5") is not None else None,
                    low_tail_cfg=pred.get("_low_tail_cfg"),
                )
                pred["p_over_line"] = round(p_over * 100, 1)
                pred["p_under_line"] = round((1 - p_over) * 100, 1)
                pred["edge"] = round(pred["predicted_total"] - pred["posted_line"], 2)

    return predictions
