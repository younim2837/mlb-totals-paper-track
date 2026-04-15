"""
Prediction report rendering and export helpers.
"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

from prediction_betting import kelly_size


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_DIR = os.path.join(BASE_DIR, "predictions")


def export_daily_prediction_reports(predictions: list[dict], target_date: str, include_all_games: bool) -> tuple[str, str]:
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    base_name = f"{target_date}-all-games" if include_all_games else target_date
    board_path = os.path.join(PREDICTIONS_DIR, f"{base_name}-board.tsv")
    picks_path = os.path.join(PREDICTIONS_DIR, f"{base_name}-picks.tsv")

    export_cols = [
        "target_date",
        "game_id",
        "commence_time",
        "away_team",
        "home_team",
        "predicted_total",
        "posted_line",
        "posted_odds",
        "edge",
        "p_over_line",
        "p_under_line",
        "bet_signal",
        "bet_confidence",
        "bet_block_reason",
        "market_line_source",
        "market_adjustment_method",
        "market_total_before_shrink",
        "market_shrink_delta",
        "market_shrink_fraction",
        "market_num_books",
        "prediction_std",
        "high_tail_prob_9p5",
        "low_tail_prob_7p5",
        "recommended_bet",
        "kalshi_model_over_pct",
        "kalshi_fair_price_pct",
        "kalshi_edge_pct",
        "kalshi_side",
        "kalshi_side_model_prob",
        "kalshi_side_market_prob",
        "kalshi_line",
        "kalshi_yes_ask",
        "kalshi_over_pct",
        "kalshi_bet_block_reason",
        "kalshi_bankroll_used",
        "kalshi_bet_pct_bankroll",
        "kalshi_raw_bet_pct_bankroll",
        "kalshi_recommended_bet",
        "kalshi_raw_bet",
    ]

    rows = []
    for pred in predictions:
        row = {col: pred.get(col) for col in export_cols}
        kelly = pred.get("kalshi_kelly") or pred.get("kelly") or {}
        kalshi_kelly = pred.get("kalshi_kelly") or {}
        row["target_date"] = target_date
        row["recommended_bet"] = kelly.get("recommended_bet", 0.0)
        row["kalshi_bankroll_used"] = kalshi_kelly.get("bankroll_used", 0.0)
        row["kalshi_bet_pct_bankroll"] = kalshi_kelly.get("recommended_bet_pct_bankroll", 0.0)
        row["kalshi_raw_bet_pct_bankroll"] = kalshi_kelly.get("raw_bet_pct_bankroll", 0.0)
        row["kalshi_recommended_bet"] = kalshi_kelly.get("recommended_bet", 0.0)
        row["kalshi_raw_bet"] = kalshi_kelly.get("raw_bet", 0.0)
        rows.append(row)

    board_df = pd.DataFrame(rows, columns=export_cols)
    board_df.to_csv(board_path, sep="\t", index=False)

    has_sportsbook_bet = board_df["bet_signal"].fillna("NO EDGE") != "NO EDGE"
    has_kalshi_bet = board_df["kalshi_recommended_bet"].fillna(0).astype(float) > 0
    picks_df = board_df[has_sportsbook_bet | has_kalshi_bet].copy()
    picks_df.to_csv(picks_path, sep="\t", index=False)
    return board_path, picks_path


def display_predictions(predictions, has_lines=False, line_source="The Odds API", cfg=None, max_bets=5):
    date_str = datetime.now().strftime("%A, %B %d, %Y")
    print("=" * 72)
    print(f"  MLB TOTAL RUNS PREDICTIONS - {date_str}")
    if has_lines:
        print(f"  (Lines from {line_source})")
    print("=" * 72)

    if not predictions:
        print("\n  No games found for today.\n")
        return

    disp_cfg = (cfg or {}).get("display", {})

    if has_lines:
        predictions = sorted(predictions, key=lambda p: abs(p.get("edge", 0)), reverse=True)

    bet_count = 0

    for i, p in enumerate(predictions):
        print(f"\n  {p['away_team']} @ {p['home_team']}")
        print(f"  Venue: {p['venue']}")
        print(f"  Pitchers: {p['away_pitcher']} (away) vs {p['home_pitcher']} (home)")

        if p.get("override_adj"):
            adj = p["override_adj"]
            print(f"  [OVERRIDE: {'+' if adj > 0 else ''}{adj} runs applied from config]")

        if p.get("dc_expected_total") is not None:
            print(f"  DC model: {p['dc_expected_total']:.1f} runs  "
                  f"(home lambda={p['dc_lambda_home']:.2f}  away lambda={p['dc_lambda_away']:.2f})")
        if p.get("prediction_std") is not None:
            print(f"  Model sigma: {p['prediction_std']:.2f} runs")
        if p.get("high_tail_prob_9p5") is not None:
            print(
                f"  High-tail risk (>9.5): {p['high_tail_prob_9p5']:.1f}%"
                f"  |  Tail sigma: {float(p.get('high_tail_sigma', p['prediction_std'])):.2f}"
            )
        if p.get("low_tail_prob_7p5") is not None:
            print(
                f"  Low-tail risk (<7.5): {p['low_tail_prob_7p5']:.1f}%"
                f"  |  Tail sigma: {float(p.get('low_tail_sigma', p['prediction_std'])):.2f}"
            )
        if p.get("predicted_away_runs") is not None and p.get("predicted_home_runs") is not None:
            print(f"  Team totals: away {p['predicted_away_runs']:.2f}  |  home {p['predicted_home_runs']:.2f}")
        if abs(float(p.get("calibration_adjustment", 0.0))) >= 0.05:
            print(f"  Calibration: {p['calibration_adjustment']:+.2f} runs")
        if abs(float(p.get("market_shrink_delta", 0.0))) >= 0.05:
            bucket = p.get("market_shrink_bucket")
            method = p.get("market_adjustment_method")
            bucket_text = f"  |  bucket {bucket}" if bucket else ""
            method_text = f"{method}: " if method and method != "shrinkage" else ""
            print(
                f"  Market adjust: {method_text}{p.get('market_total_before_shrink', p['predicted_total']):.2f} -> "
                f"{p['predicted_total']:.2f}  "
                f"({p.get('market_shrink_delta', 0.0):+.2f} runs, "
                f"alpha={p.get('market_shrink_fraction', 0.0):.2f}{bucket_text})"
            )
        if p.get("model_family") == "team_split" and p.get("dc_expected_total") is not None:
            print(
                f"  Why: DC baseline {p['dc_expected_total']:.1f} + "
                f"away residual {p.get('xgb_away_component', 0):+.2f} + "
                f"home residual {p.get('xgb_home_component', 0):+.2f}"
            )
        elif p.get("dc_expected_total") is not None:
            print(f"  Why: DC baseline {p['dc_expected_total']:.1f} + residual {p.get('xgb_residual', 0):+.2f}")
        elif p.get("xgb_residual") is not None:
            print(f"  Why: direct XGBoost total with residual signal {p['xgb_residual']:+.2f}")
        if p.get("xgb_top_buckets"):
            bucket_str = "  |  ".join(f"{label} {value:+.2f}" for label, value in p["xgb_top_buckets"])
            print(f"  Residual drivers: {bucket_str}")
        if p.get("xgb_top_features"):
            feature_str = ", ".join(f"{feat} {value:+.2f}" for feat, value in p["xgb_top_features"])
            print(f"  Top features: {feature_str}")

        if disp_cfg.get("show_elo_ratings", True) and p.get("home_dc_attack") is not None:
            print(f"  DC   ATK  DEF  |  {p['away_team'][:20]}: {p.get('away_dc_attack',0):+.3f} / {p.get('away_dc_defense',0):+.3f}"
                  f"  |  {p['home_team'][:20]}: {p.get('home_dc_attack',0):+.3f} / {p.get('home_dc_defense',0):+.3f}")
        elif disp_cfg.get("show_elo_ratings", True) and p.get("home_off_elo") is not None:
            print(f"  Elo  OFF  DEF  |  {p['away_team'][:20]}: {p.get('away_off_elo',0):.0f} / {p.get('away_def_elo',0):.0f}"
                  f"  |  {p['home_team'][:20]}: {p.get('home_off_elo',0):.0f} / {p.get('home_def_elo',0):.0f}")

        if (
            p.get("home_pitcher_avg_pitches_3g") is not None and
            p.get("away_pitcher_avg_pitches_3g") is not None and
            p.get("home_pitcher_days_rest") is not None and
            p.get("away_pitcher_days_rest") is not None
        ):
            away_rest = p.get("away_pitcher_days_rest")
            home_rest = p.get("home_pitcher_days_rest")
            away_pitches = p.get("away_pitcher_avg_pitches_3g")
            home_pitches = p.get("home_pitcher_avg_pitches_3g")
            print(
                "  Starter context: "
                f"away rest {away_rest:.0f}d / {away_pitches:.0f} pitches avg | "
                f"home rest {home_rest:.0f}d / {home_pitches:.0f} pitches avg"
            )
        if p.get("home_pitcher_short_leash_score") is not None or p.get("away_pitcher_short_leash_score") is not None:
            print(
                f"  Starter roles: away leash {float(p.get('away_pitcher_short_leash_score', 0) or 0):.2f}"
                f"{' opener' if p.get('away_pitcher_opener_flag') else ''}"
                f"  |  home leash {float(p.get('home_pitcher_short_leash_score', 0) or 0):.2f}"
                f"{' opener' if p.get('home_pitcher_opener_flag') else ''}"
            )

        if disp_cfg.get("show_umpire", True) and p.get("hp_umpire"):
            ump_avg = p.get("ump_avg_total_runs")
            ump_str = f"{ump_avg:.1f} avg runs/game" if ump_avg else "no history"
            print(f"  Umpire: {p['hp_umpire']}  ({ump_str})")

        if (disp_cfg.get("show_weather", True) and p.get("temp_f") is not None and not p.get("is_dome", False)):
            print(f"  Weather: {p['temp_f']:.0f}F  wind {p['wind_mph']:.0f}mph  "
                  f"precip {p['precip_mm']:.1f}mm  humidity {float(p.get('humidity_pct', 0.0)):.0f}%  "
                  f"dew point {float(p.get('dew_point_f', 0.0)):.0f}F")
        if p.get("first_pitch_local_hour") is not None and pd.notna(p.get("first_pitch_local_hour")):
            hour = float(p["first_pitch_local_hour"])
            hour_int = int(hour)
            minute = int(round((hour - hour_int) * 60))
            if minute == 60:
                hour_int += 1
                minute = 0
            print(
                f"  First pitch local: {hour_int:02d}:{minute:02d}  |  "
                f"{'night' if float(p.get('is_night_game', 0.0) or 0.0) >= 0.5 else 'day'} game"
            )

        if p.get("home_lineup_confirmed") or p.get("away_lineup_confirmed"):
            h_ops = p.get("home_lineup_avg_ops")
            a_ops = p.get("away_lineup_avg_ops")
            h_delta = p.get("home_lineup_delta_ops_30g")
            a_delta = p.get("away_lineup_delta_ops_30g")
            if h_ops is not None and a_ops is not None:
                away_delta_str = f"{a_delta:+.3f}" if a_delta is not None else "n/a"
                home_delta_str = f"{h_delta:+.3f}" if h_delta is not None else "n/a"
                print(
                    f"  Lineups: away OPS {a_ops:.3f} ({away_delta_str} vs team, platoon +{int(p.get('away_lineup_platoon_adv_batters', 0))})"
                    f"  |  home OPS {h_ops:.3f} ({home_delta_str} vs team, platoon +{int(p.get('home_lineup_platoon_adv_batters', 0))})"
                )

        if p.get("home_bullpen_used_pitches_3d") is not None or p.get("away_bullpen_used_pitches_3d") is not None:
            print(
                f"  Bullpens: away {int(p.get('away_bullpen_used_pitches_3d', 0) or 0)} pitches / "
                f"{int(p.get('away_bullpen_b2b_arms', 0) or 0)} b2b arms (3d)"
                f"  |  home {int(p.get('home_bullpen_used_pitches_3d', 0) or 0)} pitches / "
                f"{int(p.get('home_bullpen_b2b_arms', 0) or 0)} b2b arms (3d)"
            )
        if p.get("home_bullpen_top4_available_score") is not None or p.get("away_bullpen_top4_available_score") is not None:
            print(
                f"  Top arms: away avail {float(p.get('away_bullpen_top4_available_score', 0) or 0):.1f} / "
                f"burned {float(p.get('away_bullpen_top4_burned_score', 0) or 0):.1f} / "
                f"top2 used yday {int(p.get('away_bullpen_top2_used_yesterday', 0) or 0)}"
                f"  |  home avail {float(p.get('home_bullpen_top4_available_score', 0) or 0):.1f} / "
                f"burned {float(p.get('home_bullpen_top4_burned_score', 0) or 0):.1f} / "
                f"top2 used yday {int(p.get('home_bullpen_top2_used_yesterday', 0) or 0)}"
            )

        if has_lines and "posted_line" in p:
            line = p["posted_line"]
            edge = p["edge"]
            signal = p["bet_signal"]
            conf = p["bet_confidence"]
            direction = "+" if edge > 0 else ""
            market_source = p.get("market_line_source", "sportsbook")
            line_label = "Kalshi Line" if market_source == "kalshi" else "Posted Line"
            print(f"  Predicted Total: {p['predicted_total']}  |  {line_label}: {line}")
            print(f"  Edge: {direction}{edge}  |  P(Over {line}): {p['p_over_line']}%  P(Under {line}): {p['p_under_line']}%")
            if signal != "NO EDGE" and (max_bets == 0 or bet_count < max_bets):
                print(f"  *** BET: {signal} {line} ({conf:.1f}% confidence) ***")
                br_cfg = (cfg or {}).get("bankroll", {})
                if br_cfg.get("total"):
                    p_win = p["p_over_line"] / 100 if signal == "OVER" else p["p_under_line"] / 100
                    k = kelly_size(
                        p_win=p_win,
                        american_odds=p.get("posted_odds", -110),
                        bankroll=float(br_cfg["total"]),
                        kelly_fraction=float(br_cfg.get("kelly_fraction", 0.25)),
                        max_bet_pct=float(br_cfg.get("max_bet_pct", 5.0)),
                        min_bet=float(br_cfg.get("min_bet", 1)),
                        round_to=float(br_cfg.get("round_to", 5)),
                    )
                    if k["recommended_bet"] > 0:
                        cap_note = " (capped)" if k["was_capped"] else ""
                        print(f"  Kelly: {k['full_kelly_pct']}% full  ->  "
                              f"{k['frac_kelly_pct']}% quarter-Kelly{cap_note}  ->  "
                              f"BET ${k['recommended_bet']:.0f}  "
                              f"(edge: {k['edge_pct']:+.1f}%)")
                    else:
                        print("  Kelly: below minimum bet threshold - skip")
                bet_count += 1
            elif signal != "NO EDGE":
                print(f"  Signal: {signal} {line} (daily bet cap reached)")
            else:
                block_reason = p.get("bet_block_reason")
                block_method = p.get("market_adjustment_method")
                if block_reason == "market_adjustment_method_not_allowed" and block_method:
                    print(f"  Signal: NO EDGE ({block_method} not eligible for live bets)")
                else:
                    print("  Signal: NO EDGE")
            if "kalshi_line" in p and p["kalshi_line"] == line:
                print(f"  Kalshi market: {p['kalshi_over_pct']}% implied over {line}  "
                      f"(yes_ask ${p['kalshi_yes_ask']:.2f})")
                if p.get("kalshi_fair_price_pct") is not None:
                    print(f"  Kalshi fair price: {p['kalshi_fair_price_pct']}%  |  "
                          f"edge: {p.get('kalshi_edge_pct', 0):+.1f}%  |  side: {p.get('kalshi_side', 'OVER')}")
                if p.get("kalshi_kelly"):
                    kk = p["kalshi_kelly"]
                    if kk["recommended_bet"] > 0:
                        cap_note = " (capped)" if kk["was_capped"] else ""
                        print(f"  Kalshi Kelly: {kk['full_kelly_pct']}% full  ->  "
                              f"{kk['frac_kelly_pct']}% quarter-Kelly{cap_note}  ->  "
                              f"BUY ${kk['recommended_bet']:.0f} ({kk.get('recommended_bet_pct_bankroll', 0.0):.1f}% bankroll)  "
                              f"(edge: {kk['edge_pct']:+.1f}%)")
                    else:
                        reason = p.get("kalshi_bet_block_reason")
                        if reason == "edge_below_threshold":
                            print("  Kalshi Kelly: skipped (edge below Kalshi threshold)")
                        elif reason == "confidence_below_threshold":
                            print("  Kalshi Kelly: skipped (confidence below Kalshi threshold)")
                        else:
                            print(f"  Kalshi Kelly: raw ${kk.get('raw_bet', 0.0):.2f} below minimum bet threshold - skip")
        else:
            print(f"  Predicted Total: {p['predicted_total']}")
            if "kalshi_line" in p:
                k_line = p["kalshi_line"]
                k_pct = p["kalshi_over_pct"]
                model_pct = p.get(f"over_{k_line}")
                k_edge = ""
                if model_pct is not None:
                    diff = model_pct - k_pct
                    k_edge = f"  model edge: {'+' if diff > 0 else ''}{diff:.1f}%"
                print(f"  Kalshi line: {k_line}  |  Market: {k_pct}% over{k_edge}")
                if p.get("kalshi_fair_price_pct") is not None:
                    print(f"  Kalshi fair price: {p['kalshi_fair_price_pct']}%  |  "
                          f"edge: {p.get('kalshi_edge_pct', 0):+.1f}%  |  side: {p.get('kalshi_side', 'OVER')}")
                if p.get("kalshi_kelly"):
                    kk = p["kalshi_kelly"]
                    if kk["recommended_bet"] > 0:
                        cap_note = " (capped)" if kk["was_capped"] else ""
                        print(f"  Kalshi Kelly: {kk['full_kelly_pct']}% full  ->  "
                              f"{kk['frac_kelly_pct']}% quarter-Kelly{cap_note}  ->  "
                              f"BUY ${kk['recommended_bet']:.0f} ({kk.get('recommended_bet_pct_bankroll', 0.0):.1f}% bankroll)  "
                              f"(edge: {kk['edge_pct']:+.1f}%)")
                    else:
                        reason = p.get("kalshi_bet_block_reason")
                        if reason == "edge_below_threshold":
                            print("  Kalshi Kelly: skipped (edge below Kalshi threshold)")
                        elif reason == "confidence_below_threshold":
                            print("  Kalshi Kelly: skipped (confidence below Kalshi threshold)")
                        else:
                            print(f"  Kalshi Kelly: raw ${kk.get('raw_bet', 0.0):.2f} below minimum bet threshold - skip")
                print()
            print(f"  {'Line':<8} {'Over %':>8} {'Under %':>8}  {'Signal':>10}")
            print(f"  {'-'*38}")
            for line in [6.5, 7.5, 8.5, 9.5, 10.5, 11.5]:
                over_pct = p[f"over_{line}"]
                under_pct = p[f"under_{line}"]
                signal = ">> OVER" if over_pct >= 58 else (">> UNDER" if under_pct >= 58 else "")
                print(f"  {line:<8} {over_pct:>7.1f}% {under_pct:>7.1f}%  {signal:>10}")

        if i < len(predictions) - 1:
            print(f"\n  {'-' * 45}")

    kalshi_rows = [p for p in predictions if p.get("kalshi_line") is not None]
    if kalshi_rows:
        print("\n")
        print("  KALSHI EDGE BOARD")
        print("  " + "-" * 64)
        print(f"  {'Matchup':<24} {'Line':>5} {'Model%':>7} {'Market%':>8} {'Edge%':>7} {'Side':>6} {'Kelly$':>8} {'Raw$':>8}")
        print(f"  {'-' * 64}")
        ranked = sorted(
            kalshi_rows,
            key=lambda p: (
                (p.get("kalshi_kelly") or {}).get("recommended_bet", 0.0),
                abs(float(p.get("kalshi_edge_pct", 0.0))),
            ),
            reverse=True,
        )
        for p in ranked[:8]:
            kelly = p.get("kalshi_kelly") or {}
            kelly_amt = kelly.get("recommended_bet", 0.0) or 0.0
            raw_amt = kelly.get("raw_bet", 0.0) or 0.0
            matchup = f"{p['away_team']}@{p['home_team']}"
            print(
                f"  {matchup:<24.24} {p['kalshi_line']:>5} "
                f"{p.get('kalshi_fair_price_pct', 0.0):>7.1f} "
                f"{p.get('kalshi_over_pct', 0.0):>8.1f} "
                f"{p.get('kalshi_edge_pct', 0.0):>7.1f} "
                f"{p.get('kalshi_side', ''):>6} "
                f"{kelly_amt:>8.0f} "
                f"{raw_amt:>8.2f}"
            )
        if not any((p.get("kalshi_kelly") or {}).get("recommended_bet", 0) > 0 for p in kalshi_rows):
            print("  Note: no Kalshi Kelly sizing cleared the rounded bet threshold.")
