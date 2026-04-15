import unittest
from unittest.mock import patch

import pandas as pd

import simulate_2026_season as sim2026


class Simulate2026SeasonTests(unittest.TestCase):
    def test_find_consensus_picks_nearest_tradable_anchor(self):
        strike, price = sim2026.find_consensus({7.5: 0.64, 8.5: 0.52, 9.5: 0.43})
        self.assertEqual(strike, 8.5)
        self.assertEqual(price, 0.52)

    def test_find_consensus_falls_back_to_closest_to_fifty(self):
        strike, price = sim2026.find_consensus({7.5: 0.61, 8.5: 0.57, 9.5: 0.54})
        self.assertEqual(strike, 9.5)
        self.assertEqual(price, 0.54)

    def test_simulate_skips_bet_when_edge_is_below_threshold(self):
        preds_df = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-04-10"),
                    "game_id": 1,
                    "away_team": "Away",
                    "home_team": "Home",
                    "predicted_total": 8.5,
                    "prediction_std": 4.0,
                    "high_tail_prob_9p5": None,
                    "low_tail_prob_7p5": None,
                    "total_runs": 9.0,
                }
            ]
        )
        kalshi = {("2026-04-10", "Away", "Home"): {8.5: 0.50}}
        bankroll_cfg = {"total": 10000, "kelly_fraction": 0.25, "max_bet_pct": 0, "min_bet": 1, "round_to": 1}
        display_cfg = {"kalshi_max_line_diff": 4.5}

        def fake_add_kalshi_metrics(pred, residual_std):
            pred["kalshi_side"] = "OVER"
            pred["kalshi_side_model_prob"] = 54.0
            pred["kalshi_side_market_prob"] = 50.0
            pred["kalshi_fair_price_pct"] = 54.0
            pred["kalshi_edge_pct"] = 4.0
            pred["kalshi_kelly"] = {"recommended_bet": 200.0}
            return pred

        with patch.object(sim2026, "add_kalshi_metrics", side_effect=fake_add_kalshi_metrics):
            sim_df = sim2026.simulate(
                preds_df,
                kalshi,
                bankroll_cfg,
                display_cfg,
                min_kalshi_edge_pct=5.0,
                min_kalshi_confidence_pct=0.0,
            )

        self.assertEqual(float(sim_df.loc[0, "bet_amount"]), 0.0)
        self.assertEqual(sim_df.loc[0, "skip_reason"], "edge_below_threshold")

    def test_simulate_skips_bet_when_confidence_is_below_threshold(self):
        preds_df = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-04-10"),
                    "game_id": 1,
                    "away_team": "Away",
                    "home_team": "Home",
                    "predicted_total": 8.5,
                    "prediction_std": 4.0,
                    "high_tail_prob_9p5": None,
                    "low_tail_prob_7p5": None,
                    "total_runs": 9.0,
                }
            ]
        )
        kalshi = {("2026-04-10", "Away", "Home"): {8.5: 0.50}}
        bankroll_cfg = {"total": 10000, "kelly_fraction": 0.25, "max_bet_pct": 0, "min_bet": 1, "round_to": 1}
        display_cfg = {"kalshi_max_line_diff": 4.5}

        def fake_add_kalshi_metrics(pred, residual_std):
            pred["kalshi_side"] = "OVER"
            pred["kalshi_side_model_prob"] = 54.0
            pred["kalshi_side_market_prob"] = 50.0
            pred["kalshi_fair_price_pct"] = 54.0
            pred["kalshi_edge_pct"] = 8.0
            pred["kalshi_kelly"] = {"recommended_bet": 200.0}
            return pred

        with patch.object(sim2026, "add_kalshi_metrics", side_effect=fake_add_kalshi_metrics):
            sim_df = sim2026.simulate(
                preds_df,
                kalshi,
                bankroll_cfg,
                display_cfg,
                min_kalshi_edge_pct=0.0,
                min_kalshi_confidence_pct=55.0,
            )

        self.assertEqual(float(sim_df.loc[0, "bet_amount"]), 0.0)
        self.assertEqual(sim_df.loc[0, "skip_reason"], "confidence_below_threshold")


if __name__ == "__main__":
    unittest.main()
