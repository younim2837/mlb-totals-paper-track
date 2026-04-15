import unittest

import prediction_betting


class PredictionBettingTests(unittest.TestCase):
    def test_kalshi_kelly_size_treats_zero_max_bet_pct_as_uncapped(self):
        result = prediction_betting.kalshi_kelly_size(
            p_win=0.65,
            market_price=0.50,
            bankroll=100.0,
            kelly_fraction=0.25,
            max_bet_pct=0.0,
            min_bet=1.0,
            round_to=1.0,
        )

        self.assertGreater(result["raw_bet"], 0.0)
        self.assertGreater(result["recommended_bet"], 0.0)
        self.assertFalse(result["was_capped"])

    def test_kalshi_filter_reason_applies_edge_and_confidence_thresholds(self):
        pred = {
            "kalshi_line": 8.5,
            "predicted_total": 8.7,
            "kalshi_edge_pct": 4.0,
            "kalshi_side_model_prob": 54.0,
            "kalshi_kelly": {"recommended_bet": 25.0},
        }

        self.assertEqual(
            prediction_betting.kalshi_filter_reason(
                pred,
                max_line_diff=4.5,
                min_edge_pct=5.0,
                min_confidence_pct=0.0,
            ),
            "edge_below_threshold",
        )
        self.assertEqual(
            prediction_betting.kalshi_filter_reason(
                pred,
                max_line_diff=4.5,
                min_edge_pct=0.0,
                min_confidence_pct=55.0,
            ),
            "confidence_below_threshold",
        )

    def test_suppress_kalshi_bet_zeroes_recommended_bet_and_records_reason(self):
        pred = {
            "kalshi_kelly": {"recommended_bet": 40.0, "raw_bet": 43.25},
        }

        updated = prediction_betting.suppress_kalshi_bet(pred, "confidence_below_threshold")

        self.assertEqual(updated["kalshi_bet_block_reason"], "confidence_below_threshold")
        self.assertEqual(updated["kalshi_kelly"]["recommended_bet"], 0.0)
        self.assertEqual(updated["kalshi_kelly"]["raw_bet"], 43.25)


if __name__ == "__main__":
    unittest.main()
