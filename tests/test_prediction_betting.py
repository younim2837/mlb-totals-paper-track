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


if __name__ == "__main__":
    unittest.main()
