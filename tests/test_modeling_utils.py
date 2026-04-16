import unittest

import modeling_utils


class ModelingUtilsTests(unittest.TestCase):
    def test_infer_side_sigmas_scales_metadata_to_match_total_sigma(self):
        result = modeling_utils.infer_side_sigmas_from_total(
            total_sigma=5.0,
            side_distribution={
                "enabled": True,
                "home_sigma": 3.0,
                "away_sigma": 4.0,
                "rho": 0.0,
            },
        )

        self.assertAlmostEqual(result["sigma_home"], 3.0)
        self.assertAlmostEqual(result["sigma_away"], 4.0)
        self.assertAlmostEqual(result["total_sigma"], 5.0)
        self.assertAlmostEqual(result["margin_sigma"], 5.0)

    def test_probability_home_win_is_fifty_fifty_for_symmetric_game(self):
        result = modeling_utils.probability_home_win(
            mean_home_runs=4.5,
            mean_away_runs=4.5,
            total_sigma=4.0,
        )

        self.assertAlmostEqual(result, 0.5, places=6)

    def test_probability_home_covering_spread_uses_sportsbook_convention(self):
        home_cover = modeling_utils.probability_home_covering_spread(
            mean_home_runs=5.0,
            mean_away_runs=4.0,
            total_sigma=3.5,
            home_spread_line=-1.5,
        )
        margin_over = modeling_utils.probability_margin_over(
            mean_home_runs=5.0,
            mean_away_runs=4.0,
            total_sigma=3.5,
            line=1.5,
        )

        self.assertAlmostEqual(home_cover, margin_over, places=6)
        self.assertGreater(home_cover, 0.0)
        self.assertLess(home_cover, 1.0)


if __name__ == "__main__":
    unittest.main()
