import unittest

import numpy as np
import pandas as pd

import model_runtime
import train_model


class ModelRuntimeTests(unittest.TestCase):
    def test_get_side_residual_distribution_defaults_when_missing(self):
        result = model_runtime.get_side_residual_distribution({})

        self.assertFalse(result["enabled"])
        self.assertEqual(result["source"], "missing")
        self.assertEqual(result["samples"], 0)
        self.assertEqual(result["rho"], 0.0)

    def test_get_side_residual_distribution_clips_rho(self):
        result = model_runtime.get_side_residual_distribution(
            {"side_residual_distribution": {"rho": 5.0, "samples": 12}}
        )

        self.assertTrue(result["enabled"])
        self.assertEqual(result["samples"], 12)
        self.assertEqual(result["rho"], 0.999)

    def test_estimate_side_residual_distribution_uses_oof_side_predictions(self):
        train = pd.DataFrame(
            {
                "home_score": [5.0, 4.0, 6.0, 3.0],
                "away_score": [4.0, 2.0, 5.0, 1.0],
            }
        )
        point_oof_artifacts = {
            "oof_total_raw": pd.Series([7.0, 5.0, 9.0, 3.0]),
            "oof_side_raw": {
                "home": pd.Series([4.0, 4.0, 5.0, 2.0]),
                "away": pd.Series([3.0, 1.0, 4.0, 1.0]),
            },
        }

        result = train_model.estimate_side_residual_distribution(train, point_oof_artifacts)
        residual_home = train["home_score"].to_numpy() - point_oof_artifacts["oof_side_raw"]["home"].to_numpy()
        residual_away = train["away_score"].to_numpy() - point_oof_artifacts["oof_side_raw"]["away"].to_numpy()
        expected_rho = float(np.corrcoef(residual_home, residual_away)[0, 1])

        self.assertTrue(result["enabled"])
        self.assertEqual(result["samples"], 4)
        self.assertAlmostEqual(result["home_sigma"], float(np.std(residual_home, ddof=1)))
        self.assertAlmostEqual(result["away_sigma"], float(np.std(residual_away, ddof=1)))
        self.assertAlmostEqual(result["rho"], expected_rho)
        self.assertGreater(result["margin_sigma"], 0.0)
        self.assertGreater(result["total_sigma"], 0.0)


if __name__ == "__main__":
    unittest.main()
