import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import prediction_reporting


TEST_ROOT = Path(__file__).resolve().parents[1] / "test_artifacts"


class PredictionReportingTests(unittest.TestCase):
    def test_export_includes_kalshi_only_bets_in_picks_file(self):
        prediction = {
            "target_date": "2026-04-14",
            "game_id": 99,
            "commence_time": "2026-04-14T19:40:00Z",
            "away_team": "Chicago Cubs",
            "home_team": "Philadelphia Phillies",
            "predicted_total": 8.9,
            "bet_signal": "NO EDGE",
            "kalshi_side": "OVER",
            "kalshi_line": 8.5,
            "kalshi_over_pct": 49.0,
            "kalshi_kelly": {
                "recommended_bet": 25.0,
                "raw_bet": 24.6,
                "bankroll_used": 1000.0,
                "recommended_bet_pct_bankroll": 2.5,
                "raw_bet_pct_bankroll": 2.46,
            },
        }

        tmpdir = TEST_ROOT / f"reporting-{uuid.uuid4().hex}"
        tmpdir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: __import__("shutil").rmtree(tmpdir, ignore_errors=True))

        with patch.object(prediction_reporting, "PREDICTIONS_DIR", str(tmpdir)):
            board_path, picks_path = prediction_reporting.export_daily_prediction_reports(
                [prediction],
                target_date="2026-04-14",
                include_all_games=True,
            )

            board_df = pd.read_csv(board_path, sep="\t")
            picks_df = pd.read_csv(picks_path, sep="\t")

        self.assertEqual(Path(board_path).name, "2026-04-14-all-games-board.tsv")
        self.assertEqual(Path(picks_path).name, "2026-04-14-all-games-picks.tsv")
        self.assertEqual(len(board_df), 1)
        self.assertEqual(len(picks_df), 1)
        self.assertEqual(int(board_df.loc[0, "game_id"]), 99)
        self.assertEqual(board_df.loc[0, "commence_time"], "2026-04-14T19:40:00Z")
        self.assertEqual(float(picks_df.loc[0, "kalshi_recommended_bet"]), 25.0)
        self.assertEqual(float(picks_df.loc[0, "kalshi_bankroll_used"]), 1000.0)
        self.assertEqual(float(picks_df.loc[0, "kalshi_bet_pct_bankroll"]), 2.5)


if __name__ == "__main__":
    unittest.main()
