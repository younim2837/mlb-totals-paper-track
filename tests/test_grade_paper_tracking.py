import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import grade_paper_tracking as gpt


TEST_ROOT = Path(__file__).resolve().parents[1] / "test_artifacts"


class GradePaperTrackingTests(unittest.TestCase):
    def test_list_board_files_prefers_richer_all_games_board(self):
        tmp_path = TEST_ROOT / f"grade-{uuid.uuid4().hex}"
        tmp_path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp_path, ignore_errors=True))

        try:
            (tmp_path / "2026-04-14-board.tsv").write_text("header\n", encoding="utf-8")
            (tmp_path / "2026-04-14-all-games-board.tsv").write_text(
                "header\nrow1\nrow2\n",
                encoding="utf-8",
            )
            (tmp_path / "2026-04-15-board.tsv").write_text("header\nrow1\n", encoding="utf-8")

            with patch.object(gpt, "PREDICTIONS_DIR", tmp_path):
                files = gpt.list_board_files(2026)
        finally:
            pass

        self.assertEqual(
            [path.name for path in files],
            ["2026-04-14-all-games-board.tsv", "2026-04-15-board.tsv"],
        )

    def test_attach_actuals_falls_back_to_date_and_matchup_when_game_id_missing(self):
        board = pd.DataFrame(
            [
                {
                    "target_date": "2026-04-14",
                    "game_id": pd.NA,
                    "away_team": "Seattle Mariners",
                    "home_team": "San Diego Padres",
                    "kalshi_side": "OVER",
                }
            ]
        )
        results = pd.DataFrame(
            [
                {
                    "game_id": 12345,
                    "date": "2026-04-14",
                    "away_team": "Seattle Mariners",
                    "home_team": "San Diego Padres",
                    "away_score": 5,
                    "home_score": 4,
                    "total_runs": 9,
                }
            ]
        )

        attached = gpt.attach_actuals(board, results)

        self.assertEqual(int(attached.loc[0, "away_score"]), 5)
        self.assertEqual(int(attached.loc[0, "home_score"]), 4)
        self.assertEqual(int(attached.loc[0, "total_runs"]), 9)

    def test_build_kalshi_tracker_rolls_forward_daily_bankroll(self):
        board_rows = pd.DataFrame(
            [
                {
                    "target_date": "2026-04-14",
                    "game_id": 1,
                    "away_team": "A",
                    "home_team": "B",
                    "predicted_total": 8.9,
                    "kalshi_line": 8.5,
                    "kalshi_side": "OVER",
                    "kalshi_edge_pct": 9.0,
                    "kalshi_fair_price_pct": 59.0,
                    "kalshi_side_model_prob": 59.0,
                    "kalshi_side_market_prob": 50.0,
                    "kalshi_over_pct": 50.0,
                    "kalshi_recommended_bet": 100.0,
                    "kalshi_bankroll_used": 10000.0,
                    "kalshi_bet_pct_bankroll": 1.0,
                    "total_runs": 10.0,
                },
                {
                    "target_date": "2026-04-15",
                    "game_id": 2,
                    "away_team": "C",
                    "home_team": "D",
                    "predicted_total": 8.1,
                    "kalshi_line": 8.5,
                    "kalshi_side": "UNDER",
                    "kalshi_edge_pct": 7.0,
                    "kalshi_fair_price_pct": 57.0,
                    "kalshi_side_model_prob": 57.0,
                    "kalshi_side_market_prob": 50.0,
                    "kalshi_over_pct": 52.0,
                    "kalshi_recommended_bet": 101.0,
                    "kalshi_bankroll_used": 0.0,
                    "kalshi_bet_pct_bankroll": 0.0,
                    "total_runs": 7.0,
                },
            ]
        )

        with patch.object(gpt, "load_starting_bankroll", return_value=10000.0):
            tracker = gpt.build_kalshi_tracker(board_rows)

        self.assertAlmostEqual(float(tracker.loc[0, "pnl_dollars"]), 100.0, places=2)
        self.assertAlmostEqual(float(tracker.loc[0, "paper_bankroll_after_day"]), 10100.0, places=2)
        self.assertAlmostEqual(float(tracker.loc[1, "paper_bankroll_at_bet"]), 10100.0, places=2)
        self.assertAlmostEqual(float(tracker.loc[1, "kalshi_bet_pct_bankroll"]), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
