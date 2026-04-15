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


if __name__ == "__main__":
    unittest.main()
