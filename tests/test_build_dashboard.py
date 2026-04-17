import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import build_dashboard


TEST_ROOT = Path(__file__).resolve().parents[1] / "test_artifacts"


class BuildDashboardTests(unittest.TestCase):
    def test_build_dashboard_renders_summary_and_latest_pick(self):
        base = TEST_ROOT / f"dashboard-{uuid.uuid4().hex}"
        predictions_dir = base / "predictions"
        paper_tracking_dir = base / "paper_tracking"
        data_dir = base / "data"
        docs_dir = base / "docs"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        paper_tracking_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        docs_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(base, ignore_errors=True))

        kalshi_df = pd.DataFrame(
            [
                {
                    "target_date": "2026-04-15",
                    "month": "2026-04",
                    "game_id": 1,
                    "away_team": "Chicago Cubs",
                    "home_team": "Philadelphia Phillies",
                    "predicted_total": 9.1,
                    "kalshi_line": 8.5,
                    "kalshi_side": "OVER",
                    "kalshi_edge_pct": 10.4,
                    "kalshi_fair_price_pct": 60.4,
                    "kalshi_side_model_prob": 60.4,
                    "kalshi_side_market_prob": 50.0,
                    "kalshi_side_market_price": 0.50,
                    "kalshi_bet_pct_bankroll": 1.25,
                    "kalshi_recommended_bet": 125.0,
                    "paper_bankroll_after_day": 10125.0,
                    "away_score": 5,
                    "home_score": 4,
                    "total_runs": 9,
                    "result": "win",
                    "profit_per_contract": 0.50,
                    "roi_pct": 100.0,
                    "settled": True,
                },
                {
                    "target_date": "2026-04-15",
                    "month": "2026-04",
                    "game_id": 2,
                    "away_team": "Texas Rangers",
                    "home_team": "Athletics",
                    "predicted_total": 8.3,
                    "kalshi_line": 9.5,
                    "kalshi_side": "UNDER",
                    "kalshi_edge_pct": -13.2,
                    "kalshi_fair_price_pct": 38.8,
                    "kalshi_side_model_prob": 61.2,
                    "kalshi_side_market_prob": 48.0,
                    "kalshi_side_market_price": 0.48,
                    "kalshi_bet_pct_bankroll": 0.80,
                    "kalshi_recommended_bet": 80.0,
                    "paper_bankroll_after_day": 10125.0,
                    "result": "pending",
                    "profit_per_contract": 0.0,
                    "roi_pct": 0.0,
                    "settled": False,
                },
            ]
        )
        kalshi_df.to_csv(paper_tracking_dir / "kalshi_tracker_2026.tsv", sep="\t", index=False)

        picks_df = pd.DataFrame(
            [
                {
                    "target_date": "2026-04-16",
                    "game_id": 99,
                    "commence_time": "2026-04-16T19:40:00Z",
                    "away_team": "Seattle Mariners",
                    "home_team": "San Diego Padres",
                    "predicted_total": 8.7,
                    "kalshi_side": "UNDER",
                    "kalshi_line": 9.5,
                    "kalshi_side_market_prob": 45.0,
                    "kalshi_side_model_prob": 61.2,
                    "kalshi_fair_price_pct": 38.8,
                    "kalshi_edge_pct": -13.2,
                    "kalshi_bet_pct_bankroll": 0.8,
                    "kalshi_recommended_bet": 80.0,
                }
            ]
        )
        picks_df.to_csv(predictions_dir / "2026-04-16-pregame-board.tsv", sep="\t", index=False)

        historical_df = pd.DataFrame(
            [
                {
                    "date": "2026-04-10",
                    "game_id": 77,
                    "away_team": "Boston Red Sox",
                    "home_team": "Toronto Blue Jays",
                    "predicted_total": 8.4,
                    "kalshi_line": 7.5,
                    "kalshi_side": "OVER",
                    "kalshi_side_market_prob": 48.0,
                    "kalshi_fair_price_pct": 60.0,
                    "kalshi_edge_pct": 12.0,
                    "bet_amount": 110.0,
                    "bet_pct_bankroll": 1.1,
                    "actual_total": 10,
                    "result": "win",
                    "won": True,
                    "pnl_dollars": 119.17,
                    "roi_pct": 108.3,
                    "settled": True,
                },
                {
                    "date": "2026-04-11",
                    "game_id": 78,
                    "away_team": "Chicago White Sox",
                    "home_team": "Kansas City Royals",
                    "predicted_total": 8.1,
                    "kalshi_line": 9.5,
                    "kalshi_side": "UNDER",
                    "kalshi_side_market_prob": 48.0,
                    "kalshi_side_model_prob": 61.2,
                    "kalshi_fair_price_pct": 38.8,
                    "kalshi_edge_pct": -13.2,
                    "bet_amount": 90.0,
                    "bet_pct_bankroll": 0.9,
                    "actual_total": 7,
                    "result": "win",
                    "won": True,
                    "pnl_dollars": 95.0,
                    "roi_pct": 105.6,
                    "settled": True,
                }
            ]
        )
        historical_df.to_csv(data_dir / "season_sim_2026.tsv", sep="\t", index=False)
        (data_dir / "season_sim_summary.json").write_text(
            """{
  "total_games": 15,
  "games_with_kalshi": 14,
  "bets_placed": 9,
  "wins": 5,
  "losses": 4,
  "win_rate": 55.6,
  "roi_pct": 12.3,
  "total_pnl": 431.2,
  "total_wagered": 3500.0,
  "avg_edge_pct": 8.4,
  "kelly_fraction": 0.1,
  "max_bet_pct": 0.0,
  "min_bet": 1.0,
  "min_kalshi_edge_pct": 6.0,
  "min_kalshi_confidence_pct": 57.5
}""",
            encoding="utf-8",
        )

        with patch.object(build_dashboard, "PREDICTIONS_DIR", predictions_dir), patch.object(
            build_dashboard, "PAPER_TRACKING_DIR", paper_tracking_dir
        ), patch.object(build_dashboard, "DASHBOARD_DIR", docs_dir), patch.object(
            build_dashboard, "DATA_DIR", data_dir
        ):
            output_path = build_dashboard.build_dashboard(2026)

        html_text = output_path.read_text(encoding="utf-8")
        self.assertIn("Season Snapshot", html_text)
        self.assertIn("Seattle Mariners", html_text)
        self.assertIn("Chicago Cubs", html_text)
        self.assertIn("Latest Daily Bets", html_text)
        self.assertIn("Historical Replay", html_text)
        self.assertIn("Kalshi Rules", html_text)
        self.assertIn("57.5%", html_text)
        self.assertIn("% Roll", html_text)
        self.assertIn("Current Bankroll", html_text)
        self.assertIn("Started at $10.0k.", html_text)
        self.assertIn("Boston Red Sox", html_text)
        self.assertIn("minmax(min(100%, 220px), 1fr)", html_text)
        self.assertIn("white-space: nowrap;", html_text)
        self.assertIn("+11.8%", html_text)
        self.assertIn("+12.6%", html_text)
        self.assertIn("61.2%", html_text)
        self.assertIn("+16.2%", html_text)



if __name__ == "__main__":
    unittest.main()
