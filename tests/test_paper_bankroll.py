import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import paper_bankroll


TEST_ROOT = Path(__file__).resolve().parents[1] / "test_artifacts"


class PaperBankrollTests(unittest.TestCase):
    def test_resolve_paper_bankroll_uses_latest_prior_day_balance(self):
        base = TEST_ROOT / f"paper-bankroll-{uuid.uuid4().hex}"
        paper_tracking_dir = base / "paper_tracking"
        paper_tracking_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: __import__("shutil").rmtree(base, ignore_errors=True))

        pd.DataFrame(
            [
                {"target_date": "2026-04-14", "paper_bankroll_after_day": 10100.0},
                {"target_date": "2026-04-15", "paper_bankroll_after_day": 9950.0},
            ]
        ).to_csv(paper_tracking_dir / "kalshi_tracker_2026.tsv", sep="\t", index=False)

        with patch.object(paper_bankroll, "PAPER_TRACKING_DIR", paper_tracking_dir), patch.object(
            paper_bankroll, "load_starting_bankroll", return_value=10000.0
        ):
            bankroll = paper_bankroll.resolve_paper_bankroll("2026-04-16", 2026)

        self.assertEqual(bankroll, 9950.0)


if __name__ == "__main__":
    unittest.main()
