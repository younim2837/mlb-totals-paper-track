import tempfile
import unittest

import pandas as pd

import collect_lines_historical_oddsapi as oddsapi


class CollectLinesHistoricalOddsApiTests(unittest.TestCase):
    def test_parse_historical_payload_extracts_featured_markets(self):
        payload = {
            "timestamp": "2024-08-20T17:55:00Z",
            "previous_timestamp": "2024-08-20T17:45:00Z",
            "data": [
                {
                    "id": "evt-1",
                    "commence_time": "2024-08-20T22:41:00Z",
                    "home_team": "Miami Marlins",
                    "away_team": "Arizona Diamondbacks",
                    "bookmakers": [
                        {
                            "key": "betmgm",
                            "last_update": "2024-08-20T17:54:21Z",
                            "markets": [
                                {
                                    "key": "h2h",
                                    "last_update": "2024-08-20T17:54:21Z",
                                    "outcomes": [
                                        {"name": "Arizona Diamondbacks", "price": -150},
                                        {"name": "Miami Marlins", "price": 125},
                                    ],
                                },
                                {
                                    "key": "spreads",
                                    "last_update": "2024-08-20T17:54:21Z",
                                    "outcomes": [
                                        {"name": "Arizona Diamondbacks", "price": 105, "point": -1.5},
                                        {"name": "Miami Marlins", "price": -130, "point": 1.5},
                                    ],
                                },
                                {
                                    "key": "totals",
                                    "last_update": "2024-08-20T17:54:21Z",
                                    "outcomes": [
                                        {"name": "Over", "price": -115, "point": 7.5},
                                        {"name": "Under", "price": -105, "point": 7.5},
                                    ],
                                },
                            ],
                        }
                    ],
                }
            ],
        }

        frame, meta = oddsapi.parse_historical_payload(payload, requested_ts_iso="2024-08-20T18:00:00Z")

        self.assertEqual(meta["events"], 1)
        self.assertEqual(meta["rows"], 1)
        self.assertEqual(float(frame.loc[0, "consensus_total_line"]), 7.5)
        self.assertEqual(float(frame.loc[0, "consensus_h2h_home_price"]), 125.0)
        self.assertEqual(float(frame.loc[0, "consensus_h2h_away_price"]), -150.0)
        self.assertEqual(float(frame.loc[0, "consensus_spread_home_line"]), 1.5)
        self.assertEqual(float(frame.loc[0, "consensus_spread_away_line"]), -1.5)
        self.assertEqual(float(frame.loc[0, "betmgm_line"]), 7.5)
        self.assertEqual(float(frame.loc[0, "betmgm_h2h_home_price"]), 125.0)
        self.assertEqual(float(frame.loc[0, "betmgm_spread_home_price"]), -130.0)
        self.assertEqual(int(frame.loc[0, "num_books_h2h"]), 1)
        self.assertEqual(int(frame.loc[0, "num_books_spreads"]), 1)
        self.assertEqual(int(frame.loc[0, "num_books_totals"]), 1)

    def test_finalize_snapshots_keeps_close_columns_and_merges_actuals(self):
        snapshot_df = pd.DataFrame(
            [
                {
                    "requested_snapshot_ts": "2024-04-01T18:00:00Z",
                    "snapshot_ts": "2024-04-01T17:55:00Z",
                    "previous_snapshot_ts": "2024-04-01T17:45:00Z",
                    "event_id": "evt-1",
                    "commence_time": "2024-04-01T19:05:00Z",
                    "home_team": "Home",
                    "away_team": "Away",
                    "consensus_total_line": 8.5,
                    "consensus_over_price": -110,
                    "consensus_under_price": -110,
                    "consensus_h2h_home_price": -125,
                    "consensus_h2h_away_price": 105,
                    "consensus_spread_home_line": -1.5,
                    "consensus_spread_home_price": 120,
                    "consensus_spread_away_line": 1.5,
                    "consensus_spread_away_price": -140,
                    "num_books": 1,
                    "num_books_h2h": 1,
                    "num_books_spreads": 1,
                    "num_books_totals": 1,
                    "pinnacle_line": 8.5,
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_games_path = f"{tmpdir}/mlb_games_raw.tsv"
            final_out_path = f"{tmpdir}/lines_historical_oddsapi.tsv"
            pd.DataFrame(
                [
                    {
                        "date": "2024-04-01",
                        "away_team": "Away",
                        "home_team": "Home",
                        "away_score": 3,
                        "home_score": 6,
                    }
                ]
            ).to_csv(raw_games_path, sep="\t", index=False)

            old_raw_games = oddsapi.RAW_GAMES_FILE
            old_final_out = oddsapi.FINAL_OUT
            oddsapi.RAW_GAMES_FILE = raw_games_path
            oddsapi.FINAL_OUT = final_out_path
            try:
                final = oddsapi.finalize_snapshots(snapshot_df)
            finally:
                oddsapi.RAW_GAMES_FILE = old_raw_games
                oddsapi.FINAL_OUT = old_final_out

        self.assertEqual(float(final.loc[0, "close_total_line"]), 8.5)
        self.assertEqual(float(final.loc[0, "close_h2h_home_price"]), -125.0)
        self.assertEqual(float(final.loc[0, "close_spread_home_line"]), -1.5)
        self.assertEqual(float(final.loc[0, "actual_total"]), 9.0)
        self.assertEqual(float(final.loc[0, "actual_margin"]), 3.0)
        self.assertEqual(float(final.loc[0, "actual_home_win"]), 1.0)
        self.assertEqual(bool(final.loc[0, "actual_home_cover"]), True)


if __name__ == "__main__":
    unittest.main()
