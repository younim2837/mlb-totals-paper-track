import unittest
from unittest.mock import Mock, patch

import requests

import collect_kalshi_lines as ckl


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class CollectKalshiLinesTests(unittest.TestCase):
    def test_parse_teams_from_title_supports_known_aliases(self):
        self.assertEqual(
            ckl._parse_teams_from_title("Texas vs A's: Total Runs"),
            ("Texas Rangers", "Athletics"),
        )
        self.assertEqual(
            ckl._parse_teams_from_title("Tampa Bay vs Chicago WS: Total Runs"),
            ("Tampa Bay Rays", "Chicago White Sox"),
        )

    def test_api_get_retries_timeout_and_rate_limit_before_success(self):
        responses = [
            requests.exceptions.Timeout(),
            FakeResponse(status_code=429),
            FakeResponse(status_code=200, payload={"ok": True}),
        ]

        def fake_get(*args, **kwargs):
            result = responses.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        with patch.object(ckl.requests, "get", side_effect=fake_get) as mock_get, patch.object(
            ckl.time, "sleep"
        ) as mock_sleep:
            response = ckl._api_get("https://example.test", {"foo": "bar"})

        self.assertIsNotNone(response)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_get.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    def test_fetch_kalshi_lines_handles_pagination_and_market_mapping(self):
        def fake_api_get(url, params, max_retries=3):
            if url.endswith("/events") and "cursor" not in params:
                return FakeResponse(
                    payload={
                        "events": [
                            {
                                "title": "Texas vs A's: Total Runs",
                                "event_ticker": "KXMLBTOTAL-26APR14-TEXOAK",
                            }
                        ],
                        "cursor": "page-2",
                    }
                )
            if url.endswith("/events") and params.get("cursor") == "page-2":
                return FakeResponse(
                    payload={
                        "events": [
                            {
                                "title": "Tampa Bay vs Chicago WS: Total Runs",
                                "event_ticker": "KXMLBTOTAL-26APR14-TBCHW",
                            }
                        ],
                        "cursor": None,
                    }
                )
            if url.endswith("/markets") and params.get("event_ticker") == "KXMLBTOTAL-26APR14-TEXOAK":
                return FakeResponse(
                    payload={
                        "markets": [
                            {
                                "ticker": "A",
                                "floor_strike": 8.5,
                                "yes_ask_dollars": 0.62,
                                "yes_bid_dollars": 0.60,
                                "volume_fp": 10,
                            },
                            {
                                "ticker": "B",
                                "floor_strike": 9.5,
                                "yes_ask_dollars": 0.44,
                                "yes_bid_dollars": 0.42,
                                "volume_fp": 12,
                            },
                        ]
                    }
                )
            if url.endswith("/markets") and params.get("event_ticker") == "KXMLBTOTAL-26APR14-TBCHW":
                return FakeResponse(
                    payload={
                        "markets": [
                            {
                                "ticker": "C",
                                "floor_strike": 8.5,
                                "yes_ask_dollars": 0.48,
                                "yes_bid_dollars": 0.46,
                                "volume_fp": 8,
                            }
                        ]
                    }
                )
            raise AssertionError(f"Unexpected API call: {url} {params}")

        with patch.object(ckl, "_api_get", side_effect=fake_api_get):
            lines = ckl.fetch_kalshi_lines("2026-04-14")

        self.assertEqual(len(lines), 2)
        self.assertIn(("Texas Rangers", "Athletics"), lines)
        self.assertIn(("Tampa Bay Rays", "Chicago White Sox"), lines)
        self.assertEqual(lines[("Texas Rangers", "Athletics")]["kalshi_line"], 9.5)
        self.assertAlmostEqual(lines[("Tampa Bay Rays", "Chicago White Sox")]["implied_over_pct"], 48.0)


if __name__ == "__main__":
    unittest.main()
