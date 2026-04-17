"""
Kalshi MLB Totals Integration
Fetches live over/under lines from Kalshi's prediction market.
No API key required — all market data is public.

Kalshi markets are binary contracts:
  yes_ask = $0.51 means 51% implied probability of going OVER that strike.
  The "consensus line" = strike whose yes_ask is closest to $0.50.

Usage (standalone):
    python collect_kalshi_lines.py          # show today's Kalshi lines

Imported by predict_today.py:
    from collect_kalshi_lines import fetch_kalshi_lines
"""

import requests
import re
import time
from datetime import date, datetime

KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
MLB_TOTAL_SERIES = "KXMLBTOTAL"

# Kalshi shortened team names -> full team names matching mlb_games_raw.tsv
KALSHI_TEAM_MAP = {
    "Arizona":        "Arizona Diamondbacks",
    "Arizona D":      "Arizona Diamondbacks",
    "Atlanta":        "Atlanta Braves",
    "Baltimore":      "Baltimore Orioles",
    "Baltimore O":    "Baltimore Orioles",
    "Boston":         "Boston Red Sox",
    "Boston R":       "Boston Red Sox",
    "Chicago C":      "Chicago Cubs",
    "Chicago W":      "Chicago White Sox",
    "Chicago WS":     "Chicago White Sox",
    "Cincinnati":     "Cincinnati Reds",
    "Cleveland":      "Cleveland Guardians",
    "Colorado":       "Colorado Rockies",
    "Detroit":        "Detroit Tigers",
    "Houston":        "Houston Astros",
    "Kansas City":    "Kansas City Royals",
    "KC":             "Kansas City Royals",
    "Los Angeles A":  "Los Angeles Angels",
    "Los Angeles D":  "Los Angeles Dodgers",
    "LA Angels":      "Los Angeles Angels",
    "LA Dodgers":     "Los Angeles Dodgers",
    "Miami":          "Miami Marlins",
    "Milwaukee":      "Milwaukee Brewers",
    "Minnesota":      "Minnesota Twins",
    "New York M":     "New York Mets",
    "New York Y":     "New York Yankees",
    "NY Mets":        "New York Mets",
    "NY Yankees":     "New York Yankees",
    "Oakland":        "Athletics",
    "A's":            "Athletics",
    "Athletics":      "Athletics",
    "Philadelphia":   "Philadelphia Phillies",
    "Philadelphia P": "Philadelphia Phillies",
    "Pittsburgh":     "Pittsburgh Pirates",
    "San Diego":      "San Diego Padres",
    "San Diego P":    "San Diego Padres",
    "San Francisco":  "San Francisco Giants",
    "SF Giants":      "San Francisco Giants",
    "Seattle":        "Seattle Mariners",
    "St. Louis":      "St. Louis Cardinals",
    "St Louis":       "St. Louis Cardinals",
    "Tampa Bay":      "Tampa Bay Rays",
    "TB Rays":        "Tampa Bay Rays",
    "Texas":          "Texas Rangers",
    "Texas R":        "Texas Rangers",
    "Toronto":        "Toronto Blue Jays",
    "Washington":     "Washington Nationals",
}


def _parse_teams_from_title(title: str) -> tuple[str, str] | None:
    """
    Parse 'Houston vs Seattle: Total Runs' -> ('Houston Astros', 'Seattle Mariners').
    Kalshi format is 'Away vs Home'.
    Returns (away_full, home_full) or None if unparseable.
    """
    # Strip the ': Total Runs' suffix
    match = re.match(r"^(.+?)\s+vs\s+(.+?)(?:\s*:.*)?$", title, re.IGNORECASE)
    if not match:
        return None
    away_short = match.group(1).strip()
    home_short = match.group(2).strip()
    away_full = KALSHI_TEAM_MAP.get(away_short)
    home_full = KALSHI_TEAM_MAP.get(home_short)
    if not away_full or not home_full:
        return None
    return away_full, home_full


def _event_matches_target_date(event: dict, target_date: str | None) -> bool:
    """
    Filter Kalshi events to the requested slate date. Kalshi keeps open markets
    for future dates in the same series, so we need an explicit date match.
    """
    if not target_date:
        return True

    try:
        target = datetime.strptime(target_date, "%Y-%m-%d").date()
    except ValueError:
        return True

    event_ticker = event.get("event_ticker") or event.get("ticker") or ""
    match = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", event_ticker)
    if match:
        yy, mon_abbr, dd = match.groups()
        month_map = {
            "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
            "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
        }
        month = month_map.get(mon_abbr.upper())
        if month:
            try:
                event_date = date(2000 + int(yy), month, int(dd))
                return event_date == target
            except ValueError:
                pass

    sub_title = event.get("sub_title") or ""
    match = re.search(r"\(([A-Za-z]{3})\s+(\d{1,2})\)", sub_title)
    if match:
        month = datetime.strptime(match.group(1), "%b").month
        day = int(match.group(2))
        return date(target.year, month, day) == target

    return True


def _estimate_consensus_line(markets: list[dict]) -> dict | None:
    """
    Estimate the market-implied total from the strike ladder.

    We use the yes bid/ask midpoint when available, then look for the
    over-probability crossover around 50%. This is more stable than simply
    picking the strike with ask closest to 0.50, which can fail on ladders
    that jump from ~1.00 to ~0.00.
    """
    rows = []
    for m in markets:
        try:
            strike = float(m["floor_strike"])
            yes_ask = float(m["yes_ask_dollars"])
            yes_bid = float(m["yes_bid_dollars"])
            volume = float(m.get("volume_fp") or 0.0)
        except (TypeError, ValueError, KeyError):
            continue

        midpoint = (yes_ask + yes_bid) / 2 if yes_bid > 0 else yes_ask
        rows.append({
            "strike": strike,
            "yes_ask": yes_ask,
            "yes_bid": yes_bid,
            "midpoint": midpoint,
            "volume": volume,
            "ticker": m.get("ticker", ""),
        })

    if not rows:
        return None

    rows = sorted(rows, key=lambda row: row["strike"])
    crossed = None
    for prev, curr in zip(rows, rows[1:]):
        if prev["midpoint"] >= 0.5 and curr["midpoint"] <= 0.5:
            gap = prev["midpoint"] - curr["midpoint"]
            if gap <= 0:
                est = (prev["strike"] + curr["strike"]) / 2
            else:
                frac = (prev["midpoint"] - 0.5) / gap
                est = prev["strike"] + frac * (curr["strike"] - prev["strike"])
            anchor = min(
                [prev, curr],
                key=lambda row: (abs(row["midpoint"] - 0.5), -row["volume"]),
            )
            crossed = {
                "kalshi_line": anchor["strike"],
                "consensus_estimate": round(est, 1),
                "anchor_strike": anchor["strike"],
                "yes_ask": anchor["yes_ask"],
                "yes_bid": anchor["yes_bid"],
                "midpoint": round(anchor["midpoint"], 3),
                "implied_over_pct": round(anchor["yes_ask"] * 100, 1),
                "volume": anchor["volume"],
            }
            break

    if crossed:
        return crossed

    nearest = min(
        rows,
        key=lambda row: (abs(row["midpoint"] - 0.5), -row["volume"]),
    )
    return {
        "kalshi_line": nearest["strike"],
        "consensus_estimate": nearest["strike"],
        "anchor_strike": nearest["strike"],
        "yes_ask": nearest["yes_ask"],
        "yes_bid": nearest["yes_bid"],
        "midpoint": round(nearest["midpoint"], 3),
        "implied_over_pct": round(nearest["yes_ask"] * 100, 1),
        "volume": nearest["volume"],
    }


def _get_consensus_market(markets: list) -> dict | None:
    """
    From a list of markets for one game, find the one whose yes_ask is
    closest to $0.50 — that's the effective 'line' the market is most
    uncertain about. Must have some volume to be meaningful.
    """
    return _estimate_consensus_line(markets)


def _api_get(url: str, params: dict, max_retries: int = 5) -> requests.Response | None:
    """GET with retry, exponential backoff, and jitter. Returns Response or None."""
    import random
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 200:
                # Validate the response is real JSON, not an empty/corrupt body
                try:
                    r.json()
                except ValueError:
                    wait = 2 ** attempt + random.random()
                    if attempt < max_retries - 1:
                        print(f"  Kalshi bad JSON (attempt {attempt+1}), retrying in {wait:.1f}s...")
                        time.sleep(wait)
                        continue
                    print(f"  Kalshi returned non-JSON after {max_retries} attempts")
                    return None
                return r
            if r.status_code == 429:
                wait = 2 ** attempt + random.random() * 2
                print(f"  Kalshi rate limited (attempt {attempt+1}), retrying in {wait:.1f}s...")
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                wait = 2 ** attempt + random.random()
                print(f"  Kalshi server error {r.status_code} (attempt {attempt+1}), retrying in {wait:.1f}s...")
                time.sleep(wait)
                continue
            print(f"  Kalshi API error: HTTP {r.status_code} for {url}")
            return None
        except requests.exceptions.Timeout:
            wait = 2 ** attempt + random.random()
            if attempt < max_retries - 1:
                print(f"  Kalshi timeout (attempt {attempt+1}), retrying in {wait:.1f}s...")
                time.sleep(wait)
        except requests.exceptions.ConnectionError as e:
            wait = 2 ** attempt + random.random()
            if attempt < max_retries - 1:
                print(f"  Kalshi connection error (attempt {attempt+1}), retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                print(f"  Kalshi connection failed after {max_retries} attempts: {e}")
        except Exception as e:
            wait = 2 ** attempt + random.random()
            if attempt < max_retries - 1:
                print(f"  Kalshi request error ({e}), retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                print(f"  Kalshi request failed after {max_retries} attempts: {e}")
    return None


def fetch_kalshi_lines(target_date: str | None = None) -> dict:
    """
    Fetch today's (or target_date's) Kalshi MLB total run lines.

    Returns:
        {
          (away_team, home_team): {
              "kalshi_line":    float,   # actual tradable strike shown on Kalshi
              "consensus_estimate": float, # optional interpolated 50/50 estimate
              "yes_ask":        float,   # implied prob of OVER that strike
              "yes_bid":        float,
              "midpoint":       float,   # (yes_ask + yes_bid) / 2
              "implied_over_pct": float, # yes_ask * 100
              "volume":         float,
          },
          ...
        }
    """
    # ── 1. Get all open events (with cursor pagination) ─────────────────────
    events = []
    cursor = None
    for _ in range(10):  # safety bound on pages
        params = {
            "series_ticker": MLB_TOTAL_SERIES,
            "status": "open",
            "limit": 200,
        }
        if cursor:
            params["cursor"] = cursor

        r = _api_get(f"{KALSHI_API}/events", params)
        if r is None:
            if not events:
                print("  Kalshi: failed to fetch events")
                return {}
            break

        data = r.json()
        page = data.get("events", [])
        events.extend(page)
        cursor = data.get("cursor")
        if not cursor or not page:
            break

    if not events:
        return {}

    # ── 2. For each event, fetch its markets and find the consensus line ──────
    results = {}
    skipped = []
    for event in events:
        title = event.get("title", "")
        event_ticker = event.get("event_ticker") or event.get("ticker")
        if not event_ticker:
            continue
        if not _event_matches_target_date(event, target_date):
            continue

        teams = _parse_teams_from_title(title)
        if not teams:
            skipped.append(f"parse-fail: {title}")
            continue
        away_full, home_full = teams

        # Fetch all markets for this event (with retry)
        r = _api_get(
            f"{KALSHI_API}/markets",
            {"event_ticker": event_ticker, "status": "open", "limit": 50},
        )
        if r is None:
            skipped.append(f"api-fail: {away_full} @ {home_full}")
            continue

        markets = r.json().get("markets", [])
        if not markets:
            skipped.append(f"no-markets: {away_full} @ {home_full}")
            continue

        consensus = _get_consensus_market(markets)
        if consensus is None:
            skipped.append(f"no-consensus: {away_full} @ {home_full}")
            continue

        # Build full strike ladder: {strike: yes_ask} for all valid markets
        all_strikes = {}
        for m in markets:
            try:
                s = float(m["floor_strike"])
                ask = float(m["yes_ask_dollars"])
                if 0 < ask < 1:  # sanity: valid probability
                    all_strikes[s] = ask
            except (TypeError, ValueError, KeyError):
                continue

        results[(away_full, home_full)] = {
            "kalshi_line":        consensus["kalshi_line"],
            "consensus_estimate": consensus["consensus_estimate"],
            "anchor_strike":      consensus["anchor_strike"],
            "yes_ask":            consensus["yes_ask"],
            "yes_bid":            consensus["yes_bid"],
            "midpoint":           consensus["midpoint"],
            "implied_over_pct":   consensus["implied_over_pct"],
            "volume":             consensus["volume"],
            "all_strikes":        all_strikes,
        }

    if skipped:
        print(f"  Kalshi skipped {len(skipped)} events: {'; '.join(skipped)}")

    return results


def implied_to_american(p: float) -> str:
    """Convert implied probability (0-1) to American odds string."""
    if p <= 0 or p >= 1:
        return "N/A"
    if p >= 0.5:
        return f"-{round(p / (1 - p) * 100)}"
    else:
        return f"+{round((1 - p) / p * 100)}"


def main():
    today = date.today().strftime("%Y-%m-%d")
    print(f"Fetching Kalshi MLB total run markets for {today}...\n")

    lines = fetch_kalshi_lines()

    if not lines:
        print("No Kalshi MLB total markets found for today.")
        return

    print(f"{'Matchup':<42} {'K-Line':>7} {'Yes Ask':>9} {'Imp Prob':>9} {'AmOdds':>8} {'Volume':>8}")
    print("-" * 88)

    for (away, home), d in sorted(lines.items()):
        matchup = f"{away} @ {home}"[:41]
        american = implied_to_american(d["yes_ask"])
        print(
            f"  {matchup:<40} {d['kalshi_line']:>7.1f}"
            f"  {d['yes_ask']:>8.3f}  {d['implied_over_pct']:>7.1f}%"
            f"  {american:>7}  {d['volume']:>8.0f}"
        )

    print(f"\n  {len(lines)} games found.")
    print(f"  K-Line  = Kalshi consensus line (strike closest to 50/50)")
    print(f"  Yes Ask = cost to buy OVER contract ($0.52 = 52% implied probability)")
    print(f"  AmOdds  = American odds equivalent of Yes Ask for the over")


if __name__ == "__main__":
    main()
