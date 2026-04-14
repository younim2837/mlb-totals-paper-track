"""
MLB Betting Lines Collection
- Historical (2021): Downloads from sportsbookreviewsonline.com (free)
- Live (today): Fetches from The Odds API (free tier, requires API key)

Note:
  The old SportsbookReviewOnline workbook links for 2022-2025 are now stale.
  For future historical backfills beyond 2021, use
  collect_lines_historical_oddsapi.py instead.

Outputs:
  data/lines_historical.tsv   — game-level closing O/U lines (2021)
  data/lines_today.tsv        — today's O/U lines from sportsbooks
"""

import requests
import openpyxl
import pandas as pd
import io
import os
import sys
import json
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")

SBRO_BASE = "https://www.sportsbookreviewsonline.com/wp-content/uploads/sportsbookreviewsonline_com_737/"
SBRO_SEASONS = {
    2021: "mlb-odds-2021.xlsx",
    2022: "mlb-odds-2022.xlsx",
    2023: "mlb-odds-2023.xlsx",
    2024: "mlb-odds-2024.xlsx",
    2025: "mlb-odds-2025.xlsx",
}

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"

# SBRO team abbreviation -> MLB Stats API team name
SBRO_TEAM_MAP = {
    "ATL": "Atlanta Braves", "ARI": "Arizona Diamondbacks",
    "BAL": "Baltimore Orioles", "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs", "CWS": "Chicago White Sox",
    "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies", "DET": "Detroit Tigers",
    "HOU": "Houston Astros", "KAN": "Kansas City Royals",
    "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins", "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins", "NYM": "New York Mets",
    "NYY": "New York Yankees", "OAK": "Athletics",
    "PHI": "Philadelphia Phillies", "PIT": "Pittsburgh Pirates",
    "SDP": "San Diego Padres", "SEA": "Seattle Mariners",
    "SFG": "San Francisco Giants", "STL": "St. Louis Cardinals",
    "TBR": "Tampa Bay Rays", "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays", "WSN": "Washington Nationals",
    # Common alternate abbreviations
    "CUB": "Chicago Cubs", "CHW": "Chicago White Sox",
    "CLV": "Cleveland Guardians", "GUA": "Cleveland Guardians",
    "KCR": "Kansas City Royals", "LAA": "Los Angeles Angels",
    "ANA": "Los Angeles Angels", "OAK": "Athletics",
    "ATH": "Athletics", "SDP": "San Diego Padres", "SDG": "San Diego Padres",
    "SFG": "San Francisco Giants", "SFN": "San Francisco Giants",
    "TBR": "Tampa Bay Rays", "TAM": "Tampa Bay Rays",
    "WSN": "Washington Nationals", "WAS": "Washington Nationals",
}


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}


def parse_sbro_date(date_val, year):
    """Parse SBRO date format: 401 = April 1, 1001 = October 1."""
    try:
        date_str = str(int(date_val)).zfill(4)
        month = int(date_str[:2])
        day = int(date_str[2:])
        return datetime(year, month, day).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def download_sbro_season(year):
    """Download and parse one season of SBRO data."""
    filename = SBRO_SEASONS.get(year)
    if not filename:
        print(f"  No SBRO file known for {year}")
        return pd.DataFrame()

    url = SBRO_BASE + filename
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    print(f"  Downloading {year} from SBRO...")
    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code != 200:
        print(f"  Failed: HTTP {r.status_code}")
        return pd.DataFrame()

    wb = openpyxl.load_workbook(io.BytesIO(r.content))
    ws = wb.active

    # Parse rows in pairs (V=visitor on row i, H=home on row i+1)
    records = []
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    i = 0
    while i < len(rows) - 1:
        v_row = rows[i]
        h_row = rows[i + 1]

        if v_row[2] == "V" and h_row[2] == "H":
            date_str = parse_sbro_date(v_row[0], year)
            if not date_str:
                i += 2
                continue

            away_abbr = str(v_row[3]).strip().upper() if v_row[3] else ""
            home_abbr = str(h_row[3]).strip().upper() if h_row[3] else ""

            # Columns: 0=Date,1=Rot,2=VH,3=Team,4=Pitcher,5-13=Inn1-9,
            #          14=Final,15=OpenML,16=CloseML,17=RunLine,18=RLOdds,
            #          19=OpenOU_line,20=OpenOU_odds,21=CloseOU_line,22=CloseOU_odds
            try:
                away_score = int(v_row[14]) if v_row[14] is not None else None
                home_score = int(h_row[14]) if h_row[14] is not None else None
                open_total = float(v_row[19]) if v_row[19] is not None else None
                close_total = float(v_row[21]) if v_row[21] is not None else None
                close_total_odds = int(v_row[22]) if v_row[22] is not None else None
            except (ValueError, TypeError):
                i += 2
                continue

            records.append({
                "date": date_str,
                "season": year,
                "away_team_sbro": away_abbr,
                "home_team_sbro": home_abbr,
                "away_team": SBRO_TEAM_MAP.get(away_abbr, away_abbr),
                "home_team": SBRO_TEAM_MAP.get(home_abbr, home_abbr),
                "away_score": away_score,
                "home_score": home_score,
                "actual_total": (away_score + home_score) if away_score is not None and home_score is not None else None,
                "open_total_line": open_total,
                "close_total_line": close_total,
                "close_total_odds": close_total_odds,
            })
            i += 2
        else:
            i += 1

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  Parsed {len(df)} games from {year}")
    return df


def fetch_live_lines(api_key):
    """Fetch today's MLB totals from The Odds API."""
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "totals",
        "oddsFormat": "american",
        "bookmakers": "draftkings,fanduel,betmgm,pinnacle,caesars",
    }
    r = requests.get(ODDS_API_URL, params=params, timeout=15)
    if r.status_code != 200:
        print(f"  Odds API error: {r.status_code} — {r.text[:200]}")
        return pd.DataFrame()

    remaining = r.headers.get("x-requests-remaining", "?")
    used = r.headers.get("x-requests-used", "?")
    print(f"  API credits used: {used}, remaining: {remaining}")

    games = r.json()
    records = []
    for game in games:
        home = game["home_team"]
        away = game["away_team"]
        game_time = game.get("commence_time", "")

        # Collect lines from each bookmaker
        lines = {}
        for bm in game.get("bookmakers", []):
            for market in bm.get("markets", []):
                if market["key"] == "totals":
                    for outcome in market["outcomes"]:
                        if outcome["name"] == "Over":
                            lines[bm["key"]] = outcome["point"]
                            break

        if not lines:
            continue

        # Use consensus line (median across books)
        line_values = list(lines.values())
        consensus = sorted(line_values)[len(line_values) // 2]

        records.append({
            "game_time": game_time,
            "home_team": home,
            "away_team": away,
            "consensus_total_line": consensus,
            "draftkings_line": lines.get("draftkings"),
            "fanduel_line": lines.get("fanduel"),
            "betmgm_line": lines.get("betmgm"),
            "pinnacle_line": lines.get("pinnacle"),
            "caesars_line": lines.get("caesars"),
            "num_books": len(lines),
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df["game_time"] = pd.to_datetime(df["game_time"])
    return df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    config = load_config()

    # ── Historical lines (SBRO) ──────────────────────────────────────────────
    hist_path = os.path.join(DATA_DIR, "lines_historical.tsv")
    if os.path.exists(hist_path):
        existing = pd.read_csv(hist_path, sep="\t", parse_dates=["date"])
        existing_years = set(existing["season"].unique())
        print(f"Historical lines already downloaded for: {sorted(existing_years)}")
    else:
        existing = pd.DataFrame()
        existing_years = set()

    new_seasons = []
    for year in SBRO_SEASONS:
        if year not in existing_years:
            print(f"\nDownloading {year} historical lines...")
            df = download_sbro_season(year)
            if not df.empty:
                new_seasons.append(df)

    if new_seasons:
        combined = pd.concat([existing] + new_seasons, ignore_index=True) if not existing.empty else pd.concat(new_seasons)
        combined.to_csv(hist_path, sep="\t", index=False)
        print(f"\nSaved {len(combined)} historical games to {hist_path}")
    else:
        print("No new historical seasons to download.")

    # ── Live lines (The Odds API) ────────────────────────────────────────────
    api_key = config.get("odds_api_key") or os.environ.get("ODDS_API_KEY")
    if not api_key:
        print("\n" + "="*60)
        print("To get today's live lines, add your Odds API key:")
        print("  1. Sign up free at https://the-odds-api.com")
        print("  2. Run: python collect_lines.py YOUR_API_KEY")
        print("  Or add to config.json: {\"odds_api_key\": \"your_key_here\"}")
        print("="*60)
    else:
        print("\nFetching today's live lines from The Odds API...")
        live_df = fetch_live_lines(api_key)
        if not live_df.empty:
            live_path = os.path.join(DATA_DIR, "lines_today.tsv")
            live_df.to_csv(live_path, sep="\t", index=False)
            print(f"Saved {len(live_df)} games to {live_path}")
            print(live_df[["away_team", "home_team", "consensus_total_line"]].to_string(index=False))
        else:
            print("No games found for today.")


if __name__ == "__main__":
    # Optionally accept API key as command-line arg
    if len(sys.argv) > 1:
        os.environ["ODDS_API_KEY"] = sys.argv[1]
    main()
