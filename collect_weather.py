"""
MLB Weather Data Collector
Fetches historical daily weather for each MLB venue using Open-Meteo (free, no API key).

Outputs: data/weather_historical.tsv
Columns: date, venue_id, temp_f, wind_mph, precip_mm, humidity_pct, dew_point_f, sunrise, sunset
"""

import requests
import pandas as pd
import numpy as np
import time
import os
from venue_metadata import VENUE_COORDS, VENUE_NAMES

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def fetch_daily_weather(lat: float, lon: float,
                        start_date: str, end_date: str) -> pd.DataFrame | None:
    """
    Fetch daily weather from Open-Meteo archive API (free, no key required).
    Returns DataFrame with columns including humidity/dew point and sunrise/sunset.
    or None on failure.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": (
            "temperature_2m_max,windspeed_10m_max,precipitation_sum,"
            "relative_humidity_2m_mean,dew_point_2m_mean,sunrise,sunset"
        ),
        "temperature_unit": "fahrenheit",
        "windspeed_unit": "mph",
        "timezone": "auto",
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            print(f"    HTTP {r.status_code}: {r.text[:120]}")
            return None
        data = r.json()
        daily = data.get("daily", {})
        df = pd.DataFrame({
            "date": pd.to_datetime(daily["time"]),
            "temp_f": daily["temperature_2m_max"],
            "wind_mph": daily["windspeed_10m_max"],
            "precip_mm": daily["precipitation_sum"],
            "humidity_pct": daily["relative_humidity_2m_mean"],
            "dew_point_f": daily["dew_point_2m_mean"],
            "sunrise": daily["sunrise"],
            "sunset": daily["sunset"],
        })
        return df
    except Exception as e:
        print(f"    Error: {e}")
        return None


def main():
    games_path = os.path.join(DATA_DIR, "mlb_games_raw.tsv")
    out_path = os.path.join(DATA_DIR, "weather_historical.tsv")

    print("Loading game data to find date range per venue...")
    games = pd.read_csv(games_path, sep="\t", parse_dates=["date"])

    # Group by venue_id: find min/max date
    venue_dates = (
        games.groupby("venue_id")["date"]
        .agg(["min", "max"])
        .reset_index()
    )
    venue_dates.columns = ["venue_id", "date_min", "date_max"]

    # Extend max date to today in case we want fresh data
    today = pd.Timestamp.today().strftime("%Y-%m-%d")

    # Load existing data to skip already-fetched venues
    existing = pd.DataFrame()
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path, sep="\t", parse_dates=["date"])
        required_cols = {"humidity_pct", "dew_point_f", "sunrise", "sunset"}
        if required_cols.issubset(existing.columns):
            print(f"  Found existing weather file with {len(existing)} rows")
        else:
            print("  Existing weather file is missing humidity/dew-point columns — rebuilding it.")
            existing = pd.DataFrame()

    all_rows = [existing] if not existing.empty else []
    already_have = set(existing["venue_id"].unique()) if not existing.empty else set()

    total = len(venue_dates)
    for i, row in venue_dates.iterrows():
        vid = int(row["venue_id"])
        if vid not in VENUE_COORDS:
            print(f"  [{i+1}/{total}] venue_id {vid} — no coordinates, skipping")
            continue
        if vid in already_have:
            print(f"  [{i+1}/{total}] venue_id {vid} ({VENUE_NAMES.get(vid, vid)}) — already cached, skipping")
            continue

        lat, lon = VENUE_COORDS[vid]
        name = VENUE_NAMES.get(vid, str(vid))
        start = row["date_min"].strftime("%Y-%m-%d")
        end = today

        print(f"  [{i+1}/{total}] Fetching {name} ({lat}, {lon}) {start} to {end}...")
        df = fetch_daily_weather(lat, lon, start, end)
        if df is not None and not df.empty:
            df["venue_id"] = vid
            all_rows.append(df)
            print(f"    Got {len(df)} days")
        else:
            print(f"    Failed to fetch weather for {name}")

        time.sleep(4)  # Open-Meteo free tier: ~15 req/min safe

    if not all_rows:
        print("No weather data collected.")
        return

    weather = pd.concat(all_rows, ignore_index=True)
    weather["date"] = pd.to_datetime(weather["date"])
    weather = weather.drop_duplicates(subset=["date", "venue_id"])
    weather = weather.sort_values(["venue_id", "date"]).reset_index(drop=True)

    # Round to reasonable precision
    weather["temp_f"] = weather["temp_f"].round(1)
    weather["wind_mph"] = weather["wind_mph"].round(1)
    weather["precip_mm"] = weather["precip_mm"].round(2)
    weather["humidity_pct"] = weather["humidity_pct"].round(1)
    weather["dew_point_f"] = weather["dew_point_f"].round(1)

    weather.to_csv(out_path, sep="\t", index=False)
    print(f"\nSaved {len(weather)} rows to {out_path}")
    print(f"Venues covered: {weather['venue_id'].nunique()}")
    print(f"Date range: {weather['date'].min().date()} — {weather['date'].max().date()}")


if __name__ == "__main__":
    main()
