"""
Shared MLB venue metadata used by historical feature building and live prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo


VENUE_METADATA = {
    1: {"lat": 33.8003, "lon": -117.8827, "name": "Angel Stadium", "tz": "America/Los_Angeles"},
    2: {"lat": 39.2838, "lon": -76.6218, "name": "Oriole Park at Camden Yards", "tz": "America/New_York"},
    3: {"lat": 42.3467, "lon": -71.0972, "name": "Fenway Park", "tz": "America/New_York"},
    4: {"lat": 41.8300, "lon": -87.6339, "name": "Guaranteed Rate Field", "tz": "America/Chicago"},
    5: {"lat": 41.4962, "lon": -81.6852, "name": "Progressive Field", "tz": "America/New_York"},
    7: {"lat": 39.0517, "lon": -94.4803, "name": "Kauffman Stadium", "tz": "America/Chicago"},
    10: {"lat": 37.7516, "lon": -122.2005, "name": "Oakland Coliseum", "tz": "America/Los_Angeles"},
    12: {"lat": 27.7683, "lon": -82.6534, "name": "Tropicana Field", "tz": "America/New_York"},
    14: {"lat": 43.6414, "lon": -79.3894, "name": "Rogers Centre", "tz": "America/Toronto"},
    15: {"lat": 33.4453, "lon": -112.0667, "name": "Chase Field", "tz": "America/Phoenix"},
    17: {"lat": 41.9484, "lon": -87.6553, "name": "Wrigley Field", "tz": "America/Chicago"},
    19: {"lat": 39.7560, "lon": -104.9942, "name": "Coors Field", "tz": "America/Denver"},
    22: {"lat": 34.0739, "lon": -118.2400, "name": "Dodger Stadium", "tz": "America/Los_Angeles"},
    31: {"lat": 40.4469, "lon": -80.0057, "name": "PNC Park", "tz": "America/New_York"},
    32: {"lat": 43.0280, "lon": -87.9712, "name": "American Family Field", "tz": "America/Chicago"},
    680: {"lat": 47.5914, "lon": -122.3325, "name": "T-Mobile Park", "tz": "America/Los_Angeles"},
    2392: {"lat": 29.7572, "lon": -95.3555, "name": "Minute Maid Park / Daikin Park", "tz": "America/Chicago"},
    2394: {"lat": 42.3390, "lon": -83.0485, "name": "Comerica Park", "tz": "America/New_York"},
    2395: {"lat": 37.7786, "lon": -122.3893, "name": "Oracle Park", "tz": "America/Los_Angeles"},
    2523: {"lat": 28.0341, "lon": -82.5065, "name": "George M. Steinbrenner Field", "tz": "America/New_York"},
    2529: {"lat": 38.5810, "lon": -121.5046, "name": "Sutter Health Park", "tz": "America/Los_Angeles"},
    2536: {"lat": 28.0873, "lon": -82.5646, "name": "TD Ballpark", "tz": "America/New_York"},
    2602: {"lat": 39.0979, "lon": -84.5082, "name": "Great American Ball Park", "tz": "America/New_York"},
    2680: {"lat": 32.7076, "lon": -117.1570, "name": "Petco Park", "tz": "America/Los_Angeles"},
    2681: {"lat": 39.9061, "lon": -75.1665, "name": "Citizens Bank Park", "tz": "America/New_York"},
    2735: {"lat": 40.6083, "lon": -75.4704, "name": "Coca-Cola Park / Allentown", "tz": "America/New_York"},
    2756: {"lat": 42.8956, "lon": -78.8781, "name": "Sahlen Field", "tz": "America/New_York"},
    2889: {"lat": 38.6226, "lon": -90.1928, "name": "Busch Stadium", "tz": "America/Chicago"},
    3289: {"lat": 40.7571, "lon": -73.8458, "name": "Citi Field", "tz": "America/New_York"},
    3309: {"lat": 38.8730, "lon": -77.0074, "name": "Nationals Park", "tz": "America/New_York"},
    3312: {"lat": 44.9817, "lon": -93.2780, "name": "Target Field", "tz": "America/Chicago"},
    3313: {"lat": 40.8296, "lon": -73.9262, "name": "Yankee Stadium", "tz": "America/New_York"},
    3949: {"lat": 33.5027, "lon": -86.8052, "name": "Rickwood Field", "tz": "America/Chicago"},
    4169: {"lat": 25.7781, "lon": -80.2197, "name": "loanDepot park", "tz": "America/New_York"},
    4705: {"lat": 33.8908, "lon": -84.4677, "name": "Truist Park", "tz": "America/New_York"},
    5325: {"lat": 32.7473, "lon": -97.0830, "name": "Globe Life Field", "tz": "America/Chicago"},
    5340: {"lat": 19.4800, "lon": -99.1200, "name": "Estadio Alfredo Harp Helu", "tz": "America/Mexico_City"},
    5381: {"lat": 51.5560, "lon": -0.2796, "name": "London Stadium", "tz": "Europe/London"},
    5445: {"lat": 42.5116, "lon": -91.5522, "name": "Field of Dreams", "tz": "America/Chicago"},
    6130: {"lat": 36.5099, "lon": -80.2538, "name": "Bristol Motor Speedway", "tz": "America/New_York"},
}

VENUE_COORDS = {vid: (meta["lat"], meta["lon"]) for vid, meta in VENUE_METADATA.items()}
VENUE_NAMES = {vid: meta["name"] for vid, meta in VENUE_METADATA.items()}
VENUE_TIMEZONES = {vid: meta["tz"] for vid, meta in VENUE_METADATA.items()}

DOME_VENUE_IDS = {
    12,
    14,
    15,
    32,
    680,
    2392,
    4169,
    5325,
}


def compute_local_time_features(
    game_datetime,
    venue_id,
    sunrise=None,
    sunset=None,
) -> dict[str, float]:
    hour = np.nan
    is_night = np.nan

    try:
        venue_id = int(venue_id)
    except (TypeError, ValueError):
        return {"first_pitch_local_hour": hour, "is_night_game": is_night}

    tz_name = VENUE_TIMEZONES.get(venue_id)
    ts = pd.to_datetime(game_datetime, errors="coerce", utc=True)
    if pd.isna(ts) or not tz_name:
        return {"first_pitch_local_hour": hour, "is_night_game": is_night}

    local_ts = ts.tz_convert(ZoneInfo(tz_name))
    hour = float(local_ts.hour + local_ts.minute / 60.0)

    sunrise_ts = pd.to_datetime(sunrise, errors="coerce")
    sunset_ts = pd.to_datetime(sunset, errors="coerce")
    if pd.notna(sunrise_ts) and pd.notna(sunset_ts):
        first_pitch_minutes = local_ts.hour * 60 + local_ts.minute
        sunrise_minutes = sunrise_ts.hour * 60 + sunrise_ts.minute
        sunset_minutes = sunset_ts.hour * 60 + sunset_ts.minute
        is_night = float(first_pitch_minutes >= sunset_minutes or first_pitch_minutes < sunrise_minutes)
    else:
        is_night = float(hour >= 17.0)

    return {
        "first_pitch_local_hour": hour,
        "is_night_game": is_night,
    }
