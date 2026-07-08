"""
Fetches R/V Falkor (too)'s recent track from FSU/COAPS SAMOS (Shipboard
Automated Meteorological and Oceanographic System) and saves it for use in
daily mapping scripts.

No API key required - this is a public THREDDS/OPeNDAP feed of Falkor
(too)'s own underway navigation data (call sign ZGOJ7), one file per UTC
day at 1-minute resolution. SAMOS publishes each day's file shortly after
that day ends, so the most recent fix is typically ~a few hours to ~1 day
old, not live like AIS - see SPICE_CMEMS_SAT.py's Falkor overlay, which
labels the map with the actual fix timestamp rather than implying it's
current.

Meant to run once/day via cron (get_falkor_position.sh): each run pulls the
latest available day's full track and merges it into a rolling window CSV,
building up a multi-day tail the same way ru29_staircase.py does for the
glider track.

Usage:
    python get_falkor_position.py
    TARGET_DATE=2026-07-02 python get_falkor_position.py   # backfill a specific UTC day

Output:
    falkor_track.csv (written next to this script, rolling TRACK_WINDOW_DAYS
    window) -> columns: time, lat, lon
    falkor_position.json (latest single fix, for quick reference) ->
    {"call_sign": ..., "lat": ..., "lon": ..., "timestamp_utc": ..., "fix_time_utc": ...}
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr

FALKOR_CALL_SIGN = "ZGOJ7"
SAMOS_QUICK_BASE = "https://tds.coaps.fsu.edu/thredds/dodsC/samos/data/quick"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRACK_FILE = os.path.join(SCRIPT_DIR, "falkor_track.csv")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "falkor_position.json")
MAX_LOOKBACK_DAYS = 5  # SAMOS files can have occasional gaps (e.g. in-port days)
TRACK_WINDOW_DAYS = 7  # how much history to keep in falkor_track.csv


def _fetch_day(date):
    """Return a DataFrame of every valid (time, lat, lon) fix for one UTC day, or None."""
    url = f"{SAMOS_QUICK_BASE}/{FALKOR_CALL_SIGN}/{date:%Y}/{FALKOR_CALL_SIGN}_{date:%Y%m%d}v10001.nc"
    ds = xr.open_dataset(url)
    lats = ds["lat"].values
    lons = ds["lon"].values
    times = ds["time"].values
    valid = ~np.isnan(lats) & ~np.isnan(lons)
    if not valid.any():
        raise ValueError("no valid lat/lon in file")

    lons = lons.copy()
    lons[lons > 180] -= 360  # SAMOS reports 0-360 (+E); rest of the pipeline uses -180/180

    return pd.DataFrame({
        "time": pd.to_datetime(times[valid]).tz_localize("UTC"),
        "lat": lats[valid].astype(float),
        "lon": lons[valid].astype(float),
    })


def get_position():
    target_date = os.environ.get("TARGET_DATE")
    today = pd.Timestamp(target_date, tz="UTC").normalize() if target_date else pd.Timestamp.now(tz="UTC").normalize()

    day_df = None
    fetched_date = None
    last_err = None
    for days_back in range(MAX_LOOKBACK_DAYS + 1):
        date = today - pd.Timedelta(days=days_back)
        try:
            day_df = _fetch_day(date)
            fetched_date = date
            break
        except Exception as e:
            last_err = e

    if day_df is None:
        print(
            f"No SAMOS data found for {FALKOR_CALL_SIGN} within {MAX_LOOKBACK_DAYS} days "
            f"of {today:%Y-%m-%d}. Last error: {last_err}"
        )
        return None

    # Merge into the rolling track CSV, replacing any existing rows for the
    # same UTC date so reruns stay idempotent.
    if os.path.exists(TRACK_FILE):
        track = pd.read_csv(TRACK_FILE, parse_dates=["time"])
        if track["time"].dt.tz is None:
            track["time"] = track["time"].dt.tz_localize("UTC")
        track = track[track["time"].dt.date != fetched_date.date()]
        track = pd.concat([track, day_df], ignore_index=True)
    else:
        track = day_df

    track = track.sort_values("time")
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=TRACK_WINDOW_DAYS)
    track = track[track["time"] >= cutoff].reset_index(drop=True)
    track.to_csv(TRACK_FILE, index=False)

    last = day_df.iloc[-1]
    fix_time = last["time"]
    result = {
        "call_sign": FALKOR_CALL_SIGN,
        "lat": float(last["lat"]),
        "lon": float(last["lon"]),
        "fix_time_utc": fix_time.isoformat(),
        "timestamp_utc": fix_time.isoformat(),
        "source": "SAMOS quick (FSU/COAPS)",
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {len(day_df)} fixes for {fetched_date:%Y-%m-%d}, track now {len(track)} rows. Latest: {result}")
    return result


if __name__ == "__main__":
    get_position()
