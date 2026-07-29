#!/usr/bin/env python
"""
Summarizes currently-active AVISO+ eddies within a tighter bbox around the
12N survey line (lat 7-15, lon -60 to -43 by default - see
configs/eddy_12N_forecast.yml) into a plain-text metrics table. Plain
fixed-width text (not JSON) so it reads cleanly as columns/rows when opened
directly. Which columns appear, and in what order, is driven entirely by
configs/eddy_12N_forecast.yml's `columns` list - see AVAILABLE_FIELDS below
for the full set this script knows how to compute; add/remove/reorder in
the yml, no code change needed to change what gets reported.

Reads the subset NetCDF files eddy_trajectory_download.py already wrote
(eddy_<polarity>_latest.nc, subset to the wider TROP_WTRN_ATL_EXTENT bbox +
ROLLING_WINDOW_DAYS history) - same files eddy_trajectory_plot.py reads.
This script does no network access itself and does NOT change either the
download script's or the plot script's own bbox/subsetting - it just
further filters down to a tighter bbox and derives per-eddy metrics on top.
Output lands in the same dated tree eddy_trajectory_plot.py saves its maps
into (<save_dir>/YYYY/MM/DD/eddy_trajectory/), so the table sits right next
to the map it corresponds to.

"Active" mirrors eddy_trajectory_plot.py's definition: an eddy's last known
position falls on the most recent day present in the loaded data (AVISO
never explicitly flags an eddy as ended - only inferred by a track not
reappearing).

AVISO's eddy atlas has no direct "translation speed" field - speed_average
("rotational_speed_m_s" below) is the eddy's own internal circum-averaged
rotational speed, not how fast its center is moving. Translation speed here
is derived from each eddy's last two available daily observations
(flat-earth approximation - fine at this bbox's scale). The forecasted time
to reach the 12N line uses only the *meridional* (northward) component of
that velocity, not the full translation speed - eddies in this region often
propagate mostly westward (Rossby wave behavior), so distance / total-speed
would overstate how soon they actually reach a fixed latitude line. An eddy
presently moving south, or not moving north at all, is reported as
"not_approaching" rather than a misleading ETA.

Platform distances come in two flavors per platform, both from the same
track CSV SPICE_CMEMS_SAT.py/eddy_trajectory_plot.py write/read:
  - dist_<platform>_rt_km: distance to the platform's true real-time latest
    reported position, regardless of how old the eddy data is. Answers "how
    far is this eddy from where the platform actually is right now."
  - dist_<platform>_lag_km: distance to the platform's position as of the
    eddy data's OWN (often ~2-week-lagged) date - the exact same cutoff
    windowing eddy_trajectory_plot.py's map marker uses
    (pd.Timestamp(plot_date.date()) + RUN_TS). Answers "how far was this
    eddy from the platform on the day this eddy data is dated" - this is
    the number that matches what you'd eyeball off the corresponding map.
Only platforms with "enabled": True in the PLATFORMS list below get either
distance field at all (mirrors the same on/off toggle SPICE_CMEMS_SAT.py/
eddy_trajectory_plot.py use for map overlays - e.g. Falkor (too) is off
until the official cruise starts).
"""
import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr
import yaml

_configdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
with open(os.path.join(_configdir, 'eddy_12N_forecast.yml')) as _f:
    _CFG = yaml.safe_load(_f)

BBOX = _CFG['bbox']
TARGET_LAT = _CFG['target_lat']
KM_PER_DEG_LAT = _CFG['km_per_deg_lat']
COLUMNS_CFG = _CFG.get('columns') or []

POLARITIES = ("anticyclonic", "cyclonic")

# Mirrors SPICE_CMEMS_SAT.py's/eddy_trajectory_plot.py's PLATFORMS list
# (name + track CSV + enabled flag - no marker/color needed here, this
# isn't a map). Duplicated rather than imported for the same reason noted
# in eddy_trajectory_plot.py: those scripts aren't safely importable
# modules. Keep "enabled" in sync by hand with those two scripts - it's the
# same knob (e.g. Falkor (too) is off until the official cruise starts,
# see project memory). A disabled platform's distance fields are left out
# of AVAILABLE_FIELDS entirely, so if configs/eddy_12N_forecast.yml's
# `columns` list still names one, it's skipped with a warning (same as any
# unrecognized column) rather than reporting a distance nobody's vouching
# for - and it starts working again automatically the moment this list's
# "enabled" is flipped back to True, no yml edit needed.
PLATFORMS = [
    {"name": "ru29", "csv": "ru29_latest_track.csv", "enabled": True},
    {"name": "Falkor (too)", "csv": "falkor_track.csv", "enabled": False},
]
ACTIVE_PLATFORMS = [p for p in PLATFORMS if p.get("enabled", True)]


def platform_field_key(name, suffix):
    """"ru29","rt" -> "dist_ru29_rt_km", "Falkor (too)","lag" ->
    "dist_falkor_lag_km" - the parenthetical is dropped, it's redundant in
    a field/column name."""
    return f"dist_{name.split()[0].lower()}_{suffix}_km"


# Registry of every field this script knows how to compute: key -> (header
# label, column width, decimal places or None for str/int-like values).
# This is what configs/eddy_12N_forecast.yml's `columns` list selects from -
# a PI can add any of these to that list to show it, with no code change.
# Platform distance fields are appended below since they depend on
# ACTIVE_PLATFORMS.
AVAILABLE_FIELDS = {
    "eddy_id":               ("ID", 9, None),
    "polarity":              ("POLARITY", 13, None),
    "latitude":              ("LAT", 9, 3),
    "longitude":             ("LON", 10, 3),
    "last_obs_date":         ("LAST_OBS_DATE", 16, None),
    "radius_km":             ("RADIUS_KM", 11, 2),
    "effective_radius_km":   ("EFF_RADIUS_KM", 15, 2),
    "amplitude_m":           ("AMPLITUDE_M", 13, 2),
    "rotational_speed_m_s":  ("ROT_SPD_M/S", 14, 4),
    "effective_area_km2":    ("EFF_AREA_KM2", 14, 1),
    "translation_speed_m_s": ("TRANS_SPD_M/S", 15, 4),
    "meridional_speed_m_s":  ("MERID_SPD_M/S", 15, 4),
    "distance_to_12N_km":    ("DIST_12N_KM", 13, 2),
    "eta_to_12N_hours":      ("ETA_12N_HR", 12, 1),
    "eta_to_12N_days":       ("ETA_12N_DAYS", 14, 2),
    "status":                ("STATUS", 22, None),
}
for _p in ACTIVE_PLATFORMS:
    for _suffix in ("rt", "lag"):
        _key = platform_field_key(_p["name"], _suffix)
        _label = f"DIST_{_p['name'].split()[0].upper()}_{_suffix.upper()}_KM"
        AVAILABLE_FIELDS[_key] = (_label, max(len(_label) + 2, 18), 1)


arg_parser = argparse.ArgumentParser(
    description='Summarize active eddies near the 12N survey line into a plain-text metrics table',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_parser.add_argument('-c', '--cmems_dir', dest='cmems_dir', type=str, default='./cmems_data',
                        help='Directory eddy_trajectory_download.py wrote its subset files to')
arg_parser.add_argument('-s', '--save_dir', dest='save_dir', type=str, default='./satellite_figs',
                        help='Base directory to write the metrics table to - same tree eddy_trajectory_plot.py '
                             'saves its maps into (<save_dir>/YYYY/MM/DD/eddy_trajectory/)')
args = arg_parser.parse_args()


def load_eddy_tracks(base_dir):
    """Reads the subset NetCDF files eddy_trajectory_download.py wrote (same
    files eddy_trajectory_plot.py reads) into one DataFrame per polarity,
    sorted by (track, time) - needed here, unlike the plot script, because
    translation speed is derived from consecutive rows of the same track.
    Pulls every scalar-per-observation AVISO field this script might report
    (see AVAILABLE_FIELDS) - cheap, and means adding one of them to the yml
    columns list never requires touching this function."""
    data = {}
    for polarity in POLARITIES:
        path = os.path.join(base_dir, "eddy_trajectory", f"eddy_{polarity}_latest.nc")
        if not os.path.exists(path):
            print(f"Warning: {path} not found - run eddy_trajectory_download.py first")
            continue
        ds = xr.open_dataset(path)
        df = pd.DataFrame({
            "track": ds["track"].values,
            "time": pd.to_datetime(ds["time"].values),
            "lat": ds["latitude"].values,
            "lon": np.where(ds["longitude"].values > 180, ds["longitude"].values - 360, ds["longitude"].values),
            "radius_m": ds["speed_radius"].values,
            "effective_radius_m": ds["effective_radius"].values,
            "amplitude_m": ds["amplitude"].values,
            "rotational_speed_m_s": ds["speed_average"].values,
            "effective_area_m2": ds["effective_area"].values,
        })
        data[polarity] = df.sort_values(["track", "time"]).reset_index(drop=True)
    return data


def load_platform_track(csv_name):
    """Returns a platform's full position history (time, lat, lon), sorted
    by time, or None if the CSV is missing/unreadable. Mirrors
    get_platform_track in SPICE_CMEMS_SAT.py/eddy_trajectory_plot.py -
    needed here (rather than just the latest row) so a lagged, date-matched
    position can be looked up, not just the true latest one."""
    try:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_name)
        df = pd.read_csv(csv_path)
        df["time"] = pd.to_datetime(df["time"])
        return df.sort_values("time").reset_index(drop=True)
    except Exception as e:
        print(f"Warning: could not load track from {csv_name}: {e}")
        return None


def latest_position(track_df):
    """(lat, lon, time) of a track's last row, or None if track_df is
    None/empty."""
    if track_df is None or len(track_df) == 0:
        return None
    last = track_df.iloc[-1]
    return float(last["lat"]), float(last["lon"]), last["time"]


def position_as_of(track_df, cutoff):
    """(lat, lon, time) of a track's last row at/before cutoff, or None if
    track_df is None or has no rows that early yet. cutoff must share
    track_df['time']'s tz-awareness (see the tz handling in __main__)."""
    if track_df is None or len(track_df) == 0:
        return None
    windowed = track_df[track_df["time"] <= cutoff]
    if len(windowed) == 0:
        return None
    last = windowed.iloc[-1]
    return float(last["lat"]), float(last["lon"]), last["time"]


def meridional_km(lat1, lat2):
    """North-positive distance in km between two latitudes."""
    return (lat2 - lat1) * KM_PER_DEG_LAT


def zonal_km(lon1, lon2, mean_lat):
    """East-positive distance in km between two longitudes at mean_lat."""
    return (lon2 - lon1) * KM_PER_DEG_LAT * np.cos(np.radians(mean_lat))


def flat_earth_km(lat1, lon1, lat2, lon2):
    """Straight-line distance in km between two points - same flat-earth
    approximation as meridional_km/zonal_km, fine at this bbox's scale."""
    d_merid = meridional_km(lat1, lat2)
    d_zonal = zonal_km(lon1, lon2, (lat1 + lat2) / 2)
    return float(np.hypot(d_merid, d_zonal))


def eddy_metrics(track_df, polarity, latest_obs, bbox, platform_positions_rt, platform_positions_lag):
    """Returns one metrics dict per active eddy (track) whose latest position
    falls inside bbox, keyed by every name in AVAILABLE_FIELDS (computed
    regardless of what's actually selected in the yml columns list - cheap,
    and keeps format_table/AVAILABLE_FIELDS as the single source of truth
    for what can be shown). track_df must already be sorted by (track,
    time) - see load_eddy_tracks. platform_positions_rt/_lag are
    {name: (lat, lon, time) or None}, see the module docstring for the
    real-time vs. lagged distinction."""
    lon_min, lon_max = bbox['lon_min'], bbox['lon_max']
    lat_min, lat_max = bbox['lat_min'], bbox['lat_max']
    results = []
    for track_id, grp in track_df.groupby("track"):
        last = grp.iloc[-1]
        if last["time"] != latest_obs:
            continue  # not active - this track's last fix is older than the data's latest day
        if not (lon_min <= last["lon"] <= lon_max and lat_min <= last["lat"] <= lat_max):
            continue

        translation_speed_m_s = None
        meridional_speed_m_s = None
        if len(grp) >= 2:
            prev = grp.iloc[-2]
            dt_hours = (last["time"] - prev["time"]) / pd.Timedelta(hours=1)
            if dt_hours > 0:
                d_merid_km = meridional_km(prev["lat"], last["lat"])
                d_zonal_km = zonal_km(prev["lon"], last["lon"], (prev["lat"] + last["lat"]) / 2)
                translation_speed_m_s = (np.hypot(d_merid_km, d_zonal_km) * 1000) / (dt_hours * 3600)
                meridional_speed_m_s = (d_merid_km * 1000) / (dt_hours * 3600)

        distance_to_12N_km = meridional_km(last["lat"], TARGET_LAT)

        if distance_to_12N_km <= 0:
            status = "at_or_north_of_target"
            eta_hours = 0.0
        elif meridional_speed_m_s is None:
            status = "insufficient_history"
            eta_hours = None
        elif meridional_speed_m_s <= 0:
            status = "not_approaching"
            eta_hours = None
        else:
            status = "approaching"
            eta_hours = (distance_to_12N_km * 1000) / meridional_speed_m_s / 3600

        row = {
            "eddy_id": int(track_id),
            "polarity": polarity,
            "latitude": round(float(last["lat"]), 3),
            "longitude": round(float(last["lon"]), 3),
            "last_obs_date": last["time"].strftime("%Y-%m-%d"),
            "radius_km": round(float(last["radius_m"]) / 1000, 2),
            "effective_radius_km": round(float(last["effective_radius_m"]) / 1000, 2),
            "amplitude_m": round(float(last["amplitude_m"]), 2),
            "rotational_speed_m_s": round(float(last["rotational_speed_m_s"]), 4),
            "effective_area_km2": round(float(last["effective_area_m2"]) / 1e6, 1),
            "translation_speed_m_s": round(translation_speed_m_s, 4) if translation_speed_m_s is not None else None,
            "meridional_speed_m_s": round(meridional_speed_m_s, 4) if meridional_speed_m_s is not None else None,
            "distance_to_12N_km": round(distance_to_12N_km, 2),
            "eta_to_12N_hours": round(eta_hours, 1) if eta_hours is not None else None,
            "eta_to_12N_days": round(eta_hours / 24, 2) if eta_hours is not None else None,
            "status": status,
        }
        for name, pos in platform_positions_rt.items():
            row[platform_field_key(name, "rt")] = (
                round(flat_earth_km(last["lat"], last["lon"], pos[0], pos[1]), 1) if pos is not None else None
            )
        for name, pos in platform_positions_lag.items():
            row[platform_field_key(name, "lag")] = (
                round(flat_earth_km(last["lat"], last["lon"], pos[0], pos[1]), 1) if pos is not None else None
            )
        results.append(row)
    return results


def format_table(eddies, columns_cfg):
    """Fixed-width columns/rows text table - readable straight out of a
    plain text editor, no parsing needed. columns_cfg is the ordered list
    of field names from configs/eddy_12N_forecast.yml; unrecognized names
    are skipped with a warning instead of crashing the whole report (config
    file is a PI-editable boundary, not trusted to always be a valid key)."""
    columns = []
    for key in columns_cfg:
        if key not in AVAILABLE_FIELDS:
            print(f"Warning: unknown column '{key}' in configs/eddy_12N_forecast.yml columns list - skipping. "
                  f"Available: {sorted(AVAILABLE_FIELDS)}")
            continue
        columns.append(key)
    if not columns:
        raise RuntimeError("No valid columns configured in configs/eddy_12N_forecast.yml's `columns` list")

    def cell(value, width, decimals):
        if value is None:
            s = "n/a"
        elif decimals is not None:
            s = f"{value:.{decimals}f}"
        else:
            s = str(value)
        return s.ljust(width)

    header = "".join(AVAILABLE_FIELDS[key][0].ljust(AVAILABLE_FIELDS[key][1]) for key in columns)
    rows = [header, "-" * len(header)]
    for e in eddies:
        row = "".join(
            cell(e.get(key), AVAILABLE_FIELDS[key][1], AVAILABLE_FIELDS[key][2])
            for key in columns
        )
        rows.append(row)
    return "\n".join(rows)


if __name__ == "__main__":
    run_ts = os.environ.get("RUN_TS", pd.Timestamp.now(tz="UTC").strftime("%H%M"))
    eddy_tracks = load_eddy_tracks(args.cmems_dir)
    nonempty = [df for df in eddy_tracks.values() if len(df)]
    if not nonempty:
        print("No eddy data available - nothing to summarize (run eddy_trajectory_download.py first)")
    else:
        # Same "most recent day we have any data for" convention as
        # eddy_trajectory_plot.py's is_active - joint across both polarities,
        # not per-file, so both files agree on what counts as "current".
        latest_obs = max(df["time"].max() for df in nonempty)

        platform_tracks = {p["name"]: load_platform_track(p["csv"]) for p in ACTIVE_PLATFORMS}
        platform_positions_rt = {name: latest_position(track) for name, track in platform_tracks.items()}

        # Same cutoff eddy_trajectory_plot.py's map marker uses
        # (pd.Timestamp(plot_date.date()) + RUN_TS) - this is what makes
        # dist_<platform>_lag_km match a distance eyeballed off that map.
        try:
            run_h, run_m = (int(run_ts[:2]), int(run_ts[2:])) if len(run_ts) >= 4 else (0, 0)
        except (ValueError, TypeError):
            run_h, run_m = 0, 0
        lag_cutoff_naive = pd.Timestamp(latest_obs.date()) + pd.Timedelta(hours=run_h, minutes=run_m)
        platform_positions_lag = {}
        for name, track in platform_tracks.items():
            cutoff = lag_cutoff_naive
            if track is not None and len(track) and track["time"].dt.tz is not None:
                cutoff = lag_cutoff_naive.tz_localize("UTC")
            platform_positions_lag[name] = position_as_of(track, cutoff)

        for name in platform_tracks:
            rt, lag = platform_positions_rt[name], platform_positions_lag[name]
            print(f"{name} position rt={rt if rt else 'not available'} lag(as of {latest_obs:%Y-%m-%d})={lag if lag else 'not available'}")

        # Safety net, not a confirmed root-cause fix: a track cannot
        # rotate both ways at once, so the same eddy_id showing up as
        # active in both polarities is never legitimate. A run on
        # 2026-07-29 briefly showed this (cyclonic data pixel-identical to
        # anticyclonic) but a later run on the same underlying download
        # came back clean with no overlap - so that first result may have
        # been a local read glitch rather than a real AVISO/download bug
        # (see feedback_netcdf_testing memory for a confirmed instance of
        # the same kind of false alarm). Keeping this check anyway since
        # it's cheap and correct either way: if it ever fires for real, the
        # anticyclonic copy is kept and dropped ids are reported below.
        metrics_by_polarity = {
            polarity: eddy_metrics(df, polarity, latest_obs, BBOX, platform_positions_rt, platform_positions_lag)
            for polarity, df in eddy_tracks.items()
        }
        anti_ids = {m["eddy_id"] for m in metrics_by_polarity.get("anticyclonic", [])}
        cyc_ids = {m["eddy_id"] for m in metrics_by_polarity.get("cyclonic", [])}
        duplicate_ids = sorted(anti_ids & cyc_ids)
        if duplicate_ids:
            print(f"WARNING: {len(duplicate_ids)} eddy id(s) appear as both anticyclonic AND "
                  f"cyclonic ({duplicate_ids}) - a track cannot be both, so this is a data "
                  f"problem, not real eddies. Dropping the cyclonic duplicate(s); worth "
                  f"re-running the fetch/investigating if this keeps happening.")
            metrics_by_polarity["cyclonic"] = [
                m for m in metrics_by_polarity.get("cyclonic", []) if m["eddy_id"] not in duplicate_ids
            ]

        eddies = [m for metrics in metrics_by_polarity.values() for m in metrics]
        eddies.sort(key=lambda e: (e["eta_to_12N_hours"] is None, e["eta_to_12N_hours"]))

        # Same dated tree eddy_trajectory_plot.py saves its maps into
        # (<save_dir>/YYYY/MM/DD/eddy_trajectory/), dated by the eddy data's
        # own latest observation - not wall-clock "now" - so this table
        # lands right next to the map it corresponds to. Unlike the png
        # (one per RUN_TS slot), this file has no RUN_TS in its name and
        # gets overwritten in place on every 3-hourly run within the same
        # eddy-data day - same "overwrite the one current file" convention
        # as eddy_trajectory_download.py's sla_background.nc. The eddy-
        # derived fields (radius, speeds, ETA, distance_to_12N) are
        # daily-resolution and won't actually change between runs, but
        # dist_<platform>_rt_km AND dist_<platform>_lag_km both can:
        # _rt always reflects the platform's true current position, and
        # _lag's cutoff advances with RUN_TS through the eddy-data day just
        # like eddy_trajectory_plot.py's own map-marker cutoff does, so
        # both are worth recomputing/overwriting each cron cycle, not just
        # _rt. A new eddy-data day naturally lands in a new dated folder,
        # so nothing here is ever clobbered across days, only within one.
        out_dir = os.path.join(args.save_dir, f"{latest_obs:%Y}", f"{latest_obs:%m}", f"{latest_obs:%d}", "eddy_trajectory")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "eddy_active_12N_forecast.txt")

        header_lines = [
            f"Generated: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Eddy data date: {latest_obs:%Y-%m-%d}",
            f"Target latitude (12N survey line): {TARGET_LAT}",
            f"Bbox: lon [{BBOX['lon_min']}, {BBOX['lon_max']}], lat [{BBOX['lat_min']}, {BBOX['lat_max']}]",
        ]
        if duplicate_ids:
            header_lines.append(f"WARNING: dropped {len(duplicate_ids)} duplicate cyclonic id(s): {duplicate_ids}")
        header_lines.append("")

        table_text = "\n".join(header_lines) + format_table(eddies, COLUMNS_CFG) + "\n"
        # Write to a temp file and atomically replace, rather than writing
        # out_path directly - same pattern as download_eddy_file's .part
        # file + os.replace elsewhere in this pipeline. This is the one
        # file this script overwrites in place every 3-hourly run (see the
        # out_dir/out_path comment above) - a crash mid-write must never
        # leave a truncated/corrupt file in place of the last good one.
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w") as f:
            f.write(table_text)
        os.replace(tmp_path, out_path)
        print(f"{len(eddies)} active eddies in bbox, wrote {out_path}")