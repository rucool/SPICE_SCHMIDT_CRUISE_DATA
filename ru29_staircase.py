#!/usr/bin/env python
import argparse
import yaml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from erddapy import ERDDAP
import gsw
import seawater
import datetime as dt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean.cm as cmo
import os
from itertools import cycle
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from thermohalinesteps.detect_staircases import classify_staircase, identify_staircases_from_layers
from tqdm import tqdm
# import geopandas as gpd
import warnings


# MONKEY PATCH (Crucial for legacy append support) ---
if not hasattr(pd.DataFrame, 'append'):
    def _append(self, other, ignore_index=False, verify_integrity=False, sort=False):
        if isinstance(other, dict):
            other = pd.DataFrame([other])
        elif isinstance(other, list) and len(other) > 0 and isinstance(other[0], dict):
            other = pd.DataFrame(other)
        return pd.concat([self, other], ignore_index=ignore_index, 
                         verify_integrity=verify_integrity, sort=sort)
    pd.DataFrame.append = _append


arg_parser = argparse.ArgumentParser(description='Detect thermohaline staircases in ru29 glider profiles and save hovmoller figures (zonal variant: plots vs longitude instead of distance-along-track when the glider is within the fixed zonal survey latitude band)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_parser.add_argument('-s', '--save_dir',
                        dest='save_dir',
                        type=str,
                        default='./satellite_figs',
                        help='Full file path to save directory for figures')
args = arg_parser.parse_args()

# Tunable numeric limits (colorbar ranges, station reach) live in this
# config file so they can be adjusted without editing python - see
# configs/staircase_vars.yml.
_configdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
with open(os.path.join(_configdir, 'staircase_vars.yml')) as _f:
    PLOT_VARS_CFG = yaml.safe_load(_f)


def get_erddap_dataset(ds_id, server, variables=None, constraints=None, filetype=None):
    ## Written by Mike Smith
    """
    Returns a netcdf dataset for a specified dataset ID (or dataframe if dataset cannot be converted to xarray)
    :param ds_id: dataset ID e.g. ng314-20200806T2040
    :param variables: optional list of variables
    :param constraints: optional list of constraints
    :param filetype: optional filetype to return, 'nc' (default) or 'dataframe'
    :return: netcdf dataset
    """
    variables = variables or None
    constraints = constraints or None
    filetype = filetype or 'nc'
    #ioos_url = 'https://data.ioos.us/gliders/erddap'


    e = ERDDAP(server,
               protocol='tabledap',
               response='nc')
    e.dataset_id = ds_id
    if constraints:
        e.constraints = constraints
    if variables:
        e.variables = variables
    if filetype == 'nc':
        try:
            ds = e.to_xarray()
            ds = ds.sortby(ds.time)
        except OSError:
            print('No dataset available for specified constraints: {}'.format(ds_id))
            ds = []
        except TypeError:
            print('Cannot convert to xarray, providing dataframe: {}'.format(ds_id))
            ds = e.to_pandas().dropna()
    elif filetype == 'dataframe':
        #ds = e.to_pandas().dropna()
        ds = e.to_pandas().dropna(how='all')
    else:
        print('Unrecognized filetype: {}. Needs to  be "nc" or "dataframe"'.format(filetype))

    return ds

ds_id = 'ru29-20260623T2102-profile-sci-rt'
# ds_id = 'ru29-20250715T1838-profile-sci-delayed'


## Load flight data
variables = ['time','profile_time','profile_id','depth', 'latitude', 'longitude', 'salinity','temperature','pressure']
gdf = get_erddap_dataset(ds_id, server='http://slocum-data.marine.rutgers.edu/erddap', variables = variables, filetype='dataframe')
print(f"ERDDAP returned shape={gdf.shape}, columns={list(gdf.columns)}")
if len(gdf.columns) != len(variables):
    raise RuntimeError(
        f"ERDDAP returned {len(gdf.columns)} columns {list(gdf.columns)} but "
        f"expected {len(variables)} matching {variables}. erddapy's output shape "
        f"changed (likely a version difference) - fix the 'variables' list or "
        f"erddapy pin before rerunning, rather than silently mislabeling columns."
    )
gdf.columns = variables
gdf=gdf.rename(columns={'latitude':'lat','longitude':'lon'})
gdf['time']=pd.to_datetime(gdf.time)
gdf['profile_time']=pd.to_datetime(gdf.profile_time)
gdf=gdf.set_index('time')
print('RU29 data retrieved')


def convert_per_profile(group):
    group = group.copy()
    group['absolute_salinity'] = gsw.SA_from_SP(
        group.salinity, group.pressure, group.lon, group.lat
    )
    group['conservative_temperature'] = gsw.CT_from_t(
        group.absolute_salinity, group.temperature, group.pressure
    )
    return group

# NOTE: pandas >=3.0 silently drops the grouping column (profile_id) from what
# gets passed into/returned by the applied function (the old include_groups=True
# behavior was removed, not just deprecated - there's no flag left to restore
# it). Selecting gdf.columns explicitly before .apply() keeps profile_id in
# scope on both pandas 2.2.x and 3.0.x, so this is safe across the version bump.
gdf = gdf.groupby('profile_id', group_keys=False)[gdf.columns].apply(convert_per_profile)

# --- 1. SORT ONCE ---
print("Sorting data globally...")
gdf_sorted = gdf.reset_index().sort_values(by=['profile_id', 'pressure'])

# Cut off at TARGET_DATE + RUN_TS when backfilling a past day, or at today +
# RUN_TS otherwise - always applied, not just in backfill mode, so batch-
# generating multiple 'today' slots at once (e.g. cleanup_and_rerun.sh) doesn't
# silently give every slot the same full/current data. If RUN_TS isn't set at
# all (true live cron with no RUN_TS export), fall back to the actual current
# UTC time, which is a no-op filter since the data can't be from the future.
_target = os.environ.get("TARGET_DATE", "")
_run_ts_env = os.environ.get("RUN_TS", "")
if _run_ts_env:
    _h, _m = int(_run_ts_env[:2]), int(_run_ts_env[2:])
else:
    _now = pd.Timestamp.now(tz="UTC")
    _h, _m = _now.hour, _now.minute
_base_date = pd.Timestamp(_target, tz="UTC") if _target else pd.Timestamp.now(tz="UTC").normalize()
_cutoff = _base_date + pd.Timedelta(hours=_h, minutes=_m)
gdf_sorted = gdf_sorted[pd.to_datetime(gdf_sorted["profile_time"]) <= _cutoff]
print(f"{'Backfill' if _target else 'Live'} mode: {len(gdf_sorted['profile_id'].unique())} profiles up to {_cutoff}")


# --- 2. PROCESS EACH PROFILE ---
def process_profile(name, group):
    group = group.drop_duplicates(subset='pressure')

    if np.isinf(group[['pressure', 'conservative_temperature', 'absolute_salinity']]).any().any():
        group = group.replace([np.inf, -np.inf], np.nan)

    group = group.dropna(subset=['pressure', 'conservative_temperature', 'absolute_salinity'])
    group = group.sort_values('pressure')

    # skip profiles too shallow to contain staircases
    if len(group) < 5 or group.pressure.max() < 50:
        return None

    pid = group['profile_id'].iloc[0]
    pt  = group['profile_time'].iloc[0]

    # regrid to 1 dbar even spacing (required by classify_staircase)
    p_min = np.ceil(group.pressure.min())
    p_max = np.floor(group.pressure.max())
    p_reg = np.arange(p_min, p_max + 1, 1.0)

    if len(p_reg) < 5:
        return None

    ct_reg = np.interp(p_reg, group.pressure.values, group.conservative_temperature.values)
    sa_reg = np.interp(p_reg, group.pressure.values, group.absolute_salinity.values)

    try:
        df_out, mixes, grads = classify_staircase(
            p_reg,
            ct_reg,
            sa_reg,
            temp_flag_only=True,
            show_steps=False
        )

        if df_out is None or len(df_out) == 0:
            return None

        if not isinstance(df_out, pd.DataFrame):
            df_out = pd.DataFrame(df_out)
        df_out = df_out.copy()
        df_out['profile_id']   = pid
        df_out['profile_time'] = pt

        mixes_df = grads_df = None

        if mixes is not None:
            if not isinstance(mixes, pd.DataFrame):
                mixes = pd.DataFrame(mixes)
            mixes_df = mixes.copy()
            mixes_df['profile_id']   = pid
            mixes_df['profile_time'] = pt

        if grads is not None:
            if not isinstance(grads, pd.DataFrame):
                grads = pd.DataFrame(grads)
            grads_df = grads.copy()
            grads_df['profile_id']   = pid
            grads_df['profile_time'] = pt

        if mixes_df is None or grads_df is None:
            return df_out, mixes_df, grads_df, None, None

        staircase_list, ct_list = identify_staircases_from_layers(
            df=df_out.copy(),
            df_mixed_layers=mixes_df.copy(),
            df_gradient_layers=grads_df.copy(),
            max_allowable_gap=1,
            show_plot=False
        )

        stair_stats_profile = []
        stair_ct_profile    = []

        for i, st_df in enumerate(staircase_list, start=1):
            tmp = st_df.copy()
            tmp['profile_id']   = pid
            tmp['profile_time'] = pt
            tmp['staircase_id'] = i
            stair_stats_profile.append(tmp)

        for i, ct_df in enumerate(ct_list, start=1):
            tmp = ct_df.copy()
            tmp['profile_id']   = pid
            tmp['profile_time'] = pt
            tmp['staircase_id'] = i
            stair_ct_profile.append(tmp)

        stair_stats_df = (
            pd.concat(stair_stats_profile, ignore_index=True) if stair_stats_profile else None
        )
        stairs_ct_df = (
            pd.concat(stair_ct_profile, ignore_index=True) if stair_ct_profile else None
        )

        return df_out, mixes_df, grads_df, stair_stats_df, stairs_ct_df

    except Exception:
        return None


# --- 3. MAIN LOOP ---
print("Grouping data...")
grouped = gdf_sorted.groupby('profile_id')

df_out_all      = []
mixes_all       = []
grads_all       = []
stair_stats_all = []
staircases_ct_all = []

print("Starting processing...")

for name, group in tqdm(grouped, total=len(grouped)):
    res = process_profile(name, group)
    if res is None:
        continue

    df_out, mixes_df, grads_df, stair_stats_df, stairs_ct_df = res

    if df_out is not None:
        df_out_all.append(df_out)
    if mixes_df is not None and not mixes_df.empty:
        mixes_all.append(mixes_df)
    if grads_df is not None and not grads_df.empty:
        grads_all.append(grads_df)
    if stair_stats_df is not None and not stair_stats_df.empty:
        stair_stats_all.append(stair_stats_df)
    if stairs_ct_df is not None and not stairs_ct_df.empty:
        staircases_ct_all.append(stairs_ct_df)

print(f"Done. Profiles with results: {len(df_out_all)}")

# --- DIAGNOSTIC: find a deep profile and test classify_staircase with regridding ---
gdf_sorted = gdf.reset_index().sort_values(by=['profile_id', 'pressure'])

# Same always-applied cutoff as above (see comment there) - kept in sync
# since this diagnostic section rebuilds gdf_sorted from scratch.
_target = os.environ.get("TARGET_DATE", "")
_run_ts_env = os.environ.get("RUN_TS", "")
if _run_ts_env:
    _h, _m = int(_run_ts_env[:2]), int(_run_ts_env[2:])
else:
    _now = pd.Timestamp.now(tz="UTC")
    _h, _m = _now.hour, _now.minute
_base_date = pd.Timestamp(_target, tz="UTC") if _target else pd.Timestamp.now(tz="UTC").normalize()
_cutoff = _base_date + pd.Timedelta(hours=_h, minutes=_m)
gdf_sorted = gdf_sorted[pd.to_datetime(gdf_sorted["profile_time"]) <= _cutoff]
print(f"{'Backfill' if _target else 'Live'} mode: {len(gdf_sorted['profile_id'].unique())} profiles up to {_cutoff}")

# pick the deepest profile available
max_p_per_profile = gdf_sorted.groupby('profile_id')['pressure'].max()
test_pid = max_p_per_profile.idxmax()
test_grp = gdf_sorted[gdf_sorted['profile_id'] == test_pid].copy()
test_grp = test_grp.drop_duplicates(subset='pressure')
test_grp = test_grp.dropna(subset=['pressure', 'conservative_temperature', 'absolute_salinity'])
test_grp = test_grp.sort_values('pressure')

print(f"profile_id: {test_pid}")
print(f"  n obs:    {len(test_grp)}")
print(f"  pressure: {test_grp.pressure.min():.1f} - {test_grp.pressure.max():.1f} dbar")
print(f"  CT:       {test_grp.conservative_temperature.min():.3f} - {test_grp.conservative_temperature.max():.3f} °C")
print(f"  SA:       {test_grp.absolute_salinity.min():.3f} - {test_grp.absolute_salinity.max():.3f} g/kg")

# regrid to 1 dbar
p_min = np.ceil(test_grp.pressure.min())
p_max = np.floor(test_grp.pressure.max())
p_reg = np.arange(p_min, p_max + 1, 1.0)
ct_reg = np.interp(p_reg, test_grp.pressure.values, test_grp.conservative_temperature.values)
sa_reg = np.interp(p_reg, test_grp.pressure.values, test_grp.absolute_salinity.values)
print(f"\nRegridded to {len(p_reg)} levels ({p_reg[0]:.0f} - {p_reg[-1]:.0f} dbar, 1 dbar spacing)")

try:
    df_out, mixes, grads = classify_staircase(
        p_reg, ct_reg, sa_reg,
        temp_flag_only=True,
        show_steps=False
    )
    print(f"\nclassify_staircase: df_out has {len(df_out) if df_out is not None else 0} rows")
    print(f"  mixes: {len(mixes) if mixes is not None else 0} rows")
    print(f"  grads: {len(grads) if grads is not None else 0} rows")
    if mixes is not None:
        print(mixes.head())
except Exception:
    import traceback
    print("\nERROR in classify_staircase:")
    traceback.print_exc()


print("Saving outputs...")

if df_out_all:
    pd.concat(df_out_all, ignore_index=True).to_csv(f"{ds_id}_staircase_results.csv", index=False)

if mixes_all:
    pd.concat(mixes_all, ignore_index=True).to_csv(f"{ds_id}_mixes.csv", index=False)

if grads_all:
    pd.concat(grads_all, ignore_index=True).to_csv(f"{ds_id}_grads.csv", index=False)

if stair_stats_all:
    pd.concat(stair_stats_all, ignore_index=True).to_csv(f"{ds_id}_staircase_layer_stats.csv", index=False)

if staircases_ct_all:
    pd.concat(staircases_ct_all, ignore_index=True).to_csv(f"{ds_id}_staircases_ct.csv", index=False)

print("Done.")


# --- load results (works from memory or saved CSVs) ---
if 'stair_stats_all' in dir() and stair_stats_all:
    df_ls = pd.concat(stair_stats_all, ignore_index=True)
    df_mixes = pd.concat(mixes_all, ignore_index=True) if mixes_all else pd.DataFrame()
else:
    df_ls    = pd.read_csv(f"{ds_id}_staircase_layer_stats.csv")
    df_mixes = pd.read_csv(f"{ds_id}_mixes.csv")

# convert boolean columns that may have been read as strings from CSV
for col in ['mixed_layer', 'gradient_layer']:
    if col in df_ls.columns:
        df_ls[col] = df_ls[col].astype(bool)

# --- one lat/lon/time per profile, sorted chronologically ---
prof_coords = (
    gdf_sorted
       .groupby('profile_id', sort=False)
       .agg(lat=('lat', 'first'), lon=('lon', 'first'),
            profile_time=('profile_time', 'first'))
       .reset_index()
       .sort_values('profile_time')
       .reset_index(drop=True)
)

# cumulative along-track distance (km)
lons = prof_coords.lon.values
lats = prof_coords.lat.values
dists = np.zeros(len(lons))
for i in range(1, len(lons)):
    dists[i] = dists[i-1] + gsw.distance([lons[i-1], lons[i]], [lats[i-1], lats[i]])[0] / 1000.0
prof_coords['dist_km'] = dists
dist_map = prof_coords.set_index('profile_id')['dist_km']
lon_map = prof_coords.set_index('profile_id')['lon']

# Zonal-mode check: if the glider's recent profiles sit within the fixed
# zonal survey latitude band, plot against longitude instead of cumulative
# distance along track - more physically meaningful when the glider is
# running an east-west line at ~constant latitude, rather than transiting
# toward it. Uses the median lat of the last few profiles (not just the
# single latest one) so one noisy profile that briefly meanders outside the
# band does not flip the whole run's plotting mode. prof_coords is already
# sorted by profile_time, so .tail(n) is the most recent profiles.
ZONAL_LAT_MIN, ZONAL_LAT_MAX = 11.8, 12.2
ZONAL_CHECK_N_PROFILES = 5
_recent_lat = prof_coords['lat'].tail(ZONAL_CHECK_N_PROFILES).median()
ZONAL_MODE = ZONAL_LAT_MIN <= _recent_lat <= ZONAL_LAT_MAX
if ZONAL_MODE:
    x_col, x_label = 'lon', 'Longitude (°)'
    print(f"Zonal mode: median lat of last {ZONAL_CHECK_N_PROFILES} profiles={_recent_lat:.3f} within [{ZONAL_LAT_MIN}, {ZONAL_LAT_MAX}] - plotting vs longitude")
else:
    x_col, x_label = 'dist_km', 'Distance along track (km)'
    print(f"Distance mode: median lat of last {ZONAL_CHECK_N_PROFILES} profiles={_recent_lat:.3f} outside zonal band [{ZONAL_LAT_MIN}, {ZONAL_LAT_MAX}] - plotting vs distance along track")

# Turnaround detection (only meaningful in ZONAL_MODE): once ru29 has run
# far enough in one direction along the 12N line and then reverses, the 7
# figures below split into two stacked panels (outbound above, return
# below) instead of letting both legs overplot on the same longitude
# strip. Scoped to the FIRST turnaround only - repeat zonal surveys often
# do multiple round trips, but N-leg support is out of scope for now; a
# second reversal is flagged with a loud warning below rather than
# silently mixed into the return-leg panel.
def _detect_reversal(smoothed, direction):
    """Returns (extreme_idx, pullback) for a smoothed longitude series
    presumed to be traveling in `direction` (+1 = east/increasing lon,
    -1 = west/decreasing lon). `pullback` is how far the series has fallen
    back from its extreme in that direction - positive means it's pulling
    back (a candidate reversal), independent of which direction that is."""
    extreme_idx = smoothed.idxmax() if direction == 1 else smoothed.idxmin()
    pullback = direction * (smoothed.loc[extreme_idx] - smoothed.iloc[-1])
    return extreme_idx, pullback


TURNAROUND_MIN_REVERSAL_DEG = PLOT_VARS_CFG['turnaround']['min_reversal_deg']
TURNAROUND_DETECTED = False
turnaround_time = None
if ZONAL_MODE:
    _zonal = (prof_coords[(prof_coords['lat'] >= ZONAL_LAT_MIN) & (prof_coords['lat'] <= ZONAL_LAT_MAX)]
              .sort_values('profile_time').reset_index(drop=True))
    if len(_zonal) >= 2 * ZONAL_CHECK_N_PROFILES:
        # Same smoothing window as the ZONAL_MODE check above, for
        # consistency - not a new magic number.
        _smoothed = _zonal['lon'].rolling(ZONAL_CHECK_N_PROFILES, min_periods=1, center=True).median()
        _direction = 1 if _smoothed.iloc[-1] >= _smoothed.iloc[0] else -1  # overall travel so far: +1 east, -1 west
        _extreme_idx, _pullback = _detect_reversal(_smoothed, _direction)
        if _pullback >= TURNAROUND_MIN_REVERSAL_DEG:
            TURNAROUND_DETECTED = True
            turnaround_time = _zonal.loc[_extreme_idx, 'profile_time']
            print(f"Turnaround detected: extreme lon reached at {turnaround_time} (pulled back "
                  f"{_pullback:.3f}\u00b0 since, >= {TURNAROUND_MIN_REVERSAL_DEG}\u00b0 threshold) - "
                  f"splitting zonal figures into outbound/return panels")

            # Second-reversal check: same test applied to the profiles
            # AFTER the confirmed turnaround, now traveling in the
            # opposite (-_direction) sense. Only a warning - this script
            # doesn't attempt to split a third leg out.
            _post = _zonal[_zonal['profile_time'] > turnaround_time].reset_index(drop=True)
            if len(_post) >= 2 * ZONAL_CHECK_N_PROFILES:
                _post_smoothed = _post['lon'].rolling(ZONAL_CHECK_N_PROFILES, min_periods=1, center=True).median()
                _post_extreme_idx, _post_pullback = _detect_reversal(_post_smoothed, -_direction)
                if _post_pullback >= TURNAROUND_MIN_REVERSAL_DEG:
                    _second_turnaround_time = _post.loc[_post_extreme_idx, 'profile_time']
                    print(f"WARNING: a SECOND reversal was detected at {_second_turnaround_time} - "
                          f"this script only splits on the first turnaround, so the return/westbound "
                          f"panel below will mix both legs from that point on. Revisit N-leg support "
                          f"if this becomes a regular pattern.")

# profile_id -> bool map (True = outbound/before the turnaround) used to
# split every dataframe below into the two panels once TURNAROUND_DETECTED.
# All of gdf_dist/ml/df_ls/df_results/all_profs/profile_stats trace back to
# prof_coords via profile_id (same join key dist_map/lon_map already use
# above), so one map covers all of them. Profiles with no turnaround yet
# (or when TURNAROUND_DETECTED is False) all map to True/outbound, which is
# harmless since the split path is never taken in that case. Handled as an
# explicit branch rather than a `turnaround_time or <sentinel>` fallback -
# profile_time is tz-aware (UTC) here, and comparing it against a tz-naive
# sentinel like pd.Timestamp.max raises rather than just working.
_prof_by_id = prof_coords.set_index('profile_id')['profile_time']
if turnaround_time is not None:
    is_outbound = _prof_by_id <= turnaround_time
else:
    is_outbound = pd.Series(True, index=_prof_by_id.index)


def split_outbound_return(df, id_col='profile_id'):
    """Splits df into (outbound, return) using the is_outbound map above.
    Rows whose profile_id isn't in the map (shouldn't normally happen) are
    treated as outbound rather than silently dropped."""
    mask = df[id_col].map(is_outbound).fillna(True)
    return df.loc[mask].copy(), df.loc[~mask].copy()


if not _target:  # do not overwrite current track CSV during backfill
    prof_coords[["lat", "lon", "profile_time"]].rename(columns={"profile_time": "time"}).to_csv(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "ru29_latest_track.csv"), index=False)

# add distance to all dataframes
gdf_dist         = gdf_sorted.copy()
gdf_dist['dist_km'] = gdf_dist['profile_id'].map(dist_map)
df_ls['dist_km'] = df_ls['profile_id'].map(dist_map)
df_ls['lon'] = df_ls['profile_id'].map(lon_map)
if not df_mixes.empty:
    df_mixes['dist_km'] = df_mixes['profile_id'].map(dist_map)
    df_mixes['lon'] = df_mixes['profile_id'].map(lon_map)

print(f"Track length: {dists[-1]:.1f} km  |  {len(prof_coords)} profiles")
print(f"Staircase layers found: {len(df_ls)}  (mixed: {df_ls['mixed_layer'].sum()}, gradient: {df_ls['gradient_layer'].sum()})")

from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist as scipy_cdist

run_time = dt.datetime.utcnow()
FIG_BASE_DIR = args.save_dir
_plot_date = pd.Timestamp(_target) if _target else run_time
# Filename date stamp must track _plot_date (the day being backfilled), not
# run_time (today) - otherwise every backfilled day's files get stamped with
# today's date instead of the day they belong to.
run_ts = _plot_date.strftime("%Y%m%d_") + os.environ.get("RUN_TS", "") + "00" if os.environ.get("RUN_TS") else run_time.strftime("%Y%m%d_%H%M%S")
daily_dir = os.path.join(FIG_BASE_DIR, _plot_date.strftime("%Y"), _plot_date.strftime("%m"), _plot_date.strftime("%d"))

# Datetime shown above every plot title below: always the rounded RUN_TS
# boundary (backfill cutoff date + RUN_TS, or today's date + RUN_TS when
# live) rather than the actual wall-clock time this line executes at - the
# ERDDAP fetch + staircase detection above can take a couple minutes, so
# using raw run_time here made titles drift past the top of the hour under
# load instead of matching the RUN_TS-rounded filename.
if os.environ.get("RUN_TS"):
    _title_h, _title_m = int(os.environ.get("RUN_TS")[:2]), int(os.environ.get("RUN_TS")[2:])
    _title_base_date = pd.Timestamp(_target, tz="UTC") if _target else pd.Timestamp(run_time, tz="UTC").normalize()
    _title_dt = _title_base_date + pd.Timedelta(hours=_title_h, minutes=_title_m)
else:
    _title_dt = pd.Timestamp(run_time, tz="UTC")
title_datetime_str = _title_dt.strftime("%Y-%m-%d %H:%M") + " UTC"

ru29_plot_vars = ["CT", "ml_height", "turner", "sigma", "classification", "counts", "depth_range"]
for v in ru29_plot_vars:
    os.makedirs(os.path.join(daily_dir, f"ru29_{v}"), exist_ok=True)


# Station list lives in configs/cruise_stations.yml - edit that file directly
# to adjust positions/notes/argo/drifter flags, no code changes needed.
with open(os.path.join(_configdir, 'cruise_stations.yml')) as _f:
    stations = yaml.safe_load(_f)['stations']

STATION_REACH_KM = PLOT_VARS_CFG['station_reach_km']

# Per-station color cycles through tab10's 10 colors, combined with a marker
# shape that changes every 10 stations - e.g. stations 1 and 11 share a
# color but station 1 is a down-triangle and station 11 is an up-triangle,
# so the (color, shape) pair is unique across up to 60 stations. Shape
# doesn't depend on color vision at all, so this stays distinguishable for
# colorblind viewers without needing 57 individually-unique hues. Avoids '*'
# (Argo) and 'D' (drifter), which are reserved for those overlay markers.
tab10 = plt.cm.tab10
STATION_MARKER_SHAPES = ['v', '^', 's', 'p', 'h', 'o']
for i, s in enumerate(stations):
    s['color'] = tab10.colors[i % 10]
    s['marker'] = STATION_MARKER_SHAPES[(i // 10) % len(STATION_MARKER_SHAPES)]

ARGO_COLOR = 'gold'
DRIFTER_COLOR = 'lime'

stn_pts    = np.array([[s['lat'], s['lon']] for s in stations])
glider_pts = prof_coords[['lat', 'lon']].values
closest    = scipy_cdist(stn_pts, glider_pts).argmin(axis=1)

for s, idx in zip(stations, closest):
    prof = prof_coords.iloc[idx]
    dist_to_stn = gsw.distance([s['lon'], prof['lon']], [s['lat'], prof['lat']])[0] / 1000.0
    s['dist_km'] = float(prof['dist_km'])
    s['reached'] = dist_to_stn <= STATION_REACH_KM

reached = [s for s in stations if s['reached']]
print(f"Stations reached so far ({len(reached)}/{len(stations)}): "
      f"{[s['name'] for s in reached] or 'none yet'}")

if ZONAL_MODE:
    # Stations already have an explicit longitude - plot each at its true
    # position, no distance-along-track de-overlap offset needed.
    for s in stations:
        s['x_plot'] = s['lon']
else:
    for s in stations:
        s['x_plot'] = s['dist_km']

    spacing_km = 2.0
    reached_by_dist = sorted(reached, key=lambda s: s['dist_km'])
    i = 0
    while i < len(reached_by_dist):
        group = [reached_by_dist[i]]
        j = i + 1
        while j < len(reached_by_dist) and reached_by_dist[j]['dist_km'] - group[0]['dist_km'] < 1.0:
            group.append(reached_by_dist[j])
            j += 1
        if len(group) > 1:
            group = sorted(group, key=lambda s: s['lon'])
            offsets = np.arange(len(group)) * spacing_km - (len(group) - 1) * spacing_km / 2
            for s, off in zip(group, offsets):
                s['x_plot'] = s['dist_km'] + off
        i = j

# Legend built by walking `stations` in natural sequence order, so entries
# appear in station order. Each reached station - regular, argo, or
# drifter - gets its own individual entry with its own color; argo/drifter
# additionally get the star/diamond shape and a "Argo Deploy"/"Drifter
# Deploy" label prefix so the category is still obvious at a glance.
legend_handles = []
for s in stations:
    if not s['reached']:
        continue
    if s.get('argo'):
        legend_handles.append(
            Line2D([0],[0], marker='*', color='w', markerfacecolor=ARGO_COLOR,
                   markersize=18, markeredgecolor='k', markeredgewidth=0.5,
                   label=f"Argo ({s['name']})")
        )
    elif s.get('drifter'):
        legend_handles.append(
            Line2D([0],[0], marker='D', color='w', markerfacecolor=DRIFTER_COLOR,
                   markersize=14, markeredgecolor='k', markeredgewidth=0.8,
                   label=f"Drifter ({s['name']})")
        )
    else:
        legend_handles.append(
            Line2D([0],[0], marker=s['marker'], color='w', markerfacecolor=s['color'],
                   markersize=16, markeredgecolor='k', markeredgewidth=0.8, label=s['name'])
        )

ml = df_ls[df_ls['mixed_layer']].copy()

GRID_KW = dict(color='gray', alpha=0.2, linewidth=0.5, zorder=0)


def _legend_row_major_order(items, ncols):
    """matplotlib fills multi-column legends column-major (top-to-bottom
    within each column, then next column) - with a partial last row that
    reads confusingly out of sequence (e.g. items 2 and 4 hidden in a
    second row under columns 1-2 instead of appearing right after 1 and 3).
    This reorders `items` so that column-major filling produces normal
    left-to-right, top-to-bottom reading order instead."""
    n = len(items)
    if n == 0 or ncols <= 1:
        return items
    nrows = -(-n // ncols)  # ceil(n / ncols)
    extra = n - (nrows - 1) * ncols  # first `extra` columns get a full nrows
    col_heights = [nrows if c < extra else nrows - 1 for c in range(ncols)]
    # existing grid cells in reading order (row-major), skipping cells that
    # don't exist because their column is one row short
    cells = [(r, c) for r in range(nrows) for c in range(ncols) if r < col_heights[c]]
    cell_to_item = dict(zip(cells, items))
    # rebuild in matplotlib's actual column-major fill order
    return [cell_to_item[(r, c)] for c in range(ncols) for r in range(col_heights[c])]


def add_station_markers(axes, fig, extra_handles=None):
    """Triangles on x-axis spine for reached stations, drawn on every ax in
    `axes` (a single Axes, or a list/tuple of them for the split-panel
    case); one combined legend below the whole figure regardless of how
    many axes were drawn on."""
    if not isinstance(axes, (list, tuple)):
        axes = [axes]
    for ax in axes:
        xform = ax.get_xaxis_transform()
        for s in stations:
            if not s['reached']:
                continue
            if s.get('argo'):
                ax.plot(s['x_plot'], 0, marker='*',
                        color=ARGO_COLOR, markersize=16,
                        transform=xform, clip_on=False, zorder=6,
                        markeredgecolor='k', markeredgewidth=0.5)
            elif s.get('drifter'):
                ax.plot(s['x_plot'], 0, marker='D',
                        color=DRIFTER_COLOR, markersize=13,
                        transform=xform, clip_on=False, zorder=6,
                        markeredgecolor='k', markeredgewidth=0.5)
            else:
                ax.plot(s['x_plot'], 0, marker=s['marker'],
                        color=s['color'], markersize=14,
                        transform=xform, clip_on=False, zorder=6,
                        markeredgecolor='k', markeredgewidth=0.8)
    all_handles = (extra_handles or []) + legend_handles
    if all_handles:
        max_ncols = 10
        nrows = -(-len(all_handles) // max_ncols)  # ceil
        ncols = -(-len(all_handles) // nrows)  # smallest ncols that still fits in nrows - avoids a mostly-empty trailing row
        leg = fig.legend(handles=_legend_row_major_order(all_handles, ncols), loc='lower center',
                          bbox_to_anchor=(0.5, 0.0), ncol=ncols,
                          frameon=True, fontsize=10, handletextpad=0.3, columnspacing=1.0)

        # Grow the bottom margin so the x-axis label of the bottom-most axes
        # clears the legend - self-correcting via directly measured pixel
        # gap rather than a guessed constant (two earlier attempts at
        # guessing a fixed-fraction buffer both underestimated the real
        # buffer needed - a legend's top edge in figure-fraction terms is
        # NOT the same as "how much bottom margin clears it", since the
        # x-axis label itself needs its own full height below the axes
        # edge too, not just a small gap above the legend). This measures
        # ax.xaxis.label's actual bottom edge vs. the legend's actual top
        # edge in pixels and shifts `bottom` by exactly the deficit -
        # exact because the label's pixel position moves in lockstep with
        # `bottom` at a 1:1 rate (delta_bottom * fig_height_px), while the
        # legend's position doesn't move at all (bbox_to_anchor=(0.5, 0.0)
        # anchors it to figure y=0 regardless of the axes' bottom margin).
        # A couple of iterations lets it converge if the first shift was
        # slightly off. Uses the bottom-most axes (axes[-1]) since that's
        # the one whose label actually sits near the legend - only
        # relevant when axes has 2 entries (the turnaround-split case).
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        min_gap_px = 12
        for _ in range(3):
            xlabel_y0 = axes[-1].xaxis.label.get_window_extent(renderer).y0
            legend_y1 = leg.get_window_extent(renderer).y1
            gap_px = xlabel_y0 - legend_y1
            if gap_px >= min_gap_px:
                break
            fig_height_px = fig.get_size_inches()[1] * fig.dpi
            new_bottom = fig.subplotpars.bottom + (min_gap_px - gap_px) / fig_height_px
            fig.subplots_adjust(bottom=new_bottom)
            fig.canvas.draw()


# Figure 1: Conservative Temperature
def _plot_ct(ax, gdf_sub, ml_sub, vmin=None, vmax=None):
    sc = ax.scatter(
        gdf_sub[x_col], gdf_sub['pressure'],
        c=gdf_sub['conservative_temperature'],
        cmap=cmo.thermal, vmin=vmin, vmax=vmax, s=0.5, rasterized=True, zorder=1
    )
    for _, row in ml_sub.iterrows():
        ax.plot([row[x_col]] * 2, [row['p_start'], row['p_end']],
                color='white', lw=3.2, alpha=0.95, zorder=3,
                solid_capstyle='round', path_effects=[pe.Stroke(linewidth=4.4, foreground='black'), pe.Normal()])
    ax.invert_yaxis()
    ax.grid(True, **GRID_KW)
    ax.set_ylabel('Pressure (dbar)')
    ax.set_xlabel(x_label)
    return sc


if TURNAROUND_DETECTED:
    gdf_out, gdf_ret = split_outbound_return(gdf_dist)
    ml_out, ml_ret = split_outbound_return(ml)
    ct_vmin, ct_vmax = gdf_dist['conservative_temperature'].min(), gdf_dist['conservative_temperature'].max()
    fig, (ax_out, ax_ret) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.subplots_adjust(bottom=0.14, hspace=0.35)
    sc = _plot_ct(ax_out, gdf_out, ml_out, ct_vmin, ct_vmax)
    _plot_ct(ax_ret, gdf_ret, ml_ret, ct_vmin, ct_vmax)
    ax_out.set_title(f"{title_datetime_str}\nConservative Temperature - Eastbound leg  |  white/black = staircase mixed layers", loc='left')
    ax_ret.set_title("Westbound leg (return)", loc='left')
    add_station_markers([ax_out, ax_ret], fig)
    cb = fig.colorbar(sc, ax=[ax_out, ax_ret], pad=0.01)
    cb.set_label('CT (°C)')
else:
    fig, ax = plt.subplots(figsize=(16, 5))
    fig.subplots_adjust(bottom=0.22)
    sc = _plot_ct(ax, gdf_dist, ml)
    ax.set_title(f"{title_datetime_str}\nConservative Temperature  |  white/black = staircase mixed layers", loc='left')
    add_station_markers(ax, fig)
    cb = plt.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label('CT (°C)')
plt.gcf().canvas.draw()  # force full render before tight-bbox crop
plt.savefig(os.path.join(daily_dir, 'ru29_CT', f'ru29_CT_{run_ts}.png'), dpi=200, bbox_inches='tight')
plt.show()


#  Figure 2: Mixed-layer height 
def _plot_ml_height(ax, ml_sub):
    sc = None
    if not ml_sub.empty:
        sc = ax.scatter(ml_sub[x_col], ml_sub['p'], c=ml_sub['layer_height'],
                        cmap=cmo.matter, s=35, zorder=2, vmin=PLOT_VARS_CFG['ml_height']['vmin'], vmax=PLOT_VARS_CFG['ml_height']['vmax'],
                        edgecolors='k', linewidths=0.3)
    ax.invert_yaxis()
    ax.grid(True, **GRID_KW)
    ax.set_ylabel('Pressure (dbar)')
    ax.set_xlabel(x_label)
    return sc


if TURNAROUND_DETECTED:
    ml_out, ml_ret = split_outbound_return(ml)
    fig, (ax_out, ax_ret) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.subplots_adjust(bottom=0.14, hspace=0.35)
    sc_out = _plot_ml_height(ax_out, ml_out)
    sc_ret = _plot_ml_height(ax_ret, ml_ret)
    ax_out.set_title(f"{title_datetime_str}\nMixed-layer height - Eastbound leg", loc='left')
    ax_ret.set_title("Westbound leg (return)", loc='left')
    sc_for_cb = sc_out if sc_out is not None else sc_ret
    add_station_markers([ax_out, ax_ret], fig)
    if sc_for_cb is not None:
        cb = fig.colorbar(sc_for_cb, ax=[ax_out, ax_ret], pad=0.01, extend='max')
        cb.set_label('Layer height (dbar)')
else:
    fig, ax = plt.subplots(figsize=(16, 5))
    fig.subplots_adjust(bottom=0.22)
    sc = _plot_ml_height(ax, ml)
    ax.set_title(f"{title_datetime_str}\nMixed-layer height", loc='left')
    add_station_markers(ax, fig)
    if sc is not None:
        cb = plt.colorbar(sc, ax=ax, pad=0.01, extend='max')
        cb.set_label('Layer height (dbar)')
plt.gcf().canvas.draw()  # force full render before tight-bbox crop
plt.savefig(os.path.join(daily_dir, 'ru29_ml_height', f'ru29_ml_height_{run_ts}.png'), dpi=200, bbox_inches='tight')
plt.show()


# Figure 3: Turner angle 
def _plot_turner(ax, df_ls_sub):
    sc = None
    if not df_ls_sub.empty:
        sc = ax.scatter(df_ls_sub[x_col], df_ls_sub['p'], c=df_ls_sub['turner_ang'],
                        cmap='RdBu_r', vmin=PLOT_VARS_CFG['turner']['vmin'], vmax=PLOT_VARS_CFG['turner']['vmax'], s=35, zorder=2,
                        edgecolors='k', linewidths=0.3)
    ax.invert_yaxis()
    ax.grid(True, **GRID_KW)
    ax.set_ylabel('Pressure (dbar)')
    ax.set_xlabel(x_label)
    return sc


if TURNAROUND_DETECTED:
    df_ls_out, df_ls_ret = split_outbound_return(df_ls)
    fig, (ax_out, ax_ret) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.subplots_adjust(bottom=0.14, hspace=0.35)
    sc_out = _plot_turner(ax_out, df_ls_out)
    sc_ret = _plot_turner(ax_ret, df_ls_ret)
    # Pin both legs to the same pressure range, computed from the combined
    # (pre-split) data - same fix as the CT/sigma vmin/vmax above, applied
    # to the y-axis instead of the color axis. df_ls only has rows where a
    # staircase layer was actually detected (unlike CT/sigma, which plot
    # every pressure bin of every profile), so each leg's own min/max can
    # differ a lot - e.g. no near-surface layers on one leg - letting the
    # two panels' y-axes silently diverge even though the underlying
    # pressure range is otherwise comparable across legs.
    if not df_ls.empty:
        _p_min, _p_max = df_ls['p'].min(), df_ls['p'].max()
        _p_pad = (_p_max - _p_min) * 0.05
        ax_out.set_ylim(_p_max + _p_pad, _p_min - _p_pad)
        ax_ret.set_ylim(_p_max + _p_pad, _p_min - _p_pad)
    ax_out.set_title(f"{title_datetime_str}\nTurner angle - Eastbound leg  (red = salt fingering >45°, blue = diffusive convection <-45°)", loc='left')
    ax_ret.set_title("Westbound leg (return)", loc='left')
    sc_for_cb = sc_out if sc_out is not None else sc_ret
    add_station_markers([ax_out, ax_ret], fig)
    if sc_for_cb is not None:
        cb = fig.colorbar(sc_for_cb, ax=[ax_out, ax_ret], pad=0.01)
        cb.set_label('Turner angle (°)')
else:
    fig, ax = plt.subplots(figsize=(16, 5))
    fig.subplots_adjust(bottom=0.22)
    sc = _plot_turner(ax, df_ls)
    ax.set_title(f"{title_datetime_str}\nTurner angle  (red = salt fingering >45°, blue = diffusive convection <-45°)", loc='left')
    add_station_markers(ax, fig)
    if sc is not None:
        cb = plt.colorbar(sc, ax=ax, pad=0.01)
        cb.set_label('Turner angle (°)')
plt.gcf().canvas.draw()  # force full render before tight-bbox crop
plt.savefig(os.path.join(daily_dir, 'ru29_turner', f'ru29_turner_{run_ts}.png'), dpi=200, bbox_inches='tight')
plt.show()



from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

# Load df_results if not already in memory 
if 'df_results' not in dir() or df_results is None:
    df_results = pd.concat(df_out_all, ignore_index=True) if df_out_all else pd.read_csv(f"{ds_id}_staircase_results.csv")

df_results['dist_km'] = df_results['profile_id'].map(dist_map)
df_results['lon'] = df_results['profile_id'].map(lon_map)

#Layer classification column 
df_results['layer_type'] = 0  # background
df_results.loc[~df_results['mixed_layer_final_mask'].astype(bool),    'layer_type'] = 1  # mixed
df_results.loc[~df_results['gradient_layer_final_mask'].astype(bool), 'layer_type'] = 2  # gradient

#  Per-profile staircase stats (mirror of argo hexbin quantities)
profile_stats = (
    df_ls.groupby('profile_id')
    .agg(
        n_staircases  = ('staircase_id', 'nunique'),
        p_min         = ('p_start', 'min'),
        p_max         = ('p_end',   'max'),
        p_mean        = ('p',       'mean'),
        p_median      = ('p',       'median'),
    )
    .reset_index()
)
profile_stats['dist_km'] = profile_stats['profile_id'].map(dist_map)
profile_stats['lon'] = profile_stats['profile_id'].map(lon_map)

# -- Surface MLD per profile: temperature threshold (0.2 degC drop from 10 dbar ref) --
def _surface_mld(group, delta_T=0.2, ref_p=10.0):
    group = group.sort_values('pressure')
    ref = group[group['pressure'] >= ref_p]
    if ref.empty:
        return np.nan
    ct_ref = ref['conservative_temperature'].iloc[0]
    below = group[(group['pressure'] > ref_p) &
                  (group['conservative_temperature'] < ct_ref - delta_T)]
    return float(below['pressure'].iloc[0]) if not below.empty else float(group['pressure'].max())

_mld_map = gdf_sorted.groupby('profile_id', group_keys=False).apply(_surface_mld)
profile_stats['mld'] = profile_stats['profile_id'].map(_mld_map)
profile_stats['p_min_clamped'] = np.maximum(
    profile_stats['p_min'],
    profile_stats['mld'].fillna(profile_stats['p_min'])
)
# all profiles: presence/absence flag (like argo has_staircase)
all_profs = prof_coords[['profile_id', 'dist_km', 'lon']].merge(
    df_ls[['profile_id']].drop_duplicates().assign(has_staircase=1),
    on='profile_id', how='left'
)
all_profs['has_staircase'] = all_profs['has_staircase'].fillna(0).astype(int)

print(f"Profiles with staircase: {all_profs.has_staircase.sum()} / {len(all_profs)}")
print(f"Max staircases in one profile: {profile_stats.n_staircases.max()}")


#Figure: Potential density hovmoller
def _plot_sigma(ax, df_results_sub, ml_sub, vmin=None, vmax=None):
    sc = ax.scatter(
        df_results_sub[x_col], df_results_sub['p'],
        c=df_results_sub['sigma1'], cmap=cmo.dense, vmin=vmin, vmax=vmax,
        s=3, rasterized=True, zorder=1
    )
    for _, row in ml_sub.iterrows():
        ax.plot([row[x_col]] * 2, [row['p_start'], row['p_end']],
                color='white', lw=3.2, alpha=0.95, zorder=3,
                solid_capstyle='round', path_effects=[pe.Stroke(linewidth=4.4, foreground='black'), pe.Normal()])
    ax.invert_yaxis()
    ax.grid(True, **GRID_KW)
    ax.set_ylabel('Pressure (dbar)')
    ax.set_xlabel(x_label)
    return sc


if TURNAROUND_DETECTED:
    df_results_out, df_results_ret = split_outbound_return(df_results)
    ml_out, ml_ret = split_outbound_return(ml)
    sigma_vmin, sigma_vmax = df_results['sigma1'].min(), df_results['sigma1'].max()
    fig, (ax_out, ax_ret) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.subplots_adjust(bottom=0.14, hspace=0.35)
    sc = _plot_sigma(ax_out, df_results_out, ml_out, sigma_vmin, sigma_vmax)
    _plot_sigma(ax_ret, df_results_ret, ml_ret, sigma_vmin, sigma_vmax)
    ax_out.set_title(f"{title_datetime_str}\nPotential density (sigma1) - Eastbound leg  |  white/black = staircase mixed layers", loc='left')
    ax_ret.set_title("Westbound leg (return)", loc='left')
    add_station_markers([ax_out, ax_ret], fig)
    cb = fig.colorbar(sc, ax=[ax_out, ax_ret], pad=0.01)
    cb.set_label(' (kg m$^{-3}$)')
else:
    fig, ax = plt.subplots(figsize=(16, 5))
    fig.subplots_adjust(bottom=0.22)
    sc = _plot_sigma(ax, df_results, ml)
    ax.set_title(f"{title_datetime_str}\nPotential density (sigma1)  |  white/black = staircase mixed layers", loc='left')
    add_station_markers(ax, fig)
    cb = plt.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label(' (kg m$^{-3}$)')
plt.gcf().canvas.draw()  # force full render before tight-bbox crop
plt.savefig(os.path.join(daily_dir, 'ru29_sigma', f'ru29_sigma_{run_ts}.png'), dpi=200, bbox_inches='tight')
plt.show()


# Figure: Layer classification hovmoller
from matplotlib.patches import Patch

class_legend = [
    Patch(facecolor='lightgray',  label='Background'),
    Patch(facecolor='steelblue',  label='Mixed layer'),
    Patch(facecolor='darkorange', label='Gradient layer'),
]

cmap_class = ListedColormap(['lightgray', 'steelblue', 'darkorange'])
norm_class  = BoundaryNorm([0, 0.5, 1.5, 2.5], cmap_class.N)

def _plot_classification(ax, df_results_sub):
    ax.scatter(
        df_results_sub.loc[df_results_sub['layer_type'] == 0, x_col],
        df_results_sub.loc[df_results_sub['layer_type'] == 0, 'p'],
        color='lightgray', s=3, rasterized=True, zorder=1
    )
    for lt, color in [(1, 'steelblue'), (2, 'darkorange')]:
        mask = df_results_sub['layer_type'] == lt
        ax.scatter(
            df_results_sub.loc[mask, x_col], df_results_sub.loc[mask, 'p'],
            color=color, s=18, rasterized=True, zorder=2, edgecolors='none'
        )
    ax.invert_yaxis()
    ax.grid(True, **GRID_KW)
    ax.set_ylabel('Pressure (dbar)')
    ax.set_xlabel(x_label)


if TURNAROUND_DETECTED:
    df_results_out, df_results_ret = split_outbound_return(df_results)
    fig, (ax_out, ax_ret) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.subplots_adjust(bottom=0.14, hspace=0.35)
    _plot_classification(ax_out, df_results_out)
    _plot_classification(ax_ret, df_results_ret)
    ax_out.set_title(f"{title_datetime_str}\nStaircase layer classification - Eastbound leg", loc='left')
    ax_ret.set_title("Westbound leg (return)", loc='left')
    add_station_markers([ax_out, ax_ret], fig, extra_handles=class_legend)
else:
    fig, ax = plt.subplots(figsize=(16, 5))
    fig.subplots_adjust(bottom=0.22)
    _plot_classification(ax, df_results)
    ax.set_title(f"{title_datetime_str}\nStaircase layer classification", loc='left')
    add_station_markers(ax, fig, extra_handles=class_legend)
plt.gcf().canvas.draw()  # force full render before tight-bbox crop
plt.savefig(os.path.join(daily_dir, 'ru29_classification', f'ru29_classification_{run_ts}.png'), dpi=200, bbox_inches='tight')
plt.show()


# Figure: Staircase count per profile
def _plot_counts(ax, all_profs_sub, profile_stats_sub, n_vmin, n_vmax):
    ax.scatter(all_profs_sub[x_col], np.zeros(len(all_profs_sub)),
               c=all_profs_sub['has_staircase'], cmap='RdYlGn',
               vmin=0, vmax=1, s=40, zorder=2, alpha=0.85,
               edgecolors='k', linewidths=0.3)
    sc = ax.scatter(
        profile_stats_sub[x_col], profile_stats_sub['n_staircases'],
        c=profile_stats_sub['n_staircases'], cmap=cmo.ice_r, vmin=n_vmin, vmax=n_vmax,
        s=70, zorder=3, edgecolors='k', linewidths=0.5
    )
    ax.grid(True, **GRID_KW)
    ax.set_ylabel('# Staircases detected')
    ax.set_xlabel(x_label)
    ax.set_ylim(-0.5)
    return sc


if TURNAROUND_DETECTED:
    all_profs_out, all_profs_ret = split_outbound_return(all_profs)
    profile_stats_out, profile_stats_ret = split_outbound_return(profile_stats)
    # Combined range, computed before splitting - the same fix as CT/sigma:
    # this scatter layer has no fixed config-driven color limits, so
    # splitting without this would let each panel silently auto-range to
    # its own leg's min/max n_staircases (colors not comparable between
    # panels).
    n_vmin, n_vmax = profile_stats['n_staircases'].min(), profile_stats['n_staircases'].max()
    fig, (ax_out, ax_ret) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.subplots_adjust(bottom=0.14, hspace=0.35)
    sc = _plot_counts(ax_out, all_profs_out, profile_stats_out, n_vmin, n_vmax)
    _plot_counts(ax_ret, all_profs_ret, profile_stats_ret, n_vmin, n_vmax)
    ax_out.set_title(f"{title_datetime_str}\nStaircase count per profile - Eastbound leg  |  bottom strip = presence (green) / absence (red)", loc='left')
    ax_ret.set_title("Westbound leg (return)", loc='left')
    add_station_markers([ax_out, ax_ret], fig)
    cb = fig.colorbar(sc, ax=[ax_out, ax_ret], pad=0.01)
    cb.set_label('# staircases')
else:
    fig, ax = plt.subplots(figsize=(16, 5))
    fig.subplots_adjust(bottom=0.22)
    sc = _plot_counts(ax, all_profs, profile_stats, None, None)
    ax.set_title(f"{title_datetime_str}\nStaircase count per profile  |  bottom strip = presence (green) / absence (red)", loc='left')
    add_station_markers(ax, fig)
    cb = plt.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label('# staircases')
plt.gcf().canvas.draw()  # force full render before tight-bbox crop
plt.savefig(os.path.join(daily_dir, 'ru29_counts', f'ru29_counts_{run_ts}.png'), dpi=200, bbox_inches='tight')
plt.show()


#  Figure: Staircase depth range per profile
count_norm = plt.Normalize(vmin=1, vmax=profile_stats['n_staircases'].max())
cmap_count = cmo.ice_r

depth_legend = [
    Line2D([0],[0], marker='^', color='w', markerfacecolor='gray',
           markersize=9, markeredgecolor='k', markeredgewidth=0.5, label='Shallowest (>=MLD)'),
    Line2D([0],[0], marker='D', color='w', markerfacecolor='gray',
           markersize=9, markeredgecolor='k', markeredgewidth=0.5, label='Median depth'),
    Line2D([0],[0], marker='v', color='w', markerfacecolor='gray',
           markersize=9, markeredgecolor='k', markeredgewidth=0.5, label='Deepest'),
]

def _plot_depth_range(ax, profile_stats_sub):
    for _, row in profile_stats_sub.iterrows():
        color = cmap_count(count_norm(row['n_staircases']))
        ax.plot([row[x_col]] * 2, [row['p_min_clamped'], row['p_max']],
                color=color, lw=4.5, zorder=2, solid_capstyle='round',
                path_effects=[pe.Stroke(linewidth=6.0, foreground='black', alpha=0.5), pe.Normal()])

    ax.scatter(profile_stats_sub[x_col], profile_stats_sub['p_min_clamped'],
               c=profile_stats_sub['n_staircases'], cmap=cmap_count, norm=count_norm,
               s=55, marker='^', zorder=4, edgecolors='k', linewidths=0.5)
    ax.scatter(profile_stats_sub[x_col], profile_stats_sub['p_median'],
               c=profile_stats_sub['n_staircases'], cmap=cmap_count, norm=count_norm,
               s=55, marker='D', zorder=4, edgecolors='k', linewidths=0.5)
    ax.scatter(profile_stats_sub[x_col], profile_stats_sub['p_max'],
               c=profile_stats_sub['n_staircases'], cmap=cmap_count, norm=count_norm,
               s=55, marker='v', zorder=4, edgecolors='k', linewidths=0.5)

    ax.invert_yaxis()
    ax.grid(True, **GRID_KW)
    ax.set_ylabel('Pressure (dbar)')
    ax.set_xlabel(x_label)


# count_norm/cmap_count are already built from the full, unsplit
# profile_stats above (line ~1027) - correct to reuse as-is for both
# panels when split, no extra shared-range computation needed here.
sm = plt.cm.ScalarMappable(cmap=cmap_count, norm=count_norm)
sm.set_array([])

if TURNAROUND_DETECTED:
    profile_stats_out, profile_stats_ret = split_outbound_return(profile_stats)
    fig, (ax_out, ax_ret) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.subplots_adjust(bottom=0.14, hspace=0.35)
    _plot_depth_range(ax_out, profile_stats_out)
    _plot_depth_range(ax_ret, profile_stats_ret)
    ax_out.set_title(f"{title_datetime_str}\nStaircase depth range per profile - Eastbound leg  |  bar = min-max, markers = shallowest / median / deepest", loc='left')
    ax_ret.set_title("Westbound leg (return)", loc='left')
    add_station_markers([ax_out, ax_ret], fig, extra_handles=depth_legend)
    cb = fig.colorbar(sm, ax=[ax_out, ax_ret], pad=0.01)
    cb.set_label('# staircases')
else:
    fig, ax = plt.subplots(figsize=(16, 5))
    fig.subplots_adjust(bottom=0.22)
    _plot_depth_range(ax, profile_stats)
    add_station_markers(ax, fig, extra_handles=depth_legend)
    cb = plt.colorbar(sm, ax=ax, pad=0.01)
    cb.set_label('# staircases')
    ax.set_title(f"{title_datetime_str}\nStaircase depth range per profile  |  bar = min-max, markers = shallowest / median / deepest", loc='left')
plt.gcf().canvas.draw()  # force full render before tight-bbox crop
plt.savefig(os.path.join(daily_dir, 'ru29_depth_range', f'ru29_depth_range_{run_ts}.png'), dpi=200, bbox_inches='tight')
plt.show()

