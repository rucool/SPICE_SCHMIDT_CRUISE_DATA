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
import cool_maps.plot as cplt
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

DRIFTER_COLOR = 'lime'

# Per-station legend entry (name + its own tab10 color), plus one
# consolidated entry each for argo/drifter deploy stations that lists which
# of those specific stations have been reached.
legend_handles = [
    Line2D([0],[0], marker=s['marker'], color='w', markerfacecolor=s['color'],
           markersize=16, markeredgecolor='k', markeredgewidth=0.8, label=s['name'])
    for s in stations if s['reached'] and not s.get('argo') and not s.get('drifter')
]
argo_reached = [s['name'] for s in stations if s.get('argo') and s['reached']]
if argo_reached:
    legend_handles.append(
        Line2D([0],[0], marker='*', color='w', markerfacecolor='gold',
               markersize=18, markeredgecolor='k', markeredgewidth=0.5,
               label=f"Argo Deploy ({', '.join(argo_reached)})")
    )
drifter_reached = [s['name'] for s in stations if s.get('drifter') and s['reached']]
if drifter_reached:
    legend_handles.append(
        Line2D([0],[0], marker='D', color='w', markerfacecolor=DRIFTER_COLOR,
               markersize=14, markeredgecolor='k', markeredgewidth=0.8,
               label=f"Drifter Deploy ({', '.join(drifter_reached)})")
    )

ml = df_ls[df_ls['mixed_layer']].copy()

GRID_KW = dict(color='gray', alpha=0.2, linewidth=0.5, zorder=0)


def add_station_markers(ax, fig, extra_handles=None):
    """Triangles on x-axis spine for reached stations; combined legend below."""
    xform = ax.get_xaxis_transform()
    for s in stations:
        if not s['reached']:
            continue
        ax.plot(s['x_plot'], 0, marker=s['marker'],
                color=s['color'], markersize=20,
                transform=xform, clip_on=False, zorder=6,
                markeredgecolor='k', markeredgewidth=0.8)
        if s.get('argo'):
            ax.plot(s['x_plot'], 0, marker='*',
                    color='gold', markersize=16,
                    transform=xform, clip_on=False, zorder=7,
                    markeredgecolor='k', markeredgewidth=0.5)
        if s.get('drifter'):
            ax.plot(s['x_plot'], 0, marker='D',
                    color=DRIFTER_COLOR, markersize=13,
                    transform=xform, clip_on=False, zorder=7,
                    markeredgecolor='k', markeredgewidth=0.5)
    all_handles = (extra_handles or []) + legend_handles
    if all_handles:
        ncols = min(10, len(all_handles))
        fig.legend(handles=all_handles, loc='lower center',
                   bbox_to_anchor=(0.5, 0.0), ncol=ncols,
                   frameon=True, fontsize=10, handletextpad=0.3, columnspacing=1.0)


# Figure 1: Conservative Temperature
fig, ax = plt.subplots(figsize=(16, 5))
fig.subplots_adjust(bottom=0.22)

sc = ax.scatter(
    gdf_dist[x_col], gdf_dist['pressure'],
    c=gdf_dist['conservative_temperature'],
    cmap=cmo.thermal, s=0.5, rasterized=True, zorder=1
)
for _, row in ml.iterrows():
    ax.plot([row[x_col]] * 2, [row['p_start'], row['p_end']],
            color='white', lw=3.2, alpha=0.95, zorder=3,
            solid_capstyle='round', path_effects=[pe.Stroke(linewidth=4.4, foreground='black'), pe.Normal()])
ax.invert_yaxis()
ax.grid(True, **GRID_KW)
ax.set_ylabel('Pressure (dbar)')
ax.set_xlabel(x_label)
cb = plt.colorbar(sc, ax=ax, pad=0.01)
cb.set_label('CT (°C)')
ax.set_title(f"{title_datetime_str}\nConservative Temperature  |  white/black = staircase mixed layers", loc='left')
add_station_markers(ax, fig)
plt.gcf().canvas.draw()  # force full render before tight-bbox crop
plt.savefig(os.path.join(daily_dir, 'ru29_CT', f'ru29_CT_{run_ts}.png'), dpi=200, bbox_inches='tight')
plt.show()


#  Figure 2: Mixed-layer height 
fig, ax = plt.subplots(figsize=(16, 5))
fig.subplots_adjust(bottom=0.22)

if not ml.empty:
    sc = ax.scatter(ml[x_col], ml['p'], c=ml['layer_height'],
                    cmap=cmo.matter, s=35, zorder=2, vmin=PLOT_VARS_CFG['ml_height']['vmin'], vmax=PLOT_VARS_CFG['ml_height']['vmax'],
                    edgecolors='k', linewidths=0.3)
    cb = plt.colorbar(sc, ax=ax, pad=0.01, extend='max')
    cb.set_label('Layer height (dbar)')
ax.invert_yaxis()
ax.grid(True, **GRID_KW)
ax.set_ylabel('Pressure (dbar)')
ax.set_xlabel(x_label)
ax.set_title(f"{title_datetime_str}\nMixed-layer height", loc='left')
add_station_markers(ax, fig)
plt.gcf().canvas.draw()  # force full render before tight-bbox crop
plt.savefig(os.path.join(daily_dir, 'ru29_ml_height', f'ru29_ml_height_{run_ts}.png'), dpi=200, bbox_inches='tight')
plt.show()


# Figure 3: Turner angle 
fig, ax = plt.subplots(figsize=(16, 5))
fig.subplots_adjust(bottom=0.22)

if not df_ls.empty:
    sc = ax.scatter(df_ls[x_col], df_ls['p'], c=df_ls['turner_ang'],
                    cmap='RdBu_r', vmin=PLOT_VARS_CFG['turner']['vmin'], vmax=PLOT_VARS_CFG['turner']['vmax'], s=35, zorder=2,
                    edgecolors='k', linewidths=0.3)
    cb = plt.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label('Turner angle (°)')
ax.invert_yaxis()
ax.grid(True, **GRID_KW)
ax.set_ylabel('Pressure (dbar)')
ax.set_xlabel(x_label)
ax.set_title(f"{title_datetime_str}\nTurner angle  (red = salt fingering >45°, blue = diffusive convection <-45°)", loc='left')
add_station_markers(ax, fig)
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
fig, ax = plt.subplots(figsize=(16, 5))
fig.subplots_adjust(bottom=0.22)

sc = ax.scatter(
    df_results[x_col], df_results['p'],
    c=df_results['sigma1'], cmap=cmo.dense,
    s=3, rasterized=True, zorder=1
)
for _, row in ml.iterrows():
    ax.plot([row[x_col]] * 2, [row['p_start'], row['p_end']],
            color='white', lw=3.2, alpha=0.95, zorder=3,
            solid_capstyle='round', path_effects=[pe.Stroke(linewidth=4.4, foreground='black'), pe.Normal()])
ax.invert_yaxis()
ax.grid(True, **GRID_KW)
ax.set_ylabel('Pressure (dbar)')
ax.set_xlabel(x_label)
cb = plt.colorbar(sc, ax=ax, pad=0.01)
cb.set_label(' (kg m$^{-3}$)')
ax.set_title(f"{title_datetime_str}\nPotential density (sigma1)  |  white/black = staircase mixed layers", loc='left')
add_station_markers(ax, fig)
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

fig, ax = plt.subplots(figsize=(16, 5))
fig.subplots_adjust(bottom=0.22)

ax.scatter(
    df_results.loc[df_results['layer_type'] == 0, x_col],
    df_results.loc[df_results['layer_type'] == 0, 'p'],
    color='lightgray', s=3, rasterized=True, zorder=1
)
for lt, color in [(1, 'steelblue'), (2, 'darkorange')]:
    mask = df_results['layer_type'] == lt
    ax.scatter(
        df_results.loc[mask, x_col], df_results.loc[mask, 'p'],
        color=color, s=18, rasterized=True, zorder=2, edgecolors='none'
    )

ax.invert_yaxis()
ax.grid(True, **GRID_KW)
ax.set_ylabel('Pressure (dbar)')
ax.set_xlabel(x_label)
ax.set_title(f"{title_datetime_str}\nStaircase layer classification", loc='left')
add_station_markers(ax, fig, extra_handles=class_legend)
plt.gcf().canvas.draw()  # force full render before tight-bbox crop
plt.savefig(os.path.join(daily_dir, 'ru29_classification', f'ru29_classification_{run_ts}.png'), dpi=200, bbox_inches='tight')
plt.show()


# Figure: Staircase count per profile
fig, ax = plt.subplots(figsize=(16, 5))
fig.subplots_adjust(bottom=0.22)

ax.scatter(all_profs[x_col], np.zeros(len(all_profs)),
           c=all_profs['has_staircase'], cmap='RdYlGn',
           vmin=0, vmax=1, s=40, zorder=2, alpha=0.85,
           edgecolors='k', linewidths=0.3)
sc = ax.scatter(
    profile_stats[x_col], profile_stats['n_staircases'],
    c=profile_stats['n_staircases'], cmap=cmo.ice_r,
    s=70, zorder=3, edgecolors='k', linewidths=0.5
)
cb = plt.colorbar(sc, ax=ax, pad=0.01)
cb.set_label('# staircases')
ax.grid(True, **GRID_KW)
ax.set_ylabel('# Staircases detected')
ax.set_xlabel(x_label)
ax.set_title(f"{title_datetime_str}\nStaircase count per profile  |  bottom strip = presence (green) / absence (red)", loc='left')
ax.set_ylim(-0.5)
add_station_markers(ax, fig)
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

fig, ax = plt.subplots(figsize=(16, 5))
fig.subplots_adjust(bottom=0.22)

for _, row in profile_stats.iterrows():
    color = cmap_count(count_norm(row['n_staircases']))
    ax.plot([row[x_col]] * 2, [row['p_min_clamped'], row['p_max']],
            color=color, lw=4.5, zorder=2, solid_capstyle='round',
            path_effects=[pe.Stroke(linewidth=6.0, foreground='black', alpha=0.5), pe.Normal()])

ax.scatter(profile_stats[x_col], profile_stats['p_min_clamped'],
           c=profile_stats['n_staircases'], cmap=cmap_count, norm=count_norm,
           s=55, marker='^', zorder=4, edgecolors='k', linewidths=0.5)
ax.scatter(profile_stats[x_col], profile_stats['p_median'],
           c=profile_stats['n_staircases'], cmap=cmap_count, norm=count_norm,
           s=55, marker='D', zorder=4, edgecolors='k', linewidths=0.5)
ax.scatter(profile_stats[x_col], profile_stats['p_max'],
           c=profile_stats['n_staircases'], cmap=cmap_count, norm=count_norm,
           s=55, marker='v', zorder=4, edgecolors='k', linewidths=0.5)

sm = plt.cm.ScalarMappable(cmap=cmap_count, norm=count_norm)
sm.set_array([])
cb = plt.colorbar(sm, ax=ax, pad=0.01)
cb.set_label('# staircases')

ax.invert_yaxis()
ax.grid(True, **GRID_KW)
ax.set_ylabel('Pressure (dbar)')
ax.set_xlabel(x_label)
ax.set_title(f"{title_datetime_str}\nStaircase depth range per profile  |  bar = min-max, markers = shallowest / median / deepest", loc='left')
add_station_markers(ax, fig, extra_handles=depth_legend)
plt.gcf().canvas.draw()  # force full render before tight-bbox crop
plt.savefig(os.path.join(daily_dir, 'ru29_depth_range', f'ru29_depth_range_{run_ts}.png'), dpi=200, bbox_inches='tight')
plt.show()

