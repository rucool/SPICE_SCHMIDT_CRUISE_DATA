#!/usr/bin/env python
"""
Run the thermohaline staircase detection algorithm (same detect_staircases
core used by gliders_staircase.py / ru29_staircase.py) against shipboard
CTD casts read from Sea-Bird SBE9/19 processed .cnv files (both up and down casts).

The parse_sbe_cnv() function extracts standardized DataFrames + metadata 
including direction-aware zero-padded profile numbers (e.g. '001_down', '001_up').
"""
import argparse
import datetime as dt
import glob
import os
import re

import gsw
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import yaml
import cmocean.cm as cmo
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.spatial.distance import cdist as scipy_cdist
from thermohalinesteps.detect_staircases import classify_staircase, identify_staircases_from_layers

BAD_FLAG_DEFAULT = -9.99e-29
MIN_CAST_PRESSURE = 50.0  # dbar - matches the shallow-profile skip threshold gliders_staircase.py uses
MIN_POINTS = 5

ARGO_COLOR = 'gold'
DRIFTER_COLOR = 'lime'

# Same config files gliders_staircase.py / ru29_staircase.py read - station
# list and colorbar-limit tuning stay in yaml, not code.
_configdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
with open(os.path.join(_configdir, 'staircase_vars.yml')) as _f:
    PLOT_VARS_CFG = yaml.safe_load(_f)
with open(os.path.join(_configdir, 'cruise_stations.yml')) as _f:
    STATIONS = yaml.safe_load(_f)['stations']


def _dm_to_decimal(dm_str):
    """Convert a Sea-Bird header lat/lon string like '11 39.31 N' or
    '061 06.19 W' (degrees + decimal minutes + hemisphere) to signed decimal degrees."""
    deg, minutes, hemi = dm_str.split()
    value = float(deg) + float(minutes) / 60.0
    if hemi.upper() in ('S', 'W'):
        value *= -1
    return value


def parse_sbe_cnv(filepath):
    """Parse a Sea-Bird processed .cnv CTD file into a standardized
    DataFrame (pressure, temperature, salinity, depth, lat, lon - plus
    whatever other channels the file has, under their raw SBE short names)
    and a metadata dict (cast_id, profile_num, cast_direction, cast_time, lat, lon).
    """
    col_names = []
    bad_flag = BAD_FLAG_DEFAULT
    header_lat = header_lon = None
    cast_time = None
    data_start = None

    with open(filepath, 'r', encoding='latin1') as f:
        lines = f.readlines()

    name_re = re.compile(r'^#\s*name\s+\d+\s*=\s*([^:]+):')
    for i, line in enumerate(lines):
        if line.startswith('*END*'):
            data_start = i + 1
            break
        m = name_re.match(line)
        if m:
            col_names.append(m.group(1).strip())
            continue
        if line.startswith('# bad_flag'):
            bad_flag = float(line.split('=')[1].strip())
        elif line.startswith('* NMEA Latitude'):
            header_lat = _dm_to_decimal(line.split('=')[1].strip())
        elif line.startswith('* NMEA Longitude'):
            header_lon = _dm_to_decimal(line.split('=')[1].strip())
        elif line.startswith('# start_time'):
            ts = line.split('=', 1)[1].split('[')[0].strip()
            cast_time = dt.datetime.strptime(ts, '%b %d %Y %H:%M:%S')

    if data_start is None:
        raise ValueError(f"No *END* header terminator found in {filepath}")
    if not col_names:
        raise ValueError(f"No column headers found in {filepath}")

    df = pd.read_csv(filepath, skiprows=data_start, sep=r'\s+', names=col_names, engine='python', encoding='latin1')
    df = df.replace(bad_flag, np.nan)

    rename_map = {
        'prDM': 'pressure', 'depSM': 'depth', 't090C': 'temperature',
        'sal00': 'salinity', 'longitude': 'lon', 'latitude': 'lat',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    missing = [c for c in ('pressure', 'temperature', 'salinity') if c not in df.columns]
    if missing:
        raise ValueError(f"{filepath}: missing required column(s) {missing}")

    if 'lat' not in df.columns:
        df['lat'] = header_lat
    if 'lon' not in df.columns:
        df['lon'] = header_lon

    filename = os.path.basename(filepath)
    cast_id = os.path.splitext(filename)[0]

    # Detect direction (up or down)
    cast_direction = 'down'
    if re.search(r'(_up|_upcast)', filename, re.IGNORECASE):
        cast_direction = 'up'

    # Extract zero-padded profile number (e.g. FKt260806_CTD_001_down.cnv -> "001")
    match = re.search(r'_CTD_(\d+)_', filename, re.IGNORECASE)
    if match:
        raw_num = match.group(1)
    else:
        digits = re.findall(r'\d+', cast_id)
        raw_num = digits[-1] if digits else "000"

    # Distinguish profile_num explicitly by direction (e.g. "001_down", "001_up")
    profile_num = f"{raw_num}_{cast_direction}"

    meta = {
        'cast_id': cast_id,
        'profile_num': profile_num,
        'cast_direction': cast_direction,
        'cast_time': pd.Timestamp(cast_time, tz='UTC') if cast_time is not None else pd.NaT,
        'lat': float(df['lat'].dropna().median()) if df['lat'].notna().any() else header_lat,
        'lon': float(df['lon'].dropna().median()) if df['lon'].notna().any() else header_lon,
    }
    return df, meta


def process_cast(filepath):
    """Read one CTD cast (up or down) and run TEOS-10 + staircase classification."""
    df, meta = parse_sbe_cnv(filepath)

    df = df.dropna(subset=['pressure', 'temperature', 'salinity', 'lat', 'lon'])
    df = df.drop_duplicates(subset='pressure').sort_values('pressure')

    if len(df) < MIN_POINTS or df.empty or df.pressure.max() < MIN_CAST_PRESSURE:
        max_p = df.pressure.max() if len(df) else float('nan')
        print(f"  skip {meta['cast_id']}: too shallow/sparse ({len(df)} pts, max {max_p:.1f} dbar)")
        return None

    df['absolute_salinity'] = gsw.SA_from_SP(df.salinity, df.pressure, df.lon, df.lat)
    df['conservative_temperature'] = gsw.CT_from_t(df.absolute_salinity, df.temperature, df.pressure)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=['pressure', 'conservative_temperature', 'absolute_salinity'])
    df = df.sort_values('pressure')

    p_min = np.ceil(df.pressure.min())
    p_max = np.floor(df.pressure.max())
    p_reg = np.arange(p_min, p_max + 1, 1.0)
    if len(p_reg) < MIN_POINTS:
        return None

    ct_reg = np.interp(p_reg, df.pressure.values, df.conservative_temperature.values)
    sa_reg = np.interp(p_reg, df.pressure.values, df.absolute_salinity.values)

    try:
        df_out, mixes, grads = classify_staircase(p_reg, ct_reg, sa_reg, temp_flag_only=True, show_steps=False)
    except Exception:
        import traceback
        print(f"  ERROR in classify_staircase for {meta['cast_id']}:")
        traceback.print_exc()
        return None

    if df_out is None or len(df_out) == 0:
        return None

    df_out = df_out.copy()
    df_out['cast_id'] = meta['cast_id']
    df_out['profile_num'] = meta['profile_num']
    df_out['cast_direction'] = meta['cast_direction']
    df_out['cast_time'] = meta['cast_time']
    df_out['lat'] = meta['lat']
    df_out['lon'] = meta['lon']

    mixes_df = grads_df = stair_stats_df = stairs_ct_df = None

    if mixes is not None and grads is not None:
        mixes_df = mixes.copy()
        mixes_df['cast_id'] = meta['cast_id']
        mixes_df['profile_num'] = meta['profile_num']
        mixes_df['cast_direction'] = meta['cast_direction']
        mixes_df['cast_time'] = meta['cast_time']

        grads_df = grads.copy()
        grads_df['cast_id'] = meta['cast_id']
        grads_df['profile_num'] = meta['profile_num']
        grads_df['cast_direction'] = meta['cast_direction']
        grads_df['cast_time'] = meta['cast_time']

        staircase_list, ct_list = identify_staircases_from_layers(
            df=df_out.copy(),
            df_mixed_layers=mixes_df.copy(),
            df_gradient_layers=grads_df.copy(),
            max_allowable_gap=1,
            show_plot=False
        )

        stair_parts, ct_parts = [], []
        for i, st_df in enumerate(staircase_list, start=1):
            tmp = st_df.copy()
            tmp['cast_id'] = meta['cast_id']
            tmp['profile_num'] = meta['profile_num']
            tmp['cast_direction'] = meta['cast_direction']
            tmp['cast_time'] = meta['cast_time']
            tmp['staircase_id'] = i
            stair_parts.append(tmp)
        for i, ct_df in enumerate(ct_list, start=1):
            tmp = ct_df.copy()
            tmp['cast_id'] = meta['cast_id']
            tmp['profile_num'] = meta['profile_num']
            tmp['cast_direction'] = meta['cast_direction']
            tmp['cast_time'] = meta['cast_time']
            tmp['staircase_id'] = i
            ct_parts.append(tmp)

        stair_stats_df = pd.concat(stair_parts, ignore_index=True) if stair_parts else None
        stairs_ct_df = pd.concat(ct_parts, ignore_index=True) if ct_parts else None

    return meta, df_out, mixes_df, grads_df, stair_stats_df, stairs_ct_df


def split_down_up(df):
    """Splits a dataframe with a cast_direction column into (down, up) subsets.

    Unlike ru29_staircase.py's outbound/return split (which needs a computed
    turnaround time), direction is already a literal column here - no
    inference needed.
    """
    down = df.loc[df['cast_direction'] == 'down'].copy()
    up = df.loc[df['cast_direction'] == 'up'].copy()
    return down, up


def _surface_mld(group, delta_T=0.2, ref_p=10.0):
    """Surface MLD via a temperature-threshold method (0.2 degC drop from a
    10 dbar reference), same definition gliders_staircase.py/ru29_staircase.py
    use - just applied to df_out's already-regridded p/ct columns instead of
    raw per-scan pressure/conservative_temperature."""
    group = group.sort_values('p')
    ref = group[group['p'] >= ref_p]
    if ref.empty:
        return np.nan
    ct_ref = ref['ct'].iloc[0]
    below = group[(group['p'] > ref_p) & (group['ct'] < ct_ref - delta_T)]
    return float(below['p'].iloc[0]) if not below.empty else float(group['p'].max())


def _legend_row_major_order(items, ncols):
    """matplotlib fills multi-column legends column-major (top-to-bottom
    within each column, then next column) - with a partial last row that
    reads confusingly out of sequence. This reorders `items` so that
    column-major filling produces normal left-to-right, top-to-bottom
    reading order instead."""
    n = len(items)
    if n == 0 or ncols <= 1:
        return items
    nrows = -(-n // ncols)  # ceil(n / ncols)
    extra = n - (nrows - 1) * ncols  # first `extra` columns get a full nrows
    col_heights = [nrows if c < extra else nrows - 1 for c in range(ncols)]
    cells = [(r, c) for r in range(nrows) for c in range(ncols) if r < col_heights[c]]
    cell_to_item = dict(zip(cells, items))
    return [cell_to_item[(r, c)] for c in range(ncols) for r in range(col_heights[c])]


def add_station_markers(axes, fig, stations, legend_handles, extra_handles=None):
    """Triangles on x-axis spine for reached stations, drawn on every ax in
    `axes` (a single Axes, or a list/tuple - here always the [down, up]
    pair); one combined legend below the whole figure."""
    if not isinstance(axes, (list, tuple)):
        axes = [axes]
    for ax in axes:
        xform = ax.get_xaxis_transform()
        for s in stations:
            if not s['reached']:
                continue
            if s.get('argo'):
                ax.plot(s['lon'], 0, marker='*', color=ARGO_COLOR, markersize=16,
                        transform=xform, clip_on=False, zorder=6,
                        markeredgecolor='k', markeredgewidth=0.5)
            elif s.get('drifter'):
                ax.plot(s['lon'], 0, marker='D', color=DRIFTER_COLOR, markersize=13,
                        transform=xform, clip_on=False, zorder=6,
                        markeredgecolor='k', markeredgewidth=0.5)
            else:
                ax.plot(s['lon'], 0, marker=s['marker'], color=s['color'], markersize=14,
                        transform=xform, clip_on=False, zorder=6,
                        markeredgecolor='k', markeredgewidth=0.8)
    all_handles = (extra_handles or []) + legend_handles
    if all_handles:
        max_ncols = 10
        nrows = -(-len(all_handles) // max_ncols)  # ceil
        ncols = -(-len(all_handles) // nrows)  # smallest ncols that still fits in nrows
        leg = fig.legend(handles=_legend_row_major_order(all_handles, ncols), loc='lower center',
                          bbox_to_anchor=(0.5, 0.0), ncol=ncols,
                          frameon=True, fontsize=10, handletextpad=0.3, columnspacing=1.0)

        # Grow the bottom margin so the bottom axis's x-axis label clears the
        # legend - same directly-measured-pixel-gap approach ru29_staircase.py
        # uses (see its add_station_markers for the full rationale), reused
        # verbatim rather than re-deriving it.
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


def main():
    arg_parser = argparse.ArgumentParser(
        description='Run thermohaline staircase detection on shipboard CTD casts (up & down .cnv files) '
                     'and save per-cast results/figures.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument('casts_dir', type=str,
                             help='Directory containing Sea-Bird processed CTD .cnv files')
    # Default pattern updated: accepts both _down.cnv and _up.cnv files
    arg_parser.add_argument('-pattern', dest='pattern', default=r'^(?!.*_(?:5m|SSM|HOS|QC)_).*_(?:down|up)\.cnv$',
                             help='regex selecting full-resolution up/down .cnv files while skipping 5m/QC variants')
    arg_parser.add_argument('-o', '--output_prefix', dest='output_prefix', default='ctd_casts',
                             help='Prefix for output CSV filenames')
    arg_parser.add_argument('-s', '--save_dir', dest='save_dir', type=str, default='./satellite_figs',
                             help='Base directory to save per-cast staircase profile figures under')
    args = arg_parser.parse_args()

    all_cnv = sorted(glob.glob(os.path.join(args.casts_dir, '*.cnv')))
    files = [f for f in all_cnv if re.search(args.pattern, os.path.basename(f))]
    if not files:
        raise SystemExit(f"No .cnv files in {args.casts_dir} matched pattern {args.pattern!r} "
                          f"(found {len(all_cnv)} .cnv file(s) total)")
    print(f"Found {len(files)} cast file(s) (up + down)")

    results = []
    for fp in files:
        print(f"Processing {os.path.basename(fp)} ...")
        res = process_cast(fp)
        if res is not None:
            results.append(res)

    print(f"Done. Casts with staircase results: {len(results)} / {len(files)}")
    if not results:
        return

    df_out_all, mixes_all, grads_all, stair_stats_all, stairs_ct_all, meta_all = [], [], [], [], [], []
    for meta, df_out, mixes_df, grads_df, stair_stats_df, stairs_ct_df in results:
        meta_all.append(meta)
        df_out_all.append(df_out)
        if mixes_df is not None and not mixes_df.empty:
            mixes_all.append(mixes_df)
        if grads_df is not None and not grads_df.empty:
            grads_all.append(grads_df)
        if stair_stats_df is not None and not stair_stats_df.empty:
            stair_stats_all.append(stair_stats_df)
        if stairs_ct_df is not None and not stairs_ct_df.empty:
            stairs_ct_all.append(stairs_ct_df)

    print("Saving outputs...")
    pd.concat(df_out_all, ignore_index=True).to_csv(f"{args.output_prefix}_staircase_results.csv", index=False)
    if mixes_all:
        pd.concat(mixes_all, ignore_index=True).to_csv(f"{args.output_prefix}_mixes.csv", index=False)
    if grads_all:
        pd.concat(grads_all, ignore_index=True).to_csv(f"{args.output_prefix}_grads.csv", index=False)
    if stair_stats_all:
        pd.concat(stair_stats_all, ignore_index=True).to_csv(f"{args.output_prefix}_staircase_layer_stats.csv", index=False)
    if stairs_ct_all:
        pd.concat(stairs_ct_all, ignore_index=True).to_csv(f"{args.output_prefix}_staircases_ct.csv", index=False)
    pd.DataFrame(meta_all).to_csv(f"{args.output_prefix}_cast_locations.csv", index=False)
    print("Done.")

    n_staircases = sum(int(s['mixed_layer'].sum()) for s in stair_stats_all) if stair_stats_all else 0
    print(f"Total mixed layers found across all casts: {n_staircases}")

    plot_dir = os.path.join(args.save_dir, 'ctd_staircase_profiles')
    os.makedirs(plot_dir, exist_ok=True)

    for meta, df_out, mixes_df, grads_df, stair_stats_df, stairs_ct_df in results:
        fig, ax_ct = plt.subplots(figsize=(6, 9))

        ct_color = cmo.thermal(0.6)
        sa_color = cmo.haline(0.35)

        ax_ct.plot(df_out['ct'], df_out['p'], color=ct_color, lw=1.3)
        ax_ct.set_xlabel('Conservative Temperature (deg C)', color=ct_color)
        ax_ct.tick_params(axis='x', colors=ct_color)

        ax_sa = ax_ct.twiny()
        ax_sa.plot(df_out['sa'], df_out['p'], color=sa_color, lw=1.3)
        ax_sa.set_xlabel('Absolute Salinity (g/kg)', color=sa_color)
        ax_sa.tick_params(axis='x', colors=sa_color)

        n_ml = 0
        if stair_stats_df is not None:
            ml = stair_stats_df[stair_stats_df['mixed_layer']]
            n_ml = len(ml)
            for _, row in ml.iterrows():
                ax_ct.axhspan(row['p_start'], row['p_end'], color='steelblue', alpha=0.25, zorder=0)

        ax_ct.invert_yaxis()
        ax_ct.set_ylabel('Pressure (dbar)')
        ax_ct.grid(True, color='gray', alpha=0.2, linewidth=0.5, zorder=-1)

        cast_time_str = meta['cast_time'].strftime('%Y-%m-%d %H:%M UTC') if pd.notna(meta['cast_time']) else 'unknown time'
        lat_str = f"{meta['lat']:.4f}" if meta['lat'] is not None else 'n/a'
        lon_str = f"{meta['lon']:.4f}" if meta['lon'] is not None else 'n/a'
        ax_ct.set_title(
            f"{meta['cast_id']} (Profile {meta['profile_num']})  |  {cast_time_str}\n"
            f"lat {lat_str}, lon {lon_str}  |  shaded = staircase mixed layers ({n_ml})",
            loc='left', fontsize=10)

        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{meta['cast_id']}_staircase_profile.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)

    print(f"Saved {len(results)} profile figure(s) to {plot_dir}")


    # =========================================================================
    # Multi-cast survey figures: CT / ml_height / turner / sigma /
    # classification / counts / depth_range, longitude-indexed, always split
    # into down-cast / up-cast panels. A CTD cast doesn't move horizontally
    # the way a glider yo does - the ship holds station while the package
    # goes down and back up - so down and up sample ~the same water column
    # at ~the same longitude. Stacking them the way ru29_staircase.py stacks
    # outbound/return legs is the only way to show both without them
    # overplotting each other.
    # =========================================================================
    df_results = pd.concat(df_out_all, ignore_index=True)
    df_results['layer_type'] = 0  # background
    df_results.loc[~df_results['mixed_layer_final_mask'].astype(bool), 'layer_type'] = 1  # mixed
    df_results.loc[~df_results['gradient_layer_final_mask'].astype(bool), 'layer_type'] = 2  # gradient

    cast_coords = pd.DataFrame(meta_all)
    lon_map = cast_coords.set_index('cast_id')['lon']
    direction_map = cast_coords.set_index('cast_id')['cast_direction']

    ls_cols = ['p', 'p_start', 'p_end', 'turner_ang', 'layer_height', 'mixed_layer',
               'gradient_layer', 'cast_id', 'cast_direction', 'staircase_id', 'lon']
    if stair_stats_all:
        df_ls = pd.concat(stair_stats_all, ignore_index=True)
        df_ls['lon'] = df_ls['cast_id'].map(lon_map)
    else:
        df_ls = pd.DataFrame(columns=ls_cols)
    ml = df_ls[df_ls['mixed_layer']].copy() if not df_ls.empty else df_ls.copy()

    print(f"Staircase layers found: {len(df_ls)}"
          + (f"  (mixed: {int(df_ls['mixed_layer'].sum())}, gradient: {int(df_ls['gradient_layer'].sum())})"
             if not df_ls.empty else ""))

    # --- station reach, computed against the down-cast (the official
    # science position for each station) ---
    stations = [dict(s) for s in STATIONS]
    tab10 = plt.cm.tab10
    STATION_MARKER_SHAPES = ['v', '^', 's', 'p', 'h', 'o']
    for i, s in enumerate(stations):
        s['color'] = tab10.colors[i % 10]
        s['marker'] = STATION_MARKER_SHAPES[(i // 10) % len(STATION_MARKER_SHAPES)]

    down_coords = cast_coords[cast_coords['cast_direction'] == 'down']
    STATION_REACH_KM = PLOT_VARS_CFG['station_reach_km']
    if not down_coords.empty:
        stn_pts = np.array([[s['lat'], s['lon']] for s in stations])
        cast_pts = down_coords[['lat', 'lon']].values
        closest = scipy_cdist(stn_pts, cast_pts).argmin(axis=1)
        for s, idx in zip(stations, closest):
            cast = down_coords.iloc[idx]
            dist_to_stn = gsw.distance([s['lon'], cast['lon']], [s['lat'], cast['lat']])[0] / 1000.0
            s['reached'] = dist_to_stn <= STATION_REACH_KM
    else:
        for s in stations:
            s['reached'] = False

    reached = [s for s in stations if s['reached']]
    print(f"Stations reached so far ({len(reached)}/{len(stations)}): "
          f"{[s['name'] for s in reached] or 'none yet'}")

    legend_handles = []
    for s in stations:
        if not s['reached']:
            continue
        if s.get('argo'):
            legend_handles.append(Line2D([0], [0], marker='*', color='w', markerfacecolor=ARGO_COLOR,
                                          markersize=18, markeredgecolor='k', markeredgewidth=0.5,
                                          label=f"Argo ({s['name']})"))
        elif s.get('drifter'):
            legend_handles.append(Line2D([0], [0], marker='D', color='w', markerfacecolor=DRIFTER_COLOR,
                                          markersize=14, markeredgecolor='k', markeredgewidth=0.8,
                                          label=f"Drifter ({s['name']})"))
        else:
            legend_handles.append(Line2D([0], [0], marker=s['marker'], color='w', markerfacecolor=s['color'],
                                          markersize=16, markeredgecolor='k', markeredgewidth=0.8, label=s['name']))

    GRID_KW = dict(color='gray', alpha=0.2, linewidth=0.5, zorder=0)
    x_label = 'Longitude (deg)'

    cast_min_t, cast_max_t = cast_coords['cast_time'].min(), cast_coords['cast_time'].max()
    if pd.notna(cast_min_t) and pd.notna(cast_max_t):
        if cast_min_t.date() == cast_max_t.date():
            title_datetime_str = cast_min_t.strftime('%Y-%m-%d')
        else:
            title_datetime_str = f"{cast_min_t:%Y-%m-%d} to {cast_max_t:%Y-%m-%d}"
    else:
        title_datetime_str = 'unknown date'

    # Same TARGET_DATE/RUN_TS-driven daily_dir + run_ts convention
    # gliders_staircase.py/ru29_staircase.py use, so these figures land in
    # the same YYYY/MM/DD/<platform>_<var>/ layout as everything else this
    # pipeline serves - "falkor" matches the lowercase platform id already
    # used elsewhere (falkor_track.csv, get_falkor_position.py), not the
    # "Falkor (too)" display name in configs/platforms.yml.
    _target = os.environ.get("TARGET_DATE", "")
    run_time = dt.datetime.utcnow()
    _plot_date = pd.Timestamp(_target) if _target else run_time
    run_ts = (_plot_date.strftime("%Y%m%d_") + os.environ.get("RUN_TS", "") + "00"
              if os.environ.get("RUN_TS") else run_time.strftime("%Y%m%d_%H%M%S"))
    daily_dir = os.path.join(args.save_dir, _plot_date.strftime("%Y"), _plot_date.strftime("%m"), _plot_date.strftime("%d"))

    ctd_plot_vars = ['CT', 'ml_height', 'turner', 'sigma', 'classification', 'counts', 'depth_range']
    for v in ctd_plot_vars:
        os.makedirs(os.path.join(daily_dir, f'falkor_{v}'), exist_ok=True)

    df_results_down, df_results_up = split_down_up(df_results)
    ml_down, ml_up = split_down_up(ml)

    # Figure 1: Conservative Temperature
    def _plot_ct(ax, df_sub, ml_sub, vmin=None, vmax=None):
        sc = ax.scatter(df_sub['lon'], df_sub['p'], c=df_sub['ct'], cmap=cmo.thermal,
                         vmin=vmin, vmax=vmax, s=8, zorder=1)
        for _, row in ml_sub.iterrows():
            ax.plot([row['lon']] * 2, [row['p_start'], row['p_end']], color='white', lw=3.2, alpha=0.95,
                    zorder=3, solid_capstyle='round',
                    path_effects=[pe.Stroke(linewidth=4.4, foreground='black'), pe.Normal()])
        ax.invert_yaxis()
        ax.grid(True, **GRID_KW)
        ax.set_ylabel('Pressure (dbar)')
        return sc

    ct_vmin, ct_vmax = df_results['ct'].min(), df_results['ct'].max()
    fig, (ax_down, ax_up) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.subplots_adjust(bottom=0.14, hspace=0.35)
    sc = _plot_ct(ax_down, df_results_down, ml_down, ct_vmin, ct_vmax)
    _plot_ct(ax_up, df_results_up, ml_up, ct_vmin, ct_vmax)
    ax_up.set_xlabel(x_label)
    ax_down.set_title(f"{title_datetime_str}{chr(10)}Conservative Temperature - Down casts  |  white/black = staircase mixed layers", loc='left')
    ax_up.set_title("Up casts", loc='left')
    cb = fig.colorbar(sc, ax=[ax_down, ax_up], pad=0.01)
    cb.set_label('CT (deg C)')
    add_station_markers([ax_down, ax_up], fig, stations, legend_handles)
    fig.canvas.draw()
    fig.savefig(os.path.join(daily_dir, 'falkor_CT', f'falkor_CT_{run_ts}.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Figure 2: Mixed-layer height
    def _plot_ml_height(ax, ml_sub):
        sc = None
        if not ml_sub.empty:
            sc = ax.scatter(ml_sub['lon'], ml_sub['p'], c=ml_sub['layer_height'], cmap=cmo.matter, s=45,
                             zorder=2, vmin=PLOT_VARS_CFG['ml_height']['vmin'], vmax=PLOT_VARS_CFG['ml_height']['vmax'],
                             edgecolors='k', linewidths=0.3)
        ax.invert_yaxis()
        ax.grid(True, **GRID_KW)
        ax.set_ylabel('Pressure (dbar)')
        return sc

    fig, (ax_down, ax_up) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.subplots_adjust(bottom=0.14, hspace=0.35)
    sc_down = _plot_ml_height(ax_down, ml_down)
    sc_up = _plot_ml_height(ax_up, ml_up)
    ax_up.set_xlabel(x_label)
    ax_down.set_title(f"{title_datetime_str}{chr(10)}Mixed-layer height - Down casts", loc='left')
    ax_up.set_title("Up casts", loc='left')
    sc_for_cb = sc_down if sc_down is not None else sc_up
    if sc_for_cb is not None:
        cb = fig.colorbar(sc_for_cb, ax=[ax_down, ax_up], pad=0.01, extend='max')
        cb.set_label('Layer height (dbar)')
    add_station_markers([ax_down, ax_up], fig, stations, legend_handles)
    fig.canvas.draw()
    fig.savefig(os.path.join(daily_dir, 'falkor_ml_height', f'falkor_ml_height_{run_ts}.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    df_ls_down, df_ls_up = split_down_up(df_ls)

    # Figure 3: Turner angle
    def _plot_turner(ax, df_ls_sub):
        sc = None
        if not df_ls_sub.empty:
            sc = ax.scatter(df_ls_sub['lon'], df_ls_sub['p'], c=df_ls_sub['turner_ang'], cmap='RdBu_r',
                             vmin=PLOT_VARS_CFG['turner']['vmin'], vmax=PLOT_VARS_CFG['turner']['vmax'], s=45,
                             zorder=2, edgecolors='k', linewidths=0.3)
        ax.invert_yaxis()
        ax.grid(True, **GRID_KW)
        ax.set_ylabel('Pressure (dbar)')
        return sc

    fig, (ax_down, ax_up) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.subplots_adjust(bottom=0.14, hspace=0.35)
    sc_down = _plot_turner(ax_down, df_ls_down)
    sc_up = _plot_turner(ax_up, df_ls_up)
    ax_up.set_xlabel(x_label)
    ax_down.set_title(f"{title_datetime_str}{chr(10)}Turner angle - Down casts  (red = salt fingering >45deg, blue = diffusive convection <-45deg)", loc='left')
    ax_up.set_title("Up casts", loc='left')
    sc_for_cb = sc_down if sc_down is not None else sc_up
    if sc_for_cb is not None:
        cb = fig.colorbar(sc_for_cb, ax=[ax_down, ax_up], pad=0.01)
        cb.set_label('Turner angle (deg)')
    add_station_markers([ax_down, ax_up], fig, stations, legend_handles)
    fig.canvas.draw()
    fig.savefig(os.path.join(daily_dir, 'falkor_turner', f'falkor_turner_{run_ts}.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Figure 4: Potential density (sigma1)
    def _plot_sigma(ax, df_sub, ml_sub, vmin=None, vmax=None):
        sc = ax.scatter(df_sub['lon'], df_sub['p'], c=df_sub['sigma1'], cmap=cmo.dense, vmin=vmin, vmax=vmax,
                         s=8, zorder=1)
        for _, row in ml_sub.iterrows():
            ax.plot([row['lon']] * 2, [row['p_start'], row['p_end']], color='white', lw=3.2, alpha=0.95,
                    zorder=3, solid_capstyle='round',
                    path_effects=[pe.Stroke(linewidth=4.4, foreground='black'), pe.Normal()])
        ax.invert_yaxis()
        ax.grid(True, **GRID_KW)
        ax.set_ylabel('Pressure (dbar)')
        return sc

    sigma_vmin, sigma_vmax = df_results['sigma1'].min(), df_results['sigma1'].max()
    fig, (ax_down, ax_up) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.subplots_adjust(bottom=0.14, hspace=0.35)
    sc = _plot_sigma(ax_down, df_results_down, ml_down, sigma_vmin, sigma_vmax)
    _plot_sigma(ax_up, df_results_up, ml_up, sigma_vmin, sigma_vmax)
    ax_up.set_xlabel(x_label)
    ax_down.set_title(f"{title_datetime_str}{chr(10)}Potential density (sigma1) - Down casts  |  white/black = staircase mixed layers", loc='left')
    ax_up.set_title("Up casts", loc='left')
    cb = fig.colorbar(sc, ax=[ax_down, ax_up], pad=0.01)
    cb.set_label('(kg m-3)')
    add_station_markers([ax_down, ax_up], fig, stations, legend_handles)
    fig.canvas.draw()
    fig.savefig(os.path.join(daily_dir, 'falkor_sigma', f'falkor_sigma_{run_ts}.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Figure 5: Layer classification
    class_legend = [
        Patch(facecolor='lightgray', label='Background'),
        Patch(facecolor='steelblue', label='Mixed layer'),
        Patch(facecolor='darkorange', label='Gradient layer'),
    ]

    def _plot_classification(ax, df_sub):
        ax.scatter(df_sub.loc[df_sub['layer_type'] == 0, 'lon'], df_sub.loc[df_sub['layer_type'] == 0, 'p'],
                   color='lightgray', s=8, zorder=1)
        for lt, color in [(1, 'steelblue'), (2, 'darkorange')]:
            mask = df_sub['layer_type'] == lt
            ax.scatter(df_sub.loc[mask, 'lon'], df_sub.loc[mask, 'p'], color=color, s=24, zorder=2,
                       edgecolors='none')
        ax.invert_yaxis()
        ax.grid(True, **GRID_KW)
        ax.set_ylabel('Pressure (dbar)')

    fig, (ax_down, ax_up) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.subplots_adjust(bottom=0.14, hspace=0.35)
    _plot_classification(ax_down, df_results_down)
    _plot_classification(ax_up, df_results_up)
    ax_up.set_xlabel(x_label)
    ax_down.set_title(f"{title_datetime_str}{chr(10)}Staircase layer classification - Down casts", loc='left')
    ax_up.set_title("Up casts", loc='left')
    add_station_markers([ax_down, ax_up], fig, stations, legend_handles, extra_handles=class_legend)
    fig.canvas.draw()
    fig.savefig(os.path.join(daily_dir, 'falkor_classification', f'falkor_classification_{run_ts}.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # --- per-cast staircase summary stats (mirror of the ru29/glider "hexbin" quantities) ---
    stats_cols = ['cast_id', 'n_staircases', 'p_min', 'p_max', 'p_mean', 'p_median']
    if not df_ls.empty:
        profile_stats = (
            df_ls.groupby('cast_id')
            .agg(n_staircases=('staircase_id', 'nunique'), p_min=('p_start', 'min'), p_max=('p_end', 'max'),
                 p_mean=('p', 'mean'), p_median=('p', 'median'))
            .reset_index()
        )
    else:
        profile_stats = pd.DataFrame(columns=stats_cols)
    profile_stats['lon'] = profile_stats['cast_id'].map(lon_map)
    profile_stats['cast_direction'] = profile_stats['cast_id'].map(direction_map)

    mld_map = df_results.groupby('cast_id', group_keys=False)[['p', 'ct']].apply(_surface_mld)
    profile_stats['mld'] = profile_stats['cast_id'].map(mld_map)
    profile_stats['p_min_clamped'] = np.maximum(
        profile_stats['p_min'], profile_stats['mld'].fillna(profile_stats['p_min']))

    has_staircase_ids = df_ls.loc[df_ls['mixed_layer'], ['cast_id']].drop_duplicates() if not df_ls.empty else pd.DataFrame(columns=['cast_id'])
    all_profs = cast_coords[['cast_id', 'cast_direction', 'lon']].merge(
        has_staircase_ids.assign(has_staircase=1), on='cast_id', how='left')
    all_profs['has_staircase'] = all_profs['has_staircase'].fillna(0).astype(int)

    print(f"Casts with staircase: {all_profs.has_staircase.sum()} / {len(all_profs)}")
    if not profile_stats.empty:
        print(f"Max staircases in one cast: {profile_stats.n_staircases.max()}")

    # Figure 6: staircase count per cast
    def _plot_counts(ax, all_profs_sub, profile_stats_sub, n_vmin, n_vmax):
        ax.scatter(all_profs_sub['lon'], np.zeros(len(all_profs_sub)), c=all_profs_sub['has_staircase'],
                   cmap='RdYlGn', vmin=0, vmax=1, s=50, zorder=2, alpha=0.85, edgecolors='k', linewidths=0.3)
        sc = ax.scatter(profile_stats_sub['lon'], profile_stats_sub['n_staircases'],
                         c=profile_stats_sub['n_staircases'], cmap=cmo.ice_r, vmin=n_vmin, vmax=n_vmax,
                         s=90, zorder=3, edgecolors='k', linewidths=0.5)
        ax.grid(True, **GRID_KW)
        ax.set_ylabel('# Staircases detected')
        ax.set_ylim(-0.5)
        return sc

    all_profs_down, all_profs_up = split_down_up(all_profs)
    profile_stats_down, profile_stats_up = split_down_up(profile_stats)
    n_vmin = profile_stats['n_staircases'].min() if not profile_stats.empty else 0
    n_vmax = profile_stats['n_staircases'].max() if not profile_stats.empty else 1
    fig, (ax_down, ax_up) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.subplots_adjust(bottom=0.14, hspace=0.35)
    sc = _plot_counts(ax_down, all_profs_down, profile_stats_down, n_vmin, n_vmax)
    _plot_counts(ax_up, all_profs_up, profile_stats_up, n_vmin, n_vmax)
    ax_up.set_xlabel(x_label)
    ax_down.set_title(f"{title_datetime_str}{chr(10)}Staircase count per cast - Down casts  |  bottom strip = presence (green) / absence (red)", loc='left')
    ax_up.set_title("Up casts", loc='left')
    cb = fig.colorbar(sc, ax=[ax_down, ax_up], pad=0.01)
    cb.set_label('# staircases')
    add_station_markers([ax_down, ax_up], fig, stations, legend_handles)
    fig.canvas.draw()
    fig.savefig(os.path.join(daily_dir, 'falkor_counts', f'falkor_counts_{run_ts}.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Figure 7: staircase depth range per cast
    count_norm = plt.Normalize(vmin=1, vmax=max(int(profile_stats['n_staircases'].max()), 1) if not profile_stats.empty else 1)
    cmap_count = cmo.ice_r

    depth_legend = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=9, markeredgecolor='k',
               markeredgewidth=0.5, label='Shallowest (>=MLD)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', markersize=9, markeredgecolor='k',
               markeredgewidth=0.5, label='Median depth'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='gray', markersize=9, markeredgecolor='k',
               markeredgewidth=0.5, label='Deepest'),
    ]

    def _plot_depth_range(ax, profile_stats_sub):
        for _, row in profile_stats_sub.iterrows():
            color = cmap_count(count_norm(row['n_staircases']))
            ax.plot([row['lon']] * 2, [row['p_min_clamped'], row['p_max']], color=color, lw=4.5, zorder=2,
                    solid_capstyle='round', path_effects=[pe.Stroke(linewidth=6.0, foreground='black', alpha=0.5), pe.Normal()])
        ax.scatter(profile_stats_sub['lon'], profile_stats_sub['p_min_clamped'], c=profile_stats_sub['n_staircases'],
                   cmap=cmap_count, norm=count_norm, s=70, marker='^', zorder=4, edgecolors='k', linewidths=0.5)
        ax.scatter(profile_stats_sub['lon'], profile_stats_sub['p_median'], c=profile_stats_sub['n_staircases'],
                   cmap=cmap_count, norm=count_norm, s=70, marker='D', zorder=4, edgecolors='k', linewidths=0.5)
        ax.scatter(profile_stats_sub['lon'], profile_stats_sub['p_max'], c=profile_stats_sub['n_staircases'],
                   cmap=cmap_count, norm=count_norm, s=70, marker='v', zorder=4, edgecolors='k', linewidths=0.5)
        ax.invert_yaxis()
        ax.grid(True, **GRID_KW)
        ax.set_ylabel('Pressure (dbar)')

    sm = plt.cm.ScalarMappable(cmap=cmap_count, norm=count_norm)
    sm.set_array([])
    fig, (ax_down, ax_up) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.subplots_adjust(bottom=0.14, hspace=0.35)
    _plot_depth_range(ax_down, profile_stats_down)
    _plot_depth_range(ax_up, profile_stats_up)
    ax_up.set_xlabel(x_label)
    ax_down.set_title(f"{title_datetime_str}{chr(10)}Staircase depth range per cast - Down casts  |  bar = min-max, markers = shallowest / median / deepest", loc='left')
    ax_up.set_title("Up casts", loc='left')
    cb = fig.colorbar(sm, ax=[ax_down, ax_up], pad=0.01)
    cb.set_label('# staircases')
    add_station_markers([ax_down, ax_up], fig, stations, legend_handles, extra_handles=depth_legend)
    fig.canvas.draw()
    fig.savefig(os.path.join(daily_dir, 'falkor_depth_range', f'falkor_depth_range_{run_ts}.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved 7 survey figure(s) to {daily_dir}")


if __name__ == '__main__':
    main()