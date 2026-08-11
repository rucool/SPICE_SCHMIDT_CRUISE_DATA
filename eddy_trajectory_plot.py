#!/usr/bin/env python
"""
Plots AVISO+ eddy centers (+ recent track tails) currently within the SPICE
survey bbox, as a standalone map - same bbox/style convention as
SPICE_CMEMS_SAT.py's other products, but a scatter/track plot rather than a
gridded pcolormesh, since eddy trajectory data is discrete per-eddy
observations, not a lat/lon grid.

Reads the subset NetCDF files written by eddy_trajectory_download.py
(eddy_<polarity>_latest.nc) - this script does no network access itself,
except for reading platform track CSVs (ru29/Falkor) already written by
their own fetch scripts.

Polarity (anticyclonic/cyclonic) is always encoded via marker/line shape
(circle vs square). Color depends on SLA_BACKGROUND_MODE below: off (the
default), each unique eddy (AVISO's "track" id) gets its own distinct
color, showing the individual character/motion of each eddy currently in
the bbox ("spaghetti"); on, eddies overlay the actual SLA field they were
derived from and are plain black instead, since color is spent on that
background.
"""
import argparse
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import cmocean.cm as cmo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml

TROP_WTRN_ATL_EXTENT = [-63, -40.75, 4, 19]

# EEZ boundary lines, drawn gray on this map too - same source/approach as
# SPICE_CMEMS_SAT.py (see that script's comment for why cartopy.io.shapereader
# rather than geopandas: not installed in the spice_data env this runs in).
# Duplicated rather than imported for the same reason the rest of this
# script's helpers are duplicated - see the PLATFORMS comment below.
EEZ_SHP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'World_EEZ_v11_20191118',
                             'eez_boundaries_v11.shp')


def load_eez_geometries(bbox, pad_deg=2.0):
    """Returns a list of shapely LineString geometries from EEZ_SHP_PATH
    whose bounding box intersects `bbox` (padded by pad_deg). Missing
    shapefile or read error -> empty list rather than crashing the run."""
    lon_min, lon_max, lat_min, lat_max = bbox
    lon_min, lon_max = lon_min - pad_deg, lon_max + pad_deg
    lat_min, lat_max = lat_min - pad_deg, lat_max + pad_deg
    try:
        records = list(shpreader.Reader(EEZ_SHP_PATH).records())
    except Exception as e:
        print(f"Warning: could not load EEZ shapefile from {EEZ_SHP_PATH}: {e} - skipping EEZ overlay")
        return []
    geoms = []
    for rec in records:
        geom = rec.geometry
        if geom is None:
            continue
        minx, miny, maxx, maxy = geom.bounds
        if maxx >= lon_min and minx <= lon_max and maxy >= lat_min and miny <= lat_max:
            geoms.append(geom)
    print(f"EEZ overlay: {len(geoms)} boundary segments loaded from {os.path.basename(EEZ_SHP_PATH)}")
    return geoms


EEZ_GEOMETRIES = load_eez_geometries(TROP_WTRN_ATL_EXTENT)

# Toggle: overlay eddy positions on the actual SLA field they were derived
# from (written by eddy_trajectory_download.py's fetch_sla_background,
# matching SPICE_CMEMS_SAT.py's sla styling), vs. the plain white-background
# view with unique per-eddy colors. When True, eddies switch to solid black
# (shape-only polarity distinction) since color is spent on the SLA
# background instead - a per-eddy rainbow would clash against it.
SLA_BACKGROUND_MODE = True
SLA_CLIM = (-0.2, 0.2)  # matches SPICE_CMEMS_SAT.py's variable_clims['sla']

# Anticyclonic = warm-core, rotates clockwise in the N. hemisphere;
# cyclonic = cold-core, rotates counter-clockwise. Distinguished by marker
# shape (not color) since color here encodes per-eddy identity instead.
POLARITY_STYLE = {
    "anticyclonic": {"marker": "o"},
    "cyclonic": {"marker": "s"},
}

TAIL_DAYS = 21  # how much of each eddy's recent track to draw as a tail

# Platforms to overlay: single shared source of truth in
# configs/platforms.yml - also read by SPICE_CMEMS_SAT.py and
# eddy_12N_forecast.py, so toggling "enabled" there (e.g. re-enabling
# Falkor, adding another glider) takes effect everywhere at once, no
# hand-sync needed across scripts anymore.
_configdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
with open(os.path.join(_configdir, 'platforms.yml')) as _f:
    PLATFORMS = yaml.safe_load(_f)['platforms']
ACTIVE_PLATFORMS = [p for p in PLATFORMS if p.get("enabled", True)]
PLATFORM_TAIL_DAYS = 7  # trims only the drawn tail line, never the latest-position marker

arg_parser = argparse.ArgumentParser(description='Plot AVISO+ eddy trajectories in the SPICE survey bbox',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_parser.add_argument('-s', '--save_dir', dest='save_dir', type=str, default='./satellite_figs',
                        help='Full file path to directory where figures are written')
arg_parser.add_argument('-c', '--cmems_dir', dest='cmems_dir', type=str, default='./cmems_data',
                        help='Directory eddy_trajectory_download.py wrote its subset files to')
args = arg_parser.parse_args()


def load_eddy_data(base_dir):
    data = {}
    for polarity in POLARITY_STYLE:
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
        })
        data[polarity] = df
    return data


def load_sla_background(base_dir):
    """Reads back the SLA snapshot eddy_trajectory_download.py's
    fetch_sla_background wrote for this same eddy data's latest observation
    date. Returns None (not an error) if missing - e.g. an older download
    run before this feature existed, or SLA_BACKGROUND_MODE was off at
    fetch time - so plot_eddies can fall back to the plain background
    gracefully instead of crashing."""
    path = os.path.join(base_dir, "eddy_trajectory", "sla_background.nc")
    if not os.path.exists(path):
        print(f"Warning: {path} not found - run eddy_trajectory_download.py to fetch it")
        return None
    return xr.open_dataset(path)


def get_platform_track(csv_name):
    """Mirrors SPICE_CMEMS_SAT.py's function of the same name - see the
    PLATFORMS comment above for why this is duplicated rather than
    imported."""
    try:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_name)
        df = pd.read_csv(csv_path)
        df["time"] = pd.to_datetime(df["time"])
        return df.sort_values("time").reset_index(drop=True)
    except Exception as e:
        print(f"Warning: could not load track from {csv_name}: {e}")
        return None


def _track_color_map(track_ids):
    """Assigns each unique eddy track id its own maximally-distinct color so
    individual eddies stay visually separable in a dense spaghetti plot.
    Draws from matplotlib's tab20/tab20b/tab20c (60 qualitative colors
    designed to be distinct from each other) before falling back to
    continuous hsv sampling if a single map ever has more eddies than that
    (not expected for a 14-day tail in this bbox, but harmless either way)."""
    ids = sorted(track_ids)
    n = len(ids)
    if n == 0:
        return {}
    qualitative = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)
    colors = qualitative[:n] if n <= len(qualitative) else [plt.cm.hsv(i / n) for i in range(n)]
    return dict(zip(ids, colors))


def _legend_row_major_order(items, ncols):
    """matplotlib fills multi-column legends column-major (top-to-bottom
    within each column, then next column) - with a partial last row that
    reads confusingly out of sequence. Mirrors the helper of the same name
    in ru29_staircase.py/gliders_staircase.py (duplicated, not imported -
    those are standalone top-level scripts, not safely importable modules).
    Reorders `items` so column-major filling produces normal left-to-right,
    top-to-bottom reading order instead."""
    n = len(items)
    if n == 0 or ncols <= 1:
        return items
    nrows = -(-n // ncols)  # ceil(n / ncols)
    extra = n - (nrows - 1) * ncols  # first `extra` columns get a full nrows
    col_heights = [nrows if c < extra else nrows - 1 for c in range(ncols)]
    cells = [(r, c) for r in range(nrows) for c in range(ncols) if r < col_heights[c]]
    cell_to_item = dict(zip(cells, items))
    return [cell_to_item[(r, c)] for c in range(ncols) for r in range(col_heights[c])]


def plot_eddies(eddy_data, sla_ds=None, bbox=TROP_WTRN_ATL_EXTENT, base_dir=args.save_dir, run_ts=""):
    lon_min, lon_max, lat_min, lat_max = bbox
    date = pd.Timestamp.now(tz="UTC")
    sla_mode = SLA_BACKGROUND_MODE and sla_ds is not None
    if SLA_BACKGROUND_MODE and sla_ds is None:
        print("SLA_BACKGROUND_MODE is on but no SLA background was loaded - falling back to plain background")

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": ccrs.Mercator()})
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                                 edgecolor='black', facecolor='lightgray'))
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # EEZ boundary lines - zorder above the SLA background (zorder=1 below)
    # but below eddy markers/tracks (40/41) and platform overlays (50/51).
    if EEZ_GEOMETRIES:
        ax.add_geometries(EEZ_GEOMETRIES, ccrs.PlateCarree(),
                           edgecolor='gray', facecolor='none', linewidth=0.7, zorder=4)

    sla_date = None
    if sla_mode:
        sla_da = sla_ds["sla"]
        if "time" in sla_da.dims:
            # Captured before indexing away the time dim - fetch_sla_background
            # has its own small lookback if the exact eddy date has a gap in
            # SLA coverage, so the actual date here isn't guaranteed to equal
            # plot_date. Shown in the title below rather than assumed, so the
            # title can't silently mislead the way the pre-fix eddy title did.
            sla_date = pd.Timestamp(sla_da.time.values[-1])
            sla_da = sla_da.isel(time=-1)
        sla_vmin, sla_vmax = SLA_CLIM
        sla_im = ax.pcolormesh(
            sla_ds.longitude.values, sla_ds.latitude.values, sla_da.values,
            cmap=cmo.balance, vmin=sla_vmin, vmax=sla_vmax,
            transform=ccrs.PlateCarree(), zorder=1,
        )
        fig.colorbar(sla_im, ax=ax, orientation="horizontal", label="sla (m)", shrink=0.8, pad=0.08)

    legend_handles = []
    legend_labels = []
    n_eddies_total = 0

    # Cutoff is relative to the latest observation actually present in the
    # data, not wall-clock "now" - eddy_trajectory_download.py already
    # windows to the last TAIL_DAYS relative to the AVISO file's own latest
    # timestamp (that file can lag real-time by ~2 weeks), so re-filtering
    # here relative to wall-clock "now" would hit the identical problem:
    # silently plotting nothing whenever that lag reaches TAIL_DAYS. Same
    # class of bug already fixed once for platform tracks elsewhere in this
    # pipeline (see get_platform_track's docstring in SPICE_CMEMS_SAT.py).
    nonempty = [df for df in eddy_data.values() if len(df)]
    if nonempty:
        latest_obs = max(df["time"].max() for df in nonempty)
        cutoff = latest_obs - pd.Timedelta(days=TAIL_DAYS)
    else:
        latest_obs = None
        cutoff = pd.Timestamp.min

    # Folder/filename/title are dated by the eddy data's own latest
    # observation, not wall-clock "now" - matches SPICE_CMEMS_SAT.py's
    # convention of dating output by the actual data date (which can also
    # lag "today" for slow-updating products like sss/sargassum), and
    # avoids the exact misleading-title problem just fixed for the cutoff
    # above: printing "now" on a map whose data is really ~2 weeks old.
    # Falls back to wall-clock now only in the edge case of zero eddy data
    # at all (nothing meaningful to date it by).
    plot_date = latest_obs if latest_obs is not None else date

    # In sla_mode, every eddy is plain black (shape-only polarity
    # distinction) since color is spent on the SLA background instead - a
    # per-eddy rainbow would clash against it. Otherwise, colors are
    # assigned per unique eddy across BOTH polarities at once, so no two
    # eddies on the map ever collide in color even across the
    # anticyclonic/cyclonic split.
    all_recent = {polarity: df[df["time"] >= cutoff] for polarity, df in eddy_data.items()}
    if sla_mode:
        track_colors = {}
    else:
        all_track_ids = set()
        for recent in all_recent.values():
            all_track_ids.update(recent["track"].unique())
        track_colors = _track_color_map(all_track_ids)

    # (polarity, "active"/"ended") -> count, so the legend can show the full
    # shape x color cross-tab (e.g. "active cyclonic", "ended anticyclonic")
    # matching exactly what's drawn on the map, rather than two separate
    # partial-encoding legend groups that don't show the combination.
    status_counts = {}
    for polarity, recent in all_recent.items():
        style = POLARITY_STYLE[polarity]

        for track_id, grp in recent.groupby("track"):
            grp = grp.sort_values("time")
            if sla_mode:
                # Active = this eddy's last known position falls on the
                # most recent day we have any data for (latest_obs) - AVISO
                # doesn't flag "this eddy ended" anywhere (confirmed by
                # inspecting the raw file directly: termination is only
                # ever inferred by a track simply not appearing again), so
                # "ended within our TAIL_DAYS window" just means its last
                # row is dated earlier than the most current day available.
                is_active = grp["time"].max() == latest_obs
                status = "active" if is_active else "ended"
                color = "cyan" if is_active else "black"
                status_counts[(polarity, status)] = status_counts.get((polarity, status), 0) + 1
            else:
                color = track_colors[track_id]
            ax.plot(grp["lon"], grp["lat"], '-', color=color, lw=1.2, alpha=0.7,
                    transform=ccrs.PlateCarree(), zorder=40)
            ax.plot(grp["lon"].iloc[-1], grp["lat"].iloc[-1], style["marker"], color=color,
                    markersize=7, markeredgecolor='k', markeredgewidth=0.5,
                    transform=ccrs.PlateCarree(), zorder=41)
            n_eddies_total += 1

        if not sla_mode and len(recent):
            legend_handles.append(plt.Line2D([0], [0], marker=style["marker"], color='w',
                                              markerfacecolor="gray", markersize=8,
                                              markeredgecolor='k', markeredgewidth=0.5))
            legend_labels.append(f"{polarity} ({recent['track'].nunique()} eddies)")

    if sla_mode:
        for polarity in POLARITY_STYLE:
            style = POLARITY_STYLE[polarity]
            for status, status_color in (("active", "cyan"), ("ended", "black")):
                count = status_counts.get((polarity, status), 0)
                if count == 0:
                    continue
                legend_handles.append(plt.Line2D([0], [0], marker=style["marker"], color='w',
                                                  markerfacecolor=status_color, markersize=8,
                                                  markeredgecolor='k', markeredgewidth=0.5))
                legend_labels.append(f"{status} {polarity} ({count} eddies)")

    # Platform overlays (ru29/Falkor): black tail + a marker at the latest
    # reported position, same convention as SPICE_CMEMS_SAT.py (including
    # cutoff = plot_date + RUN_TS, not wall-clock "now" - matters here
    # specifically because plot_date can lag ~2 weeks behind real time, and
    # showing the glider's real-time position stapled onto that stale eddy
    # data would be a temporal mismatch. Using plot_date+RUN_TS instead
    # shows where the glider actually was at that same lagged point in
    # time, consistent with how SPICE_CMEMS_SAT.py already handles this for
    # its own lagging products like sss).
    try:
        run_h, run_m = (int(run_ts[:2]), int(run_ts[2:])) if len(run_ts) >= 4 else (0, 0)
    except (ValueError, TypeError):
        run_h, run_m = 0, 0
    platform_cutoff_naive = pd.Timestamp(plot_date.date()) + pd.Timedelta(hours=run_h, minutes=run_m)

    for platform in ACTIVE_PLATFORMS:
        track = get_platform_track(platform["csv"])
        if track is None or len(track) == 0:
            continue
        try:
            cutoff_platform = (platform_cutoff_naive if track['time'].dt.tz is None
                                else platform_cutoff_naive.tz_localize('UTC'))
            plot_track = track[track['time'] <= cutoff_platform]
            if len(plot_track) == 0:
                continue  # platform had not reported any position yet

            tail_start = cutoff_platform - pd.Timedelta(days=PLATFORM_TAIL_DAYS)
            tail_track = plot_track[plot_track['time'] >= tail_start]
            if len(tail_track) == 0:
                tail_track = plot_track.tail(1)

            tail_lons = tail_track['lon'].values.astype(float)
            tail_lats = tail_track['lat'].values.astype(float)
            last = plot_track.iloc[-1]
            lon_last, lat_last = float(last['lon']), float(last['lat'])
            # White matches SPICE_CMEMS_SAT.py's convention when the SLA
            # background is present (same cmo.balance colormap, proven to
            # contrast well there); black when there's no colored fill
            # behind it (plain white background), where white would vanish.
            tail_color = "white" if sla_mode else "black"
            ax.plot(tail_lons, tail_lats, '-', color=tail_color, lw=2.0,
                    transform=ccrs.PlateCarree(), zorder=50)
            marker, = ax.plot(lon_last, lat_last, platform["marker"], color=platform["color"],
                               markersize=platform.get("markersize", 8),
                               markeredgecolor='k', markeredgewidth=0.8,
                               transform=ccrs.PlateCarree(), zorder=51)
            t_str = last['time'].strftime('%Y-%m-%d %H:%M')
            legend_handles.append(marker)
            legend_labels.append(f"{platform['name']}\n{t_str} UTC")
        except Exception as e:
            print(f"  ERROR in {platform['name']} overlay: {e}")

    if legend_handles:
        legend_ncol = 3
        ordered = _legend_row_major_order(list(zip(legend_handles, legend_labels)), legend_ncol)
        ordered_handles, ordered_labels = zip(*ordered)
        fig.legend(handles=list(ordered_handles), labels=list(ordered_labels),
                   loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=legend_ncol,
                   frameon=True, fontsize=7, handletextpad=0.6, columnspacing=1.2)

    # Date only, no time-of-day - AVISO's eddy atlas is daily-resolution
    # (each obs is one eddy's position on one day), so a "%H:%M" would
    # imply false precision that isn't actually in the source data.
    title = f"Eddy trajectories (last {TAIL_DAYS}d as of {plot_date:%Y-%m-%d})"
    if sla_mode:
        # SLA's own date, not assumed to equal plot_date - see the
        # fetch_sla_background lookback note above.
        title += f"\nSLA {sla_date:%Y-%m-%d}"
    ax.set_title(title)
    ax._autotitlepos = False
    ax.title.set_position((0.5, 1.02))

    out_dir = os.path.join(base_dir, f"{plot_date:%Y}", f"{plot_date:%m}", f"{plot_date:%d}", "eddy_trajectory")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"eddy_trajectory_{plot_date:%Y%m%d}_{run_ts}.png")
    fig.canvas.draw()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path} ({n_eddies_total} eddies plotted)")
    return out_path


if __name__ == "__main__":
    run_ts = os.environ.get("RUN_TS", pd.Timestamp.now(tz="UTC").strftime("%H%M"))
    eddy_data = load_eddy_data(args.cmems_dir)
    sla_ds = load_sla_background(args.cmems_dir) if SLA_BACKGROUND_MODE else None
    if eddy_data:
        plot_eddies(eddy_data, sla_ds=sla_ds, run_ts=run_ts)
    else:
        print("No eddy data available - nothing plotted")