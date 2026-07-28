#!/usr/bin/env python
import argparse
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean.cm as cmo
import dask
import os
import glob


arg_parser = argparse.ArgumentParser(description='Create CMEMS SSH SST CHL maps',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_parser.add_argument('-s', '--save_dir',
                        dest='save_dir',
                        type=str,
                        default='./satellite_figs',
                        help='Full file path to directory where figures are written')
arg_parser.add_argument('-c', '--cmems_dir',
                        dest='cmems_dir',
                        type=str,
                        default='./cmems_data',
                        help='Full file path to directory where downloaded CMEMS data is read from')
args = arg_parser.parse_args()

CMEMS_BASE_DIR = args.cmems_dir

# Shared bounding box for every map: [lon_min, lon_max, lat_min, lat_max]
TROP_WTRN_ATL_EXTENT = [-63, -40.75, 4, 19]

# product -> variables to plot (matches the folder names cmems_download.py writes into)
PRODUCT_PLOT_VARS = {
    "aviso_ssh": ["sla"],
    "sst": ["analysed_sst"],
    "ocean_color": ["CHL"],
    "sss": ["sos", "dos"],
    "sargassum": ["nfai"],
}


def load_latest(product_name, base_dir=CMEMS_BASE_DIR):
    """Open the most recently downloaded NetCDF for a product. cmems_download.py
    is what actually fetches the data - this just reads back what it wrote.
    Returns None (not an error) if no files exist yet - e.g. a newly added
    product whose download hasn't run yet, or a one-off missed fetch cycle -
    so one product's temporarily-missing data doesn't crash every other
    product's figures too (confirmed this could happen: sargassum's first
    cmems_download run on admin hadn't completed yet when SPICE_CMEMS_SAT.py
    ran, taking down sla/sst/CHL/sos/dos plotting along with it)."""
    files = sorted(glob.glob(os.path.join(base_dir, product_name, "*.nc")), key=os.path.getmtime)
    if not files:
        print(f"Warning: no downloaded files found for {product_name} in {base_dir} - skipping")
        return None
    return xr.open_dataset(files[-1])


daily_data = {
    product_name: load_latest(product_name) for product_name in PRODUCT_PLOT_VARS
}


def percentile(da, min=2, max=98):
    vmin = np.floor(np.nanpercentile(da, min))
    vmax = np.ceil(np.nanpercentile(da, max))
    return vmin, vmax


def _kelvin_to_celsius(da):
    da = da - 273.15
    da.attrs["units"] = "degC"
    return da


# colormap per variable, reused regardless of which product it came from.
# nfai deliberately does NOT use set_under() - tried pure white for the
# no-signal floor, but it looked too harsh/artificial against the natural
# gradient. Just clips to cmo.algae's own palest shade instead (its default
# behavior for values at/below vmin).
variable_cmaps = {
    "sla": cmo.balance,
    "adt": cmo.balance,
    "analysed_sst": cmo.thermal,
    "CHL": cmo.algae,
    "sos": cmo.haline,
    "dos": cmo.dense,
    "nfai": cmo.algae,
}

# display units per variable, after any conversion in variable_transforms below
variable_units = {
    "sla": "m",
    "adt": "m",
    "analysed_sst": "°C",
    "CHL": "mg m$^{-3}$",
    "sos": "psu",
    "dos": "kg m$^{-3}$",
}

# applied to the data right before plotting (e.g. analysed_sst arrives in Kelvin)
variable_transforms = {
    "analysed_sst": _kelvin_to_celsius,
}

# static cbar limits per variable, for day-to-day comparability
variable_clims = {
    'sla': (-0.2, 0.2),
    'analysed_sst': (25.0, 29.0),
    'CHL': (0.0, 7.0),
    'sos': (30.0, 37.0),
    'dos': (1018.0, 1025.0),
    # nfai is heavily floored at -0.5 ("confirmed water, no algae" - 97.9%
    # of valid pixels in a real test fetch sat at/below -0.4) with a hard
    # jump straight to real signal - confirmed zero pixels anywhere in
    # (-0.5, 0.02), so there's no gradient to preserve down there. vmin=0
    # sits right at that gap (any value in the empty range works
    # identically) paired with cmap.set_under('white') above, so the floor
    # renders as clean white instead of blending into pale green - real
    # detections then use the full 0-0.2 gradient (vmax tightened from the
    # true max of 0.5, which is a rare outlier - p99.9=0.225 is a much
    # better match for where real signal actually concentrates).
    'nfai': (0.0, 0.2),
}

# Pending PI feedback on which view of nfai communicates better: the
# continuous gradient above, or a simple binary detected/not-detected map
# (real signal is so sparse and jumps so sharply from the -0.5 floor - see
# variable_clims comment - that a binary view may read more honestly than
# implying a smooth gradient that isn't really there). Flip to True once
# decided; threshold matches the continuous mode's vmin (exactly where
# real signal starts).
NFAI_BINARY_MODE = True
NFAI_BINARY_THRESHOLD = 0.0

# contour line levels per variable - only drawn for variables listed here
variable_contour_levels = {
    'sla': np.arange(-0.2, 0.21, 0.1),
    'analysed_sst': np.arange(25.0, 29.1, 1.0),
    'sos': np.arange(30.0, 37.1, 0.5),
    'dos': np.arange(1018.0, 1025.1, 0.5),
}

FIG_BASE_DIR = args.save_dir

# Platforms to overlay on every map: track CSV (time, lat, lon columns,
# written by that platform's own fetch script) + legend marker style for its
# latest position. Tails are always white; only the latest-position marker
# varies so platforms stay distinguishable as more get added (e.g. more
# gliders alongside ru29 and Falkor (too) later in the cruise).
PLATFORMS = [
    {"name": "ru29", "csv": "ru29_latest_track.csv", "marker": "*", "color": "gold", "markersize": 10, "enabled": True},
    # off until the official cruise starts - flip to True to bring Falkor back
    {"name": "Falkor (too)", "csv": "falkor_track.csv", "marker": "^", "color": "magenta", "markersize": 8, "enabled": False},
]
ACTIVE_PLATFORMS = [p for p in PLATFORMS if p.get("enabled", True)]

# How much trailing history to draw as the white track line on each map. This
# only trims the drawn *line* and is applied relative to the date being
# plotted (see plot_and_save_variable), never to wall-clock "now" - it never
# hides the latest-position marker itself.
PLATFORM_TAIL_DAYS = 7


def get_platform_track(csv_name):
    """Read a platform's full position history from a CSV (time, lat, lon columns).

    Intentionally does NOT window by wall-clock "now": some products (e.g.
    weekly-updated SSS) plot a date over a week old, and a platform's last
    reported fix can itself be stale (e.g. during a network outage). Windowing
    here relative to "now" made the overlay silently vanish on both of those
    plots - see plot_and_save_variable for the actual (date-relative) windowing.
    """
    try:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_name)
        df = pd.read_csv(csv_path)
        df["time"] = pd.to_datetime(df["time"])
        return df.sort_values("time").reset_index(drop=True)
    except Exception as e:
        print(f"Warning: could not load track from {csv_name}: {e}")
        return None


def plot_and_save_variable(ds, var, bbox=TROP_WTRN_ATL_EXTENT, base_dir=FIG_BASE_DIR, platform_tracks=None, run_ts=""):
    """Save one standalone map per day present in ds. Returns the list of paths saved."""
    lon_min, lon_max, lat_min, lat_max = bbox
    paths = []
    platform_tracks = platform_tracks or {}

    for t in np.atleast_1d(ds.time.values):
        date = pd.Timestamp(t)
        data = ds[var].sel(time=t)
        if "depth" in data.dims:
            data = data.isel(depth=0)
        if var in variable_transforms:
            data = variable_transforms[var](data)

        # Copernicus can auto-clamp a partially out-of-range request instead
        # of erroring (see the SATELLITE_PRODUCTS registry comment in
        # cmems_download.py) - but for some products this still returns a
        # placeholder time step for a day beyond the product's real
        # coverage, filled entirely with NaN, rather than omitting that day.
        # Without this check, the day still gets a dated folder and a saved
        # PNG - coastlines/gridlines/title all draw fine since they don't
        # depend on the data - but pcolormesh renders nothing, producing a
        # blank-looking figure that's easy to mistake for a real (empty)
        # observation instead of "no data existed for this day at all".
        if bool(np.all(np.isnan(data.values))):
            print(f"Warning: {var} on {date:%Y-%m-%d} is all-NaN (no real data for this day) - skipping")
            continue

        vmin, vmax = variable_clims.get(var, (None, None))
        if vmin is None:
            vmin, vmax = percentile(data)

        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": ccrs.Mercator()})
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

        # Natural Earth features - avoids GSHHS download (404s on old cartopy URLs)
        ax.add_feature(cfeature.NaturalEarthFeature(
            'physical', 'land', '10m',
            edgecolor='black', facecolor='lightgray'
        ))
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)

        # Gridlines with labels
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False

        binary_mode = var == "nfai" and NFAI_BINARY_MODE
        if binary_mode:
            # NaN (no-data, e.g. cloud cover) must stay NaN, not get
            # silently folded into "no detection" - those mean different
            # things (confirmed clear water vs. we simply don't know).
            raw = data.values
            plot_values = np.where(raw > NFAI_BINARY_THRESHOLD, 1.0, 0.0)
            plot_values = np.where(np.isnan(raw), np.nan, plot_values)
            cmap = mcolors.ListedColormap(["white", "darkgreen"])
            vmin, vmax = -0.5, 1.5  # centers the 2 discrete cells on 0 and 1
        else:
            plot_values = data.values
            cmap = variable_cmaps.get(var, "viridis")

        im = ax.pcolormesh(
            ds.longitude.values, ds.latitude.values, plot_values,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree(),
        )

        levels = variable_contour_levels.get(var)
        if levels is not None and not binary_mode:
            cs = ax.contour(
                ds.longitude.values, ds.latitude.values, data.values,
                levels=levels, colors='k', linewidths=0.5,
                transform=ccrs.PlateCarree(),
            )
            ax.clabel(cs, inline=True, fontsize=6, fmt='%.1f')

        # Platform overlays: white tail up to figure date/time + a marker
        # (shape/color set per platform) at the latest position. Each
        # platform's latest fix feeds a legend entry in the upper right.
        legend_handles = []
        legend_labels = []
        for platform in ACTIVE_PLATFORMS:
            track = platform_tracks.get(platform["name"])
            if track is None or len(track) == 0:
                continue
            try:
                h, m = int(run_ts[:2]), int(run_ts[2:]) if len(run_ts) >= 4 else (0, 0)
            except (ValueError, TypeError):
                h, m = 0, 0
            cutoff = pd.Timestamp(date.date()) + pd.Timedelta(hours=h, minutes=m)
            _times = track['time']
            if _times.dt.tz is not None:
                cutoff = cutoff.tz_localize('UTC')
            plot_track = track[_times <= cutoff]
            if len(plot_track) == 0:
                continue  # platform had not reported any position yet as of this map's date

            # Trailing line is capped to PLATFORM_TAIL_DAYS before cutoff (for
            # readability); the latest-position marker below always uses
            # plot_track's last row regardless, so a lagged product (e.g.
            # weekly SSS plotting a date >7 days old) or a stale/outage-frozen
            # fix never makes the marker disappear - only the tail shortens.
            tail_start = cutoff - pd.Timedelta(days=PLATFORM_TAIL_DAYS)
            tail_track = plot_track[plot_track['time'] >= tail_start]
            if len(tail_track) == 0:
                tail_track = plot_track.tail(1)

            try:
                tail_lons = tail_track['lon'].values.astype(float)
                tail_lats = tail_track['lat'].values.astype(float)
                last = plot_track.iloc[-1]
                lon_last, lat_last = float(last['lon']), float(last['lat'])
                print(f"  {platform['name']} overlay: {len(tail_lons)} tail pts, latest=({lat_last:.2f},{lon_last:.2f}) @ {last['time']}")
                # White works for every other product (all have a colored
                # pcolormesh fill behind it), but nfai's binary mode renders
                # "no detection" as plain white, which would make a white
                # tail invisible - use black there instead.
                tail_color = 'black' if binary_mode else 'white'
                ax.plot(tail_lons, tail_lats, '-', color=tail_color, lw=2.0,
                        transform=ccrs.PlateCarree(), zorder=50)
                marker, = ax.plot(lon_last, lat_last, platform["marker"], color=platform["color"],
                                   markersize=platform.get("markersize", 8),
                                   markeredgecolor='k', markeredgewidth=0.8,
                                   transform=ccrs.PlateCarree(), zorder=51)
                _t = last['time']
                t_str = _t.strftime('%Y-%m-%d %H:%M') if hasattr(_t, 'strftime') else str(_t)[:16]
                legend_handles.append(marker)
                legend_labels.append(f"{platform['name']}\n{t_str} UTC")
            except Exception as _e:
                print(f"  ERROR in {platform['name']} overlay: {_e}")

        if legend_handles:
            ax.legend(legend_handles, legend_labels, loc='upper right', fontsize=6,
                      framealpha=0.85, handletextpad=0.6, borderpad=0.6)

        unit = variable_units.get(var, "")
        cbar_label = f"{var} ({unit})" if unit else var
        if binary_mode:
            cbar = fig.colorbar(im, ax=ax, orientation="horizontal", label=cbar_label, shrink=0.8, pad=0.08,
                                 ticks=[0, 1])
            cbar.ax.set_xticklabels(["no detection", "detected"])
        else:
            fig.colorbar(im, ax=ax, orientation="horizontal", label=cbar_label, shrink=0.8, pad=0.08)
        ax.set_title(f"{var} {date:%Y-%m-%d %H:%M}")
        # cartopy 0.25.0 bug workaround: GeoAxes hides its xaxis, which makes
        # matplotlib's automatic title-positioning logic compute an infinite
        # y-position for the title (falls back badly instead of cleanly), which
        # then poisons ax.get_tightbbox() (used by bbox_inches="tight") - so the
        # saved PNG could get cropped down to near-blank. No fixed cartopy
        # release exists on PyPI yet as of this writing.
        ax._autotitlepos = False
        ax.title.set_position((0.5, 1.02))

        out_dir = os.path.join(base_dir, f"{date:%Y}", f"{date:%m}", f"{date:%d}", var)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{var}_{date:%Y%m%d}_{run_ts}.png")
        # Force a full render before the tight-bbox crop, otherwise lazily-drawn
        # GeoAxes features (coastlines/land) can end up excluded from the saved
        # bbox on some matplotlib/cartopy version combos, cropping to near-blank.
        fig.canvas.draw()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(out_path)

    return paths


# Fetch every platform's track once to overlay on all figures
run_ts = os.environ.get("RUN_TS", pd.Timestamp.now(tz="UTC").strftime("%H%M"))
platform_tracks = {}
for platform in ACTIVE_PLATFORMS:
    track = get_platform_track(platform["csv"])
    platform_tracks[platform["name"]] = track
    print(f"{platform['name']} track: {len(track)} positions" if track is not None else f"{platform['name']} track: not available (check log above)")

# standalone plot per variable per day, saved to satellite_figs/yyyy/mm/dd/variable/
saved_paths = []
for product_name, plot_vars in PRODUCT_PLOT_VARS.items():
    ds = daily_data[product_name]
    if ds is None:
        continue
    for var in plot_vars:
        saved_paths.extend(plot_and_save_variable(ds, var, platform_tracks=platform_tracks, run_ts=run_ts))

print("Saved figures:", saved_paths)
