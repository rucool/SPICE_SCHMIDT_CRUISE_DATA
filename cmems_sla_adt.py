#!/usr/bin/env python
import argparse
import simplekml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import xarray as xr
import glob
import numpy.ma as ma
from netCDF4 import Dataset, date2index, num2date
import cmocean.cm as cmo
import pandas as pd
from pathlib import Path
import os
from datetime import date, timedelta
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


arg_parser = argparse.ArgumentParser(description='Create CMEMS SLA/ADT and CHL kmz imagery',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_parser.add_argument('-s', '--save_dir',
                        dest='save_dir',
                        type=str,
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help='Full file path to base output directory (dated cmems_YYYY_MM_DD folders written here)')
arg_parser.add_argument('-c', '--cmems_dir',
                        dest='cmems_dir',
                        type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cmems_data'),
                        help='Full file path to directory where downloaded CMEMS data is read from')
args = arg_parser.parse_args()

CMEMS_BASE_DIR = args.cmems_dir

# Shared bounding box for every map: [lon_min, lon_max, lat_min, lat_max]
TROP_WTRN_ATL_EXTENT = [-63, -40.75, 4, 19]


def load_latest(product_name, base_dir=CMEMS_BASE_DIR):
    """Open the most recently downloaded NetCDF for a product. cmems_download.py
    is what actually fetches the data - this just reads back what it wrote."""
    files = sorted(glob.glob(os.path.join(base_dir, product_name, "*.nc")), key=os.path.getmtime)
    # .load() + .close() eagerly pulls data into memory and frees the file
    # handle immediately - this function is now called for multiple products
    # in one process (sla then CHL), and leaving prior files open/lazy caused
    # a real cross-product variable mixup on this machine when opened back-
    # to-back in the same process.
    ds = xr.open_dataset(files[-1])
    ds.load()
    ds.close()
    return ds


def gearth_fig(llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat, pixels=1024):
    """Return a Matplotlib fig and ax handles for a Google-Earth Image."""
    aspect = np.cos(np.mean([llcrnrlat, urcrnrlat]) * np.pi/180.0)
    xsize = np.ptp([urcrnrlon, llcrnrlon]) * aspect
    ysize = np.ptp([urcrnrlat, llcrnrlat])
    aspect = ysize / xsize

    if aspect > 1.0:
        figsize = (10.0 / aspect, 10.0)
    else:
        figsize = (10.0, 10.0 * aspect)

    if False:
        plt.ioff()
    fig = plt.figure(figsize=figsize,
                     frameon=False,
                     dpi=pixels//10)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(llcrnrlon, urcrnrlon)
    ax.set_ylim(llcrnrlat, urcrnrlat)
    return fig, ax



def lon180to360(array):
    array = np.array(array)
    return np.mod(array, 360)

def lon360to180(array):
    array = np.array(array)
    return np.mod(array+180, 360)-180


def save_colorbar_legend(cfg, var_name, out_path):
    """Standalone colorbar-only image for one variable (color limits are
    static, so one legend covers every frame). Kept separate from the
    georeferenced ground-overlay frames on purpose - baking the colorbar
    into the map image itself covers whatever real data sits underneath it;
    this instead becomes a fixed KML ScreenOverlay that sits in a screen
    corner and never overlaps the map."""
    vmin, vmax = cfg['clim']
    if cfg['log_scale']:
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cfg['cmap'], norm=norm)

    fig = plt.figure(figsize=(3.4, 0.9), dpi=150)
    fig.patch.set_facecolor('black')
    cbaxes = fig.add_axes([0.08, 0.5, 0.84, 0.3])
    cbar = fig.colorbar(sm, cax=cbaxes, orientation='horizontal', ticks=cfg['ticks'])
    if cfg['ticks'] is not None:
        cbar.ax.set_xticklabels([str(t) for t in cfg['ticks']])
    cbar.ax.xaxis.set_tick_params(color='white', labelcolor='white')
    cbar.set_label(f"{var_name} ({cfg['units']})", color='white')
    fig.savefig(out_path, facecolor=fig.get_facecolor(), format='png')
    plt.close(fig)


# Create main folder with today's date inside the script's directory
today = date.today()
base_folder = os.path.join(args.save_dir, today.strftime("cmems_%Y_%m_%d"))
os.makedirs(base_folder, exist_ok=True)

# Create subfolders for each plotted variable plus 'kmz'
subfolders = ['sla', 'CHL', 'kmz']
for sub in subfolders:
    os.makedirs(os.path.join(base_folder, sub), exist_ok=True)


lonmin, lonmax, latmin, latmax = TROP_WTRN_ATL_EXTENT
pixels = 1024

# Per-variable config: which downloaded product folder to read (matches
# cmems_download.py's SATELLITE_PRODUCTS keys), colormap, color limits,
# and contour levels. CHL uses a log color scale (LogNorm, not vmin/vmax -
# the two are mutually exclusive on pcolormesh) since chlorophyll commonly
# spans 2+ orders of magnitude in one map (open-ocean vs. coastal/river-
# plume waters) that a linear scale would wash out - matches CHL_LOG_CLIM /
# CHL_LOG_TICKS / variable_contour_levels['CHL'] in SPICE_CMEMS_SAT.py so
# this kmz reads the same as the static maps.
var_config = {
    'sla': {
        'product': 'aviso_ssh',
        'cmap': cmo.balance,
        'clim': (-0.2, 0.2),
        'log_scale': False,
        'ticks': None,
        'contour_levels': np.arange(-0.2, 0.21, 0.1),
        'units': 'm',
    },
    'CHL': {
        'product': 'ocean_color',
        'cmap': cmo.algae,
        'clim': (0.03, 10.0),
        'log_scale': True,
        'ticks': [0.03, 0.1, 0.3, 1, 3, 10],
        'contour_levels': [0.1, 1, 5],
        'units': 'mg m$^{-3}$',
    },
}
var_list = ['sla', 'CHL']

for var_name in var_list:
    print(f"Processing variable: {var_name}")

    cfg = var_config[var_name]
    ds = load_latest(cfg['product'])
    lat = ds.latitude.data
    lon = ds.longitude.data
    vmin, vmax = cfg['clim']
    levels = cfg['contour_levels']

    timestamps = []
    fig_paths = []

    for i in range(len(ds.time)):
        var = ds[var_name][i, :, :]
        time_val = pd.to_datetime(var.time.values)
        timestamps.append(time_val.strftime("%Y-%m-%dT%H:%M:%SZ"))

        fig, ax = gearth_fig(llcrnrlon=lonmin, llcrnrlat=latmin,
                            urcrnrlon=lonmax, urcrnrlat=latmax, pixels=pixels)

        pcolormesh_kwargs = {'cmap': cfg['cmap'], 'shading': 'auto'}
        if cfg['log_scale']:
            pcolormesh_kwargs['norm'] = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            pcolormesh_kwargs['vmin'] = vmin
            pcolormesh_kwargs['vmax'] = vmax
        cb = ax.pcolormesh(lon, lat, var, **pcolormesh_kwargs)

        if levels is not None:
            cs = ax.contour(lon, lat, var, levels=levels, colors='k', linewidths=0.5)
            ax.clabel(cs, inline=True, fontsize=6, fmt='%.1f')

        ax.set_xticks([])
        ax.set_yticks([])

        ax.text(0.5, 0.97, f'{var_name} {time_val.strftime("%Y-%m-%d %H:%M UTC")}',
                transform=ax.transAxes, ha='center', va='top',
                color='white', fontsize=16, fontweight='bold',
                path_effects=[pe.withStroke(linewidth=3, foreground='black')])

        fname = f'{base_folder}/{var_name}/{var_name}_{time_val.strftime("%Y%m%dT%H%M%S")}.png'
        fig.canvas.draw()  # force full render before tight-bbox crop
        fig.savefig(fname, bbox_inches='tight', transparent=True, format='png')
        plt.close(fig)
        fig_paths.append(fname)



    fig_list = sorted(fig_paths)

    legend_path = f'{base_folder}/kmz/{var_name}_legend.png'
    save_colorbar_legend(cfg, var_name, legend_path)

    kml = simplekml.Kml()
    for ii, fig_path in enumerate(fig_list):
        fname = Path(fig_path).stem
        ground = kml.newgroundoverlay(name=fname)
        ground.draworder = ii + 1
        ground.icon.href = fig_path
        ground.gxlatlonquad.coords = [(lonmin, latmin), (lonmax, latmin),
                                    (lonmax, latmax), (lonmin, latmax)]

        ts_begin = pd.to_datetime(timestamps[ii]).strftime("%Y-%m-%dT%H:%M:%SZ")
        if ii < len(timestamps) - 1:
            ts_end = (pd.to_datetime(timestamps[ii + 1]) - pd.Timedelta(seconds=0.5)).strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            ts_end = (pd.to_datetime(timestamps[ii]) + pd.Timedelta(seconds=0.5)).strftime("%Y-%m-%dT%H:%M:%SZ")

        ground.timespan.begin = ts_begin
        ground.timespan.end = ts_end

    # Legend as a fixed on-screen overlay (not draped on the globe) - stays
    # pinned to the lower-left of the viewport regardless of zoom/tilt, and
    # never sits on top of real map data the way a baked-in colorbar would.
    screen = kml.newscreenoverlay(name=f'{var_name} legend')
    screen.icon.href = legend_path
    screen.overlayxy = simplekml.OverlayXY(x=0, y=0, xunits=simplekml.Units.fraction, yunits=simplekml.Units.fraction)
    screen.screenxy = simplekml.ScreenXY(x=0.02, y=0.02, xunits=simplekml.Units.fraction, yunits=simplekml.Units.fraction)
    screen.size.x = -1
    screen.size.y = -1
    screen.size.xunits = simplekml.Units.fraction
    screen.size.yunits = simplekml.Units.fraction

    model_name = f"CMEMS_{var_name}"
    kml.savekmz(f"{base_folder}/kmz/{model_name}.kmz")