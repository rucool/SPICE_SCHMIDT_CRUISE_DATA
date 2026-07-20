#!/usr/bin/env python
import argparse
import simplekml
import numpy as np
import matplotlib.pyplot as plt
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


arg_parser = argparse.ArgumentParser(description='Create CMEMS SLA/ADT kmz imagery',
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
    return xr.open_dataset(files[-1])


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




# Create main folder with today's date inside the script's directory
today = date.today()
base_folder = os.path.join(args.save_dir, today.strftime("cmems_%Y_%m_%d"))
os.makedirs(base_folder, exist_ok=True)

# Create subfolders for 'sla' and 'kmz'
subfolders = ['sla', 'kmz']
for sub in subfolders:
    os.makedirs(os.path.join(base_folder, sub), exist_ok=True)


ds = load_latest("aviso_ssh")


timestamps = []
lat = ds.latitude.data
lon = ds.longitude.data
lonmin, lonmax, latmin, latmax = TROP_WTRN_ATL_EXTENT
pixels = 1024
var_list = ['sla']

variable_clims = {'sla': (-0.2, 0.2)}
contour_levels = {'sla': np.arange(-0.2, 0.21, 0.1)}

for var_name in var_list:
    print(f"Processing variable: {var_name}")

    vmin, vmax = variable_clims[var_name]
    fig_paths = []

    for i in range(len(ds.time)):
        var = ds[var_name][i, :, :]
        time_val = pd.to_datetime(var.time.values)
        timestamps.append(time_val.strftime("%Y-%m-%dT%H:%M:%SZ"))

        fig, ax = gearth_fig(llcrnrlon=lonmin, llcrnrlat=latmin,
                            urcrnrlon=lonmax, urcrnrlat=latmax, pixels=pixels)
        cb = ax.pcolormesh(lon, lat, var, cmap=cmo.balance, vmin=vmin, vmax=vmax, shading='auto')

        levels = contour_levels.get(var_name)
        if levels is not None:
            cs = ax.contour(lon, lat, var, levels=levels, colors='k', linewidths=0.5)
            ax.clabel(cs, inline=True, fontsize=6, fmt='%.1f')

        ax.set_xticks([])
        ax.set_yticks([])

        cbaxes = fig.add_axes([0.25, 0.91, 0.5, 0.02])
        cbar = plt.colorbar(cb, cax=cbaxes, orientation='horizontal')
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.xaxis.tick_top()
        cbar.ax.xaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='white')
        cbar.set_label(f'{var_name} m {time_val.strftime("%Y-%m-%dT%H:%M:%SZ")}', color='w', labelpad=20)

        fname = f'{base_folder}/{var_name}/{var_name}_{time_val.strftime("%Y%m%dT%H%M%S")}.png'
        fig.canvas.draw()  # force full render before tight-bbox crop
        fig.savefig(fname, bbox_inches='tight', transparent=True, format='png')
        plt.close(fig)
        fig_paths.append(fname)



    fig_list = sorted(fig_paths)

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

    model_name = f"CMEMS_{var_name}"
    kml.savekmz(f"{base_folder}/kmz/{model_name}.kmz")

