#!/usr/bin/env python

"""
Author: Lori Garzio on 7/15/2026
Last modified: 7/15/2026
Generate T-S diagrams of real-time glider data colored by depth.
The full timeseries, last 24 hours, and last 48 hours
can be plotted. The default is to plot the full timeseries.
"""

import os
import argparse
import sys
import numpy as np
import pandas as pd
import yaml
import cmocean as cmo
import gsw
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import functions.common as cf
plt.rcParams.update({'font.size': 12})


def main(args):
    dsid = args.deployment
    server = args.server
    time_range = args.time_range
    save_dir = args.save_dir

    ds = cf.get_erddap_dataset(server, dsid)
    if ds is None:
        print(f'No dataset returned for {dsid}')
        sys.exit(1)
    ds = ds.drop_dims(['trajectory', 'profile'])
    ds = ds.swap_dims({'obs': 'profile_time'})
    ds = ds.sortby(ds.profile_time)
    ds = cf.add_teos10_variables(ds)
    
    deploy = ds.attrs['deployment']
    glider = ds.platform.attrs['id']

    save_dir = os.path.join(save_dir, deploy, 'TS', time_range)
    os.makedirs(save_dir, exist_ok=True)

    if time_range == 'last_24h':
        ds = ds.where(ds.time >= (np.nanmax(ds.time) - np.timedelta64(1, 'D')), drop=True)
    elif time_range == 'last_48h':
        ds = ds.where(ds.time >= (np.nanmax(ds.time) - np.timedelta64(2, 'D')), drop=True)
    
    t0str = pd.to_datetime(np.nanmin(ds.time)).strftime('%Y-%m-%dT%H:%M')
    t1str = pd.to_datetime(np.nanmax(ds.time)).strftime('%Y-%m-%dT%H:%M')

    t0title = pd.to_datetime(np.nanmin(ds.time)).strftime('%Y-%m-%d %H:%M')
    t1title = pd.to_datetime(np.nanmax(ds.time)).strftime('%Y-%m-%d %H:%M')

    # get config file for constraining x and y axes
    root_dir = os.path.dirname(os.path.abspath(__file__))
    configdir = os.path.join(root_dir, 'configs')

    # check if there is a deployment-specific config file, otherwise use the default
    if os.path.isfile(os.path.join(configdir, f'TS_glider{deploy}.yml')):
        configfile = os.path.join(configdir, f'TS_glider{deploy}.yml')
    else:
        if os.path.isfile(os.path.join(configdir, 'TS_glider.yml')):
            configfile = os.path.join(configdir, 'TS_glider.yml')
        else:
            raise FileNotFoundError(f'No config file found for deployment {deploy} and no default config file found in {configdir}.')
    with open(configfile) as f:
        config_file = yaml.safe_load(f)

    try:
        sal = ds.salinity
    except KeyError:
        print(f'Salinity not found in dataset: {dsid}')
        sys.exit(1)
    
    try:
        temp = ds.temperature
    except KeyError:
        print(f'Temperature not found in dataset: {dsid}')
        sys.exit(1)

    try:
        d = ds.depth_interpolated
    except KeyError:
        print(f'Interp depth not found in dataset: {dsid}. Trying to use depth instead')
        try:
            d = ds.depth
        except KeyError:
            print(f'Depth not found in dataset: {dsid}')
            sys.exit(1)

    scatter_args = dict(c=d, cmap=cmo.cm.deep, s=10, edgecolor='None')
    
    # limit the colorbar if specified in the config file
    if config_file['depth']['min'] is not None:
        scatter_args['vmin'] = config_file['depth']['min']
    if config_file['depth']['max'] is not None:
        scatter_args['vmax'] = config_file['depth']['max']

    fig, ax = plt.subplots(figsize=(10, 8))
    # ax.set_facecolor('#e8e8e8')  # set background color of plot to light gray
    plt.subplots_adjust(left=0.125, right=0.9)
    xc = ax.scatter(sal, temp, **scatter_args)
    
    # limit the x-axis
    xmin, xmax = ax.get_xlim()
    if config_file['salinity']['min'] is not None:
        xmin = config_file['salinity']['min']
    if config_file['salinity']['max'] is not None:
        xmax = config_file['salinity']['max']
    ax.set_xlim(xmin, xmax)

    # limit the y-axis
    ymin, ymax = ax.get_ylim()
    if config_file['temperature']['min'] is not None:
        ymin = config_file['temperature']['min']
    if config_file['temperature']['max'] is not None:
        ymax = config_file['temperature']['max']
    ax.set_ylim(ymin, ymax)

    # format labels
    ax.set_ylabel(f'{config_file["temperature"]["title"]} ({temp.attrs["units"]})')
    ax.set_xlabel(f'{config_file["salinity"]["title"]} ({sal.attrs["units"]})')
    ax.ticklabel_format(useOffset=False)  # don't use scientific notation for ticks
    ttl = f'{glider} T-S diagram\n{t0title} to {t1title}'
    ax.set_title(ttl, fontsize=14)

    # add colorbar to the right side of the plot for depth
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size='5%', pad=0.1, axes_class=plt.Axes)
    fig.add_axes(cax)
    cb = plt.colorbar(xc, cax=cax, label='Depth (m)')
    cb.ax.invert_yaxis()

    # plot the density contours
    n = 499

    tempL = np.linspace(ymin, ymax, n)
    salL = np.linspace(xmin, xmax, n)
    T, S = np.meshgrid(tempL, salL)

    sigma_theta = gsw.sigma0(S, T) + 1000

    lmin = np.floor(np.nanmin(sigma_theta))
    lmax = np.ceil(np.nanmax(sigma_theta))
    levels = np.arange(lmin, lmax, 0.5)
    contour_args = dict(colors='grey', alpha=0.3, linestyles='dashed', zorder=1, levels=levels)
    cs = ax.contour(S, T, sigma_theta, **contour_args)
    
    ax.clabel(cs, levels, fontsize=12, inline=True, fmt='%.1f')

    sfilename = f'{deploy}_TS_{time_range}.png'
    sfile = os.path.join(save_dir, sfilename)
    plt.savefig(sfile, dpi=300)
    plt.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Plot profiles of real time glider data',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('deployment',
                            type=str,
                            help='Glider deployment ID formatted as glider-YYYYmmddTHHMM-profile-sci-rt')
    
    arg_parser.add_argument('-server',
                            dest='server',
                            default='http://slocum-data.marine.rutgers.edu//erddap',
                            help='ERDDAP server (default is slocum-data)')

    arg_parser.add_argument('-time_range',
                            dest='time_range',
                            default='synoptic',
                            choices=['synoptic', 'last_24h', 'last_48h'],
                            help='Time range for plotting (default is full)')

    arg_parser.add_argument('-s', '--save_dir',
                            dest='save_dir',
                            type=str,
                            help='Full file path to save directory')

    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))
