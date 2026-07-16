#!/usr/bin/env python

"""
Author: Lori Garzio on 7/10/2026
Last modified: 7/14/2026
Plot profiles of real-time glider data, colored by time.
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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import functions.common as cf
import functions.plotting as pf
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
    ds = ds.assign_coords(depth_interpolated=ds.depth_interpolated)
    ds = ds.sortby(ds.profile_time)
    ds = cf.add_teos10_variables(ds)
    
    deploy = ds.attrs['deployment']
    glider = ds.platform.attrs['id']

    save_dir = os.path.join(save_dir, deploy, 'profiles', time_range)
    os.makedirs(save_dir, exist_ok=True)

    if time_range == 'last_24h':
        ds = ds.where(ds.time >= (np.nanmax(ds.time) - np.timedelta64(1, 'D')), drop=True)
    elif time_range == 'last_48h':
        ds = ds.where(ds.time >= (np.nanmax(ds.time) - np.timedelta64(2, 'D')), drop=True)
    
    t0str = pd.to_datetime(np.nanmin(ds.time)).strftime('%Y-%m-%dT%H:%M')
    t1str = pd.to_datetime(np.nanmax(ds.time)).strftime('%Y-%m-%dT%H:%M')

    t0title = pd.to_datetime(np.nanmin(ds.time)).strftime('%Y-%m-%d %H:%M')
    t1title = pd.to_datetime(np.nanmax(ds.time)).strftime('%Y-%m-%d %H:%M')

    # get plotting variable config file
    root_dir = os.path.dirname(os.path.abspath(__file__))
    configdir = os.path.join(root_dir, 'configs')

    # check if there is a deployment-specific config file, otherwise use the default
    if os.path.isfile(os.path.join(configdir, f'plot_vars_glider_{deploy}.yml')):
        configfile = os.path.join(configdir, f'plot_vars_glider_{deploy}.yml')
    else:
        if os.path.isfile(os.path.join(configdir, 'plot_vars_glider.yml')):
            configfile = os.path.join(configdir, 'plot_vars_glider.yml')
        else:
            raise FileNotFoundError(f'No config file found for deployment {deploy} and no default config file found in {configdir}.')
    with open(configfile) as f:
        plt_vars = yaml.safe_load(f)

    # find profiles
    ptimes = np.unique(ds.profile_time.values)

    # make a color map for profiles based on profile time
    cmap = plt.cm.rainbow
    ptime_nums = mdates.date2num(pd.to_datetime(ptimes))
    vmin = np.nanmin(ptime_nums)
    vmax = np.nanmax(ptime_nums)
    if vmin == vmax:
        vmax = vmin + 1
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    for pv, info in plt_vars.items():
        try:
            variable = ds[pv]
        except KeyError:
            continue

        scatter_args = dict(s=10, edgecolor='None')

        # plot profiles, colored by time
        if np.sum(~np.isnan(variable.values)) > 1:
            fig, ax = plt.subplots(figsize=(8, 10))
            plt.subplots_adjust(left=0.125, right=0.9)

            for i, pt in enumerate(ptimes):
                vpt = variable.sel(profile_time=pt)
                if np.sum(~np.isnan(vpt.values)) > 1:
                    ptime_num = mdates.date2num(pd.to_datetime(pt))
                    scatter_args['color'] = cmap(norm(ptime_num))
                    xc = ax.scatter(vpt.values, vpt.depth_interpolated.values, **scatter_args)

            # limit the x-axis to the min and max of the variable values
            xmin, xmax = ax.get_xlim()
            if info['min'] is not None:
                xmin = info['min']
            if info['max'] is not None:
                xmax = info['max']
            ax.set_xlim(xmin, xmax)
            
            ax.invert_yaxis()
            ax.set_ylabel('Depth (m)')
            ax.grid(ls='--', lw=.5)
            ax.ticklabel_format(useOffset=False)  # don't use scientific notation for ticks
            ax.set_xlabel(f'{info["title"]} ({variable.attrs["units"]})')
            figttl_profile = f'{glider}: {info["title"]}\n{t0title} to {t1title}'
            ax.set_title(figttl_profile, fontsize=14)

            # add colorbar to the right side of the plot for profile time
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.01)
            cbar.set_label('Profile Time (UTC)', labelpad=12)
            cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))

            sfilename = f'{deploy}_profile_{pv}_{time_range}.png'
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
