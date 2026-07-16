#!/usr/bin/env python

"""
Author: Lori Garzio on 7/10/2026
Last modified: 7/14/2026
Plot cross-sections of real-time glider data
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
    ds = ds.swap_dims({'obs': 'time'})
    ds = ds.sortby(ds.time)
    ds = cf.add_teos10_variables(ds)
    
    deploy = ds.attrs['deployment']
    glider = ds.platform.attrs['id']

    save_dir = os.path.join(save_dir, deploy, 'xsection', time_range)
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

    for pv, info in plt_vars.items():
        try:
            variable = ds[pv]
        except KeyError:
            continue

        # plot xsection
        if np.sum(~np.isnan(variable.values)) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.subplots_adjust(left=0.1)
            figttl_xsection = f'{glider}: {info["title"]}\n{t0title} to {t1title}'
            clab = f'{info["title"]} ({variable.attrs["units"]})'
            xargs = dict()
            xargs['clabel'] = clab
            xargs['title'] = figttl_xsection
            xargs['date_fmt'] = '%m-%d\n%H:%M'
            xargs['grid'] = True
            xargs['cmap'] = info['cmap']
            xargs['markersize'] = 20
            xargs['cbar_min'] = info['min']
            xargs['cbar_max'] = info['max']
            pf.xsection(fig, ax, ds.time.values, ds.depth_interpolated.values, variable.values, **xargs)

            sfilename = f'{deploy}_xsection_{pv}_{time_range}.png'
            sfile = os.path.join(save_dir, sfilename)
            plt.savefig(sfile, dpi=300)
            plt.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Plot cross sections of real time glider data',
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
