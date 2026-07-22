#! /usr/bin/env python

"""
Author: Lori Garzio on 7/10/2026
Last modified: 7/21/2026
"""

import gsw
from erddapy import ERDDAP
import numpy as np
import xarray as xr


def add_profile_time(ds, profile_var='profile_num'):
    """Add a profile_time variable to a glider dataset which is the 
    mean time of each profile"""
    profile_times = np.array([])
    profiles = np.unique(ds[profile_var].values)
    for p in profiles:
        idx = np.where(ds[profile_var].values == p)[0]
        pt = float(np.nanmean(ds.time.values[idx].astype('datetime64[s]').astype('int64')))
        profile_times = np.append(profile_times, [pt] * len(idx))       
    
    attrs = {
        'long_name': 'Profile Time', 
        'standard_name': 'profile_time', 
        'units': 'seconds since 1970-01-01 00:00:00',
        'comment': 'Unique identifier of the profile. The profile ID is the mean profile timestamp'
        }
    da = xr.DataArray(profile_times, coords=ds[profile_var].coords, dims=ds[profile_var].dims,
                                  name='profile_time', attrs=attrs)
    
    ds['profile_time'] = da
    return ds


def get_erddap_dataset(server, ds_id, variables=None, constraints=None):
    e = ERDDAP(server=server,
               protocol='tabledap',
               response='nc')
                      
    e.dataset_id = ds_id
    if constraints:
        e.constraints = constraints
    if variables:
        e.variables = variables
    
    try:
        ds = e.to_xarray(requests_kwargs={"timeout": 60})  # timeout 60 seconds
    except Exception as err:
        print(f'Error downloading dataset {ds_id} from ERDDAP server {server}: {err}')
        ds = None
    
    return ds


def add_teos10_variables(ds):
    """Add TEOS-10 absolute salinity, conservative temperature, and potential
    density (referenced to the sea surface) to a glider dataset that has
    'salinity' (practical), 'temperature' (in-situ), 'pressure',
    'latitude', and 'longitude' variables. Added as NEW variables
    (absolute_salinity, conservative_temperature, potential_density) -
    the raw pulled salinity/temperature/density are left untouched, so
    callers that need the original practical/in-situ values still have
    them."""
    sa = gsw.SA_from_SP(ds['salinity'].values, ds['pressure'].values,
                         ds['longitude'].values, ds['latitude'].values)
    ct = gsw.CT_from_t(sa, ds['temperature'].values, ds['pressure'].values)
    potential_density = gsw.rho(sa, ct, 0)

    ds['absolute_salinity'] = ds['salinity'].copy(data=sa)
    ds['absolute_salinity'].attrs.update(units='g kg-1', long_name='Absolute Salinity',
                                          standard_name='sea_water_absolute_salinity')

    ds['conservative_temperature'] = ds['temperature'].copy(data=ct)
    ds['conservative_temperature'].attrs.update(units='degrees_Celsius', long_name='Conservative Temperature',
                                                 standard_name='sea_water_conservative_temperature')

    ds['potential_density'] = ds['density'].copy(data=potential_density)
    ds['potential_density'].attrs.update(units='kg m-3', long_name='Potential Density',
                                          standard_name='sea_water_potential_density')

    return ds
