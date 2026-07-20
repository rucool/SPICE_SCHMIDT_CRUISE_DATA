#! /usr/bin/env python

"""
Author: Lori Garzio on 7/10/2026
Last modified: 7/10/2026
"""

import gsw
from erddapy import ERDDAP


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
    """Convert 'salinity' (practical -> absolute) and 'temperature'
    (in-situ -> conservative) to TEOS-10 quantities in place, and replace the
    raw pulled in-situ 'density' with potential density (referenced to the
    sea surface). Variable names are kept the same - only the values and
    units/attrs change - so filenames built from these names (e.g. in
    plot_vars_glider.yml) do not change."""
    sa = gsw.SA_from_SP(ds['salinity'].values, ds['pressure'].values,
                         ds['longitude'].values, ds['latitude'].values)
    ct = gsw.CT_from_t(sa, ds['temperature'].values, ds['pressure'].values)
    potential_density = gsw.rho(sa, ct, 0)

    ds['salinity'] = ds['salinity'].copy(data=sa)
    ds['salinity'].attrs.update(units='g kg-1', long_name='Absolute Salinity',
                                 standard_name='sea_water_absolute_salinity')

    ds['temperature'] = ds['temperature'].copy(data=ct)
    ds['temperature'].attrs.update(units='degrees_Celsius', long_name='Conservative Temperature',
                                    standard_name='sea_water_conservative_temperature')

    ds['density'] = ds['density'].copy(data=potential_density)
    ds['density'].attrs.update(units='kg m-3', long_name='Potential Density',
                               standard_name='sea_water_potential_density')

    return ds
