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
    """Add TEOS-10 absolute salinity, conservative temperature, and potential
    density (referenced to the sea surface) to a glider dataset that has
    'salinity' (practical), 'temperature' (in-situ), 'pressure', 'latitude',
    and 'longitude' variables. Added as new variables alongside the raw
    pulled ones (not overwritten), so plot_vars_glider.yml can select which
    to plot."""
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
