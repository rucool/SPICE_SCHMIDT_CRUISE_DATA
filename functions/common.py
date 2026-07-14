#! /usr/bin/env python

"""
Author: Lori Garzio on 7/10/2026
Last modified: 7/10/2026
"""

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
