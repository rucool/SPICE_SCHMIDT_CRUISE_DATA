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
    
    ds = e.to_xarray(requests_kwargs={"timeout": 600})  # increase timeout to 10 minutes
    return ds
