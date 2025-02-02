#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filename: preprocess_utility.py
Author: Lukas Kaleta
Date: 2025-01-31
Version: 1.0
Description: 
    This script contains functions for cml dataset preprocessing

License: 
Contact: 211312@vutbr.cz
"""


""" Imports """
# Import python libraries

import numpy as np
import xarray as xr

# Import external packages


# Import own modules


""" Variable definitions """


""" Function definitions """

def cml_preprocess(cml_set:xr.Dataset, interp_max_gap='5min'):
    """
    Preprocess cml dataset: remove fault values, calculate TRSL, interpolate missing values,
    and standardise values.

    Parameters
    cml_set : xarray.dataset containing several CMLs with rsl, tsl, timestamps and metadata
    interp_max_gap : string, default value = 5min, maximal gap in data to be interpolated
    
    Returns
    cml_set : xarray.dataset
    """

    cml_set['tsl'] = cml_set.tsl.where(cml_set.tsl != 255.0)
    cml_set['rsl'] = cml_set.rsl.where(cml_set.rsl != -99.9)
    cml_set['trsl'] = cml_set.tsl - cml_set.rsl
    cml_set['trsl'] = cml_set.trsl.interpolate_na(dim='time', method='linear', max_gap=interp_max_gap)

    cml_set = cml_set.transpose('cml_id', 'channel_id', 'time')

    # Standardising by subtracting median
    if 'trsl_st' in cml_set:
        cml_set = cml_set.reset_coords(['trsl_st'], drop=True)    # if trsl_st exists, erase it. (issue fix)
    cml_set['trsl_st'] = cml_set.trsl.copy()

    for i in range(len(cml_set.cml_id)):
        meanI = cml_set.trsl[i,0].mean()
        cml_set['trsl_st'][i,0] = (cml_set.trsl[i,0] - meanI) / (cml_set.trsl[i,0].max() - meanI)
        meanJ = cml_set.trsl[i,1].mean()
        cml_set['trsl_st'][i,1] = (cml_set.trsl[i,1] - meanJ) / (cml_set.trsl[i,1].max() - meanJ)

    return cml_set




def ref_preprocess(ref_set:xr.Dataset, interp_max_gap='20min', resample=10):
    """
    Preprocess reference rainfall dataset: iterpolate missing values,
    resample, and create wet/dry flag

    Parameters
    ref_set : xarray.dataset containing reference raifall data for given CMLs
    interp_max_gap : string, default value = 20min, maximal gap in data to be interpolated
    resample : int, default value = 10 (minutes), new sample rate of rainfall data

    Returns
    ref_set : xarray.dataset
    """
    ref_set['rain'] = ref_set.rain.interpolate_na(dim='time', method='linear', max_gap=interp_max_gap)

    ref_set = ref_set.resample(time=(str(resample)+'min')).sum()

    ref_set = ref_set.transpose('cml_id', 'time')
    # From reference rain rate create boolean reference Wet/Dry signal
    ref_set['ref_wd'] = ref_set.rain.where(ref_set.rain == 0, True).astype(bool)

    return ref_set




def build_dataset(cml_set:xr.Dataset, ref_set:xr.Dataset, sample_size = 10, num_cmls = float('nan')):
    """
    Connect cml and reference data into dataset

    Parameters
    cml_set : xarray.dataset containing several CMLs with rsl, tsl, trsl, timestamps and metadata
    ref_set : xarray.dataset containing reference rainrate and wet/dry data for given CMLs
    sample_size : int, default value = 10, length of samples for wet/dry classification in minutes
    num_cmls : float, optional, number of cmls to be inclued into dataset, default value NaN = all 

    Returns
    ds : xarray.dataset
    """

    # extract metadata:
    cml_id = cml_set.cml_id.values
    length = cml_set.length.values
    frequency = cml_set.frequency.values
    polarization = cml_set.polarization.values

    n_samples = len(cml_set.time) // sample_size    # get number of samples 
    # cutoff = len(cml_set.time)-n_samples*sample_size # last few points cant make a sample of full length. Will be cut off.


    # Reshape dataset to a new shape with the sample_id dimension 
    trsl_reshaped = cml_set['trsl'].values[:,:,:(n_samples*sample_size)].reshape(len(cml_set.cml_id),2, n_samples, sample_size)
    trsl_st_reshaped = cml_set['trsl_st'].values[:,:,:(n_samples*sample_size)].reshape(len(cml_set.cml_id),2, n_samples, sample_size)

    # reshape and add vector of time into dataset to remember the time stamp
    time_reshaped = cml_set['time'].values[:(n_samples*sample_size)].reshape(n_samples, sample_size)

    # how many cmls include into dataset for NaN include all
    if np.isnan(num_cmls):
        num_cmls = len(cml_id)

    ds = xr.Dataset({
        'trsl': (('cml_id', 'channel_id', 'sample_num', 'timestep'), trsl_reshaped[:num_cmls]),
        'trsl_st': (('cml_id', 'channel_id', 'sample_num', 'timestep'), trsl_st_reshaped[:num_cmls]),
        'rain': (('cml_id', 'sample_num'), ref_set['rain'].values[:num_cmls,:(n_samples*sample_size)]),
        'ref_wd': (('cml_id', 'sample_num'), ref_set['ref_wd'].values[:num_cmls,:(n_samples*sample_size)]),
        'time': (('sample_num', 'timestep'), time_reshaped)
    }, coords={'cml_id': cml_id[:num_cmls],
            'channel_id': np.arange(2),
            'sample_num': np.arange(n_samples),
            'timestep': np.arange(sample_size),
            'length':(('cml_id'), length[:num_cmls]),
            'frequency': (('cml_id', 'channel_id'), frequency[:num_cmls]),
            'polarization': (('cml_id', 'channel_id'), polarization[:num_cmls])
    })

    # for tensors we need to make sample_num the first dimension
    ds['trsl_st'] = ds.trsl_st.transpose('cml_id', 'sample_num', 'channel_id', 'timestep')

    return ds

## TODO: 
# exclude long dry periods. Make dataset 50:50

## TODO:
# Handle missing values
def exclude_missing_values(ds:xr.Dataset):
    """
    from given dataset, exclude cmls with missing values

    Parameters
    cml_set : xarray.dataset containing both several CMLs with rsl, tsl, trsl, 
        timestamps and metadata, reference rainrate and wet/dry data for given CMLs

    Returns
    ds : xarray.dataset
    """

    # get cml ids with NaN gaps
    ds['trsl_gap'] = (
            ('cml_id', 'sample_num', 'timestep'), 
            np.logical_or(np.isnan(ds.trsl.isel(channel_id=0).values), np.isnan(ds.trsl.isel(channel_id=1).values))
    )

    has_gap = np.any(ds.trsl_gap.values, axis=(1,2))
 
    ds = ds.assign_coords({'has_gap':(('cml_id'), has_gap)})
    ds = ds.where(~ds["has_gap"], drop=True)
    
    # When vals are dropped, bool variables turn into int, has to be turned back
    ds['trsl_gap'] = ds.trsl_gap.astype(bool)
    ds['ref_wd'] = ds.ref_wd.astype(bool)

    return ds

