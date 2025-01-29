#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filename: train_full_dataset.py
Author: Lukas Kaleta
Date: 2025-01-21
Version: 1.0
Description: 
    This script showcases the typical workflow of training an CNN module
    for rain event detection using data from CML.

License: 
Contact: 211312@vutbr.cz
"""


""" Imports """
# Import libraries

import math
import numpy as np
import itertools
import matplotlib.pyplot as plt

import xarray as xr
import pandas as pd

import torch
import torch.nn as nn
import sklearn.metrics as skl
from sklearn.utils import shuffle
from tqdm import tqdm
#from IPython.display import clear_output

# Import externat packages
import pycomlink as pycml

# Import own modules
import modul.cnn as cnn


""" Variable definitions """


""" Function definitions """



""" main """


# load 500 CMLs with 1 min time step
cml_set = xr.open_dataset('example_data/example_cml_data.nc', engine='netcdf4') 

cml_set = cml_set.reset_coords(['site_a_latitude','site_b_latitude','site_a_longitude','site_b_longitude'], drop=True)

cml_set['tsl'] = cml_set.tsl.where(cml_set.tsl != 255.0)
cml_set['rsl'] = cml_set.rsl.where(cml_set.rsl != -99.9)
cml_set['trsl'] = cml_set.tsl - cml_set.rsl
cml_set['trsl'] = cml_set.trsl.interpolate_na(dim='time', method='linear', max_gap='5min')

cml_set = cml_set.transpose('cml_id', 'channel_id', 'time')

# load path averaged reference RADOLAN data aligned with all 500 CML IDs with 5 min time step
ref_set = xr.open_dataset('example_data/example_path_averaged_reference_data.nc', engine='netcdf4')
ref_set = ref_set.rename_vars({'rainfall_amount':'rain'})

ref_set['rain'] = ref_set.rain.interpolate_na(dim='time', method='linear', max_gap='20min')
ref_set = ref_set.resample(time="10min").sum()

ref_set = ref_set.transpose('cml_id', 'time')

# From reference rain rate create boolean reference Wet/Dry signal
ref_set['ref_wd'] = ref_set.rain.where(ref_set.rain == 0, True).astype(bool)


# get cml ids with NaN gaps
cml_set['trsl_gap'] = (
        ('cml_id', 'time'), 
        np.logical_or(np.isnan(cml_set.trsl.isel(channel_id=0).values), np.isnan(cml_set.trsl.isel(channel_id=1).values))
)

has_gap = np.any(cml_set.trsl_gap.values, axis=1)
cmls_without_gap = np.where(~has_gap)[0]
num_correct_cmls = len(cmls_without_gap)

cml_set = cml_set.assign_coords({'has_gap':(('cml_id'), has_gap)})
ref_set = ref_set.assign_coords({'has_gap':(('cml_id'), has_gap)})

# drop cmls with NaN gaps
if 1:
    cml_set = cml_set.where(~cml_set["has_gap"], drop=True)
    ref_set = ref_set.where(~ref_set["has_gap"], drop=True)
    # When vmls are dropped, bool variables turn into int, has to be turned back
    cml_set['trsl_gap'] = cml_set.trsl_gap.astype(bool)


# Build dataset
# extract metadata:
cml_id = cml_set.cml_id.values
length = cml_set.length.values
frequency = cml_set.frequency.values
polarization = cml_set.polarization.values


sample_size = 10 # = minutes
n_samples = len(cml_set.time) // sample_size    # get number of samples 
cutoff = len(cml_set.time)-n_samples*sample_size # last few points cant make a sample of full length. Will be cut off.


# Reshape dataset to a new shape with the sample_id dimension 
trsl_reshaped = cml_set['trsl'].values[:,:,:(n_samples*sample_size)].reshape(len(cml_set.cml_id),2, n_samples, sample_size)

# reshape and add vector of time into dataset to remember the time stamp
time_reshaped = cml_set['time'].values[:(n_samples*sample_size)].reshape(n_samples, sample_size)

# how many cmls include into dataset
num_cmls = 20

ds = xr.Dataset({
    'trsl': (('cml_id', 'channel_id', 'sample_num', 'timestep'), trsl_reshaped[:num_cmls]),
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


# Standardising subtracting median
if 'trsl_st' in ds:
    ds = ds.reset_coords(['trsl_st'], drop=True)    # if trsl_st exists, erase it. (issue fix)
ds['trsl_st'] = ds.trsl.copy()

for i in range(num_cmls):
    ds['trsl_st'][i,0] = (ds.trsl[i,0] - ds.trsl[i,0].mean()) / (ds.trsl[i,0].max() - ds.trsl[i,0].mean())
    ds['trsl_st'][i,1] = (ds.trsl[i,1] - ds.trsl[i,1].mean()) / (ds.trsl[i,1].max() - ds.trsl[i,1].mean())


# for tensors we need to make dataset 3D and make sample_num the first dimension
ds['trsl_st'] = ds.trsl_st.transpose('cml_id', 'sample_num', 'channel_id', 'timestep')

# reshape to 3D
trsl = ds.trsl_st.values.reshape(num_cmls*n_samples,2 , sample_size)
ref = ds.ref_wd.values.reshape(num_cmls*n_samples)


