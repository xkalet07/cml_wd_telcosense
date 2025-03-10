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

import math
import numpy as np
import xarray as xr
import pandas as pd
import os
import csv

from scipy.signal import find_peaks
from sklearn.utils import shuffle
# Import external packages


# Import own modules


""" Variable definitions """


""" Function definitions """

def find_missing_ref_rainrate(path = 'TelcoRain/merged_data/summit/'):
    """
    Detect merged cml-rainGauge files which are missing refference rainrate. 
    Meaning: print meteo stations without rain gauge.

    Parameters:
    path : str, default value = 'TelcoRain/merged_data/summit/', default directory of files
    """

    # Get the list of all csvs
    file_list = os.listdir(path)

    print('meteo stations missing rain gauge:')

    for file in file_list:
        with open(path+file, 'r') as fp:
            s = fp.read()
            if 'SRA10M' not in s:
                print(file)
            fp.close()
    
def cml_preprocess(cml:pd.DataFrame, interp_max_gap = 10, conv_threshold = 20.0, window_size = 10, std_threshold = 5.0, z_threshold = 10.0):
    """
    Preprocess cml dataset: Interpolate gaps, Exclude NaN values,
    standardise values by subtracting mean and scaling to 0-1.
    Optional:
        remove fault extreme values in rsl series using STD or Z method,
        detect steps in rsl mean value and alighn periods with extremely different mean value,
        
    Parameters
    cml : Pandas.DataFrame, containing two aligned adjacent CMLs and rainrate reference with timestamps.
    interp_max_gap : int, default value = 10, maximal gap in each data column to be interpolated.
    window_size : int, default = 10, Window size for rolling STD. Adjust based on your data characteristics.
    conv_threshold : float, default = 20.0, threshold for Large steps detection using convolution.
        Adjust this based on normal data fluctuations
    std_threshold : float, default = 5.0, threshold for extreme value detection using STD.
        Adjust this based on normal data fluctuations
    z_threshold : float, default = 10.0, threshold for extreme value detection using Z method.
        Adjust this based on fluctuations of your data
    
    Returns
    cml : Pandas.DataFrame
    """
    # First interpolation both rsl, and R and drop missing values
    cml = cml.interpolate(axis=0, method='linear', limit = interp_max_gap)
    cml = cml.dropna(axis=0, how = 'all', subset=['rsl_A','rsl_B'])
    cml = cml.interpolate(axis=0, method='linear')
    
    # Anomaly handling
    cml = cml_suppress_step(cml,conv_threshold)
    cml = cml_suppress_extremes_std(cml, window_size, std_threshold)
    cml = cml_suppress_extremes_z(cml, z_threshold)


    # standardisation
    cml['rsl_A'] = cml.rsl_A.values / cml.rsl_A.max()
    
    return cml



def cml_suppress_extremes_std(cml:pd.DataFrame, window_size = 10, std_threshold = 5.0):
    """
    Remove fault extreme values in rsl series by calculating floating window std,
    interpolate missing values

    Parameters
    cml : Pandas.DataFrame, containing two aligned adjacent CMLs and rainrate reference with timestamps.
    window_size : int, default = 10, Window size for rolling STD. Adjust based on your data characteristics.
    std_threshold : float, default = 5.0, threshold for Large steps and extreme value detection using STD.
        Adjust this based on normal data fluctuations
    
    Returns
    cml : Pandas.DataFrame
    """
    # calculate rolling STD
    for rsl in ['rsl_A', 'rsl_B']:
        rolling_std = cml[rsl].rolling(window=window_size, center=True).std()

        # Fill NaN values at the edges
        rolling_std.fillna(method='bfill', inplace=True)
        rolling_std.fillna(method='ffill', inplace=True)

        # drop values with STD above the threshold
        cml[rsl] = cml[rsl].where(np.abs(rolling_std) < std_threshold)

    # interpolation both rsl, and R
    cml = cml.interpolate(axis=0, method='linear')

    return cml



def cml_suppress_extremes_z(cml:pd.DataFrame, z_threshold = 10.0):
    """
    Remove fault extreme values in rsl series based on Z method:
    Z = (x-mean)/std
    interpolate missing values

    Parameters
    cml : Pandas.DataFrame, containing two aligned adjacent CMLs and rainrate reference with timestamps.
    z_threshold : float, default = 10.0, threshold for extreme value detection using Z method.
        Adjust this based on fluctuations of your data
    
    Returns
    cml : Pandas.DataFrame
    """
    for rsl in ['rsl_A', 'rsl_B']:
        # Drop faulty single extreme values by Z method (non detected by std)
        z_param = (cml[rsl]-cml[rsl].mean())/cml[rsl].std()
        cml[rsl] = cml[rsl].where(z_param < z_threshold)
        cml[rsl] = cml[rsl].where(z_param > -3.0)

    # interpolation both rsl, and R
    cml = cml.interpolate(axis=0, method='linear')

    return cml




def cml_suppress_step(cml:pd.DataFrame, conv_threshold = 20.0):
    """
    detect steps in rsl mean value and alighn periods with extremely different mean value

    Parameters
    cml : Pandas.DataFrame, containing two aligned adjacent CMLs and rainrate reference with timestamps.
    conv_threshold : float, default = 20.0, Threshold for detecting step from convolution result
        pick values between (10;50)

    Returns
    cml : Pandas.DataFrame
    """
    step = np.hstack((np.ones(100), -1*np.ones(100)))

    for rsl in ['rsl_A', 'rsl_B']:
        # standardisation
        cml_min = cml[rsl].min()
        cml_max = cml[rsl].max()
        cml[rsl] = (cml[rsl].values-cml_min) / (cml_max-cml_min)

        conv = np.abs(np.convolve(cml[rsl], step, mode='valid'))
        conv = np.append(np.append(np.zeros(100),conv),np.zeros(99))
        cml[rsl+'_conv'] = conv 

        step_mask = (conv > conv_threshold)
        
        # TODO: delete few values around step
        #cml[rsl] = cml[rsl].where(~step_mask)
                
        # Find indices where convolution reaches maximum
        step_loc,_ = find_peaks( cml[rsl+'_conv'].where(step_mask), prominence=1)
        step_loc = np.append(0,step_loc)

        # If rsl step is present, align values
        for i in range(len(step_loc)):
            if i < len(step_loc)-1:
                cml[rsl][step_loc[i]:step_loc[i+1]] = cml[rsl][step_loc[i]:step_loc[i+1]] - cml[rsl][step_loc[i]:step_loc[i+1]].mean()
            elif i >= len(step_loc)-1:
                cml[rsl][step_loc[i]:] = cml[rsl][step_loc[i]:] - cml[rsl][step_loc[i]:].mean()

    return cml


def ref_preprocess(cml:pd.DataFrame, comp_lin_interp = False, upsampled_n_times = int(0)):
    """
    Create wet/dry flag from rain rate reference included in cml DataFrame.
    Optional feature: zero out last N-2 values of rainy periods, caused by upsampling
    and interpolation between last nonzero and first zero sample, vausing non zero leakage 
    into zero values after rainy period.

    Parameters
    cml : Pandas.DaraFrame containing cml data and corresponding reference raifall data
    comp_lin_interp : bool, default = False, Compensate for nonzero leakage if True
    upsampled_n_times : int, default value = 0, number of times upsampled. Defining 
        number of samples to be zeroed.

    Returns
    cml : Pandas.DataFrame
    """
    # Compensating linear interpolation:
    # Find indices where values transition from nonzero to zero (end of rain patterns)
    if comp_lin_interp & (upsampled_n_times >= 2):
        nonzero_mask = cml.rain != 0
        shifted_mask = np.roll(nonzero_mask, -1)
        shifted_mask[-1] = False  # Prevent wraparound issues
        last_indices = np.where(nonzero_mask & ~shifted_mask)[0]

        # Zero out interpolated nonzero values at the end of rain patter
        for idx in last_indices:
            cml.rain[max(0, idx - (upsampled_n_times-2)): idx + 1] = 0  # Ensure we don't go out of bounds

    # create reference WD flag
    cml['ref_wd'] = cml.rain.where(cml.rain == 0, True).astype(bool)

    return cml


## TODO: 
# exclude long dry periods. Make dataset 50:50
def balance_wd_classes(cml_set:xr.Dataset, ref_set:xr.Dataset, sample_size = 60):
    """
    undersample wd reference and exclude large dry periods from both rainfall and cml data

    Parameters
    cml_set : xarray.dataset containing several CMLs with rsl, tsl, trsl, timestamps and metadata
    ref_set : xarray.dataset containing reference rainrate and wet/dry data for given CMLs
    sample_size : int, default value = 60 [min], length of samples

    Returns
    cml_set : xarray.dataset containing several CMLs with rsl, tsl, trsl, timestamps and metadata
    ref_set : xarray.dataset containing reference rainrate and wet/dry data for given CMLs
    """
    '''def balance_classes(a, boo):
        """
        From https://github.com/jpolz/cml_wd_pytorch
        """
        boo = boo[0,:]
        lsn=len(a.sample_num)
        ind = np.arange(lsn)
        #ind_true = np.empty((len(a.cml_id),lsn))
        #ind_false = np.empty((len(a.cml_id),lsn))
        #for i in range(len(a.cml_id)):
        ind_true = shuffle(ind[boo])
        ind_false = ind[~boo]
        ind_true = ind_true[:np.sum(~boo)]
        print(1-(2*len(ind_false)/lsn))
        return a.isel(sample_num=np.concatenate([ind_true,ind_false]))'''


