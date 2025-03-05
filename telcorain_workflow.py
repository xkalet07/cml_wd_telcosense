#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filename: telcorain_workflow.py
Author: Lukas Kaleta
Date: 2025-02-19
Version: 1.0
Description: 
    This script showcases the typical workflow of training an CNN module
    for rain event detection using CML data from czech republic and CHMI.

License: 
Contact: 211312@vutbr.cz
"""


""" Imports """
# Import python libraries

import math
import numpy as np
import itertools
import matplotlib.pyplot as plt

import xarray as xr
import pandas as pd
import os

import torch
import torch.nn as nn
import sklearn.metrics as skl
from sklearn.utils import shuffle
from tqdm import tqdm
import datetime

# Import external packages

# Import own modules
import modul.cnn_orig as cnn
import preprocess_utility
import cnn_utility
import plot_utility


""" ConstantVariable definitions """

""" Function definitions """


""" Main """

# problems:
# WARNING: meteo 0-203-0-11515 has no SRA10M value. Therefore unusable!
# TODO: bad gauge reference interpolation
# TODO: rain gauge reference SRA10M is missing in some technologies
# TODO: single extreme rsl values like 90 dB or so.
# TODO: calculate new mean value when skipping large nan gaps, causing steps in rsl data
# TODO: filter extreme values
# TODO: floating standardization
# DONE: filtered metadata is duplicative. each cml is there 2 times identically
# tip: cmlAip and cmlBip are next to each other and cmlBip is always cmlAip+1

## LOADING DATA
# Loading metadata
metadata_all = pd.read_csv('TelcoRain/filtered_radius1.0km_offset1.0_CML.csv')
metadata_all = metadata_all.drop_duplicates()          # clean duplicative rows

# !!!!!! Summit technology rsl [-dB]

# Get the list of all csvs
path = 'TelcoRain/merged_data/summit/'
file_list = sorted(os.listdir(path))                   # sort alphanumerically

#for k in range(100):
i = 34      # multiples of 2 up to 102
# problematic: 2, 4, 62

cml_A_ip = file_list[i][file_list[i].rfind('CML_')+4:-4]

metadata = metadata_all.loc[metadata_all['IP_address_A'] == cml_A_ip]

cml = pd.read_csv(path+file_list[i], usecols=['time','SRA10M','cml_PrijimanaUroven'])   #,'cml_MaximalniRychlostRadia(modulace)' cml_Teplota,cml_RxDatovyTok,cml_KvalitaSignalu,cml_Uptime
cml = cml.rename(columns={'SRA10M':'rain', 'cml_PrijimanaUroven':'rsl_A','cml_Uptime':'uptime_A'})
# TODO: load cml B using its IP, not i+1
cml['rsl_B'] = pd.read_csv(path+file_list[i+1], usecols=['cml_PrijimanaUroven'])

# make copies
cml['rsl_A_orig'] = cml.rsl_A.copy()
cml['rsl_B_orig'] = cml.rsl_B.copy()

## PREPROCESS
# First interpolation both rsl, and R and drop missing values
cml = cml.interpolate(axis=0, method='linear', limit = 10)
cml = cml.dropna(axis=0, how = 'all', subset=['rsl_A','rsl_B'])

# calculate rolling STD
window_size = 10  # Adjust based on your data characteristics
threshold = 5     # Adjust this based on normal data fluctuations

for rsl in ['rsl_A', 'rsl_B']:
    rolling_std = cml[rsl].rolling(window=window_size, center=True).std()
    # Fill NaN values at the edges
    rolling_std.fillna(method='bfill', inplace=True)
    rolling_std.fillna(method='ffill', inplace=True)

    # threshold for step detection
    step_mask = np.abs(rolling_std) > threshold
    shifted_mask = np.roll(step_mask, -1)
    shifted_mask[-1] = False  # Prevent wraparound issues
    step_loc = np.where(step_mask & ~shifted_mask)[0]
    step_loc = np.append(0,step_loc)

    # drop values around the step
    cml[rsl] = cml[rsl].where(~step_mask)

    # If rsl step is present, align values
    for i in range(len(step_loc)):
        if i < len(step_loc)-1:
            cml[rsl][step_loc[i]:step_loc[i+1]] = cml[rsl][step_loc[i]:step_loc[i+1]] - cml[rsl][step_loc[i]:step_loc[i+1]].mean()
        elif i >= len(step_loc)-1:
            cml[rsl][step_loc[i]:] = cml[rsl][step_loc[i]:] - cml[rsl][step_loc[i]:].mean()
    

    # Drop faulty single extreme values by Z method (non detected by std)
    cml[rsl] = cml[rsl].where(abs((cml[rsl]-cml[rsl].mean())/cml[rsl].std()) < 10.0)

    # standardisation
    cml[rsl] = cml[rsl].values / cml[rsl].max()

# interpolation both rsl, and R
cml = cml.interpolate(axis=0, method='linear')




## WD reference
# Find indices where values transition from nonzero to zero (end of pattern)
nonzero_mask = cml.rain != 0
shifted_mask = np.roll(nonzero_mask, -1)
shifted_mask[-1] = False  # Prevent wraparound issues
last_indices = np.where(nonzero_mask & ~shifted_mask)[0]

# Zero out the last 20 values of each pattern
for idx in last_indices:
    cml.rain[max(0, idx - 19): idx + 1] = 0  # Ensure we don't go out of bounds

cml['ref_wd'] = cml.rain.where(cml.rain == 0, True).astype(bool)*1




fig, axs = plt.subplots(figsize=(12, 6))
#fig.tight_layout(h_pad = 3)
cml.plot(ax=axs,subplots=True)                          #xlim=[737500,742500]
#from matplotlib.widgets import Cursor
#cursor = Cursor(ax=axs, useblit=True, color='red', linewidth=2)
plt.show()



## TRAINING

## CLASSIFICATION



input("Press Enter to continue...")