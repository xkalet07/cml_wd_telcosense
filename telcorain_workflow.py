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
i = 90      # multiples of 2 up to 103

cml_A_ip = file_list[i][file_list[i].rfind('CML_')+4:-4]

metadata = metadata_all.loc[metadata_all['IP_address_A'] == cml_A_ip]

cml = pd.read_csv(path+file_list[i], usecols=['time','SRA10M','cml_PrijimanaUroven'])   #,'cml_MaximalniRychlostRadia(modulace)' cml_Teplota,cml_RxDatovyTok,cml_KvalitaSignalu,cml_Uptime
cml = cml.rename(columns={'SRA10M':'rain', 'cml_PrijimanaUroven':'rsl_A','cml_Uptime':'uptime_A'})
# TODO: load cml B using its IP, not i+1
cml['rsl_B'] = pd.read_csv(path+file_list[i+1], usecols=['cml_PrijimanaUroven'])

#cml['rsl_A'] = cml.rsl_A.where(abs((cml.rsl_A-cml.rsl_A.mean())/cml.rsl_A.std()) < 13.0)
#cml['rsl_B'] = cml.rsl_B.where(abs((cml.rsl_B-cml.rsl_B.mean())/cml.rsl_B.std()) < 13.0)


## PREPROCESS
# interpolation both rsl, and R
cml = cml.interpolate(axis=0, method='linear', limit = 10)
# skip rows with missing rsl values
cml = cml.dropna(axis=0, how = 'all', subset=['rsl_A','rsl_B'])
#cml = cml.interpolate(axis=0, method='nearest', limit = 10)

# Convert to Pandas Series for easy rolling median calculation
window_size = 10  # Adjust based on your data characteristics
rolling_median = cml.rsl_A.rolling(window=window_size, center=True).median()

# Fill NaN values at the edges
rolling_median.fillna(method='bfill', inplace=True)
rolling_median.fillna(method='ffill', inplace=True)

# Define threshold for step detection
threshold = 20  # Adjust this based on normal data fluctuations

# Find indices where step change occurs
step_mask = np.abs(cml.rsl_A - rolling_median) > threshold

# Suppress steps by replacing them with the rolling median
rsl_corrected = cml.rsl_A.copy()
rsl_corrected[step_mask] = rolling_median[step_mask]

cml['roll_median'] = rolling_median
cml['rslA_correct'] = rsl_corrected
cml['step_mask'] = step_mask*1


# standardisation
#cml['srsl_A'] = (cml.rsl_A.values - cml.rsl_A.min()) / (cml.rsl_A.max() - cml.rsl_A.min())
#cml['srsl_B'] = (cml.rsl_B.values - cml.rsl_B.min()) / (cml.rsl_B.max() - cml.rsl_B.min())

## WD reference
# Find indices where values transition from nonzero to zero (end of pattern)
nonzero_mask = cml.rain != 0
shifted_mask = np.roll(nonzero_mask, -1)
shifted_mask[-1] = False  # Prevent wraparound issues
last_indices = np.where(nonzero_mask & ~shifted_mask)[0]

# Zero out the last 20 values of each pattern
for idx in last_indices:
    cml.rain[max(0, idx - 19): idx + 1] = 0  # Ensure we don't go out of bounds

cml['ref_wd'] = cml.rain.where(cml.rain == 0, True).astype(bool)




fig, axs = plt.subplots(figsize=(12, 6))
#axs.set_xlim(cml.time.values[500000], cml.time.values[-1])    
#fig.tight_layout(h_pad = 3)
cml.plot(ax=axs,x='time',subplots=True)
from matplotlib.widgets import Cursor
cursor = Cursor(ax=axs, useblit=True, color='red', linewidth=2)
plt.show()



## TRAINING

## CLASSIFICATION



input("Press Enter to continue...")