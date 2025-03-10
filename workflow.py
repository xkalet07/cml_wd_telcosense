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
import telcosense_classification.module.cnn_orig as cnn
from telcosense_classification import preprocess_utility
from telcosense_classification import cnn_utility
from telcosense_classification import plot_utility


""" ConstantVariable definitions """

""" Function definitions """


""" Main """

# problems:
# DONE: bad gauge reference interpolation
# DONE: rain gauge reference SRA10M is missing in some technologies
# TODO: skip cmls without SRA10M inside the main skript: check for it
# DONE-doublecheck: single extreme rsl values like 90 dB or so.
# DONE-doublecheck: calculate new mean value when skipping large nan gaps, causing steps in rsl data
# tip: floating standardization excludes long term fluctuation
# TODO: load cml B using its IP, not i+1
# DONE: filtered metadata is duplicative. each cml is there 2 times identically
# tip: cmlAip and cmlBip are next to each other and cmlBip is always cmlAip+1
# TODO: copy data into NaN gaps from adjacent cml #cml['rsl_A'] = cml.rsl_A + cml.rsl_B.where(np.isnan(cml.rsl_A))
# TODO: delete few values around step
# TODO: better interpolation: https://stackoverflow.com/questions/30533021/interpolate-or-extrapolate-only-small-gaps-in-pandas-dataframe

## LOADING DATA
# Loading metadata
metadata_all = pd.read_csv('TelcoRain/filtered_radius1.0km_offset1.0_CML.csv')
metadata_all = metadata_all.drop_duplicates()          # clean duplicative rows

# !!!!!! Summit technology rsl [-dB]

# Get the list of all csvs
path = 'TelcoRain/merged_data/summit/'
file_list = sorted(os.listdir(path))                   # sort alphanumerically

#for k in range(100):
i = 100      # multiples of 2 up to 102
# problematic: 2, 4, 62, 32, 34, 100
# nice: 78
# ideal showcase: 100

cml_A_ip = file_list[i][file_list[i].rfind('CML_')+4:-4]

metadata = metadata_all.loc[metadata_all['IP_address_A'] == cml_A_ip]

cml = pd.read_csv(path+file_list[i], usecols=['time','SRA10M','cml_PrijimanaUroven'])   #,,'cml_KvalitaSignalu','cml_Teplota',cml_RxDatovyTok,cml_Uptime
cml = cml.rename(columns={'SRA10M':'rain', 'cml_PrijimanaUroven':'rsl_A'})
cml['rsl_B'] = pd.read_csv(path+file_list[i+1], usecols=['cml_PrijimanaUroven'])

# make copies for presentation only
cml['rsl_A_orig'] = cml.rsl_A.copy()
cml['rsl_B_orig'] = cml.rsl_B.copy()

## PREPROCESS
cml = preprocess_utility.cml_preprocess(cml, interp_max_gap = 10, 
                   suppress_step = True, conv_threshold = 20.0, 
                   std_method = True, window_size = 10, std_threshold = 5.0, 
                   z_method = True, z_threshold = 10.0
                   )

## WD reference
cml = preprocess_utility.ref_preprocess(cml, comp_lin_interp=True, upsampled_n_times=20)

# plot
fig, axs = plt.subplots(2,1, figsize=(12, 2))
#fig.tight_layout(h_pad = 3)
#cml.plot(ax=axs,subplots=True)                          #x='time', 
cml.rsl_A.plot(ax=axs[0])   
cml.rsl_B.plot(ax=axs[0]) 
cml.rain.plot(ax=axs[1])
#axs.set_xlim(cml.rsl_A.values[0], cml.rsl_A.values[-1])
#from matplotlib.widgets import Cursor
#cursor = Cursor(ax=axs, useblit=True, color='red', linewidth=2)

ref_wet_start = np.roll(cml.ref_wd, -1) & ~cml.ref_wd
ref_wet_end = np.roll(cml.ref_wd, 1) & ~cml.ref_wd
for start_i, end_i in zip(
    ref_wet_start.values.nonzero()[0],
    ref_wet_end.values.nonzero()[0],
):
    axs[0].axvspan(start_i, end_i, color='b', alpha=0.5, linewidth=0, label='_'*start_i+'true wet') 







# plot real bool wet/dry
wet_start = np.roll(my_ref.ref_wd, -1) & ~my_ref.ref_wd
wet_end = np.roll(my_ref.ref_wd, 1) & ~my_ref.ref_wd
for wet_start_i, wet_end_i in zip(
    wet_start.values.nonzero()[0],
    wet_end.values.nonzero()[0],
):
    axs[1].axvspan(my_ref.time.values[wet_start_i], my_ref.ref_wd.time.values[wet_end_i], color='b', alpha=0.2, linewidth=0); # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axvspan.html
    axs[0].axvspan(my_ref.time.values[wet_start_i], my_ref.ref_wd.time.values[wet_end_i], color='b', alpha=0.2, linewidth=0); # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axvspan.html


# axes limits source: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xlim.html
axs[1].set_xlim(my_cml.time.values[0], my_cml.time.values[-1])
axs[0].set_xlabel('')
axs[1].set_title("")















plt.show()





# setup figure
fig, axs = plt.subplots(1, sharex=True, figsize=(12,6))

ax1.set_xlim(ds.time.values[0,0], ds.time.values[-1,-1])                    # change to [0,0,0] and [0,-1,-1] if excluding fault cmls
fig.tight_layout(h_pad = 3)



## TRAINING

## CLASSIFICATION


