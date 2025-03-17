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

""" Notes """

# DONE: bad gauge reference interpolation
# DONE: rain gauge reference SRA10M is missing in some technologies
# DONE: skip cmls without SRA10M inside the main skript: check for it. Long computation!!!
# DONE-doublecheck: single extreme rsl values like 90 dB or so.
# DONE-doublecheck: calculate new mean value when skipping large nan gaps, causing steps in rsl data
# tip: floating standardization excludes long term fluctuation
# TODO: load cml B using its IP, not i+1
# DONE: filtered metadata is duplicative. each cml is there 2 times identically
# tip: cmlAip and cmlBip are next to each other and cmlBip is always cmlAip+1
# TODO: copy data into NaN gaps from adjacent cml #cml['rsl_A'] = cml.rsl_A + cml.rsl_B.where(np.isnan(cml.rsl_A))
# TODO: copy adjacent data, if large chunk of rsl is missing: 1s10: 12
# DONE: delete few values around step
# TODO: better interpolation: https://stackoverflow.com/questions/30533021/interpolate-or-extrapolate-only-small-gaps-in-pandas-dataframe
# DONE: detect uptime resets
# DONE: dry 10min segments during light rainfall
# tip: Summit technology rsl is [-dB]
# TODO: Feed cnn constant metadata such as frequency, technology, length...
# TODO: Feed cnn the cml temperature
# DONE: If uptime is constant drop values
# TODO: spikes remaining around step after supressing the step in preprocessing (especially 1s10)
# tip: all datetime is in UTC time
# TODO: Different preprocess tresholds for different cml technologies



""" Imports """
# Import python libraries

import math
import numpy as np
import itertools
import matplotlib.pyplot as plt

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


technology = '1s10'           # ['summit', 'summit_bt', '1s10', 'ceragon_ip_10', 'ceragon_ip_20']
path = 'TelcoRain/merged_data/'+technology+'/'

for k in range(12):
    i = k*2
    # SUMMIT (0,102)
    # problematic/weird: 12, 28, 30, 36, 54, 62, 68, 74, 76, 94, 96
    # nice:  78, ideal showcase:  100, 16,2 
    # SUMMIT_BT (0,32) - showcase: 12,24,26,28
    # ceragon_ip_10 (4)
    # ceragon_ip_20 (42) - problematic: 2, 10, 14(uptime cons), 28, 34(symetric), 40(step, thr=210) nice: 8,16, overall higher peaks
    # 1s10 (26) - nice:0,2, problematic:4, 12, 14, 16,18,22, 24 overall more extreme peaks
    """ Function definitions """


    """ Main """

 
    ## LOADING DATA
    # Loading metadata

    # Get the list of all files
    file_list = sorted(os.listdir(path))
    # Check for missing column
    #missng_rain = preprocess_utility.find_missing_column('SRA10M', path)
    #file_list = sorted(list(set(file_list) - set(missng_rain)))

    metadata_all = pd.read_csv('TelcoRain/filtered_radius1.0km_offset1.0_CML.csv')
    metadata_all = metadata_all.drop_duplicates()          # clean duplicative rows

    cml_A_ip = file_list[i][file_list[i].rfind('CML_')+4:-4]

    metadata = metadata_all.loc[metadata_all['IP_address_A'] == cml_A_ip]


    if (technology=='summit') | (technology=='summit_bt'):
        cml = pd.read_csv(path+file_list[i], usecols=['time','SRA10M','cml_PrijimanaUroven','cml_Uptime'])   #,,'cml_KvalitaSignalu','cml_Teplota',cml_RxDatovyTok,cml_Uptime
        cml = cml.rename(columns={'SRA10M':'rain', 
                                'cml_PrijimanaUroven':'trsl_A',
                                'cml_Uptime':'uptime_A'})
        cml[['trsl_B','uptime_B']] = pd.read_csv(path+file_list[i+1], usecols=['cml_PrijimanaUroven','cml_Uptime'])
    if (technology=='ceragon_ip_10'):
        cml = pd.read_csv(path+file_list[i], usecols=['time','SRA10M','cml_PrijimanaUroven','cml_Uptime','cml_VysilaciVykon']) 
        cml = cml.rename(columns={'SRA10M':'rain', 
                                'cml_PrijimanaUroven':'rsl_A',
                                'cml_Uptime':'uptime_A',
                                'cml_VysilaciVykon':'tsl_A'})
        cml[['rsl_B','uptime_B','tsl_B']] = pd.read_csv(path+file_list[i+1], usecols=['cml_PrijimanaUroven','cml_Uptime','cml_VysilaciVykon'])
        cml['trsl_A'] = cml.tsl_A - cml.rsl_A
        cml['trsl_B'] = cml.tsl_B - cml.rsl_B
    if (technology=='ceragon_ip_20'):
        cml = pd.read_csv(path+file_list[i], usecols=['time','SRA10M','cml_Signal','cml_Uptime']) 
        cml = cml.rename(columns={'SRA10M':'rain', 
                                'cml_Signal':'trsl_A',
                                'cml_Uptime':'uptime_A'})
        cml[['trsl_B','uptime_B']] = pd.read_csv(path+file_list[i+1], usecols=['cml_Signal','cml_Uptime'])
        cml['trsl_A'] = -cml.trsl_A
        cml['trsl_B'] = -cml.trsl_B
    if (technology=='1s10'):
        cml = pd.read_csv(path+file_list[i], usecols=['time','SRA10M','cml_PrijimanaUroven','cml_Uptime']) 
        cml = cml.rename(columns={'SRA10M':'rain', 
                                'cml_PrijimanaUroven':'trsl_A',
                                'cml_Uptime':'uptime_A'})
        cml[['trsl_B','uptime_B']] = pd.read_csv(path+file_list[i+1], usecols=['cml_PrijimanaUroven','cml_Uptime'])
        cml['trsl_A'] = -cml.trsl_A
        cml['trsl_B'] = -cml.trsl_B


    # make copies for presentation only
    cml['trsl_A_orig'] = cml.trsl_A.copy()
    cml['trsl_B_orig'] = cml.trsl_B.copy()






    ## PREPROCESS
    cml = preprocess_utility.cml_preprocess(cml, interp_max_gap = 10, 
                    suppress_step = True, conv_threshold = 250.0, 
                    std_method = True, window_size = 10, std_threshold = 5.0, 
                    z_method = True, z_threshold = 10.0,
                    reset_detect=True
                    )

    ## WD reference
    cml = preprocess_utility.ref_preprocess(cml, 
                                            comp_lin_interp=True, upsampled_n_times=20,
                                            supress_single_zeros=True
                                            )


    ## PLOT
    fig, axs = plt.subplots(4,1, figsize=(12, 5))
    #fig.tight_layout(h_pad = 3)
    #cml.plot(ax=axs,subplots=True)                          #x='time', 
    ax1 = axs[0].twinx()
    axs[0].set_title((cml_A_ip + ', ' + str(i)))

    cml.trsl_A.plot(ax=axs[0])   
    cml.trsl_B.plot(ax=axs[0]) 
    cml.rain.plot(ax=ax1, color='black', linewidth=0.5)
    cml.uptime_A.plot(ax=axs[1])
    cml.uptime_B.plot(ax=axs[1])
    cml.trsl_A_conv.plot(ax=axs[2])   
    cml.trsl_B_conv.plot(ax=axs[2]) 
    cml.trsl_A_orig.plot(ax=axs[3])   
    cml.trsl_B_orig.plot(ax=axs[3])

    #axs.set_xlim(cml.rsl_A.values[0], cml.rsl_A.values[-1])
    #from matplotlib.widgets import Cursor
    #cursor = Cursor(ax=axs[1], useblit=True, color='red', linewidth=2)

    ref_wet_start = np.roll(cml.ref_wd, -1) & ~cml.ref_wd
    ref_wet_end = np.roll(cml.ref_wd, 1) & ~cml.ref_wd
    for start_i, end_i in zip(
        ref_wet_start.values.nonzero()[0],
        ref_wet_end.values.nonzero()[0],
    ):
        axs[0].axvspan(start_i, end_i, color='b', alpha=0.2, linewidth=0) 



    plt.show()





## TRAINING

## CLASSIFICATION


input('press enter to continue')