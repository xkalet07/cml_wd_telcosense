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
# TODO: period of trsl == reference wet/dry. Meaning, for each trsl point there will be wet/dry flag predicted.  
# TODO: Forward and backward memory implementation will be needed.  
# TODO: This approach should bring better learning performance. For longer wet/dry periods there are ocasions, where the period is wet, but trsl shows rain pattern for only fraction of the period.  



""" Imports """
# Import python libraries

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import os

# Import external packages

# Import own modules
from telcosense_classification import data_loading_utility
from telcosense_classification import preprocess_utility
from telcosense_classification import cnn_utility
from telcosense_classification import plot_utility


""" ConstantVariable definitions """

technology = '1s10'      # ['summit', 'summit_bt', '1s10', 'ceragon_ip_10', 'ceragon_ip_20']
dir = 'TelcoRain/merged_data/'
path = dir + technology+'/'

# list of all files
file_list = sorted(os.listdir(path))
# Check for missing column
missng_rain = data_loading_utility.find_missing_column('SRA10M', path)
file_list = sorted(list(set(file_list) - set(missng_rain)))

# Loading metadata
metadata_all = pd.read_csv('TelcoRain/filtered_radius1.0km_offset1.0_CML.csv')
metadata_all = metadata_all.drop_duplicates()          # clean duplicative rows

""" Function definitions """


""" Main """

i = 0
# SUMMIT (0,102)
# problematic/weird: 12, 28, 30, 36, 54, 62, 68, 74, 76, 94, 96
# nice:  78, ideal showcase:  100, 16,2 
# SUMMIT_BT (0,32) - showcase: 12,24,26,28
# ceragon_ip_10 (4)
# ceragon_ip_20 (42) - problematic: 2, 10, 14(uptime cons), 28, 34(symetric), 40(step, thr=210) nice: 8,16, overall higher peaks
# 1s10 (26) - nice:0,2, problematic:4, 12, 14, 16,18,22, 24 overall more extreme peaks


## LOADING DATA 
# metadata
cml_A_ip = file_list[i][file_list[i].rfind('CML_')+4:-4]
metadata = metadata_all.loc[metadata_all['IP_address_A'] == cml_A_ip]

# Loading cml
cml = data_loading_utility.load_cml(dir, technology, i)

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

## WD REFERENCE
cml = preprocess_utility.ref_preprocess(cml, 
                                        comp_lin_interp=True, upsampled_n_times=20,
                                        supress_single_zeros=True
                                        )


#plot_utility.plot_cml(cml, columns=['rain', 'ref_wd', 'trsl', 'uptime', 'temp'])

"""
cml['mean_A'] = cml.trsl_A.rolling(window=10000, center=True).mean()
cml['mean_B'] = cml.trsl_B.rolling(window=10000, center=True).mean()

cml['post_A'] = cml.trsl_A - cml.mean_A
cml['post_B'] = cml.trsl_B - cml.mean_B

cml['med_A'] = cml.trsl_A.rolling(window=10000, center=True).median()
cml['med_B'] = cml.trsl_B.rolling(window=10000, center=True).median()

cml['postx_A'] = cml.trsl_A - cml.med_A
cml['postx_B'] = cml.trsl_B - cml.med_B

plot_utility.plot_cml(cml, columns=['rain', 'ref_wd', 'trsl', 'mean', 'post', 'med', 'postx'])
"""






## CLASS BALANCE
cml = preprocess_utility.balance_wd_classes(cml)

## PLOT
plot_utility.plot_cml(cml, columns=['rain', 'ref_wd', 'trsl', 'uptime', 'temp'])


## TRAINING
# TODO: batchsize
cnn_utility.cnn_train(cml, sample_size=100, epochs = 300, resume_epoch = 0, save_param = False)

## CLASSIFICATION


input('press enter to continue')