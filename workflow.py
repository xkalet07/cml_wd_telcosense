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
# DONE: period of trsl == reference wet/dry. Meaning, for each trsl point there will be wet/dry flag predicted.  
# TODO: Forward and backward memory implementation will be needed.  
# TODO: This approach should bring better learning performance. For longer wet/dry periods there are ocasions, where the period is wet, but trsl shows rain pattern for only fraction of the period.  
# TODO: problem with ceragon_ip_10
# DONE: implement pooling, we need shorttime-longtime pattern reckognition
# TODO: Plots doesnt show cml ip or any id as a title


""" Imports """
# Import python libraries

import math
import numpy as np
import pandas as pd
import xarray as xr
import itertools
import matplotlib.pyplot as plt

import datetime
import os

import torch
import torch.nn as nn
from tqdm import tqdm


# Import external packages

# Import own modules
from telcosense_classification import data_loading_utility
from telcosense_classification import preprocess_utility
from telcosense_classification import cnn_utility
from telcosense_classification import plot_utility


""" Constant Variable definitions """

technology = 'summit_bt'      # ['summit', 'summit_bt', '1s10', 'ceragon_ip_10', 'ceragon_ip_20']
dir = 'TelcoRain/merged_data/'
i = 24
# SUMMIT (0,102)    # problematic/weird: 6, 12, 28, 30, 36, 54, 62, 68, 74, 76, 94, 96     # nice:  78, ideal showcase:  100, 16,2 
# SUMMIT_BT (0,32)  # showcase: 12,24,26,28
# ceragon_ip_10 (4) # doesnt work so far
# ceragon_ip_20 (42)# problematic: 2, 10, 14(uptime cons), 28, 34(symetric), 40(step, thr=210) nice: 8,16, overall higher peaks
# 1s10 (26)         # nice:0,2, problematic:4, 12, 14, 16,18,22, 24 overall more extreme peaks

# Training CNN parameters
num_channels = 2
sample_size = 60            # 60 keep lower than FC num of neurons
batchsize = 128             # 128 most smooth (64)
epochs = 50                 # 50
resume_epoch = 0 
learning_rate = 0.0005      # 0.0005 - 0.001 
dropout_rate = 0.001         # 0.04 train loss: 
kernel_size = 3
n_conv_filters = [24, 48, 96, 192]
n_fc_neurons = 128          # 64 better FP, 128 better TP
save_param = False


# constant parameters
upsampled_n_times = 20

""" Function definitions """


""" Main """
path = dir + technology+'/'

# list of all files
file_list = sorted(os.listdir(path))

## Check for missing column, WARNING: long execution
# missng_rain = data_loading_utility.find_missing_column('SRA10M', path)
# file_list = sorted(list(set(file_list) - set(missng_rain)))


## LOADING DATA 
# Loading metadata
metadata_all = pd.read_csv('TelcoRain/filtered_radius1.0km_offset1.0_CML.csv')
metadata_all = metadata_all.drop_duplicates()          # clean duplicative rows
cml_A_ip = file_list[i][file_list[i].rfind('CML_')+4:-4]
metadata = metadata_all.loc[metadata_all['IP_address_A'] == cml_A_ip]

# Loading cml
cml = data_loading_utility.load_cml(dir, technology, i)

# make copies for presentation only
#cml['trsl_A_orig'] = cml.trsl_A.copy()
#cml['trsl_B_orig'] = cml.trsl_B.copy()


## PREPROCESS
cml = preprocess_utility.cml_preprocess(cml, interp_max_gap = 10, 
                suppress_step = True, conv_threshold = 250.0, 
                std_method = True, window_size = 10, std_threshold = 5.0, 
                z_method = True, z_threshold = 10.0,
                reset_detect=True,
                subtract_median=True
                )

## WD REFERENCE
cml = preprocess_utility.ref_preprocess(cml, 
                                        comp_lin_interp=True, upsampled_n_times = upsampled_n_times,
                                        supress_single_zeros=True
                                        )


#plot_utility.plot_cml(cml, columns=['rain', 'ref_wd', 'trsl', 'uptime', 'temp'])


## CLASS BALANCE
cml = preprocess_utility.balance_wd_classes(cml)

## PLOT
#plot_utility.plot_cml(cml, columns=['rain', 'ref_wd', 'trsl', 'uptime', 'temp'])

## SHUFFLE DATASET
#cml = preprocess_utility.shuffle_dataset(cml, segment_size = 20000)

#plot_utility.plot_cml(cml, columns=['rain', 'ref_wd', 'trsl', 'uptime', 'temp'])

## save the preprocessed cml
#cml.to_csv('TelcoRain/evaluating_dataset_meanMax/'+technology+'_'+str(i)+'_'+cml_A_ip+'.csv', index=False)  

## TRAINING
cnn_wd_threshold = 0.5
cnn_out, train_loss, test_loss = cnn_utility.cnn_train_period(cml, 
                                                num_channels,
                                                sample_size,
                                                batchsize, 
                                                epochs, 
                                                resume_epoch, 
                                                learning_rate, 
                                                dropout_rate,
                                                kernel_size,
                                                n_conv_filters,
                                                n_fc_neurons,
                                                save_param
                                                )

## CNN output
cutoff = len(cml) % sample_size
cml['cnn_out'] = np.append(np.repeat(cnn_out, sample_size), np.zeros(cutoff))

# sort CML back from previous shuffle
#cml = cml.sort_values(['segment_id', 'time']).reset_index(drop=True)

cml['cnn_wd'] = cml.cnn_out > cnn_wd_threshold

# predicted true wet
cml['true_wet'] = cml.cnn_wd & cml.ref_wd 
# cnn false alarm
cml['false_alarm'] = cml.cnn_wd & ~cml.ref_wd
# cnn missed wet
cml['missed_wet'] = ~cml.cnn_wd & cml.ref_wd

print('TP: ' + str(sum(cml.true_wet)/sum(cml.ref_wd)))
print('FP: ' + str(sum(cml.false_alarm)/sum(cml.ref_wd)))
print('FN: ' + str(sum(cml.missed_wet)/sum(cml.ref_wd)))


## PLOT
plot_utility.plot_cnn_classification(cml, cnn_wd_threshold)


'''
if cutoff == 0:
    ref_wd = cml.ref_wd.values[:][::sample_size]   
else:
    ref_wd = cml.ref_wd.values[:-cutoff][::sample_size]   

cnn_out_upsampled = np.repeat(cnn_out, upsampled_n_times, axis=1)



cnn_wd = cnn_out > cnn_wd_threshold
true_wet = cnn_wd & ref_wd 
false_alarm = cnn_wd & ~ref_wd

TP = sum(true_wet)/sum(ref_wd)
FP = sum(false_alarm)/sum(ref_wd)

print('TP: '+str(TP))
print('FP: '+str(FP))

'''





## CLASSIFICATION

input('press enter to continue')