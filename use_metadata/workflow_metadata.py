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
# DONE: delete few values around step
# DONE: detect uptime resets
# TODO: some CML still has NaN gaps, find it and exclude the NaN samples.
# DONE: dry 10min segments during light rainfall
# tip: Summit technology rsl is [-dB]
# TODO: Feed cnn constant metadata such as frequency, technology, length...
# DONE: Feed cnn the cml temperature -> -10% worse TPR with this CNN model
# DONE: If uptime is constant drop values
# tip: all datetime is in UTC time
# DONE: period of trsl == reference wet/dry. Meaning, for each trsl point there will be wet/dry flag predicted.  
# TODO: Forward and backward memory implementation will be needed.  
# TODO: This approach should bring better learning performance. For longer wet/dry periods there are ocasions, where the period is wet, but trsl shows rain pattern for only fraction of the period.  
# TODO: problem with ceragon_ip_10
# DONE: implement pooling, we need shorttime-longtime pattern reckognition
# TODO: Plots doesnt show cml ip or any id as a title
# TODO: Data augmentation: noise injecting, time warp, Random scaling, mixUp/cutMix

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
import sklearn.metrics as skl

# Import external packages

# Import own modules
from telcosense_classification import data_loading_utility
from telcosense_classification import preprocess_utility
from telcosense_classification import cnn_utility
from telcosense_classification import plot_utility
from telcosense_classification import metrics_utility


""" Constant Variable definitions """

technology = 'summit'      # ['summit', 'summit_bt', '1s10', 'ceragon_ip_10', 'ceragon_ip_20']
dir = 'TelcoRain/merged_data/'
i = 78 #
# SUMMIT (0,102)    # problematic/weird: 6, 12, 28, 30, 36, 54, 62, 68, 74, 76, 94, 96     # nice:  78, ideal showcase:  100, 16,2 
# 1dB, 1°C
# SUMMIT_BT (0,32)  # showcase: 12,24,26,28
# 1dB, 1°C
# ceragon_ip_10 (4) # doesnt work so far
# 1dB, 1°C
# ceragon_ip_20 (42)# problematic: 2, 10, 14(uptime cons), 28, 34(symetric), 40(step, thr=210) nice: 8,16, overall higher peaks
# 1dB, 1°C
# 1s10 (26)         # nice:0,2, problematic:4, 12, 14, 16,18,22, 24 overall more extreme peaks
# 0.01dB, 0.01°C

# Training CNN parameters
num_channels = 2
sample_size = 60            # 60 keep lower than FC num of neurons
batchsize = 256             # 128 most smooth (64)
epochs = 20                 # 50
resume_epoch = 0 
learning_rate = 0.0002      # 0.0002 or 0.0003
dropout_rate = 0.001        # 0.001
kernel_size = 3             # 3 - best performance
n_conv_filters = [16, 32, 64, 128]#, 96, 192]     # [48, 96, 192, 384] 4% worse
n_fc_neurons = 64#128          # 128 (64 better FP, 128 better TP)
single_output = True
shuffle = False             # use True (for testloss: 1.3)
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
constant_parameters = ['IP_address_A', 'IP_address_B', 'technology', 'distance', 'frequency_A', 'frequency_B', 'polarization'] # 5+1+2+3
metadata_all = pd.read_csv('TelcoRain/filtered_radius1.0km_offset1.0_CML.csv', usecols=constant_parameters)
metadata_all = metadata_all.drop_duplicates()          # clean duplicative rows
cml_A_ip = file_list[i][file_list[i].rfind('CML_')+4:-4]


# standardize metadata
# distance 0-max
metadata_all.distance = metadata_all.distance/metadata_all.distance.max()

# frequency 0-max
max_freq = max(metadata_all.frequency_B.max(), metadata_all.frequency_A.max())
metadata_all.frequency_A = metadata_all.frequency_A/max_freq
metadata_all.frequency_B = metadata_all.frequency_B/max_freq

# technology and polarization onehot, https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html:
metadata_all = pd.get_dummies(metadata_all, columns=['technology', 'polarization'])

metadata = metadata_all.loc[metadata_all['IP_address_A'] == cml_A_ip]
metadata.drop(columns='IP_address_A')

# Loading cml
cml = data_loading_utility.load_cml(dir, technology, i)


## WD REFERENCE PREPROCESS
cml = preprocess_utility.ref_preprocess(cml, sample_size,
                                        comp_lin_interp=False, upsampled_n_times = upsampled_n_times,
                                        supress_single_zeros=True
                                        )



## PREPROCESS
cml = preprocess_utility.cml_preprocess(cml, interp_max_gap = 10, 
                                        suppress_step = True, conv_threshold = 250.0, 
                                        std_method = True, window_size = 10, std_threshold = 5.0, 
                                        z_method = True, z_threshold = 10.0,
                                        reset_detect=True,
                                        subtract_median=True
                                        )




'''
fig, axs = plt.subplots(1,1, figsize=(10, 2.5))
axs.set_xlabel('sample index [-]')
axs.set_ylabel('trsl [-]')
cml['trsl_A'].plot(ax=axs)   
cml['trsl_B'].plot(ax=axs)
fig.subplots_adjust(bottom=0.2)
plt.show()
'''


# plot trsl, rain and ref
fig, axs = plt.subplots(1,1, figsize=(10, 2.5))

axs.set_xlabel('sample index [-]')

cml.trsl_A.plot(ax=axs, label = 'channel 0')
cml.trsl_B.plot(ax=axs, label = 'channel 1')
axs.set_ylabel('trsl [dB]')

ax1 = axs.twinx()
cml.rain.plot(ax=ax1, color='black', linewidth=0.5, label = 'rain intensity')
ax1.set_ylabel('rain intensity [mm/30min]')
#axs.legend(['channel 0', 'channel 1', 'rain amount'],loc='upper right')
lines, labels = axs.get_legend_handles_labels()
lines2, labels2 = ax1.get_legend_handles_labels()
axs.legend(lines + lines2, labels + labels2, loc='upper right')


ref_wet_start = np.roll(cml.ref_wd, -1) & ~cml.ref_wd
ref_wet_end = np.roll(cml.ref_wd, 1) & ~cml.ref_wd
for start_i, end_i in zip(
    ref_wet_start.values.nonzero()[0],
    ref_wet_end.values.nonzero()[0],
):
    axs.axvspan(start_i, end_i, color='b', alpha=0.2, linewidth=0) 

plt.show()


#plot_utility.plot_cml(cml, columns=['rain', 'ref_wd', 'trsl', 'uptime', 'temp']) #,


## CLASS BALANCE
cml = preprocess_utility.balance_wd_classes(cml,600)

meta_repeated = pd.concat([metadata] * len(cml), ignore_index=True)
# Combine with the trsl data
cml = pd.concat([cml.reset_index(drop=True), meta_repeated], axis=1)


'''
## -------------------------
## already preprocessed data
#dir = 'TelcoRain/evaluating_dataset_short/'
dir = 'TelcoRain/merged_data_preprocessed_short/1s10/'
file_list = os.listdir(dir)

ds = []
for i in range(len(file_list)):
    cmli = pd.read_csv(dir+file_list[i])
    ds.append(cmli)

cml = pd.concat(ds, ignore_index=True) 
'''

## PLOT
#plot_utility.plot_cml(cml, columns=['rain', 'ref_wd', 'trsl', 'uptime', 'temp'])

## SHUFFLE DATASET
cml = preprocess_utility.shuffle_dataset(cml, segment_size = batchsize*sample_size)


## TRAINING
cnn_wd_threshold = 0.5
cnn_out, train_loss, test_loss = cnn_utility.cnn_train(cml,
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
                                                single_output,
                                                shuffle,
                                                save_param
                                                )

# plot output
cutoff = len(cml) % sample_size
if cutoff == 0:
    ref_wd = cml.ref_wd.values[:][::sample_size]
    cml['ref_wd'] = np.repeat(ref_wd, sample_size).astype(bool)
    cml['cnn_out'] = np.repeat(cnn_out, sample_size)   
else:
    ref_wd = cml.ref_wd.values[:-cutoff][::sample_size]
    cml['ref_wd'] = np.append(np.repeat(ref_wd, sample_size), np.zeros(cutoff)).astype(bool)
    cml['cnn_out'] = np.append(np.repeat(cnn_out, sample_size), np.zeros(cutoff))      
cml['cnn_wd'] = cml.cnn_out > cnn_wd_threshold
plot_utility.plot_cnn_classification(cml[int(len(cml)*0.8):].reset_index(drop=True),cnn_wd_threshold)

# use only last 20% which is Testdata
cnn_out = cnn_out[int(len(cnn_out)*0.8):]
ref_wd = ref_wd[int(len(ref_wd)*0.8):]

cnn_wd = cnn_out > cnn_wd_threshold
true_wet = cnn_wd & ref_wd 
false_alarm = cnn_wd & ~ref_wd

TP_test = sum(true_wet)/sum(ref_wd)
FP_test = sum(false_alarm)/sum(ref_wd)
print('testset performance')
print('TP: ' + str(TP_test))
print('FP: ' + str(FP_test))

### Metrics
# source: https://github.com/jpolz/cnn_cml_wet-dry_example/blob/master/CNN_for_CML_example_nb.ipynb
print('CNN scores')

# ROC curve 
roc_curve = metrics_utility.calculate_roc_curve(cnn_out,ref_wd,0,1)
roc_surface = metrics_utility.calculate_roc_surface(roc_curve).round(decimals=3)
print('ROC surface A:', roc_surface)
metrics_utility.plot_roc_curve(roc_curve,cnn_wd_threshold)

# confusion matrix 
cm = skl.confusion_matrix(ref_wd, cnn_wd, labels=[0,1], normalize='true').round(decimals=3)
print('normalized confusion matrix:\n',cm)
print('TNR:', cm[0,0])
print('TPR:', cm[1,1])
metrics_utility.plot_confusion_matrix(cm)

# Matthews Correlation Coeficient
mcc = skl.matthews_corrcoef(ref_wd, cnn_wd).round(decimals=3)
print('MCC:', mcc)

# ACC 
acc = np.round(skl.accuracy_score(ref_wd, cnn_wd),decimals=3)
print('ACC:', acc)

f1 = np.round(skl.f1_score(ref_wd, cnn_wd),decimals=3)
print('F1:', f1)





## CLASSIFICATION

input('press enter to continue')