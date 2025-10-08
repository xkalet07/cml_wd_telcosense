#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filename: inference_workflow.py
Author: Lukas Kaleta
Date: 2025-05-26
Version: 2.0t
Description: 
    This script showcases the typical workflow of utilizing the trained CNN module
    for rain event detection using CML data from czech republic and CHMI.

License: 
Contact: 211312@vutbr.cz
"""

""" Notes """


""" Imports """
# Import python libraries

import numpy as np
import pandas as pd

import os

import sklearn.metrics as skl

# Import external packages

# Import local modules
from telcosense_classification import data_loading_utility
from telcosense_classification import preprocess_utility
from telcosense_classification import cnn_utility
from telcosense_classification import plot_utility
from telcosense_classification import metrics_utility


""" Constant Variable definitions """

# CML file location
technology = ''                     # technology dir
dir = 'TelcoRain/merged_data/'      # common dir
i = 12                              # file index        

# CNN parameters
param_dir = 'cnn_polz_ds_cz_param_2025-05-13_17;19'     # used in the thesis

# Training CNN parameters
num_channels = 2
sample_size = 60            # keep lower than num of FC neurons
batchsize = 256 
kernel_size = 3             
n_conv_filters = [16, 32, 64, 128]
n_fc_neurons = 64           # 64 better FP, 128 better TP
single_output = True

# other parameters
upsampled_n_times = 20

""" Function definitions """


""" Main """

## ---------------------------------------- LOADING DATA ---------------------------------------------

# list of all files
path = dir + technology+'/'
file_list = sorted(os.listdir(path))

# Loading cml
cml = data_loading_utility.load_cml(dir, technology, i)


## ---------------------------------------- PREPROCESSING --------------------------------------------
# WD rference preprocessing
cml = preprocess_utility.ref_preprocess(cml, sample_size,
                                        comp_lin_interp=False, upsampled_n_times = upsampled_n_times,
                                        supress_single_zeros=True
                                        )

# CML preprocessing
cml = preprocess_utility.cml_preprocess(cml, interp_max_gap = 10, 
                                        suppress_step = True, conv_threshold = 250.0, 
                                        std_method = True, window_size = 10, std_threshold = 5.0, 
                                        z_method = True, z_threshold = 10.0,
                                        reset_detect=True,
                                        subtract_median=True
                                        )

# Balance WD classes
if 0:
    cml = preprocess_utility.balance_wd_classes(cml,600)

# plot input trsl with ref rain data
plot_utility.plot_input_oneplot(cml)

# Shuffle the dataset
if 0:
    cml = preprocess_utility.shuffle_dataset(cml, segment_size = batchsize*sample_size)


## ---------------------------------------- WD CLASSIFICATION ----------------------------------------

cnn_out, loss =  cnn_utility.cnn_classify(cml, 
                        param_dir,
                        num_channels,
                        sample_size, 
                        batchsize, 
                        kernel_size,
                        n_conv_filters,
                        n_fc_neurons,
                        single_output
                        )

# aligning the output back into dataset
cutoff = len(cml) % sample_size     # number of values cut off by sample grouping 
cnn_wd_threshold = 0.5

if cutoff == 0:
    ref_wd = cml.ref_wd.values[:][::sample_size]
    cml['cnn_out'] = np.repeat(cnn_out, sample_size)   
else:
    ref_wd = cml.ref_wd.values[:-cutoff][::sample_size]
    cml['cnn_out'] = np.append(np.repeat(cnn_out, sample_size), np.zeros(cutoff))      
cml['cnn_wd'] = cml.cnn_out > cnn_wd_threshold

plot_utility.plot_cnn_classification(cml.reset_index(drop=True),cnn_wd_threshold)



## ------------------------------------ CNN PERFORMANCE METRICS -----------------------------------------
#  source: https://github.com/jpolz/cnn_cml_wet-dry_example/blob/master/CNN_for_CML_example_nb.ipynb
cnn_wd = cnn_out > cnn_wd_threshold
true_wet = cnn_wd & ref_wd 
false_alarm = cnn_wd & ~ref_wd

print('CNN scores')

# ROC curve 
roc_curve = metrics_utility.calculate_roc_curve(cnn_out,ref_wd,0,1)
roc_surface = metrics_utility.calculate_roc_surface(roc_curve).round(decimals=3)
print('ROC surface A:', roc_surface)
metrics_utility.plot_roc_curve(roc_curve,cnn_wd_threshold)

# confusion matrix 
cm = skl.confusion_matrix(ref_wd, cnn_wd, labels=[1,0], normalize='true').round(decimals=3)
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


input('press enter to continue')