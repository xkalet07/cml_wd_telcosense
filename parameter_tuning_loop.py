#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filename: parameter_tuning_loop.py
Author: Lukas Kaleta
Date: 2025-03-29
Version: 1.0
Description: 
    This script runs CNN training script in a for loop while tuning hyperparameters 
    or CNN structure. Sript evaluates CNN performance for each given parameter.

License: 
Contact: 211312@vutbr.cz
"""

""" Notes """


""" Imports """
# Import python libraries

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import datetime


# Import external packages

# Import own modules
from telcosense_classification import cnn_utility
from telcosense_classification import plot_utility
from telcosense_classification import preprocess_utility


""" Constant Variable definitions """

#dir = 'TelcoRain/merged_data_preprocessed_long/1s10/'
#dir = 'TelcoRain/evaluating_dataset/'   # directory containing 10 preprocessed CMLs across technologies

# Training CNN parameters
num_channels = 2
sample_size = 60            # 60 keep lower than FC num of neurons
batchsize = 128             # 128 most smooth (64)
epochs = 40                 # 50
resume_epoch = 0 
learning_rate = 0.0002      # or 0.0003
dropout_rate = 0.001        # 0.001
kernel_size = 3             # 3 - best performance
n_conv_filters = [24, 48, 96, 192]     # [16, 32, 64, 128] 2% worse TPR but lower testloss -> less overfitting
n_fc_neurons = 128          # 128 (64 better FP, 128 better TP)
single_output = True
shuffle = False             # use True (for testloss: 1.3)
save_param = False

# DONE: weight decay
# DONE: dropout between conv layers
# TODO: early stopping
# DONE: leaky relu

""" Function definitions """


""" Main """



# load x cml at once
cnn_wd_threshold = 0.5
mean_results = [['train_loss', 'test_loss', 'TP', 'FP']]
## TRAINING sample
for technology in ['summit', 'summit_bt', '1s10', 'ceragon_ip_20']:
    ## LOADING DATA 
    dir = 'TelcoRain/merged_data_preprocessed/'+technology+'/'
    file_list = os.listdir(dir)

    ds = []
    for i in range(10):
        cmli = pd.read_csv(dir+file_list[i])
        ds.append(cmli)

    cml = pd.concat(ds, ignore_index=True) 

    cml = preprocess_utility.shuffle_dataset(cml, segment_size = batchsize*sample_size)

    results = [['train_loss', 'test_loss', 'TP', 'FP']]
    for param in range(10):
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
                                            single_output,
                                            shuffle,
                                            save_param
                                            )

        cutoff = len(cml) % sample_size
        if cutoff == 0:
            ref_wd = cml.ref_wd.values[:][::sample_size]   
        else:
            ref_wd = cml.ref_wd.values[:-cutoff][::sample_size]   

        cnn_wd = cnn_out > cnn_wd_threshold
        true_wet = cnn_wd & ref_wd 
        false_alarm = cnn_wd & ~ref_wd

        TP = sum(true_wet)/sum(ref_wd)
        FP = sum(false_alarm)/sum(ref_wd)
        cml_res = [train_loss, test_loss, TP, FP]
        results.append(cml_res)
    
    df = pd.DataFrame(results, columns=['train_loss', 'test_loss', 'TP', 'FP'])
    df.to_csv('results/results'+technology+'.csv')     # +datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    mean_results.append(np.mean(results[1:],0))
df = pd.DataFrame(mean_results, columns=['train_loss', 'test_loss', 'TP', 'FP'])
df.to_csv('results/mean_repeat.csv')



# load cmls separately
'''
for learning_rate in [0.0001,0.0005,0.001,0.002,0.003]:
    mean_results = [['train_loss', 'test_loss', 'TP', 'FP']]
    for param in range(10):
        ## TRAINING sample
        results = [['train_loss', 'test_loss', 'TP', 'FP']]
        for i in range(len(file_list)):
            cnn_out, train_loss, test_loss = cnn_utility.cnn_train_period(ds[i], 
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

            cutoff = len(ds[i]) % sample_size
            if cutoff == 0:
                ref_wd = ds[i].ref_wd.values[:][::sample_size]   
            else:
                ref_wd = ds[i].ref_wd.values[:-cutoff][::sample_size]   
            
            cnn_wd = cnn_out > cnn_wd_threshold
            true_wet = cnn_wd & ref_wd 
            false_alarm = cnn_wd & ~ref_wd

            TP = sum(true_wet)/sum(ref_wd)
            FP = sum(false_alarm)/sum(ref_wd)
            cml_res = [train_loss, test_loss, TP, FP]
            results.append(cml_res)

        mean_results.append(np.mean(results[1:],0))
        df = pd.DataFrame(results, columns=['train_loss', 'test_loss', 'TP', 'FP'])
        df.to_csv('results/'+str(learning_rate)+'/results_repeat_'+str(param)+'.csv')

        ## TRAINING timestep
'''
## CLASSIFICATION


#input('press enter to continue')