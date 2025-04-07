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


# Import external packages

# Import own modules
from telcosense_classification import cnn_utility
from telcosense_classification import plot_utility


""" Constant Variable definitions """

dir = 'TelcoRain/evaluating_dataset/'   # directory containing 10 preprocessed CMLs across technologies

# Training CNN parameters
num_channels = 2
sample_size = 60            # 60 keep lower than FC num of neurons
batchsize = 128             # 128 most smooth (64)
epochs = 30                 # 50
resume_epoch = 0 
#learning_rate = 0.0008      # 0.0005 - 0.001 
dropout_rate = 0.003         # 0.04 train loss: 0.5158175 test loss: 0.43611026
kernel_size = 3             # 3 - best performance
n_conv_filters = [24, 48, 96, 192]     # [48, 96, 192, 384] 4% worse
n_fc_neurons = 128          # 64 better FP, 128 better TP
save_param = False

""" Function definitions """


""" Main """

## LOADING DATA 
file_list = os.listdir(dir)

ds = []

for i in range(len(file_list)):
    cml = pd.read_csv(dir+file_list[i])
    ds.append(cml) 
#plot_utility.plot_cml(ds[0], columns=['rain', 'ref_wd', 'trsl', 'uptime', 'temp'])


cnn_wd_threshold = 0.5


for learning_rate in [0.0001,0.0005,0.001,0.002,0.003]:
    mean_results = [['train_loss', 'test_loss', 'TP', 'FP']]
    for param in range(10):
        ## TRAINING sample
        results = [['train_loss', 'test_loss', 'TP', 'FP']]
        for i in range(len(file_list)):
            cnn_out, train_loss, test_loss = cnn_utility.cnn_train_period_classification(ds[i], 
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
        results = [['train_loss', 'test_loss', 'TP', 'FP']]
        for i in range(len(file_list)):
            ds[i]['cnn_out'], train_loss, test_loss = cnn_utility.cnn_train_timestep_classification(ds[i], 
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

            ds[i]['cnn_wd'] = ds[i].cnn_out > cnn_wd_threshold
            ds[i]['true_wet'] = ds[i].cnn_wd & ds[i].ref_wd 
            ds[i]['false_alarm'] = ds[i].cnn_wd & ~ds[i].ref_wd

            TP = sum(ds[i].true_wet)/sum(ds[i].ref_wd)
            FP = sum(ds[i].false_alarm)/sum(ds[i].ref_wd)
            cml_res = [train_loss, test_loss, TP, FP]
            results.append(cml_res)
        
        df = pd.DataFrame(results, columns=['train_loss', 'test_loss', 'TP', 'FP'])
        df.to_csv('results/results_timestepout_'+str(param)+'.csv')
        '''
    #mean_results.append(np.mean(mean_results[1:],0))
    df = pd.DataFrame(mean_results, columns=['train_loss', 'test_loss', 'TP', 'FP'])
    df.to_csv('results/'+str(learning_rate)+'/mean_repeat_'+str(param)+'.csv')
## CLASSIFICATION

#input('press enter to continue')