#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filename: model_comparison_pipeline.py
Author: Lukas Kaleta
Date: 2025-05-05
Version: 1.0
Description: 
    This script compares several methods of Wet/Dry classification using CML data
    Methods in comparison:
        Developed CNN model
        Reference CNN model from https://github.com/jpolz/cml_wd_pytorch 
        State of the art, non-ML method using rolling window STD: https://github.com/pycomlink/pycomlink
  
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
import sklearn.metrics as skl

# Import external packages
import pycomlink

# Import own modules
from telcosense_classification import cnn_utility
from telcosense_classification import plot_utility
from telcosense_classification import preprocess_utility
from telcosense_classification import metrics_utility


""" Constant Variable definitions """

dir = 'TelcoRain/merged_data_preprocessed_short_old/whole_dataset_shuffled.csv'     # shuffled by 256*60
param_dir = 'cnn_param_thesis_v22_2025-05-15_22;01'

# Training CNN parameters
num_channels = 2
sample_size = 60            # 60 keep lower than FC num of neurons
batchsize = 256             # 128 most smooth
epochs = 15                 # 50
resume_epoch = 0 
learning_rate = 0.0002      # 0.0002 or 0.0003
dropout_rate = 0.001        # 0.001
kernel_size = 3             # 3 - best performance
n_conv_filters = [16, 32, 64, 128]     # [16, 32, 64, 128]
n_fc_neurons = 64          # 64 better FP, 128 better TP
single_output = True
shuffle = False
save_param = True

""" Function definitions """


""" Main """
cml = pd.read_csv(dir)















cnn_wd_threshold = 0.5

# Training
if 0:  
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
    cutoff = len(cml) % sample_size
    if cutoff == 0:
        ref_wd = cml.ref_wd.values[:][::sample_size]
        cml['cnn_out'] = np.repeat(cnn_out, sample_size)   
    else:
        ref_wd = cml.ref_wd.values[:-cutoff][::sample_size]
        cml['cnn_out'] = np.append(np.repeat(cnn_out, sample_size), np.zeros(cutoff))      
    cml['cnn_wd'] = cml.cnn_out > cnn_wd_threshold
    plot_utility.plot_cnn_classification(cml[int(len(cml)*0.8):].reset_index(drop=True),cnn_wd_threshold)
   
    # use only last 20% which is Testdata
    cnn_out = cnn_out[int(len(cnn_out)*0.8):]
    ref_wd = ref_wd[int(len(ref_wd)*0.8):]

    cnn_wd = cnn_out > cnn_wd_threshold
    true_wet = cnn_wd & ref_wd 
    false_alarm = cnn_wd & ~ref_wd

    
elif 1:
    # Classification
    cml = cml[int(len(cml)*0.8):].reset_index(drop=True)



    #plot_utility.plot_input_oneplot(cml)


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
    
    cutoff = len(cml) % sample_size
    if cutoff == 0:
        ref_wd = cml.ref_wd.values[:][::sample_size]
        cml['cnn_out'] = np.repeat(cnn_out, sample_size)   
    else:
        ref_wd = cml.ref_wd.values[:-cutoff][::sample_size]
        cml['cnn_out'] = np.append(np.repeat(cnn_out, sample_size), np.zeros(cutoff))      
    cml['cnn_wd'] = cml.cnn_out > cnn_wd_threshold
    plot_utility.plot_cnn_classification(cml.reset_index(drop=True),cnn_wd_threshold)
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
'''
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
'''
ref_wd = ref_wd[591100//60:646000//60]
cnn_out = cnn_out[591100//60:646000//60]
cnn_wd = cnn_wd[591100//60:646000//60]

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