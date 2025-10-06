#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filename: preprocess_all_workflow.py
Author: Lukas Kaleta
Date: 2025-05-26
Version: 2.0t
Description: 
    This script showcases the typical workflow of preprocessing all of the 
    avalable CML data and storing them as separate .csv files for classification
    or one .csv file to use as training dataset
    
    
License: 
Contact: 211312@vutbr.cz
"""

""" Notes """

""" Imports """
# Import python libraries

import os
import pandas as pd


# Import external packages

# Import own modules
from telcosense_classification import data_loading_utility
from telcosense_classification import preprocess_utility


""" Constant Variable definitions """

sample_size = 60
batchsize = 256

""" Function definitions """


""" Main """
#source and target file dir. both bust exist and contain folders with technology names
source_dir = 'TelcoRain/merged_data/'
target_dir = 'TelcoRain/merged_data_preprocessed/'  
# technologies list
technologies = ['', '', '', '']         # include list of strings of technology folder names 

# define list to store all CML dataframes
ds = []

for technology in technologies: 
    file_list = sorted(os.listdir(source_dir + technology + '/'))
    for k in range(len(file_list)//2):    
        i = 2*k
        
        ## LOADING DATA 
        cml_A_ip = file_list[i][file_list[i].rfind('CML_')+4:-4]
        cml_B_ip = file_list[i+1][file_list[i+1].rfind('CML_')+4:-4]
        cml = data_loading_utility.load_cml(source_dir, technology, i)

        ## WD REFERENCE
        cml = preprocess_utility.ref_preprocess(cml, sample_size=60,
                                                comp_lin_interp=True, upsampled_n_times = 20,
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

        ## CLASS BALANCE
        cml = preprocess_utility.balance_wd_classes(cml,600)

        ## save the one preprocessed cml
        cml.to_csv(target_dir+technology+'/'+str(i)+'_ipA_'+cml_A_ip+'_ipB_'+cml_B_ip+'.csv', index=False) 

        ## append CML into whole ds list
        cutoff_i = len(cml) % sample_size       # truncate the cml to keep sampling aligned
        ds.append(cml[:-cutoff_i])

# Concatenate CMLs into ane pandas Dataframe
one_dataset = pd.concat(ds, ignore_index=True) 

# Shuffle the whole dataset
if 0:
    cml = preprocess_utility.shuffle_dataset(cml, segment_size = batchsize*sample_size)

## save the preprocessed cml dataset
cml.to_csv(target_dir+'whole_dataset_shuffled.csv', index=False) 

