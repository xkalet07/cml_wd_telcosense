#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filename: preprocess_all.py
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

""" Imports """
# Import python libraries

import os


# Import external packages

# Import own modules
from telcosense_classification import data_loading_utility
from telcosense_classification import preprocess_utility


""" Constant Variable definitions """


""" Function definitions """


""" Main """
source_dir = 'TelcoRain/merged_data/'
target_dir = 'TelcoRain/merged_data_preprocessed_short/'


for technology in ['summit', 'summit_bt', '1s10', 'ceragon_ip_20']: # 'ceragon_ip_10' #doesnt have a temperature 
    file_list = sorted(os.listdir(source_dir + technology + '/'))
    for k in range(len(file_list)//2):    
        i = 2*k
        
        ## LOADING DATA 
        cml_A_ip = file_list[i][file_list[i].rfind('CML_')+4:-4]
        cml_B_ip = file_list[i+1][file_list[i+1].rfind('CML_')+4:-4]
        cml = data_loading_utility.load_cml(source_dir, technology, i)

        ## PREPROCESS
        cml = preprocess_utility.cml_preprocess(cml, interp_max_gap = 10, 
                        suppress_step = True, conv_threshold = 250.0, 
                        std_method = True, window_size = 10, std_threshold = 5.0, 
                        z_method = True, z_threshold = 10.0,
                        reset_detect=True,
                        subtract_median=True
                        )

        ## WD REFERENCE
        cml = preprocess_utility.ref_preprocess(cml, sample_size=60,
                                                comp_lin_interp=True, upsampled_n_times = 20,
                                                supress_single_zeros=True
                                                )

        ## CLASS BALANCE
        cml = preprocess_utility.balance_wd_classes(cml,600)

        ## save the preprocessed cml
        cml.to_csv(target_dir+technology+'/'+str(i)+'_ipA_'+cml_A_ip+'_ipB_'+cml_B_ip+'.csv', index=False) 

