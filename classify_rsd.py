#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filename: parameter_tuning_loop.py
Author: Lukas Kaleta
Date: 2025-05-29
Version: 1.0
Description: 


License: 
Contact: 211312@vutbr.cz
"""

""" Notes """


""" Imports """
# Import python libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import sklearn.metrics as skl
import os


# Import external packages

# Import own modules
from telcosense_classification import cnn_utility
from telcosense_classification import plot_utility
from telcosense_classification import preprocess_utility
from telcosense_classification import preprocess_utility_old

from telcosense_classification import metrics_utility
from telcosense_classification import data_loading_utility



""" Constant Variable definitions """

sample_size = 60
""" Function definitions """


""" Main """


## REF DATASET

# load 500 CMLs with 1 min time step
cml_set = xr.open_dataset('example_data/example_cml_data.nc', engine='netcdf4') 
cml_set = preprocess_utility_old.cml_preprocess(cml_set, interp_max_gap='5min')


# load path averaged reference RADOLAN data aligned with all 500 CML IDs with 5 min time step
ref_set = xr.open_dataset('example_data/example_path_averaged_reference_data.nc', engine='netcdf4')
ref_set = ref_set.rename_vars({'rainfall_amount':'rain'})
ref_set = preprocess_utility_old.ref_preprocess(ref_set, interp_max_gap='20min', resample=sample_size)




 # get cml ids with NaN gaps
cml_set['trsl_gap'] = (
        ('cml_id', 'time'), 
        np.logical_or(np.isnan(cml_set.trsl.isel(channel_id=0).values), np.isnan(cml_set.trsl.isel(channel_id=1).values))
)

has_gap = np.any(cml_set.trsl_gap.values, axis=1)
cmls_without_gap = np.where(~has_gap)[0]
num_correct_cmls = len(cmls_without_gap)

cml_set = cml_set.assign_coords({'has_gap':(('cml_id'), has_gap)})
ref_set = ref_set.assign_coords({'has_gap':(('cml_id'), has_gap)})

# 222, 301, 494

cml_set = cml_set.where(~cml_set["has_gap"], drop=True)
ref_set = ref_set.where(~ref_set["has_gap"], drop=True)
num_cmls = len(cml_set.cml_id)

ref_set['ref_wd'] = ref_set.rain.where(ref_set.rain == 0, True).astype(bool)





# ----------------------------------------------------------------------------------------------
cml_set = cml_set.sel(cml_id='492')
ref_set = ref_set.sel(cml_id='492')
#cml_set = cml_set.isel(cml_id = range(int(500*0.8),497))
#ref_set = ref_set.isel(cml_id = range(int(500*0.8),497))
#num_cmls = len(cml_set.cml_id)

trsl = cml_set.isel(channel_id=0).trsl.values.reshape(-1)
rain = np.repeat(ref_set.rain.values.reshape(-1),60)
ref_wd = np.repeat(ref_set.ref_wd.values.reshape(-1).astype(bool),60)


d = {'trsl_A': trsl, 'rain': rain, 'ref_wd': ref_wd}
cml = pd.DataFrame(data = d)







## CZECH DATASET
'''
i = 1
dir = 'TelcoRain/merged_data/'
file_list = os.listdir(dir)
'''
'''
ds = []
for file in file_list:
    cmli = pd.read_csv(dir+file).reset_index(drop=True)
    ds.append(cmli)

cml = pd.concat(ds, ignore_index=True) 

'''
'''
cml = data_loading_utility.load_cml(dir, '1s10', i)

# exclude extreme values
cml['trsl_A'] = cml.trsl_A.where(cml.trsl_A < 99.0)
cml['trsl_B'] = cml.trsl_B.where(cml.trsl_B < 99.0)

# First interpolation both trsl, and R and drop missing values
cml = cml.interpolate(axis=0, method='linear', limit = 60)
cml = cml.dropna(axis=0, how = 'all', subset=['trsl_A','trsl_B'])
cml = cml.reset_index(drop=True)
cml = cml.interpolate(axis=0, method='linear')

cml = preprocess_utility.ref_preprocess(cml, 60,
                                        comp_lin_interp=False, upsampled_n_times = 20,
                                        supress_single_zeros=True
                                        )

cml = preprocess_utility.balance_wd_classes(cml, max_zero_length = 600)

plot_utility.plot_input_oneplot(cml)
'''
#RSD method
#cml = cml[int(len(cml)*0.8):].reset_index(drop=True)

RSD_threshold = 0.58

rolling_std = cml['trsl_A'].rolling(window=60, center=True).std()

# Fill NaN values at the edges
rolling_std.fillna(method='bfill', inplace=True)
rolling_std.fillna(method='ffill', inplace=True)

# RSD output
cml['cnn_out'] = rolling_std
cml['cnn_wd'] = cml.cnn_out > RSD_threshold
 
plot_utility.plot_cnn_classification(cml.reset_index(drop=True),cnn_wd_threshold=RSD_threshold)

cnn_out = cml.cnn_out.values
ref_wd = cml.ref_wd.values
cnn_wd = cml.cnn_wd.values
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
metrics_utility.plot_roc_curve(roc_curve,RSD_threshold)

# confusion matrix 
cm = skl.confusion_matrix(ref_wd, cnn_wd, labels=[1,0], normalize='true').round(decimals=3)
print('normalized confusion matrix:\n',cm)
print('TPR:', cm[0,0])
print('TNR:', cm[1,1])
metrics_utility.plot_confusion_matrix(cm)

# Matthews Correlation Coeficient
mcc = skl.matthews_corrcoef(ref_wd, cnn_wd).round(decimals=3)
print('MCC:', mcc)

# ACC 
acc = np.round(skl.accuracy_score(ref_wd, cnn_wd),decimals=3)
print('ACC:', acc)

f1 = np.round(skl.f1_score(ref_wd, cnn_wd),decimals=3)
print('F1:', f1)

input("Press Enter to continue...")