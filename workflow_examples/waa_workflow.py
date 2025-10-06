#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filename: waa_workflow.py
Author: Lukas Kaleta
Date: 2025-05-26
Version: 2.0t
Description: 
    Predict WD class using CML data and to estimate rainrate from rain induced attenuation 
    This script compares several methods of Wet Antenna Attenuation (WAA) compensations
        methods: Schleiss, Leijnse and Pastorek 
    Using modules imported from Pycomlink and code based on Pycomlink Tutorials
License: 
Contact: 211312@vutbr.cz
"""

""" Notes """


""" Imports """
# Import python libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Import external packages
import pycomlink

# Import local modules
from telcosense_classification import data_loading_utility
from telcosense_classification import preprocess_utility
from telcosense_classification import cnn_utility
from telcosense_classification import plot_utility
from telcosense_classification import metrics_utility

""" Constant Variable definitions """

# cml data dir
technology = ''                     # technology dir
dir = 'TelcoRain/merged_data/'      # common dir
i = 12                              # file index       

# CNN parameters
param_dir = 'cnn_polz_ds_cz_param_2025-05-13_17;19'     # used in the thesis

# other parameters
sample_size=60
upsampled_n_times = 20

""" Function definitions """

def hexbinplot(R_radar_along_cml, R_cml, ax, color='k', title=None, loglog=True):
    '''
    Plot scatter plot of the CML estimated rainrate vs the ref rainrate
    source: https://github.com/jpolz/cnn_cml_wet-dry_example
    '''
    R_cml.values[R_cml.values < 0] = 0
    
    ax.scatter(
        R_radar_along_cml.where(R_radar_along_cml > 0).values,
        R_cml.where(R_cml > 0).values,
        c=color,
        s=10,
        alpha=0.7,
    )
    if loglog:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_title(title)
    ax.set_xlabel('Ref. rain rate along CML [mm/h]')
    ax.set_ylabel('CML rain rate [mm/h]')


""" Main """

## --------------------------------- LOADING DATA ---------------------------------------
# list of all files
file_list = sorted(os.listdir(dir+'/'+technology))

# Loading metadata
constant_parameters = ['IP_address_A', 'IP_address_B', 'technology', 'distance', 'frequency_A', 'frequency_B', 'polarization']
metadata_all = pd.read_csv('TelcoRain/filtered_radius1.0km_offset1.0_CML.csv', usecols=constant_parameters)
# clean duplicative rows
metadata_all = metadata_all.drop_duplicates() 
cml_A_ip = file_list[i][file_list[i].rfind('CML_')+4:-4]
metadata = metadata_all.loc[metadata_all['IP_address_A'] == cml_A_ip]   

length = metadata.distance.values[0]               # m
frequency = metadata.frequency_A.values[0]*1e6     # Hz 
polarization = metadata.polarization.values[0]

# Loading cml 
cml = data_loading_utility.load_cml(dir, technology, i)


## --------------------------------- PREPROCESSING -------------------------------------------
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


# --------------------------------- WD CLASSIFICATION --------------------------------------
cnn_out, _ =  cnn_utility.cnn_classify(cml, 
                        param_dir,
                        num_channels=2,
                        sample_size=sample_size, 
                        batchsize=256
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


# --------------------------------- BASELINE ESTIMATION -------------------------------------
cml['baseline'] = pycomlink.processing.baseline.baseline_constant(
        trsl=cml.trsl,
        wet=cml.cnn_wd,
        n_average_last_dry=10
)


# --------------------------------- WAA calculation ------------------------------------
# estimate rain induced attenuation without WAA
# needed for Leijnse and Pastorek WAA model
cml['A_rain'] = cml.trsl - cml.baseline
cml['A_rain'].values[cml.A_rain < 0] = 0

# R using k-R relation without WAA comp.
cml['R'] = pycomlink.processing.k_R_relation.calc_R_from_A(
        A = cml.A_rain, 
        L_km=length/1e3,
        f_GHz=frequency/1e9,
        pol=polarization
)

# Schleiss WAA estimation
cml['waa_schleiss'] = pycomlink.processing.wet_antenna.waa_schleiss_2013(
    rsl=cml.trsl, 
    baseline=cml.baseline, 
    wet=cml.cnn_wd, 
    waa_max=2.2, 
    delta_t=1, 
    tau=15,
)

# Leijnse WAA estimation
cml['waa_leijnse'] = pycomlink.processing.wet_antenna.waa_leijnse_2008_from_A_obs(
    A_obs=cml.A_rain,
    f_Hz=frequency,
    pol=polarization,
    L_km=length/1e3,
)

# Pastorek WAA estimation
cml['waa_pastorek'] = pycomlink.processing.wet_antenna.waa_pastorek_2021_from_A_obs(
    A_obs=cml.A_rain,
    f_Hz=frequency,
    pol=polarization,
    L_km=length/1e3,
    A_max=2.2,
)

# --------------------------------- RAIN INTENSITY ESTIMATION ----------------------------------
# for all 3 WAA methods
for waa_method in ['leijnse', 'pastorek', 'schleiss']:
    # rain induced ATT
    cml[f'A_rain_{waa_method}'] = cml.A_rain - cml[f'waa_{waa_method}']
    cml[f'A_rain_{waa_method}'] = cml[f'A_rain_{waa_method}'].where(cml[f'A_rain_{waa_method}'] >= 0, 0)    
    
    # R est with k-r relation
    cml[f'R_{waa_method}'] = pycomlink.processing.k_R_relation.calc_R_from_A(
        A=cml[f'A_rain_{waa_method}'], 
        L_km=length/1e3, 
        f_GHz=frequency/1e9, 
        pol=polarization
    )

# --------------------------------- PLOT THE OUTPUT --------------------------------------
# use only segment of cml for demonstration
# cml = cml[2300:4000]

# TRSL + rain induced attenuation
fig, axs = plt.subplots(3, 1, figsize=(15, 8.4), sharex=True)
plt.sca(axs[0])
cml.trsl.plot.line(x='time', label='trsl', color='k', linewidth=0.5, linestyle='--',zorder=10)
cml.baseline.plot.line(x='time', label='baseline', color='C0')
(cml.baseline + cml.waa_schleiss).plot.line(x='time', label='baseline + WAA_schleiss', color='C3')
(cml.baseline + cml.waa_leijnse).plot.line(x='time', label='baseline + WAA_leijnse', color='C1')
(cml.baseline + cml.waa_pastorek).plot.line(x='time', label='baseline + WAA_pastorek', color='C2')

plt.ylabel('total path attenuation [dB]')
plt.title(f'cml_length = {round(length/1e3,3)} km, frequency = {frequency/1e9} GHz')
plt.legend(loc='upper left')

# Estimated R with Reference rainrate 
plt.sca(axs[1])
(cml.rain*2).plot.line(color='k', linewidth=2.0, label='rain gauge', alpha=0.5)
cml.R.plot.line(x='time', label='no WAA', color='C0')
cml.R_leijnse.plot.line(x='time', label='with WAA_leijnse', color='C1')
cml.R_pastorek.plot.line(x='time', label='with WAA_pastorek', color='C2')
cml.R_schleiss.plot.line(x='time', label='with WAA_schleiss', color='C3')
plt.ylabel('Rain-rate [mm/h]')
plt.title('')
plt.legend(loc='upper left')

# Rain amount cumulative sum
plt.sca(axs[2])
(cml.rain*2).cumsum(axis='index').plot.line(color='k', linewidth=2.0, label='rain gauge', alpha=0.5)
(cml.R).cumsum(axis='index').plot.line(x='time', label='no WAA', color='C0')
(cml.R_pastorek).cumsum(axis='index').plot.line(x='time', label='with WAA_pastorek', color='C2')
(cml.R_schleiss).cumsum(axis='index').plot.line(x='time', label='with WAA_schleiss', color='C3')
(cml.R_leijnse).cumsum(axis='index').plot.line(x='time', label='with WAA_leijnse', color='C1')
plt.ylabel('Rainfall sum [mm]')
plt.title('')
plt.legend(loc='upper left')

plt.show()


# --------------------------------- PLOT HEXBINPLOT ---------------------------------------
cml['time'] = pd.to_datetime(cml['time'])
cml = cml.resample('30min',on='time').mean()

fig, axs = plt.subplots(1, 4, figsize=(22, 5), sharex=False, sharey=False)

hexbinplot(cml.rain, cml.R, axs[0], 'C0', 'Without WAA')
hexbinplot(cml.rain, cml.R_leijnse, axs[1], 'C1', 'WAA Leijnse')
hexbinplot(cml.rain, cml.R_pastorek, axs[2], 'C2', 'WAA Pastorek')
hexbinplot(cml.rain, cml.R_schleiss, axs[3], 'C3', 'WAA Schleiss')

for ax in axs:
    ax.plot([0.01, 50], [0.01, 50], 'k', alpha=0.3)
    ax.set_xlim(0.05, 50)
    ax.set_ylim(0.05, 50)

plt.show()


input('press enter to continue')