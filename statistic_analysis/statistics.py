#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filename: statistics.py
Author: Lukas Kaleta
Date: 2025-02-26
Version: 1.0
Description: 

License: 
Contact: 211312@vutbr.cz
"""


""" Imports """
# Import python libraries

import math
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import os

# Import external packages

# Import own modules

""" ConstantVariable definitions """

""" Function definitions """


""" Main """


## LOADING DATA
# Loading metadata
metadata_all = pd.read_csv('TelcoRain/filtered_radius1.0km_offset1.0_CML.csv')
metadata_all = metadata_all.drop_duplicates()         # clean duplicative rows


## analyze stationary CML parameters
# CML length, frequency, polarization, gauge distance from cmlA, cmlB, altitude, height above terrain
# frequency_A,frequency_B,distance,polarization,altitude_A,height_above_terrain_A,height_above_terrain_B,distance_A,distance_B,gauge_elevation
frequency = metadata_all.frequency_A

plt.hist(np.append(metadata_all.frequency_A.values,metadata_all.frequency_B.values))
plt.show() 

plt.hist(metadata_all.polarization)
plt.show() 

plt.hist(metadata_all.distance)
plt.show() 


## TRSL statistic analysis
# Get the list of all csvs
path = 'TelcoRain/merged_data/summit/'
file_list = os.listdir(path)

d = {'ip' : ['?'] * len(file_list),
     'std' : np.empty(len(file_list)),
     'max' : np.empty(len(file_list)),
     'min' : np.empty(len(file_list)),
     'mean' : np.empty(len(file_list)),
     'median' : np.empty(len(file_list)),
     'Z_max' : np.empty(len(file_list)),
     'max_diff' : np.empty(len(file_list))
      }
stat_data = pd.DataFrame(data=d)
    

for i in range(len(file_list)):
   
    cml_ip = file_list[i][file_list[i].rfind('CML_')+4:-4]
    cml = pd.read_csv(path+file_list[i], usecols=['cml_PrijimanaUroven'])   #,'cml_MaximalniRychlostRadia(modulace)' cml_Teplota,cml_RxDatovyTok,cml_KvalitaSignalu,cml_Uptime
    cml = cml.rename(columns={'cml_PrijimanaUroven':'rsl'})
    
    cml['rsl'] = cml.rsl.where(abs((cml.rsl-cml.rsl.mean())/cml.rsl.std()) < 13.0)

    ## PREPROCESS
    # interpolation both rsl, and R
    cml = cml.interpolate(axis=0, method='linear', limit = 10)
    # skip rows with missing rsl values
    cml = cml.dropna(axis=0, how = 'all')



    stat_data['ip'][i] = cml_ip
    stat_data['std'][i] = cml.rsl.std()
    stat_data['max'][i] = cml.rsl.max()
    stat_data['min'][i] = cml.rsl.min()
    stat_data['mean'][i] = cml.rsl.mean()
    stat_data['median'][i] = cml.rsl.median()
    stat_data['Z_max'][i] = (cml.rsl.max()-cml.rsl.mean())/cml.rsl.std()
    stat_data['max_diff'][i] = np.diff(cml.rsl).max()+np.diff(cml.rsl).min()


stat_data.to_csv('statistics_summit_diff2.csv', sep=',')

fig, axs = plt.subplots(figsize=(12, 8))
stat_data.plot(ax=axs,x='ip',subplots=True)
plt.show()



## TRAINING

## CLASSIFICATION



input("Press Enter to continue...")