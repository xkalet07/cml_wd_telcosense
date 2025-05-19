#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filename: plot_utility.py
Author: Lukas Kaleta
Date: 2025-01-31
Version: 1.0
Description: 
    This script contains function set for using matplotlib to plot 
    input/output/reference of CNN w/d classification using data from CML.

License: 
Contact: 211312@vutbr.cz
"""

""" Imports """
# Import python libraries

import numpy as np
import matplotlib.pyplot as plt

import xarray as xr
import pandas as pd

# Import external packages

# Import own modules


""" Variable definitions """


""" Function definitions """



def plot_cml(cml:pd.DataFrame, columns = ['rain', 'ref_wd', 'trsl']):
    """
    plot selected columns from cml dataframe

    Parameters
    cml : pandas.DataFrame, containing one CML with trsl, reference rain, reference WD flag,
        timestamps, temperature, uptime etc
    columns : list of strings, desired cml variables (columns) to be plotted.
        example: 'rain', 'ref_wd', 'trsl', 'uptime', 'temp', 'rsl', 'tsl'... 
        default: 'rain', 'ref_wd', 'trsl'

    Returns: none
    """
    
    if 'rain' in columns:
        rain = True
        columns.remove('rain')
    else:
        rain = False

    if 'ref_wd' in columns:
        ref = True
        ref_wet_start = np.roll(cml.ref_wd, -1) & ~cml.ref_wd
        ref_wet_end = np.roll(cml.ref_wd, 1) & ~cml.ref_wd
        columns.remove('ref_wd')
    else:
        ref = False   
    
    fig, axs = plt.subplots(len(columns),1, figsize=(12, 2*len(columns)))
    #ax1 = axs[0].twinx()
    #axs[0].set_title('ip goes here')  #(cml_A_ip + ', ' + str(i)))
    
    axs[len(columns)-1].set_xlabel('sample index [-]')

    for i in range(len(columns)):
        cml[columns[i]+'_A'].plot(ax=axs[i])   
        cml[columns[i]+'_B'].plot(ax=axs[i])
        axs[i].set_ylabel(columns[i])

        # plot ref_wd
        if ref:
            for start_i, end_i in zip(
                ref_wet_start.values.nonzero()[0],
                ref_wet_end.values.nonzero()[0],
            ):
                axs[i].axvspan(start_i, end_i, color='b', alpha=0.2, linewidth=0) 
        if rain:
            ax1 = axs[i].twinx()
            cml.rain.plot(ax=ax1, color='black', linewidth=0.5)
            ax1.set_ylabel('rain amount [mm]')

    plt.show()

def plot_input_oneplot(cml):
    # plot trsl, rain and ref
    fig, axs = plt.subplots(1,1, figsize=(10, 2.5))

    axs.set_xlabel('sample index [-]')

    cml.trsl_A.plot(ax=axs, label = 'channel 0')
    cml.trsl_B.plot(ax=axs, label = 'channel 1')
    axs.set_ylabel('trsl [dB]')
    axs.set_ylim([-0.5,1])

    ax1 = axs.twinx()
    cml.rain.plot(ax=ax1, color='black', linewidth=0.5, label = 'rain intensity')
    ax1.set_ylabel('rain intensity [mm/30min]')
    ax1.set_ylim([-0.5,5])
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




def plot_cnn_classification(cml:pd.DataFrame, cnn_wd_threshold = 0.5):
    """
    plot output of the CNN classification, shade TP, FP, FN periods with colors,

    Parameters
    cml : pandas.DataFrame, containing one CML with trsl, reference rain, reference WD flag,
        timestamps, temperature, uptime and CNN output
    
    Returns: none
    """
    # predicted true wet
    cml['true_wet'] = cml.cnn_wd & cml.ref_wd 
    # cnn false alarm
    cml['false_alarm'] = cml.cnn_wd & ~cml.ref_wd
    # cnn missed wet
    cml['missed_wet'] = ~cml.cnn_wd & cml.ref_wd


    fig, axs = plt.subplots(1,1, figsize=(9.1, 2.5))
    #ax1 = axs[0].twinx()
    #axs.set_title('ip goes here')  #(cml_A_ip + ', ' + str(i)))

    #cml['trsl_A'].plot(ax=axs)   
    #cml['trsl_B'].plot(ax=axs)
    cml['cnn_out'].plot(ax=axs, color='black', linewidth=0.7, )
    axs.axhline(cnn_wd_threshold, color='black', linestyle='--', lw=0.7)
    axs.set_ylabel('cnn_output [-]')
    axs.set_xlabel('sample index [-]')
    axs.legend(['cnn_out', '$\\tau$ ='+str(cnn_wd_threshold)],loc='upper right')


    # GREEN: plot true cnn predicted wet/dry areas
    start = np.roll(cml.true_wet, -1) & ~cml.true_wet
    end = np.roll(cml.true_wet, 1) & ~cml.true_wet
    for start_i, end_i in zip(
        start.values.nonzero()[0],
        end.values.nonzero()[0],
    ):
        axs.axvspan(start_i, end_i, color='g', alpha=0.3, linewidth=0)

    # RED: plot false alarms
    start = np.roll(cml.false_alarm, -1) & ~cml.false_alarm
    end = np.roll(cml.false_alarm, 1) & ~cml.false_alarm
    for start_i, end_i in zip(
        start.values.nonzero()[0],
        end.values.nonzero()[0],
    ):
        axs.axvspan(start_i, end_i, color='r', alpha=0.3, linewidth=0)

    # ORANGE: missed wet periods
    start = np.roll(cml.missed_wet, -1) & ~cml.missed_wet
    end = np.roll(cml.missed_wet, 1) & ~cml.missed_wet
    for start_i, end_i in zip(
        start.values.nonzero()[0],
        end.values.nonzero()[0],
    ):
        axs.axvspan(start_i, end_i, color='orange', alpha=0.3, linewidth=0)






    '''
    num_cmls = len(ds.cml_id)
    n_samples = len(ds.sample_num)

    # setup figure
    fig, axs = plt.subplots(num_cmls, 1, sharex=True, figsize=(12,num_cmls*2))
    ax1 = axs[0].twiny()
    ax1.set_xlim(ds.time.values[0,0], ds.time.values[-1,-1])                    # change to [0,0,0] and [0,-1,-1] if excluding fault cmls
    fig.tight_layout(h_pad = 3)

    for n in range(num_cmls):    
        axs[n].set_xlim(0, n_samples)
        axs[n].set_ylim(0,1.2);    
        # plot cnn prediction
        ds.cnn_out[n].plot.line(x='sample_num', ax=axs[n], label = 'TL',color='black', lw=0.5);

        #cnn threshold
        axs[n].axhline(cnn_wd_threshold, color='black', linestyle='--', lw=0.5)

        # GREEN: plot true cnn predicted wet/dry areas
        # tip from stack oveflow: https://stackoverflow.com/questions/44632903/setting-multiple-axvspan-labels-as-one-element-in-legend
        start = np.roll(ds.true_wet[n], -1) & ~ds.true_wet[n]
        end = np.roll(ds.true_wet[n], 1) & ~ds.true_wet[n]
        for start_i, end_i in zip(
            start.values.nonzero()[0],
            end.values.nonzero()[0],
        ):
            axs[n].axvspan(ds.sample_num.values[start_i], ds.sample_num.values[end_i], color='g', alpha=0.5, linewidth=0, label='_'*start_i+'true wet') 
        
        # RED: plot false alarms
        start = np.roll(ds.false_alarm[n], -1) & ~ds.false_alarm[n]
        end = np.roll(ds.false_alarm[n], 1) & ~ds.false_alarm[n]
        for start_i, end_i in zip(
            start.values.nonzero()[0],
            end.values.nonzero()[0],
        ):
            axs[n].axvspan(ds.sample_num.values[start_i], ds.sample_num.values[end_i], color='r', alpha=0.5, linewidth=0, label='_'*start_i+'false alarm') 
            
        # ORANGE: plot missed wet 
        start = np.roll(ds.missed_wet[n], -1) & ~ds.missed_wet[n]
        end = np.roll(ds.missed_wet[n], 1) & ~ds.missed_wet[n]
        for start_i, end_i in zip(
            start.values.nonzero()[0],
            end.values.nonzero()[0],
        ):
            axs[n].axvspan(ds.sample_num.values[start_i], ds.sample_num.values[end_i], color='orange', alpha=0.5, linewidth=0, label='_'*start_i+'missed wet')

    plt.show()


'''


# # select one cml:
# cml_k = '16'
# my_cml = cml_set.sel(cml_id = cml_k)        # .sel is not int indexing, but selecting specific 'label' 
# my_ref = ref_set.sel(cml_id = cml_k)

# # shaded refernece wet periods from Pycomlink
# # set first and last value with zero for correct plotting
# my_ref['ref_wd'][0] = False
# #my_ref['ref_wd'][-1] = False

# # setup figure
# fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12,4))
# #ax1 = axs[0].twiny()
# #fig.tight_layout()

# # plot TRSL
# my_cml.trsl.plot.line(x='time', ax=axs[0], label = 'TL');
# # plot Rain rate 
# my_ref.rain.plot.line(x='time', ax=axs[1], label = 'TL');

# # plot real bool wet/dry with 5min precission
# wet_start = np.roll(my_ref.ref_wd, -1) & ~my_ref.ref_wd
# wet_end = np.roll(my_ref.ref_wd, 1) & ~my_ref.ref_wd
# for wet_start_i, wet_end_i in zip(
#     wet_start.values.nonzero()[0],
#     wet_end.values.nonzero()[0],
# ):
#     axs[1].axvspan(my_ref.time.values[wet_start_i], my_ref.ref_wd.time.values[wet_end_i], color='b', alpha=0.2, linewidth=0); # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axvspan.html
#     axs[0].axvspan(my_ref.time.values[wet_start_i], my_ref.ref_wd.time.values[wet_end_i], color='b', alpha=0.2, linewidth=0); # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axvspan.html

# # plot trsl gaps
# my_cml['trsl_gap'][0] = False
# my_cml['trsl_gap'][-1] = False

# gap_start = np.roll(my_cml.trsl_gap, -1) & ~my_cml.trsl_gap
# gap_end = np.roll(my_cml.trsl_gap, 1) & ~my_cml.trsl_gap
# for gap_start_i, gap_end_i in zip(
#     gap_start.values.nonzero()[0],
#     gap_end.values.nonzero()[0],
# ):
#     axs[0].axvspan(my_cml.time.values[gap_start_i], my_cml.time.values[gap_end_i], color='r', alpha=0.7, linewidth=0); # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axvspan.html
   

# # axes limits source: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xlim.html
# axs[1].set_xlim(my_cml.time.values[0], my_cml.time.values[-1])
# axs[0].set_xlabel('')
# axs[1].set_title("")

# fig.savefig('cml_aligned_ds.svg')