#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filename: plot_utility.py
Author: Lukas Kaleta
Date: 2025-05-26
Version: 2.0t
Description: 
    This script contains function set for using matplotlib to plot 
    input/output/reference of CNN w/d classification using data from CML.

License: 
Contact: 211312@vutbr.cz
"""

""" Imports """
# Import python libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import external packages

# Import local modules


""" Variable definitions """


""" Function definitions """

def plot_cml(cml:pd.DataFrame, columns = ['rain', 'ref_wd', 'trsl']):
    """
    plot selected columns from cml dataframe in subplot

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
    cml['cnn_out'].plot(ax=axs, color='black', linewidth=0.7, )
    axs.axhline(cnn_wd_threshold, color='black', linestyle='--', lw=0.7)
    axs.set_ylabel('RSD [-]')
    axs.set_xlabel('sample index [-]')
    axs.legend(['RSD', '$\\tau_{RSD}$ ='+str(cnn_wd_threshold)],loc='upper right')

    axs.set_xlim(0,15839)

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


