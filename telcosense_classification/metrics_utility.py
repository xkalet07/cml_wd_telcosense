#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filename: metrics_utility.py
Author: Lukas Kaleta
Date: 2025-05-06
Version: 2.0t
Description: 
    This script contains function set to quantize performance of CNN model
    for WD classification using CML data
    source: https://github.com/jpolz/cnn_cml_wet-dry_example
License: 
Contact: 211312@vutbr.cz
"""


""" Imports """
# Import python libraries

import numpy as np

import itertools
import matplotlib.pyplot as plt


# Import external packages
# Import local modules

""" Variable definitions """


""" Function definitions """

def calculate_roc_curve(cnn_pred:np.array, ref_wd:np.array, tr_start:float, tr_end:float):
    """
    Calculate ROC curve of the given CNN output,
    ROC curve shows a tradeoff between TPR and FPR when adjusting the 1/0 comparison threshold

    Source: https://github.com/jpolz/cnn_cml_wet-dry_example/blob/master/CNN_for_CML_example_nb.ipynb
    
    Parameters
    cnn_pred : np.array, CNN prediction output of floats in range 0-1
    ref_wd : np.arra, Boolean WD reference
    tr_start : float, minimal threshold value
    tr_end . float, maximal threshold value

    Returns
    roc : np.array, Clalculated ROC curve
    """
    roc = []
    for i in range(tr_start*1000,1+tr_end*1000,1):
        t = i/1000
        y_predicted=np.ravel(cnn_pred>t)  
        true_pos = np.sum(np.logical_and(ref_wd==1, y_predicted==1))
        true_neg = np.sum(np.logical_and(ref_wd==0, y_predicted==0))
        false_pos = np.sum(np.logical_and(ref_wd==0, y_predicted==1))
        false_neg = np.sum(np.logical_and(ref_wd==1, y_predicted==0))
        cond_neg = true_neg+false_pos
        cond_pos = true_pos+false_neg
        roc.append([true_pos/cond_pos,
                    false_pos/cond_neg])
    roc.append([0,0])
    
    return np.array(roc)



def calculate_roc_surface(roc:np.array):
    """
    Calculate surface under the ROC curve of the given CNN output,
    ROC surface is in range 0-1 and is a classification CNN performance quantizer
    
    Source: https://github.com/jpolz/cnn_cml_wet-dry_example/blob/master/CNN_for_CML_example_nb.ipynb
    
    Parameters
    roc : np.array, Clalculated ROC curve

    Returns
    area : float, Clalculated area under the ROC curve
    """
    k = len(roc)
    area=0
    for i in range(k-1):
        area= area+(np.abs(roc[i,1]-roc[i+1,1]))*0.5*(roc[i+1,0]+roc[i,0])
    
    return area


def plot_roc_curve(roc_dict: dict):
    """
    Plot one or multiple calculated ROC curves of given models outputs,
    ROC curve shows a tradeoff between TPR and FPR when adjusting the 1/0 comparison threshold

    Parameters
        roc_dict: dict of {'method_name': {'roc': np.array, 'threshold': float}}
        example:        
        roc_dict = {
            'modelA': {
                'roc': np.array,  # shape (N, 2)
                'threshold': float(0.5)
            },
            'modelB': {
                'roc': np.array,  # shape (N, 2)
                'threshold': float(0.7)
            }
        }
    """   
    plt.figure(figsize=(7,7))
    for name, data in roc_dict.items():
        roc = data['roc']
        thr = data['threshold']
        plt.plot(roc[:,1], roc[:,0], label=f'{name}, $\\tau$={thr}, AUC: {np.round(calculate_roc_surface(roc), 3)}', zorder=2, lw=2)
        idx = int(thr * 1000)
        if idx < len(roc):
            plt.scatter(roc[idx,1], roc[idx,0], color='black', marker='o', zorder=3)
    
    plt.plot([0,0,1,0,1,1],[0,1,1,0,0,1], 'k-', linewidth=0.3, zorder=1)
    plt.title('TPR = f(TNR)')
    plt.xlabel('False positive rate [-]')
    plt.ylabel('True positive rate [-]')
    plt.legend(loc='lower right', frameon=False)
    plt.grid()
    plt.yticks(np.arange(0, 1.01, 0.1))
    plt.xticks(np.arange(0, 1.01, 0.1))
    plt.show()


def plot_confusion_matrix(cm:np.array):
    """
    Plot calculated ¨Confusion Matrix showing TPR, FNR, FPR and FNR of the given CNN output,

    Source: https://github.com/jpolz/cnn_cml_wet-dry_example/blob/master/CNN_for_CML_example_nb.ipynb
    
    Parameters
    cm : np.array, Clalculated confusion matrix, shape: [[TP,FN],[FP,TN]]
    """
    labels = ['wet', 'dry']

    fig, ax1 = plt.subplots(figsize=(3,3), sharex=True)
    #ax1 = fig.add_subplot(131)

    cax = ax1.matshow(cm, cmap=plt.cm.Blues)
    ax1.set_xticklabels([''] + labels)
    ax1.set_yticklabels([''] + labels)
    fmt = '.3f'                             # value format, set to 3 decimal places float
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment='center',
                color='white' if cm[i, j] > thresh else 'black'
    )
        
    plt.xlabel('Predicted')
    plt.ylabel('True')
    ax1.xaxis.set_label_position('top') 
    plt.tight_layout()
    plt.title('CNN', pad=50)
    plt.show()

