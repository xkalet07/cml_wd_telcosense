#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filename: data_loading_utility.py
Author: Lukas Kaleta
Date: 2025-05-26
Version: 2.0t
Description: 
    This script contains function set for loading cml data for rain event detection. 
    designed for CML data from czech republic and reference CHMI.

License: 
Contact: 211312@vutbr.cz
"""

""" Notes """
# TODO: load cml B using its IP, not i+1
# TODO: load metadata

""" Imports """
# Import python libraries
import pandas as pd
import os

# Import external packages

# Import local modules

""" Variable definitions """


""" Function definitions """

def find_missing_column(parameter_name:str, path:str):
    """
    Search merged cml-rainGauge files in given directory for specific column name.
    Returns list of filenames, which are missing given parameter column.
    Typical names: 'time', 'SRA10M', 'cml_PrijimanaUroven', 'cml_Uptime', 'cml_Teplota'

    Parameters:
    parameter_name : str, specific parameter name to be searched
    path : str, directory to search

    Returns:
    output_list : list, list of files missing given parameter (column) name
    """

    # Get the list of all files
    file_list = os.listdir(path)
    
    output_list = []

    for file in file_list:
        with open(path+file, 'r') as fp:
            s = fp.read()
            if parameter_name not in s:
                output_list.append(file)
            fp.close()

    return output_list



def load_cml(dir:str, technology:str, i = int):
    """
    Search directory and load one merged cml-rainGauge file. 
    Returns timestamp, trsl and rainfall data of one cml

    Parameters:
    dir : str, directory containing cml folders
    technology : str, folder containing cmls of one technology
    i : int, index of cml in the filelist

    Returns:
    cml : pandas.DataFrame, containing columns: 
        'time', 'rain', 'trsl_A', 'trsl_B', 'uptime_A', 'uptime_B', 'temp_A', 'temp_B'
    """
    path = dir + technology+'/'
    file_list = sorted(os.listdir(path))
    
    if (technology=='summit') | (technology=='summit_bt'):
        cml = pd.read_csv(path+file_list[i], usecols=['time','SRA10M','cml_PrijimanaUroven','cml_Teplota','cml_Uptime'])   #,,'cml_KvalitaSignalu','cml_Teplota',cml_RxDatovyTok,cml_Uptime
        cml = cml.rename(columns={'SRA10M':'rain', 
                                'cml_PrijimanaUroven':'trsl_A',
                                'cml_Teplota':'temp_A',
                                'cml_Uptime':'uptime_A'})
        cml[['trsl_B','temp_B','uptime_B']] = pd.read_csv(path+file_list[i+1], usecols=['cml_PrijimanaUroven','cml_Teplota','cml_Uptime'])
    elif (technology=='ceragon_ip_10'):
        cml = pd.read_csv(path+file_list[i], usecols=['time','SRA10M','cml_PrijimanaUroven','cml_VysilaciVykon','cml_Uptime']) 
        cml = cml.rename(columns={'SRA10M':'rain', 
                                'cml_PrijimanaUroven':'rsl_A',
                                'cml_VysilaciVykon':'tsl_A',
                                'cml_Uptime':'uptime_A'})
        cml[['rsl_B','tsl_B','uptime_B']] = pd.read_csv(path+file_list[i+1], usecols=['cml_PrijimanaUroven','cml_VysilaciVykon','cml_Uptime'])
        cml['trsl_A'] = cml.tsl_A - cml.rsl_A
        cml['trsl_B'] = cml.tsl_B - cml.rsl_B
    elif (technology=='ceragon_ip_20'):
        cml = pd.read_csv(path+file_list[i], usecols=['time','SRA10M','cml_Signal','cml_Teplota','cml_Uptime']) 
        cml = cml.rename(columns={'SRA10M':'rain', 
                                'cml_Signal':'trsl_A',
                                'cml_Teplota':'temp_A',
                                'cml_Uptime':'uptime_A'})
        cml[['trsl_B','temp_B','uptime_B']] = pd.read_csv(path+file_list[i+1], usecols=['cml_Signal','cml_Teplota','cml_Uptime'])
        cml['trsl_A'] = -cml.trsl_A
        cml['trsl_B'] = -cml.trsl_B
    elif (technology=='1s10'):
        cml = pd.read_csv(path+file_list[i], usecols=['time','SRA10M','cml_PrijimanaUroven','cml_Teplota','cml_Uptime']) 
        cml = cml.rename(columns={'SRA10M':'rain', 
                                'cml_PrijimanaUroven':'trsl_A',
                                'cml_Teplota':'temp_A',
                                'cml_Uptime':'uptime_A'})
        cml[['trsl_B','temp_B','uptime_B']] = pd.read_csv(path+file_list[i+1], usecols=['cml_PrijimanaUroven','cml_Teplota','cml_Uptime'])
        cml['trsl_A'] = -cml.trsl_A
        cml['trsl_B'] = -cml.trsl_B
    else:
        print('technology: ' + technology + ' is not defined. \n ' + 
            'Available technologies are: summit, summit_bt, 1s10, ceragon_ip_10, ceragon_ip_20. \n' +
            'Please choose different technology or check for misspelling.')
        return None
    
    return cml