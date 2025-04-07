#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filename: cnn_utility.py
Author: Lukas Kaleta
Date: 2025-01-31
Version: 1.0
Description: 
    Function set for training and using CNN module
    for rain event classification using data from CML.

License: 
Contact: 211312@vutbr.cz
"""


""" Imports """
# Import python libraries

import math
import numpy as np
import itertools
import matplotlib.pyplot as plt

import pandas as pd

import torch
import torch.nn as nn
import sklearn.metrics as skl
from sklearn.utils import shuffle
from tqdm import tqdm
import datetime
#from IPython.display import clear_output

# Import external packages


# Import own modules
import telcosense_classification.module.cnn_telcorain_v21 as cnn_sing
import telcosense_classification.module.cnn_telcorain_v11 as cnn_cont

""" Variable definitions """


""" Function definitions """

## TODO: shuffle train and test data 

def cnn_train_period_classification(ds:pd.DataFrame, 
                                    num_channels = 2,
                                    sample_size = 100, 
                                    batchsize = 20, 
                                    epochs = 20, 
                                    resume_epoch = 0, 
                                    learning_rate = 0.01, 
                                    dropout_rate = 0.1,
                                    kernel_size = 3,
                                    n_conv_filters = 128,
                                    n_fc_neurons = 64,
                                    save_param = False
                                    ):
    """
    Train given cnn modul on given cml dataset over given number of epochs. 
    perform testing for each epoch, save training parameters. 
    Classification output is one WD probability value for whole sample
    
    Parameters
    ds : pandas.DataFrame containing CML data and reference rain data for training and testing
    num_channels : int, default = 2, number of variables for cnn to classify from (2*trsl, 2*temperature)
    samplesize : int, default = 100, number of values to be grouped in a sample
    batchsize : int, default = 20, number of samples per batch, to be feed into cnn at once
    epochs : int, default = 20, number of training epochs
    resume_epoch : int, default = 0, if training was performed previouslyover xy epochs,
        continue training at epoch xy+1 
    learning_rate : float, default = 0.01, cnn's optimizer learning rate
    dropout_rate : float, default = 0.1
    kernel_size : int, default = 3
    n_conv_filters : number of convolutional layer inputs and outputs
    n_fc_neurons : number of neurons in FC layer
    save_param: boolean, default = False, save training parameters after training
        
    Returns
    cnn_output: np.array containing 0-1 float classification probability output of CNN
    train_loss: trtainloss during last epoch
    test_loss: testloss during last epoch
    """
    
    n_samples = len(ds) // sample_size
    cutoff = len(ds) % sample_size

    if cutoff == 0:
        if num_channels == 2:
            trsl = np.concatenate((ds.trsl_A.values[:], ds.trsl_B.values[:])).reshape((-1, num_channels), order='F')
            trsl = trsl.reshape(n_samples, sample_size, num_channels).transpose(0,2,1)
        elif num_channels == 4:
            trsl = np.concatenate((ds.trsl_A.values[:], 
                                ds.trsl_B.values[:],
                                ds.temp_A.values[:],
                                ds.temp_B.values[:])
                                ).reshape((-1, num_channels), order='F')
            trsl = trsl.reshape(n_samples, sample_size, num_channels).transpose(0,2,1)
        ref = ds.ref_wd.values[:][::sample_size]   

    else:
        if num_channels == 2:
            trsl = np.concatenate((ds.trsl_A.values[:-cutoff], ds.trsl_B.values[:-cutoff])).reshape((-1, num_channels), order='F')
            trsl = trsl.reshape(n_samples, sample_size, num_channels).transpose(0,2,1)
        elif num_channels == 4:
            trsl = np.concatenate((ds.trsl_A.values[:-cutoff], 
                                ds.trsl_B.values[:-cutoff],
                                ds.temp_A.values[:-cutoff],
                                ds.temp_B.values[:-cutoff])
                                ).reshape((-1, num_channels), order='F')
            trsl = trsl.reshape(n_samples, sample_size, num_channels).transpose(0,2,1)
        ref = ds.ref_wd.values[:-cutoff][::sample_size]   



    k_train = 0.8     # fraction of training data
    train_size = int(len(trsl)*k_train/batchsize)*batchsize

    # Storing as tensors [2]
    train_data = torch.Tensor(trsl[:train_size])
    test_data = torch.Tensor(trsl[train_size:])
    train_ref = torch.Tensor(ref[:train_size])
    test_ref = torch.Tensor(ref[train_size:])

    # Turning into TensorDataset
    dataset = torch.utils.data.TensorDataset(train_data, train_ref)
    testset = torch.utils.data.TensorDataset(test_data, test_ref)

    trainloader = torch.utils.data.DataLoader(dataset, batchsize, shuffle = False)   #!!!!!!!!!!!! potential problem: batchsize
    testloader = torch.utils.data.DataLoader(testset, batchsize, shuffle = False)

    model = cnn_sing.cnn_class(channels=num_channels, 
                                sample_size=sample_size, 
                                kernel_size=kernel_size, 
                                dropout = dropout_rate, 
                                n_fc_neurons = n_fc_neurons,                    # 64
                                n_filters = n_conv_filters
                                )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # if resuming training
    if resume_epoch == 0:
        loss_dict = {}
        loss_dict['train'] = {}
        loss_dict['test'] = {}
        for key in ['train','test']:
            loss_dict[key]['loss'] = []

    # training loop
    cnn_prediction = []
    for epoch in range(resume_epoch, epochs):
        # training
        train_losses = []
        for inputs, targets in tqdm(trainloader):
            optimizer.zero_grad()
            pred = model(inputs)
            pred = nn.Flatten(0,1)(pred)
            # getting the output
            if epoch == epochs-1: cnn_prediction = cnn_prediction+pred.tolist()
            
            # calculating the loss function        
            if ~np.isnan(pred.tolist()).any():              # exclude NaN values for loss calculation
                loss = nn.BCELoss()(pred, targets)          # BCE = binary cross entropy
                loss.backward()
            optimizer.step()
            train_losses.append(loss.detach().numpy())
        loss_dict['train']['loss'].append(np.mean(train_losses))
    
        # testing
        test_losses = []
        with torch.no_grad():
            for inputs, targets in tqdm(testloader):
                pred = model(inputs)
                pred = nn.Flatten(0,1)(pred)
                # getting the output
                if epoch == epochs-1: cnn_prediction = cnn_prediction+pred.tolist()
                
                if np.isnan(pred.tolist()).any():            # exclude NaN values for loss calculation
                    targets = targets[~np.isnan(pred.tolist())]
                    pred = pred[~np.isnan(pred.tolist())]
                loss = nn.BCELoss()(pred, targets)
                if ~np.isnan(loss.tolist()).any():           # this case prevents appending nan loss to lossfunc array
                    test_losses.append(loss.detach().numpy())
            loss_dict['test']['loss'].append(np.mean(test_losses))
            
        # learning curve
        print(epoch)
        print('train loss:', np.mean(train_losses))
        print('test loss:', np.mean(test_losses))
        print('min test loss:', np.min(loss_dict['test']['loss']))
        
        plt.ion()
        if epoch == 0: fig, axs = plt.subplots(1,1, figsize=(4,4))
        axs.cla()
        for key in loss_dict.keys():
            for k, key2 in enumerate(loss_dict[key].keys()):
                axs.plot(loss_dict[key][key2], label=key)
                axs.set_title(key2)
        # axs.set_yscale('log')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.show()
        plt.pause(0.0001)
        fig.tight_layout(pad=1.0)
        resume_epoch = epoch
    # export training curve plot
    fig.savefig('results/'+str(learning_rate)+'/loss_curve'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.png')
    plt.close()
    # save cnn parameters
    if save_param:
        path = 'modul/trained_cnn_param/'
        date = datetime.datetime.now().strftime('%Y-%m-%d_%H;%M')
        torch.save(model.state_dict(), (path+date))

    cnn_out = np.array(cnn_prediction).reshape(-1)
    return cnn_out, np.mean(train_losses), np.mean(test_losses)



def cnn_train_timestep_classification(ds:pd.DataFrame, 
                                    num_channels = 2,
                                    sample_size = 100, 
                                    batchsize = 20, 
                                    epochs = 20, 
                                    resume_epoch = 0, 
                                    learning_rate = 0.01, 
                                    dropout_rate = 0.1,
                                    kernel_size = 3,
                                    n_conv_filters = 128,
                                    n_fc_neurons = 64,
                                    save_param = False
                                    ):
    """
    Train given cnn modul on given cml dataset over given number of epochs. 
    perform testing for each epoch, save training parameters.
    This approach assigns one WD probability classification value to every input timestep.
    
    Parameters
    ds : pandas.DataFrame containing CML data and reference rain data for training and testing
    num_channels : int, default = 2, number of variables for cnn to classify from (2*trsl, 2*temperature)
    samplesize : int, default = 100, number of values to be grouped in a sample
    batchsize : int, default = 20, number of samples per batch, to be feed into cnn at once
    epochs : int, default = 20, number of training epochs
    resume_epoch : int, default = 0, if training was performed previouslyover xy epochs,
        continue training at epoch xy+1 
    learning_rate : float, default = 0.01, cnn's optimizer learning rate
    dropout_rate : float, default = 0.1
    kernel_size : int, default = 3
    n_conv_filters : number of convolutional layer inputs and outputs
    n_fc_neurons : number of neurons in FC layer
    save_param: boolean, default = False, save training parameters after training
        
    Returns
    cnn_output: np.array containing 0-1 float classification probability output of CNN
    train_loss: trtainloss during last epoch
    test_loss: testloss during last epoch
    """
    
    n_samples = len(ds) // sample_size
    cutoff = len(ds) % sample_size

    if cutoff == 0:
        if num_channels == 2:
            trsl = np.concatenate((ds.trsl_A.values[:], ds.trsl_B.values[:])).reshape((-1, num_channels), order='F')
            trsl = trsl.reshape(n_samples, sample_size, num_channels).transpose(0,2,1)
        elif num_channels == 4:
            trsl = np.concatenate((ds.trsl_A.values[:], 
                                ds.trsl_B.values[:],
                                ds.temp_A.values[:],
                                ds.temp_B.values[:])
                                ).reshape((-1, num_channels), order='F')
            trsl = trsl.reshape(n_samples, sample_size, num_channels).transpose(0,2,1)
        ref = ds.ref_wd.values[:].reshape((n_samples,sample_size))
        
    else:
        if num_channels == 2:
            trsl = np.concatenate((ds.trsl_A.values[:-cutoff], ds.trsl_B.values[:-cutoff])).reshape((-1, num_channels), order='F')
            trsl = trsl.reshape(n_samples, sample_size, num_channels).transpose(0,2,1)
        elif num_channels == 4:
            trsl = np.concatenate((ds.trsl_A.values[:-cutoff], 
                                ds.trsl_B.values[:-cutoff],
                                ds.temp_A.values[:-cutoff],
                                ds.temp_B.values[:-cutoff])
                                ).reshape((-1, num_channels), order='F')
            trsl = trsl.reshape(n_samples, sample_size, num_channels).transpose(0,2,1)
        ref = ds.ref_wd.values[:-cutoff].reshape((n_samples,sample_size))
        

    k_train = 0.8     # fraction of training data
    train_size = int(len(trsl)*k_train/batchsize)*batchsize

    # Storing as tensors [2]
    train_data = torch.Tensor(trsl[:train_size])
    test_data = torch.Tensor(trsl[train_size:])
    train_ref = torch.Tensor(ref[:train_size])
    test_ref = torch.Tensor(ref[train_size:])

    # Turning into TensorDataset
    dataset = torch.utils.data.TensorDataset(train_data, train_ref)
    testset = torch.utils.data.TensorDataset(test_data, test_ref)

    trainloader = torch.utils.data.DataLoader(dataset, batchsize, shuffle = False)   #!!!!!!!!!!!! potential problem: batchsize
    testloader = torch.utils.data.DataLoader(testset, batchsize, shuffle = False)

    model = cnn_cont.cnn_class(channels=num_channels, 
                                sample_size=sample_size, 
                                kernel_size=kernel_size, 
                                dropout = dropout_rate, 
                                n_fc_neurons = n_fc_neurons,                    # 64
                                n_filters = n_conv_filters
                                )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, amsgrad=True)           #AdamW
    '''
    optim = lambda *ar, **kw: torch.optim.AdamW(*ar, **kw, amsgrad=True)
    
    optim = lambda *ar, **kw: torch.optim.SGD(*ar, **kw, nesterov=True, momentum=MAGIC_NUM)
    '''


    # if resuming training
    if resume_epoch == 0:
        loss_dict = {}
        loss_dict['train'] = {}
        loss_dict['test'] = {}
        for key in ['train','test']:
            loss_dict[key]['loss'] = []

    # training loop
    cnn_prediction = []
    for epoch in range(resume_epoch, epochs):
        # training
        train_losses = []
        for inputs, targets in tqdm(trainloader):
            optimizer.zero_grad()
            pred = model(inputs)
            
            # getting the output
            if epoch == epochs-1: cnn_prediction = cnn_prediction+pred.tolist()
            
            # calculating the loss function        
            if ~np.isnan(pred.tolist()).any():              # exclude NaN values for loss calculation
                loss = nn.BCELoss()(pred, targets)          # BCE = binary cross entropy
                loss.backward()
            optimizer.step()
            train_losses.append(loss.detach().numpy())
        loss_dict['train']['loss'].append(np.mean(train_losses))
    
        # testing
        test_losses = []
        with torch.no_grad():
            for inputs, targets in tqdm(testloader):
                pred = model(inputs)
                
                # getting the output
                if epoch == epochs-1: cnn_prediction = cnn_prediction+pred.tolist()
                
                if np.isnan(pred.tolist()).any():            # exclude NaN values for loss calculation
                    targets = targets[~np.isnan(pred.tolist())]
                    pred = pred[~np.isnan(pred.tolist())]
                loss = nn.BCELoss()(pred, targets)
                if ~np.isnan(loss.tolist()).any():           # this case prevents appending nan loss to lossfunc array
                    test_losses.append(loss.detach().numpy())
            loss_dict['test']['loss'].append(np.mean(test_losses))
            
        # learning curve
        print(epoch)
        print('train loss:', np.mean(train_losses))
        print('test loss:', np.mean(test_losses))
        print('min test loss:', np.min(loss_dict['test']['loss']))
        
        plt.ion()
        if epoch == 0: fig, axs = plt.subplots(1,1, figsize=(4,4))
        axs.cla()
        for key in loss_dict.keys():
            for k, key2 in enumerate(loss_dict[key].keys()):
                axs.plot(loss_dict[key][key2], label=key)
                axs.set_title(key2)
        # axs.set_yscale('log')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.show()
        plt.pause(0.0001)
        fig.tight_layout(pad=1.0)
        resume_epoch = epoch
    # export training curve plot
    #fig.savefig('loss_curve'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.svg')
    plt.close()
    # save cnn parameters
    if save_param:
        path = 'modul/trained_cnn_param/'
        date = datetime.datetime.now().strftime('%Y-%m-%d_%H;%M')
        torch.save(model.state_dict(), (path+date))

    cnn_out = np.append(np.array(cnn_prediction).reshape(-1), np.zeros(cutoff))
    return cnn_out, np.mean(train_losses), np.mean(test_losses)





























'''

def cnn_classify(ds:xr.Dataset, sample_size:int, batchsize = 20, param_dir = 'default'):
    """
    Classify rainy periods from trsl data of CML, using trained cnn modul
    
    Parameters
    ds : xarray.dataset containing CML data and optionally reference rain data for validation
    sample_size : int, length of samples for wet/dry classification in minutes
    batchsize : int, default = 20, number of samples in the batch, given to cnn
    param_dir: str, default = 'default', 
        
    Returns
    total_loss: float, total cnn prediction loss (w/d reference - cnn output)
    """

    n_samples = len(ds.sample_num)
    num_cmls = len(ds.cml_id)

    trsl = ds.trsl_st.values.reshape(num_cmls*n_samples,2 , sample_size)
    ref = ds.ref_wd.values.reshape(num_cmls*n_samples)

    # Storing as tensors [2]
    trsl_data = torch.Tensor(trsl)
    ref_data = torch.Tensor(ref)

    # Turning into TensorDataset
    dataset = torch.utils.data.TensorDataset(trsl_data, ref_data)

    validloader = torch.utils.data.DataLoader(dataset, batch_size = batchsize, shuffle = False)
    

    # loading the model parameters:
    path = 'modul/trained_cnn_param/'
    model = cnn.cnn_class()
    model.load_state_dict(torch.load(path+param_dir))

    cnn_output = []
    valid_losses = []
    with torch.no_grad():
        for inputs, targets in tqdm(validloader):
            pred = model(inputs)
            pred = nn.Flatten(0,1)(pred)
            cnn_output = cnn_output + pred.tolist()

            if np.isnan(pred.tolist()).any():            # exclude NaN values for loss calculation
                targets = targets[~np.isnan(pred.tolist())]
                pred = pred[~np.isnan(pred.tolist())]
            loss = nn.BCELoss()(pred, targets)
            if ~np.isnan(loss.tolist()).any():           # this case prevents appending nan loss to lossfunc array
                valid_losses.append(loss.detach().numpy())
        total_loss = np.mean(valid_losses)
        
    print(total_loss)
    return cnn_output
'''