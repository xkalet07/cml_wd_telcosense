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

## TODO: make 1 CNN architecture with choice of one output value for whole sample
#           or output size = sample size. input parameter
# TODO: patch sample size output testloss is NaN

def cnn_train_period(ds:pd.DataFrame, 
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
                    single_output = True,
                    shuffle = False,
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
    n_conv_filters : int, number of convolutional layer inputs and outputs
    n_fc_neurons : int, number of neurons in FC layer
    single_output : bool, classification output of the CNN is single value if True, otherwise matches sample_size
    shuffle : bool, default = False, enable torch dataLoader to perform shuffle between epochs if True.
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
        if single_output:
            ref = ds.ref_wd.values[:][::sample_size]   
        else:
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
        if single_output:
            ref = ds.ref_wd.values[:-cutoff][::sample_size]   
        else:
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

    trainloader = torch.utils.data.DataLoader(dataset, batchsize, shuffle)
    testloader = torch.utils.data.DataLoader(testset, batchsize, shuffle)

    # CNN model
    model = cnn_sing.cnn_class(channels=num_channels, 
                                sample_size=sample_size, 
                                kernel_size=kernel_size, 
                                dropout = dropout_rate, 
                                n_fc_neurons = n_fc_neurons,
                                n_filters = n_conv_filters,
                                single_output=single_output
                                )
    
    # used optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # weight decay: chatgpt, +1 % TP, lower testloss and FP
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)         # each 10 epoch multiply lr by 0.5
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

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
            # flatten prediction only for single value output
            if single_output: pred = nn.Flatten(0,1)(pred)
            # getting the output
            if epoch == epochs-1: cnn_prediction = cnn_prediction+pred.tolist()
            
            # calculating the loss function        
            if ~np.isnan(pred.tolist()).any():              # exclude NaN values for loss calculation
                loss = nn.BCELoss()(pred, targets)          # BCE = binary cross entropy
                loss.backward()

            optimizer.step()
            train_losses.append(loss.detach().numpy())
        scheduler.step()
        loss_dict['train']['loss'].append(np.mean(train_losses))
    
        # testing
        test_losses = []
        with torch.no_grad():
            for inputs, targets in tqdm(testloader):
                pred = model(inputs)
                # flatten prediction only for single value output
                if single_output: pred = nn.Flatten(0,1)(pred)
                # getting the output
                if epoch == epochs-1: cnn_prediction = cnn_prediction+pred.tolist()
                
                if np.isnan(pred.tolist()).any():            # exclude NaN values for loss calculation
                    targets = targets[~np.isnan(pred.tolist())]
                    pred = pred[~np.isnan(pred.tolist())]
                loss = nn.BCELoss()(pred, targets)

                # early stopping implementation: https://chatgpt.com/share/67fb88df-05f4-800a-b4ed-576cff8743bd
                # cant be used on validation data, as long as testloss doesnt decrease
                #early_stopping(loss, model)             
                #if early_stopping.early_stop:
                #    print("Early stopping triggered.")
                #    break

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
    fig.savefig('results/loss_curve_'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.png')
    plt.close()
    # save cnn parameters
    if save_param:
        path = 'modul/trained_cnn_param/'
        date = datetime.datetime.now().strftime('%Y-%m-%d_%H;%M')
        torch.save(model.state_dict(), (path+date))

    cnn_out = np.array(cnn_prediction).reshape(-1)
    return cnn_out, np.mean(train_losses), np.mean(test_losses)



class EarlyStopping:
    # Early stopping on test-loss (validation-loss) implementation
    # from: https://chatgpt.com/share/67fb88df-05f4-800a-b4ed-576cff8743bd
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = None
        self.best_model_state = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_model_state = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_model_state is not None:
                    model.load_state_dict(self.best_model_state)



