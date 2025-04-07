#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filename: cnn_telcorain.py
Author: Lukas Kaleta
Date: 2025-03-31
Version: 2.0
Description: 
    CNN architecture for the purpose of Telcorain CML precipitation detection.
    Model designed for signle output WD value classification for sample period.
    Inspired by: https://github.com/jpolz/cml_wd_pytorch/tree/main

License: 
Contact: 211312@vutbr.cz
"""

""" Imports """
# Import python libraries
import torch
import torch.nn as nn
import torch.nn.functional as F


""" CNN module architecture definition """

class ConvBlock(nn.Module):
    def __init__(self, kernel_size, dim_in, dim, dim_out):
        super().__init__()
        self.kernelsize = kernel_size
        self.dim = dim
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.proj1 = nn.Conv1d(dim_in, dim, self.kernelsize, padding='same') 	# padding same padds the input data to match the output dimension 
        nn.init.xavier_uniform_(self.proj1.weight)                              # initializing weights
        self.proj2 = nn.Conv1d(dim, dim_out, self.kernelsize, padding='same')
        nn.init.xavier_uniform_(self.proj1.weight)                              # initializing weights
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

    def forward(self, x,):
        x = self.proj1(x)            
        x = self.act1(x)
        x = self.proj2(x)            
        x = self.act2(x)
        return x

class cnn_class(nn.Module):
    def __init__(self, channels=2, sample_size= 60, kernel_size = 3, dropout = 0.2, n_fc_neurons = 64, n_filters = [24, 48, 96, 192],):
        super().__init__()
        self.channels = channels
        self.kernelsize = kernel_size
        self.dropout = dropout
        self.n_fc_neurons = n_fc_neurons
        self.n_filters = n_filters

        # ConvBlock: def __init__(self, kernel_size, dim_in, dim, dim_out):
        self.cb1 = ConvBlock(self.kernelsize, self.channels, self.n_filters[0], self.n_filters[0])      # 2 input channels, 24 filters of size 3
        self.cb2 = ConvBlock(self.kernelsize, self.n_filters[0], self.n_filters[1], self.n_filters[1])  # 24 input filters, 48 output filters
        self.cb3 = ConvBlock(self.kernelsize, self.n_filters[1], self.n_filters[2], self.n_filters[2])  # 48 in, 48 out
        self.cb4 = ConvBlock(self.kernelsize, self.n_filters[2], self.n_filters[3], self.n_filters[3])
        ### Fully Connected part 
        # no pooling implemented: not need
        self.act = nn.ReLU()                                                                            # Activation function: ReLU, after each convolution
        self.dense1 = nn.Linear(n_filters[3],n_fc_neurons)
        self.drop1 = nn.Dropout(p=dropout)
        self.dense2 = nn.Linear(n_fc_neurons, n_fc_neurons)
        self.drop2 = nn.Dropout(dropout)
        self.denseOut = nn.Linear(n_fc_neurons, 1)                     # single value on the output
        self.final_act = nn.Sigmoid()                                  # Sigmoid function to add nonlinearity for output classification as 1/0

    
    def forward(self, x):
        x = self.cb1(x)
        x = self.cb2(x)
        x = self.cb3(x)
        x = self.cb4(x)
        x = torch.mean(x,dim=-1)
        
        ### FC part
        x = self.act(self.dense1(x))
        x = self.drop1(x)
        x = self.act(self.dense2(x))
        x = self.drop2(x)
        x = self.final_act(self.denseOut(x))

        return x
