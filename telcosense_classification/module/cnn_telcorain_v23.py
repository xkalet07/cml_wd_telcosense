#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filename: cnn_telcorain.py
Author: Lukas Kaleta
Date: 2025-05-12
Version: 2.3
Description: 
    CNN architecture for the purpose of Telcorain CML precipitation detection.
    Model designed for signle output WD value classification for sample period.
    Included constant parameters.
    Inspired by: https://github.com/jpolz/cml_wd_pytorch/tree/main

License: 
Contact: 211312@vutbr.cz
"""

""" Imports """
# Import python libraries
import torch
import torch.nn as nn
import torch.nn.functional as F


""" Function definitions """

# https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/models.py
def init_layer(layer):
    # Initialize a Convolutional layer
    nn.init.xavier_uniform_(layer.weight)
    
def init_bn(bn):
    # Initialize a Batchnorm layer
    bn.bias.data.zero_()
    bn.weight.data.fill_(1.)  

def output_size(sample_size:int, single_output=True):
    # If True, make CNN output 1 value for whole sample
    # otherwise match output size to sample size
    if single_output:
        return 1
    else:
        return sample_size
    

""" CNN module architecture """

class ConvBlock(nn.Module):
    def __init__(self, kernel_size, channels_in, channels_out, dropout):
        super().__init__()
        self.kernelsize = kernel_size
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.dropout = dropout
        
        self.conv1 = nn.Conv1d(self.channels_in, self.channels_out, self.kernelsize, padding='same') 	# padding same padds the input data to match the output dimension 
        self.conv2 = nn.Conv1d(self.channels_out, self.channels_out, self.kernelsize, padding='same')
        
        self.drop1 = nn.Dropout(p=self.dropout)
        self.drop2 = nn.Dropout(p=self.dropout)

        self.init_weight()
        
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)

    def forward(self, input, pool_size=2):
        x = input
        x = self.conv1(x)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.act2(x)
        
        #x = F.max_pool1d(x, kernel_size=pool_size)

        return x

class cnn_class(nn.Module):
    def __init__(self, channels=2, const_parameters=11, sample_size=60, kernel_size=3, dropout=0.001, n_fc_neurons=64, n_filters=[24, 48, 96, 192], single_output=True):
        super().__init__()
        self.channels = channels
        self.const_parameters = const_parameters
        self.kernelsize = kernel_size
        self.dropout = dropout
        self.n_fc_neurons = n_fc_neurons
        self.n_filters = n_filters
        self.output = output_size(sample_size,single_output)

        ### Convolutional part
        self.cb1 = ConvBlock(self.kernelsize, self.channels, self.n_filters[0], self.dropout)
        self.cb2 = ConvBlock(self.kernelsize, self.n_filters[0], self.n_filters[1], self.dropout)
        self.cb3 = ConvBlock(self.kernelsize, self.n_filters[1], self.n_filters[2], self.dropout)
        self.cb4 = ConvBlock(self.kernelsize, self.n_filters[2], self.n_filters[3], self.dropout)
        self.bnc1 = nn.BatchNorm1d(self.n_filters[0], eps=0, momentum=0.1, affine=True, track_running_stats=True)#,track_running_stats=False) # !!!!!!!!!!!!!!!!!!!!!!!!!!
        init_bn(self.bnc1)
   

        ### Fully Connected part 
        self.act = nn.ReLU()
        self.dense1 = nn.Linear(self.n_filters[3]+self.const_parameters, self.n_fc_neurons)
        self.drop1 = nn.Dropout(self.dropout)
        self.dense2 = nn.Linear(self.n_fc_neurons, self.n_fc_neurons)
        self.drop2 = nn.Dropout(self.dropout)
        self.denseOut = nn.Linear(self.n_fc_neurons, self.output)                     # single value on the output
        self.final_act = nn.Sigmoid()                                  # Sigmoid function to add nonlinearity for output classification as 1/0
        self.init_fc()

    def init_fc(self):
        init_layer(self.dense1)
        init_layer(self.dense2)
        init_layer(self.denseOut)
    
    def forward(self, x_trsl, x_const):
        ### Conv part 
        x = self.cb1(x_trsl, pool_size=1)
        x = self.bnc1(x) 
        x = self.cb2(x, pool_size=1)
        x = self.cb3(x, pool_size=1)
        x = self.cb4(x, pool_size=1)
        x = torch.mean(x,dim=-1)
        
        ### FC part
        y = torch.cat((x, x_const),dim=1)         # concatenate inputs
        y = self.act(self.dense1(y))
        y = self.act(self.dense2(y))
        y = self.final_act(self.denseOut(y))    
        return y
