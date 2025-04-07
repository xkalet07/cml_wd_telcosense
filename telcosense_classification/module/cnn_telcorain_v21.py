#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filename: cnn_telcorain.py
Author: Lukas Kaleta
Date: 2025-04-03
Version: 2.1
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

# https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/models.py
def init_layer(layer):
    """ Initialize a Convolutional layer """
    nn.init.xavier_uniform_(layer.weight)

    
def init_bn(bn):
    """ Initialize a Batchnorm layer """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)



class ConvBlock(nn.Module):
    def __init__(self, kernel_size, channels_in, channels_out):
        super().__init__()
        self.kernelsize = kernel_size
        self.channels_in = channels_in
        self.channels_out = channels_out
        
        self.conv1 = nn.Conv1d(channels_in, channels_out, self.kernelsize, padding='same') 	# padding same padds the input data to match the output dimension 
        self.conv2 = nn.Conv1d(channels_out, channels_out, self.kernelsize, padding='same')
        
        # https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/models.py
        self.bn1 = nn.BatchNorm1d(channels_out)         # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
        self.bn2 = nn.BatchNorm1d(channels_out)

        self.init_weight()
        
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=2):
        x = input
        
        x = self.conv1(x)
        x = self.bn1(x)     
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)            
        x = self.act2(x)

        x = F.max_pool1d(x, kernel_size=pool_size)

        return x

class cnn_class(nn.Module):
    def __init__(self, channels=2, sample_size= 60, kernel_size = 3, dropout = 0.2, n_fc_neurons = 64, n_filters = [24, 48, 96, 192],):
        super().__init__()
        self.channels = channels
        self.kernelsize = kernel_size
        self.dropout = dropout
        self.n_fc_neurons = n_fc_neurons
        self.n_filters = n_filters

        ### Convolutional part
        # ConvBlock: def __init__(self, kernel_size, dim_in, dim, dim_out):
        self.cb1 = ConvBlock(self.kernelsize, self.channels, self.n_filters[0])
        self.cb2 = ConvBlock(self.kernelsize, self.n_filters[0], self.n_filters[1])
        self.cb3 = ConvBlock(self.kernelsize, self.n_filters[1], self.n_filters[2])
        self.cb4 = ConvBlock(self.kernelsize, self.n_filters[2], self.n_filters[3])
        
        ### Fully Connected part 
        self.act = nn.ReLU()
        self.dense1 = nn.Linear(n_filters[3],n_fc_neurons)
        self.drop1 = nn.Dropout(p=dropout)
        self.dense2 = nn.Linear(n_fc_neurons, n_fc_neurons)
        self.drop2 = nn.Dropout(dropout)
        self.denseOut = nn.Linear(n_fc_neurons, 1)                     # single value on the output
        self.final_act = nn.Sigmoid()                                  # Sigmoid function to add nonlinearity for output classification as 1/0

    
    def forward(self, x):
        x = self.cb1(x, pool_size=2)
        x = self.cb2(x, pool_size=2)
        x = self.cb3(x, pool_size=2)
        x = self.cb4(x, pool_size=2)
        x = torch.mean(x,dim=-1)
        
        ### FC part
        x = self.act(self.dense1(x))
        x = self.drop1(x)
        x = self.act(self.dense2(x))
        x = self.drop2(x)
        x = self.final_act(self.denseOut(x))

        return x
