# source: https://github.com/jpolz/cml_wd_pytorch/tree/main

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, kernel_size, dim_in, dim, dim_out):
        super().__init__()
        self.kernelsize = kernel_size
        self.dim = dim
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.proj1 = nn.Conv1d(dim_in, dim, self.kernelsize, padding='same')
        self.proj2 = nn.Conv1d(dim, dim_out, self.kernelsize, padding='same')
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

    def forward(self, x,):
        x = self.proj1(x)            
        x = self.act1(x)
        x = self.proj2(x)            
        x = self.act2(x)
        return x

class cnn_class(nn.Module):
    def __init__(self, kernel_size = 3, dropout = 0.4, n_fc_neurons = 64, n_filters = [24, 48, 48, 96, 192],):
        super().__init__()
        self.channels = 2                      # 2 input cml channels
        self.kernelsize = kernel_size
        self.dropout = dropout
        self.n_fc_neurons = n_fc_neurons
        self.n_filters = n_filters

        # ConvBlock: def __init__(self, kernel_size, dim_in, dim, dim_out):
        self.cb1 = ConvBlock(self.kernelsize, self.channels, self.n_filters[0], self.n_filters[0])      # 2 input channels, 24 filters of size 3
        self.cb2 = ConvBlock(self.kernelsize, self.n_filters[0], self.n_filters[1], self.n_filters[1])  # 24 input filters, 48 output filters
        self.cb3 = ConvBlock(self.kernelsize, self.n_filters[1], self.n_filters[2], self.n_filters[2])  # 48 in, 48 out
        self.conv5a = nn.Conv1d(n_filters[2],n_filters[4],kernel_size,padding='same')
        self.conv5b = nn.Conv1d(n_filters[4],n_filters[4],kernel_size,padding='same')
        self.act = nn.ReLU()                                                                            # Activation function: ReLU, after each convolution

        ### Fully Connected part 
        # no pooling implemented: not needed
        # neuron layers interlieved with dropout functions
        self.dense1 = nn.Linear(192,n_fc_neurons)
        self.drop1 = nn.Dropout(p=dropout)
        self.dense2 = nn.Linear(n_fc_neurons, n_fc_neurons)
        self.drop2 = nn.Dropout(dropout)
        self.denseOut = nn.Linear(n_fc_neurons, 1)                     # single value on the output
        self.final_act = nn.Sigmoid()                                  # Sigmoid function to clasiify output val as 1/0

    
    def forward(self, x):
        x = self.cb1(x)
        x = self.cb2(x)
        x = self.cb3(x)
        
        x = self.act(self.conv5a(x))
        x = self.act(self.conv5b(x))
        x = torch.mean(x,dim=-1)
        
        ### FC part
        x = self.act(self.dense1(x))
        x = self.drop1(x)
        x = self.act(self.dense2(x))
        x = self.drop2(x)
        x = self.final_act(self.denseOut(x))

        return x
