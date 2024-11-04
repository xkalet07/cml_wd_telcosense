# Rain activity detection using microwave link data with convolutional neural network

Repository for Masters thesis 2024/2025
Lukáš Kaleta  

## attempt_01
__goal:__ train existing CNN(1) on open sense cml and reference RADOLAN data from Germany(2).  

__status:__ resolving issue: dataset dimensions are not matching cnn's. Need to check: correct number of channels is 2, getting 10 (which is a time dim. batch size).  

__sources:__  
(1) CML wet/dry using Pytorch: https://github.com/jpolz/cml_wd_pytorch/blob/main/wd_pytorch/train_cnn.ipynb  
(2) Pycomlink: https://github.com/pycomlink/pycomlink  
