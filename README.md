# Rain activity detection using microwave link data with convolutional neural network

Repository for Masters thesis 2024/2025
Lukáš Kaleta  

## attempt_01
__goal:__ train existing CNN(1) on open sense cml and reference RADOLAN data from Germany(2).  

__status:__  
Working: importing external data in netCDF, choosing one cml, converting to torch Tensor.   
Resolving issue: dataset dimensions are not matching cnn's.  
existing implementation acording to (1) uses 1 wet/dry flag for 180 time stamps (one of 100 samples). Why? My implementation aims to use one flag for one trsl measurement. Is that a good thinking? Compare the 2 aproaches.  


__TODO:__  
Need to check: correct number of channels is 2, getting 10 (which is a time dim. batch size).  

## attempt_02
In the next attempt. Implement period of trsl != reference wet/dry?  
cml trsl: 1 min, Radolan ref: 5 min  


__sources:__  
(1) CML wet/dry using Pytorch: https://github.com/jpolz/cml_wd_pytorch/blob/main/wd_pytorch/train_cnn.ipynb  
(2) Pycomlink: https://github.com/pycomlink/pycomlink  
