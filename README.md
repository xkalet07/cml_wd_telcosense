# Rain activity detection using microwave link data with convolutional neural network

Repository for Masters thesis 2024/2025
Lukáš Kaleta  

## attempt_01
__goal:__ train existing CNN(1) on open sense cml and reference RADOLAN data from Germany(2).  

__status:__  
Prediction working with terrible performance.
Solved array size missmatch.  

Working: Importing external data in netCDF, choosing one cml, converting to torch Tensor.   
existing implementation acording to (1) uses 1 wet/dry flag for 180 time stamps (one of 100 samples). Why? My implementation aims to use one flag for one trsl measurement. Is that a good thinking? Compare the 2 aproaches.  


__TODO:__  

## attempt_02
In the next attempt. Implement it that period of trsl == reference wet/dry?  
Meaning, for each trsl point there will be wet/dry flag predicted.  


## CNN architecture:
Input (2 channels) → Convolution Block 1 → Convolution Block 2 → Convolution Block 3 → Convolution 5a → Convolution 5b → Flatten → Dense Layer 1 → Dropout 1 → Dense Layer 2 → Dropout 2 → Output Layer → Sigmoid Activation → Final Output (0 or 1).  


__sources:__  
(1) CML wet/dry using Pytorch: https://github.com/jpolz/cml_wd_pytorch/blob/main/wd_pytorch/train_cnn.ipynb  
(2) Pycomlink: https://github.com/pycomlink/pycomlink  
