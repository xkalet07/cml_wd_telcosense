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


__TODO: optimize learning__  
- dropout rate: 0.4 is far to high, causes high learning curve ripple: set 0, later can be increased.  
- learning rate lowered: 0.0001, learning is fast but convergs to high values.  
- changed standardising: to min-max = 0-1, performance improved significantly!


## attempt_02
In the next attempt. Implement it that period of trsl == reference wet/dry?  
Meaning, for each trsl point there will be wet/dry flag predicted.  


## CNN architecture:
Input (2 channels) → Convolution Block 1 → Convolution Block 2 → Convolution Block 3 → Convolution 5a → Convolution 5b → Flatten → Dense Layer 1 → Dropout 1 → Dense Layer 2 → Dropout 2 → Output Layer → Sigmoid Activation → Final Output (0 or 1).  

#### Convolutional Part:
- 3 convolutional blocks with increasing filter sizes.  
- Final 2 convolutional layers (conv5a and conv5b) to capture more complex patterns.  
- ReLU activations after each convolution.  
- no pooling implemented.  

#### Fully Connected Part:
- Two fully connected layers with 64 neurons.  
- Dropout applied after each fully connected layer to avoid overfitting.  
- Final output produced through a single neuron with Sigmoid activation (ideal for binary classification).  

__sources:__  
(1) CML wet/dry using Pytorch: https://github.com/jpolz/cml_wd_pytorch/blob/main/wd_pytorch/train_cnn.ipynb  
(2) Pycomlink: https://github.com/pycomlink/pycomlink  
(3) JPolz: CML wet/dry using Tensorflow: https://github.com/jpolz/cnn_cml_wet-dry_example/blob/master/CNN_for_CML_example_nb.ipynb  
