# Rain activity detection using microwave link data with convolutional neural network

Repository of Masters thesis CNN implementation 2024/2025  
author: Lukáš Kaleta  

See recent result figures [here](https://drive.google.com/drive/folders/11z7vx3xHwLl-xq6UOi-dIngluj5o-5Ou?usp=sharing)  

## objectives:

DONE: bad gauge reference interpolation  
DONE: rain gauge reference SRA10M is missing in some technologies  
DONE: skip cmls without SRA10M inside the main skript: check for it. Long computation!!!  
DONE-doublecheck: single extreme rsl values like 90 dB or so.  
DONE-doublecheck: calculate new mean value when skipping large nan gaps, causing steps in rsl data  
tip: floating standardization excludes long term fluctuation  
TODO: load cml B using its IP, not i+1  
DONE: filtered metadata is duplicative. each cml is there 2 times identically  
tip: cmlAip and cmlBip are next to each other and cmlBip is always cmlAip+1  
TODO: copy data into NaN gaps from adjacent cml #cml['rsl_A'] = cml.rsl_A + cml.rsl_B.where(np.isnan(cml.rsl_A))  
TODO: copy adjacent data, if large chunk of rsl is missing: 1s10: 12  
DONE: delete few values around step  
TODO: better interpolation: https://stackoverflow.com/questions/30533021/interpolate-or-extrapolate-only-small-gaps-in-pandas-dataframe  
DONE: detect uptime resets  
DONE: dry 10min segments during light rainfall  
tip: Summit technology rsl is [-dB]  
TODO: Feed cnn constant metadata such as frequency, technology, length...  
TODO: Feed cnn the cml temperature  
DONE: If uptime is constant drop values  
TODO: spikes remaining around step after supressing the step in preprocessing (especially 1s10)  
tip: all datetime is in UTC time  
TODO: Different preprocess tresholds for different cml technologies  
TODO: period of trsl == reference wet/dry. Meaning, for each trsl point there will be wet/dry flag predicted.  
TODO: Forward and backward memory implementation will be needed.  
TODO: This approach should bring better learning performance. For longer wet/dry periods there are ocasions, where the period is wet, but trsl shows rain pattern for only fraction of the period.  
TODO: some CML still has NaN gaps, find it and exclude the NaN samples.  
TODO: problem with ceragon_ip_10  
DONE: implement pooling, we need shorttime-longtime pattern reckognition  
TODO: Plots doesnt show cml ip or any id as a title  
TODO: Data augmentation: noise injecting, time warp, Random scaling, mixUp/cutMix  
TODO: Improve WD ref preprocessing by including single dry samples between 2 rainy as wet, if R is nonzero  
TODO: In the CNN modul, investigate the time pooling: x = torch.mean(x,dim=-1). It completely averages one dimension in one step.  
TODO: 2 outputs instead of single output? classify based on 2 probabilities pW and pD instead of one pW?  


## Hyperparameter tuning  
- different optimizers: AdamW had better performance than SGD  
- pooling: has highly positive results  
- batchnormalisation: crusial improvement from TP 0.4 to 0.8  
- weight initialisation with Xavier: 2% TP and LR improvement, 4% FP decrease.  
- sample size: Higher sample size, slightly higher TP, but more overfitting. Under 30 lower performance. Ideal 60.  
- Standardization: +3% better TP results with mean-max standardization  
- Weight_decay: using 1e-4 weight decay in optimizer, enhances TP+1% lowers testloss to 1.6 and lowers FP  
- convolutional filters best performing [24,48,96,192], less layers or constant layer sizes lead to worse performance. But using [16, 32, 64, 128] lead to -2 % worse TPR witth great decrease of trainloss to 1.45   
- dropout layers in convolutional layers: needs to be low value, same as for FC: dropout=0.001. Same TPR=0.86, lower testloss of 1.42.  
- LeakyReLU has worse performance than classic ReLU, of TPR-2%, FPR+2%, and both losses worse.  
- dataset balancing: Excluding long dry periods helps memory capacity during training, and balances wet and dry classes in the cml signal. Warning: not using wd balance leads to ilusion of lower losses, but real performance is worse. Keeping long dry periods doesnt prevent overfitting.  
- LR scheduler has positive results. StepLR is better than ReduceLROnPlateau  
- It was tested that temperature included in dataset causes erratic behaviour.  
- 

## RADOLAN and Pycomlink data 

### 01 training CNN on one CML
#### Goal:
- train existing CNN(1) on open sense cml and reference RADOLAN data from Germany(2). 
- calculate WAA using pycomlink function from Schleiss 2013. 

#### status:   
- Prediction works with test loss around 0.2.  
- Solved array size missmatch.  
- choosing one cml, converting to torch Tensor.   
- 

#### Optimize learning:   
- dropout rate: 0.4 is far to high, causes high learning curve ripple: set 0, later can be increased.  
- learning rate lowered: 0.0001, learning is fast but convergs to high values.  
- changed standardising: to min-max = 0-1, performance improved significantly!


### 02 training CNN on a dataset of 20 CMLs

#### status:
- Added more cmls to dataset. current: 20   

#### TODO:
- sample shuffle: increases learning speed and precission.  
- wet/dry 50/50 for faster learning and more accurate TPR/TNR results  
- CNN threshold optimalization algorhythm, currently set to 0.5.
- improve the CNN architecture.
 
### current CNN architecture (1):
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

## sources:  
(1) CML wet/dry using Pytorch: https://github.com/jpolz/cml_wd_pytorch/blob/main/wd_pytorch/train_cnn.ipynb  
(2) Pycomlink: https://github.com/pycomlink/pycomlink  
(3) JPolz: CML wet/dry using Tensorflow: https://github.com/jpolz/cnn_cml_wet-dry_example/blob/master/CNN_for_CML_example_nb.ipynb  
