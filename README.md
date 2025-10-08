# Rain activity detection using microwave link data with convolutional neural network

Python package for rain event detection, using signal loss data from Commercial Microwave Links, with the use of Convolutional Neuron Network. Package includes utility for CNN model inference, CML data preprocessing, results evaluation, CNN model trainingu.  

Project by TelcoSense

author: Lukáš Kaleta  


## Abstract: 

Package built as a Masters thesis in years 2024/2025 for rain event detection using CML data from czech republic and CHMI.   

This thesis presents an implementation of Convolutional Neural Network (CNN) model
for rain event detection, using signal loss from Commercial Microwave Link (CML).
The study combines theoretical aspects of CMLs, including their attenuation-rainfall
𝑘-R relationship, Wet Antenna Attenuation (WAA), and machine learning principles,
alongside a review of prior ML-based CML rain detection implementations.
The CNN model was trained and evaluated on a large dataset from the Czech Republic,
containing a full year of data from 100 CMLs, aligned with rain gauge measured rain-rate
as a reference. Even though the model outperformed chosen reference state-of-the-art
methods, achieving overall TPR of 0.65 and TNR of 0.97, its performance strongly
decreased (TPR = 0.33 and TNR = 0.93) when used on the Czech data. This outcome
suggests that the current CNN model is well-suited for robust rain detection, but the
CML data quality and available methods of data collection have to be improved.


## Example scripts:
  
`inference_workflow.py` - The typical workflow of utilizing the trained CNN module for rain event detection using CML data from czech republic and CHMI.

`train_workflow.py` - The typical workflow of data preprocesssing and training a CNN module for rain event detection using CML data from czech republic and CHMI.

`preprocess_all_workflow.py` - The typical workflow of preprocessing all of the avalable CML data in directory and storing them as separate .csv files for classification     or one .csv file to use as training dataset.
    
`model_comparison_pipeline.py` - This script compares developed CNN model, with reference Wet/Dry classification methods. Using already preprocessed CML data.  
Reference methods:  
- Reference CNN model from https://github.com/jpolz/cml_wd_pytorch   
- State of the art, non-ML method RSD using rolling window STD: https://github.com/pycomlink/pycomlink  

`waa_workflow.py` - Predict WD class using CML data and to estimate rainrate from rain induced attenuation. This script compares several methods of Wet Antenna Attenuation (WAA) compensations: Schleiss, Leijnse and Pastorek.  


## objectives:

TODO: load cml B using its IP, not i+1  
TODO: copy data into NaN gaps from adjacent CML #cml['rsl_A'] = cml.rsl_A + cml.rsl_B.where(np.isnan(cml.rsl_A)), example: 1s10: 12  
TODO: better [interpolation](https://stackoverflow.com/questions/30533021/interpolate-or-extrapolate-only-small-gaps-in-pandas-dataframe)  
TODO: Add constant CML metadata such as frequency, technology, length... as and CNN input  
TODO: Test CML end device's temperature as an CNN input  


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
 

## sources:  
(1) CML wet/dry using Pytorch: https://github.com/jpolz/cml_wd_pytorch/blob/main/wd_pytorch/train_cnn.ipynb  
(2) Pycomlink: https://github.com/pycomlink/pycomlink  
(3) CML wet/dry using Tensorflow: https://github.com/jpolz/cnn_cml_wet-dry_example/blob/master/CNN_for_CML_example_nb.ipynb  
