# Rain activity detection using microwave link data with convolutional neural network

Repository of Masters thesis CNN implementation 2024/2025  

Project by TelcoSense

author: Lukáš Kaleta  


## Abstract: 

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

`waa_workflow.py` - Predict WD class using CML data and to estimate rainrate from rain induced attenuation. This script compares several methods of Wet Antenna Attenuation (WAA) compensations: Schleiss, Leijnse and Pastorek.  

`train_workflow.py` - The typical workflow of data preprocesssing and training a CNN module for rain event detection using CML data from czech republic and CHMI.

`preprocess_all_workflow.py` - The typical workflow of preprocessing all of the avalable CML data in directory and storing them as separate .csv files for classification     or one .csv file to use as training dataset.
    
`model_comparison_pipeline.py` - This script compares developed CNN model, with reference Wet/Dry classification methods. Using already preprocessed CML data.  
Reference methods:  
- Reference CNN model from https://github.com/jpolz/cml_wd_pytorch   
- State of the art, non-ML method RSD using rolling window STD: https://github.com/pycomlink/pycomlink  
  
`classify_workflow.py` - The typical workflow of utilizing the trained CNN module for rain event detection using CML data from czech republic and CHMI.


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
 

## sources:  
(1) CML wet/dry using Pytorch: https://github.com/jpolz/cml_wd_pytorch/blob/main/wd_pytorch/train_cnn.ipynb  
(2) Pycomlink: https://github.com/pycomlink/pycomlink  
(3) JPolz: CML wet/dry using Tensorflow: https://github.com/jpolz/cnn_cml_wet-dry_example/blob/master/CNN_for_CML_example_nb.ipynb  
