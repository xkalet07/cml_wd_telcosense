





cnn_wd_threshold = 0.5

## TODO: if theres no TP, FN, FN create them
ds['cnn_wd'] = (('cml_id', 'sample_num'), ds.cnn_out.values > cnn_wd_threshold)

# predicted true wet
ds['true_wet'] = ds.cnn_wd & ds.ref_wd 
# cnn false alarm
ds['false_alarm'] = ds.cnn_wd & ~ds.ref_wd
# cnn missed wet
ds['missed_wet'] = ~ds.cnn_wd & ds.ref_wd


# #### ROC curve ... calculated for whole testing dataset
# source: https://github.com/jpolz/cnn_cml_wet-dry_example/blob/master/CNN_for_CML_example_nb.ipynb  

def roc_curve(y_pred, y_true, tr_start, tr_end):
    '''
    Compute the ROC curve for the CNN. The minimum threshold is tr_start and the maximum threshold is tr_end.
    '''
    roc = []
    for i in range(tr_start*1000,1+tr_end*1000,1):
        t = i/1000
        y_predicted=np.ravel(y_pred>t)  
        true_pos = np.sum(np.logical_and(y_true==1, y_predicted==1))
        true_neg = np.sum(np.logical_and(y_true==0, y_predicted==0))
        false_pos = np.sum(np.logical_and(y_true==0, y_predicted==1))
        false_neg = np.sum(np.logical_and(y_true==1, y_predicted==0))
        cond_neg = true_neg+false_pos
        cond_pos = true_pos+false_neg
        roc.append([true_pos/cond_pos,
                    false_pos/cond_neg])
    roc.append([0,0])
    
    return np.array(roc)

def roc_surface(roc):
    '''
    Compute the Area under a ROC curve.
    '''
    k = len(roc)
    surf=0
    for i in range(k-1):
        surf= surf+(np.abs(roc[i,1]-roc[i+1,1]))*0.5*(roc[i+1,0]+roc[i,0])
    
    return surf



# select testing cmls
first_test_cml_id = int(math.ceil(k_train * num_cmls))

# ROC curve
roc = roc_curve(ds.cnn_out[first_test_cml_id:].values.reshape(-1), ds.ref_wd[first_test_cml_id:].values.reshape(-1), 0, 1)

plt.figure(figsize=(5,5))
plt.plot(roc[:,1],roc[:,0], color='green', label='CNN Area: '+str(np.round(roc_surface(roc), decimals=2)), zorder=2, lw=3)

# plot point of cnn threshold for optimalisation
plt.scatter(roc[int(cnn_wd_threshold*1000),1],roc[int(cnn_wd_threshold*1000),0], color='black', marker='h', s=75, label='$\\tau$ ='+str(cnn_wd_threshold), zorder=3)
thr = 0.25
plt.scatter(roc[int(thr*1000),1],roc[int(thr*1000),0], color='black', marker='.', s=75, label='$\\tau$ ='+str(thr), zorder=3)

plt.plot([0,0,1,0,1,1],[0,1,1,0,0,1], 'k-', linewidth=0.3, zorder=1)
plt.title('ROC curve, TPR = f(TNR)')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc='lower right', ncol=2, frameon=False)
plt.grid()
plt.yticks(np.arange(0, 1.01, 0.1))
plt.xticks(np.arange(0, 1.01, 0.1))
plt.tight_layout()
plt.show()


# #### MCC, ACC, confusion matrix ... calculated for whole testing dataset
# source: https://github.com/jpolz/cnn_cml_wet-dry_example/blob/master/CNN_for_CML_example_nb.ipynb

print('CNN scores')

# confusion matrix 
cm = skl.confusion_matrix(ds.ref_wd[first_test_cml_id:].values.reshape(-1), ds.cnn_wd[first_test_cml_id:].values.reshape(-1), labels=[0,1], normalize='true').round(decimals=2)
print('normalized confusion matrix:\n',cm)
print('TNR:', cm[0,0])
print('TPR:', cm[1,1])

# Matthews Correlation Coeficient
mcc = skl.matthews_corrcoef(ds.ref_wd[first_test_cml_id:].values.reshape(-1), ds.cnn_wd[first_test_cml_id:].values.reshape(-1)).round(decimals=2)
print('MCC:', mcc)

# 
acc = np.round(skl.accuracy_score(ds.ref_wd[first_test_cml_id:].values.reshape(-1), ds.cnn_wd[first_test_cml_id:].values.reshape(-1)), decimals=2)
print('ACC:', acc)

f1 = skl.f1_score(ds.ref_wd[first_test_cml_id:].values.reshape(-1), ds.cnn_wd[first_test_cml_id:].values.reshape(-1)).round(decimals=2)
print('F1:', f1)

# ROC curve surface
a = roc_surface(roc).round(decimals=2)
print('ROC surface A:', a)


# plot the confusion matrix
labels = ['dry', 'wet']

fig, ax1 = plt.subplots(figsize=(3,3), sharex=True)
#ax1 = fig.add_subplot(131)

cax = ax1.matshow(cm, cmap=plt.cm.Blues)
ax1.set_xticklabels([''] + labels)
ax1.set_yticklabels([''] + labels)
fmt = '.2f'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black"
)
    
plt.xlabel('Predicted')
plt.ylabel('True')
ax1.xaxis.set_label_position('top') 
plt.tight_layout()
plt.title('CNN', pad=50)