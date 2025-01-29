
# ## Data validation
# prediction output, confusion matrix, MCC...
# #### Plot the prediction output
# #### <span style="color:red">TODO: </span>Add Legends to figures

# In[36]:


ds['cnn_out'] = (('cml_id', 'sample_num'), np.array(cnn_prediction).reshape(num_cmls,-1))


# In[37]:


cnn_wd_threshold = 0.5

ds['cnn_wd'] = (('cml_id', 'sample_num'), ds.cnn_out.values > cnn_wd_threshold)


# In[38]:


# predicted true wet
ds['true_wet'] = ds.cnn_wd & ds.ref_wd 
# cnn false alarm
ds['false_alarm'] = ds.cnn_wd & ~ds.ref_wd
# cnn missed wet
ds['missed_wet'] = ~ds.cnn_wd & ds.ref_wd


# In[39]:


# setup figure
fig, axs = plt.subplots(num_cmls, 1, sharex=True, figsize=(12,num_cmls*2))
ax1 = axs[0].twiny()
ax1.set_xlim(ds.time.values[0,0], ds.time.values[-1,-1])
fig.tight_layout(h_pad = 3)

for n in range(num_cmls):    
    axs[n].set_xlim(0, n_samples + cutoff)
    axs[n].set_ylim(0,2);    
    # plot cnn prediction
    ds.cnn_out[n].plot.line(x='sample_num', ax=axs[n], label = 'TL',color='black', lw=0.5);

    #cnn threshold
    axs[n].axhline(cnn_wd_threshold, color='black', linestyle='--', lw=0.5)

    # GREEN: plot true cnn predicted wet/dry areas
    # tip from stack ovefrolw: https://stackoverflow.com/questions/44632903/setting-multiple-axvspan-labels-as-one-element-in-legend
    start = np.roll(ds.true_wet[n], -1) & ~ds.true_wet[n]
    end = np.roll(ds.true_wet[n], 1) & ~ds.true_wet[n]
    for start_i, end_i in zip(
        start.values.nonzero()[0],
        end.values.nonzero()[0],
    ):
        axs[n].axvspan(ds.sample_num.values[start_i], ds.sample_num.values[end_i], color='g', alpha=0.5, linewidth=0, label='_'*start_i+'true wet') 
    
    # RED: plot false alarms
    start = np.roll(ds.false_alarm[n], -1) & ~ds.false_alarm[n]
    end = np.roll(ds.false_alarm[n], 1) & ~ds.false_alarm[n]
    for start_i, end_i in zip(
        start.values.nonzero()[0],
        end.values.nonzero()[0],
    ):
        axs[n].axvspan(ds.sample_num.values[start_i], ds.sample_num.values[end_i], color='r', alpha=0.5, linewidth=0, label='_'*start_i+'false alarm') 
        
    # ORANGE: plot missed wet 
    start = np.roll(ds.missed_wet[n], -1) & ~ds.missed_wet[n]
    end = np.roll(ds.missed_wet[n], 1) & ~ds.missed_wet[n]
    for start_i, end_i in zip(
        start.values.nonzero()[0],
        end.values.nonzero()[0],
    ):
        axs[n].axvspan(ds.sample_num.values[start_i], ds.sample_num.values[end_i], color='orange', alpha=0.5, linewidth=0, label='_'*start_i+'missed wet')





# In[64]:


# setup figure
fig, axs = plt.subplots(4, 1, sharex=True, figsize=(12,8))
ax1 = axs[0].twiny()
ax1.set_xlim(ds.time.values[0,0], ds.time.values[-1,-1])
fig.tight_layout(h_pad = 3)

for n in range(4):    
    axs[n].set_xlim(0, n_samples + cutoff)
    axs[n].set_ylim(0,1.5);    
    # plot cnn prediction
    ds.cnn_out[16+n].plot.line(x='sample_num', ax=axs[n], label = 'TL',color='black', lw=0.5);

    #cnn threshold
    axs[n].axhline(cnn_wd_threshold, color='black', linestyle='--', lw=0.5)

    # GREEN: plot true cnn predicted wet/dry areas
    # tip from stack ovefrolw: https://stackoverflow.com/questions/44632903/setting-multiple-axvspan-labels-as-one-element-in-legend
    start = np.roll(ds.true_wet[16+n], -1) & ~ds.true_wet[16+n]
    end = np.roll(ds.true_wet[16+n], 1) & ~ds.true_wet[16+n]
    for start_i, end_i in zip(
        start.values.nonzero()[0],
        end.values.nonzero()[0],
    ):
        axs[n].axvspan(ds.sample_num.values[start_i], ds.sample_num.values[end_i], color='g', alpha=0.5, linewidth=0, label='_'*start_i+'true wet') 
    
    # RED: plot false alarms
    start = np.roll(ds.false_alarm[16+n], -1) & ~ds.false_alarm[16+n]
    end = np.roll(ds.false_alarm[16+n], 1) & ~ds.false_alarm[16+n]
    for start_i, end_i in zip(
        start.values.nonzero()[0],
        end.values.nonzero()[0],
    ):
        axs[n].axvspan(ds.sample_num.values[start_i], ds.sample_num.values[end_i], color='r', alpha=0.5, linewidth=0, label='_'*start_i+'false alarm') 
        
    # ORANGE: plot missed wet 
    start = np.roll(ds.missed_wet[16+n], -1) & ~ds.missed_wet[16+n]
    end = np.roll(ds.missed_wet[16+n], 1) & ~ds.missed_wet[16+n]
    for start_i, end_i in zip(
        start.values.nonzero()[0],
        end.values.nonzero()[0],
    ):
        axs[n].axvspan(ds.sample_num.values[start_i], ds.sample_num.values[end_i], color='orange', alpha=0.5, linewidth=0, label='_'*start_i+'missed wet')




# #### ROC curve ... calculated for whole testing dataset
# 
# source: https://github.com/jpolz/cnn_cml_wet-dry_example/blob/master/CNN_for_CML_example_nb.ipynb  

# #### <span style="color:red">TODO:</span> Add ROC and CM for all testing, all training, one best and one worst

# In[ ]:


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


# In[ ]:


# select testing cmls
first_test_cml_id = int(math.ceil(k_train * num_cmls))


# In[ ]:


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

# In[ ]:


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


# In[ ]:



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


# #### old: Metrics for one cml

# In[ ]:


cml_k = '16'


# In[ ]:


# Calculate ROC curve for all training + testing cmls
roc_c = np.empty([num_cmls, 1002,2])
roc_surface = np.empty([num_cmls])

for i in range(num_cmls):
    roc_c[i] = roc_curve(ds.cnn_out[i], ds.ref_wd[i], 0, 1)
    roc_surface[i] = np.round(roc_surface(roc_curve[i]), decimals=2)


# In[ ]:





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


# In[ ]:


print('CNN scores')

# confusion matrix 
cm = skl.confusion_matrix(ds.ref_wd.sel(cml_id=cml_k), ds.cnn_wd.sel(cml_id=cml_k), labels=[0,1], normalize='true').round(decimals=2)
print('normalized confusion matrix:\n',cm)

print('confusion matrix:\n',skl.confusion_matrix(ds.ref_wd.sel(cml_id=cml_k), ds.cnn_wd.sel(cml_id=cml_k)).round(decimals=2))
print('TNR:', cm[0,0])
print('TPR:', cm[1,1])

# Matthews Correlation Coeficient
mcc = skl.matthews_corrcoef(ds.ref_wd.sel(cml_id=cml_k), ds.cnn_wd.sel(cml_id=cml_k)).round(decimals=2)
print('MCC:', mcc)

# 
acc = np.round(skl.accuracy_score(ds.ref_wd.sel(cml_id=cml_k), ds.cnn_wd.sel(cml_id=cml_k)), decimals=2)
print('ACC:', acc)

f1 = skl.f1_score(ds.ref_wd.sel(cml_id=cml_k), ds.cnn_wd.sel(cml_id=cml_k)).round(decimals=2)
print('F1:', f1)

# ROC curve surface
a = roc_surface(roc).round(decimals=2)
print('ROC surface A:', a)


# In[ ]:


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