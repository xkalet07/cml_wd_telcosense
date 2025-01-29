
# In[25]:


batchsize = 50
k_train = 0.8     # fraction of training data

train_size = int(len(trsl)*k_train/batchsize)* batchsize
train_size


# Shuffling data disabled, splitting and turning to arrays

# In[26]:


train_data = trsl[:train_size]
test_data = trsl[train_size:]
train_ref = ref[:train_size]
test_ref = ref[train_size:]


# Storing as tensors [2]

# In[27]:


train_data = torch.Tensor(train_data)
test_data = torch.Tensor(test_data)

train_ref = torch.Tensor(train_ref)
test_ref = torch.Tensor(test_ref)


# Turning into dataset, shuffling

# In[28]:


dataset = torch.utils.data.TensorDataset(train_data, train_ref)
testset = torch.utils.data.TensorDataset(test_data, test_ref)

trainloader = torch.utils.data.DataLoader(dataset, batch_size = batchsize, shuffle = False)    # shuffle the training data, once more? True
testloader = torch.utils.data.DataLoader(testset, batch_size = batchsize, shuffle = False)


# ## Training the CNN

# In[29]:


model = cnn.cnn_class()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

# try lower training rate and no dropout


# In[33]:


epochs = 20    # experimentally max epochs: 30, overfitting around 15
resume = 19     # continue learning at epoch xx 
# Low epochs make cnn output low? around 0.5 max


# In[34]:


if resume == 0:
    loss_dict = {}
    loss_dict['train'] = {}
    loss_dict['test'] = {}
    for key in ['train','test']:
        loss_dict[key]['loss'] = []


# In[35]:


for epoch in range(resume, epochs):
    # training
    cnn_prediction = []
    train_losses = []
    for inputs, targets in tqdm(trainloader):
        optimizer.zero_grad()
        pred = model(inputs)
        pred = nn.Flatten(0,1)(pred)            # transpose column data into row
       
        # getting the output
        if epoch == epochs-1: cnn_prediction = cnn_prediction+pred.tolist()
        
        # calculating the loss function        
        loss = nn.BCELoss()(pred, targets)      # Targets and Imputs size must match
        loss.backward()
        optimizer.step()
        train_losses.append(loss.detach().numpy())
    loss_dict['train']['loss'].append(np.mean(train_losses))

    
    # testing
    test_losses = []
    with torch.no_grad():
        for inputs, targets in tqdm(testloader):
            pred = model(inputs)
            # pred = pred.round()
            pred = nn.Flatten(0,1)(pred)
            
            # getting the output
            if epoch == epochs-1: cnn_prediction = cnn_prediction+pred.tolist()
                
            loss = nn.BCELoss()(pred, targets)
            test_losses.append(loss.detach().numpy())
        loss_dict['test']['loss'].append(np.mean(test_losses))
        
    # printing
    clear_output(wait=True)
    print(epoch)
    print('')
    print('train loss:', np.mean(train_losses))
    print('test loss:', np.mean(test_losses))
    print('min test loss:', np.min(loss_dict['test']['loss']))
    fig, axs = plt.subplots(1,1, figsize=(3,4))
    for key in loss_dict.keys():
        for k, key2 in enumerate(loss_dict[key].keys()):
            axs.plot(loss_dict[key][key2], label=key)
            axs.set_title(key2)
    # axs.set_yscale('log')
    plt.legend()
    plt.show()
    resume = epoch


# In[65]:


print(epoch)
print('')
print('train loss:', np.mean(train_losses))
print('test loss:', np.mean(test_losses))
print('min test loss:', np.min(loss_dict['test']['loss']))
fig, axs = plt.subplots(1,1, figsize=(4,4))
for key in loss_dict.keys():
    for k, key2 in enumerate(loss_dict[key].keys()):
        axs.plot(loss_dict[key][key2], label=key)
        axs.set_title((key2+' function'))
# axs.set_yscale('log')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
resume = epoch
fig.tight_layout(pad=1.0)
fig.savefig('loss_curve.svg')


# #### Option to save the trained model parameters:
# source: https://stackoverflow.com/questions/42703500/how-do-i-save-a-trained-model-in-pytorch

# In[50]:


if 1:
    import datetime
    path = 'modul/trained_cnn_param/x'
    date = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')
    torch.save(model.state_dict(), (path))

# then for loading the model parameters:
# model = cnn.cnn_class(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))


# In[44]:


print('modul/trained_cnn_param/x', datetime.datetime.now())


