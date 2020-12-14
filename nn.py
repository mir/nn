import data_generator as dg
import torch
from collections import OrderedDict
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from sin_dataset import SinDataset

os.environ['KMP_DUPLICATE_LIB_OK']='True'

input_size = 1024
dataset_size = 1024*8
valset_size = 1024
batch_size = 256
output_size = 1
epochs = 30
learning_rate = 0.01

# train_loader = DataLoader(train_ds, batch_size, shuffle=True)
# val_loader = DataLoader(val_ds, batch_size)
def accuracy(output,labels):
    diff = torch.abs(output - labels)
    diff = conertFreqsDiffBack(diff)  
    maxes, indexes = torch.max(diff, dim = 0)  
    return maxes
# Logistic regression model
class SinModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(OrderedDict([
            ('0', nn.Linear(input_size, 2)),
            ('1',nn.ReLU()),
            ('2',nn.Linear(2,2)),
            ('3',nn.ReLU()),
            ('4',nn.Linear(2,3)),
            ('5',nn.ReLU()),
            ('6',nn.Linear(3,output_size))
            ]))
        
    def forward(self, xb):
        out = self.network(xb)
        return out
    
    def training_step(self, batch):
        data, labels = batch 
        labels = getFrequencies(labels)
        out = self(data)                  # Generate predictions
        loss_fn = nn.MSELoss()        
        return loss_fn(out,labels)
    
    def validation_step(self, batch):
        data, labels = batch 
        labels = getFrequencies(labels)        
        out = self(data)
        loss_fn = nn.MSELoss()        # Generate predictions
        loss = loss_fn(out,labels)              
        acc = accuracy(out,labels)         
        return {'val_loss': loss, 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses    

        batch_acc = [x['val_acc'] for x in outputs]
        epoch_acc, _ = torch.max(torch.stack(batch_acc), dim = 0)   # Combine acc        
        epoch_acc.tolist()
        epoch_acc = epoch_acc[0]
        return {'val_loss': epoch_loss.item(),'val_acc': epoch_acc}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}".format(epoch, result['val_loss']))
        print("Max differences:")        
        print("    frequency: {:.4f}".format(result['val_acc']))

def evaluate(model, val_set):
    """Evaluate the model's performance on the validation set"""
    outputs = [model.validation_step(batch) for batch in val_set]
    return model.validation_epoch_end(outputs)

def fit(epochs, start_learning_rate, model, train_set, val_set, opt_func=torch.optim.Adam):
    """Train the model using gradient descent"""
    losses = []
    epoch_acc = []
    lr = learning_rate    
    for epoch in range(epochs):
        # Training Phase 
        optimizer = opt_func(model.parameters(), lr)
        for batch in train_set:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        # Validation phase
        result = evaluate(model, val_set)
        model.epoch_end(epoch, result)
        epoch_acc.append(result['val_acc'])        

    return losses, epoch_acc

def getFrequencies(labels):
    return (labels[:,1].view(-1,1) - 45)/10

def conertFreqsDiffBack(freqs):
    return freqs * 10

# generate data and data loaders
print('Genering train_set')
train_set = SinDataset(size = dataset_size,
 samples = input_size,
 batch_size = batch_size)
# train_set.saveCsv('train')

print('Genering val_set')
val_set = SinDataset(size = valset_size,
 samples = input_size,
 batch_size = batch_size)
# val_set.saveCsv('validation')

model = SinModel()

fig, (ax1, ax2, ax3) = plt.subplots(3, 2)
fig.suptitle('NN training')
ax1[0].set_title('Training data examples')
ax1[0].set_ylabel('Amplitude')
ax1[0].set_xlabel('time (s)')
for data,labels in train_set:    
    print(data.shape)    
    ax1[0].plot(train_set.time,data[0,:])
    ax1[0].plot(train_set.time,data[1,:])
    ax1[0].plot(train_set.time,data[2,:])
    print(labels.shape)
    print(getFrequencies(labels).shape)
    out = model(data)
    print(out.shape)
    break

print('Train NN')
losses, epoch_acc = fit(epochs,
  learning_rate,
  model, train_set, val_set)

ax1[1].set_title('Validation: NN estimate vs real')
data, label = val_set[0]
out = model(data)
ax1[1].plot(val_set.time,data)
signal = torch.sin(val_set.time *(out*10 + 45)*2*np.pi)
ax1[1].plot(val_set.time,signal.detach())

ax2[0].set_title('Training losses')
ax2[0].set_ylabel('Loss')
ax2[0].set_xlabel('batch')
ax2[0].plot(losses)

ax2[1].set_title('Epoch validation accuracy')
ax2[1].set_ylabel('Frequency deviation')
ax2[1].set_xlabel('epoch')
ax2[1].plot(epoch_acc)

ax3[0].set_title('Weights: layer 1, neuron 1')
ax3[0].plot(model.network[0].weight.detach()[0,:].flatten(),'.')
ax3[1].set_title('Weights: layer 1  neuron 2')
ax3[1].plot(model.network[0].weight.detach()[1,:].flatten(),'.')

plt.tight_layout()
plt.show()