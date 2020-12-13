import data_generator as dg
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from sin_dataset import SinDataset

os.environ['KMP_DUPLICATE_LIB_OK']='True'

input_size = 256
hidden_channels = 3
dataset_size = 8196*8
batch_size = 256
num_classes = 3
epochs = 10
start_learning_rate = 0.1

# train_loader = DataLoader(train_ds, batch_size, shuffle=True)
# val_loader = DataLoader(val_ds, batch_size)
def accuracy(output,labels):
    diff = torch.abs(output - labels)  
    maxes, indexes = torch.max(diff, dim = 0)  
    return maxes
# Logistic regression model
class SinModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(1, hidden_channels,
             kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, 1,
             kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(input_size, num_classes))            
        
    def forward(self, xb):
        out = self.network(xb)        
        return out
    
    def training_step(self, batch):
        data, labels = batch 
        out = self(data)                  # Generate predictions
        loss_fn = nn.MSELoss()
        return loss_fn(out,labels)
    
    def validation_step(self, batch):
        data, labels = batch 
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
        return {'val_loss': epoch_loss.item(),'val_acc': epoch_acc.tolist()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}".format(epoch, result['val_loss']))
        print("Max differences:")
        print("    amplitude: {:.4f}, frequency: {:.4f},phase: {:.4f},".format(result['val_acc'][0],
         result['val_acc'][1],
         result['val_acc'][2],
         ))

def evaluate(model, val_set):
    """Evaluate the model's performance on the validation set"""
    outputs = [model.validation_step(batch) for batch in val_set]
    return model.validation_epoch_end(outputs)

def fit(epochs, start_learning_rate, model, train_set, val_set, opt_func=torch.optim.Adam):
    """Train the model using gradient descent"""
    losses = []
    history = []
    lr = start_learning_rate    
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
        history.append(result)
        lr = lr * 0.5;

    return losses, history

# generate data and data loaders
train_set = SinDataset(size = dataset_size,
 samples = input_size,
 batch_size = batch_size)
train_set.saveCsv('train')

val_set = SinDataset(size = dataset_size,
 samples = input_size,
 batch_size = batch_size)
val_set.saveCsv('validation')

model = SinModel()

# for data,labels in train_set:    
#     print(data.shape)
#     plt.plot(data[0,0,:])
#     plt.plot(data[1,0,:])
#     plt.plot(data[2,0,:])
#     plt.show()
#     break

losses, history = fit(epochs,
 start_learning_rate,
  model, train_set, val_set)
plt.plot(losses)
plt.show()