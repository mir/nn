import data_generator as dg
import torch
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import animation
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
learning_rate = 0.1

min_frequency = 45
max_frequency = 55
delta_frequency = max_frequency - min_frequency

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
            ('input-hidden', nn.Linear(input_size, 2)),
            ('activation',nn.LeakyReLU(0.01)),            
            ('hidden-output',nn.Linear(2,output_size))
            ]))
        
    def forward(self, xb):
        out = self.network(xb)
        return out
    
    def training_step(self, batch):
        data, labels = batch 
        labels = getFrequencies(labels)
        out = self(data)                  # Generate predictions
        loss_fn = nn.MSELoss()        
        return loss_fn(out,labels).exp()
    
    def validation_step(self, batch):
        data, labels = batch 
        labels = getFrequencies(labels)        
        out = self(data)
        loss_fn = nn.MSELoss()        # Generate predictions
        loss = loss_fn(out,labels).exp()            
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

def fit(epochs, start_learning_rate,
 model,
 train_set, val_set,
 opt_func=torch.optim.SGD):
    """Train the model using gradient descent"""
    losses = []
    epoch_acc = []
    weights = []
    lr = learning_rate 
    weights.append(getWeights(model))
    for epoch in range(epochs):
        optimizer = opt_func(model.parameters(), lr)
        for batch in train_set:                        
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            weights.append(getWeights(model))
        # Validation phase
        result = evaluate(model, val_set)
        model.epoch_end(epoch, result)
        epoch_acc.append(result['val_acc'])                

    return losses, epoch_acc, weights

def getFrequencies(labels):
    return (labels[:,1].view(-1,1) - min_frequency)/delta_frequency

def conertFreqsDiffBack(freqs):
    return freqs * delta_frequency

def getWeights(model):
    w1 = model.network[0].weight.detach().clone()[0,:].flatten()    
    w2 = model.network[0].weight.detach().clone()[1,:].flatten()  
    w = torch.cat((w1,w2), 0)
    return w

def saveDataAnimation(data):
    fig = plt.figure()    
    data_size = len(data[0])    
    ax = plt.axes(xlim=(0, data_size), ylim=(torch.min(data[1])*1.5, torch.max(data[1])*1.5))
    scatter_plot = ax.scatter([], [], s=1)
     
    def init():
        print('Saving gifs')
        scatter_plot.set_offsets([])
        return scatter_plot,
    def animate(i):
        x = np.arange(data_size)        
        y = data[i].numpy() 
        offsets = np.stack((x,y)).T
        scatter_plot.set_offsets(offsets)        
        if (i%64 == 0):
            print('{}%'.format(round(100*i/len(data))))
        return scatter_plot,

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(data), interval=20, blit=False)    
    writergif = animation.PillowWriter(fps=60)
    anim.save('weights.gif',writer=writergif)
    plt.close(fig)

# generate data and data loaders
print('Genering train_set')
train_set = SinDataset(size = dataset_size,
 samples = input_size,
 batch_size = batch_size,
 min_freq = min_frequency,
 max_freq = max_frequency,
 )
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
losses, epoch_acc, weights = fit(epochs,
  learning_rate,
  model, train_set, val_set)

saveDataAnimation(weights)

ax1[1].set_title('Validation: NN estimate vs real')
data, label = val_set[0]
out = model(data)
ax1[1].plot(val_set.time,data)
signal = torch.sin(val_set.time *(out*delta_frequency + min_frequency)*2*np.pi)
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

# fig, (ax1, ax2, ax3) = plt.subplots(3, 2)
# fig.suptitle('NN training')
# ax1[0].set_title('Weights: layer 2, neuron 1')
# ax1[0].plot(model.network[2].weight.detach()[0,:].flatten(),'.')
# ax1[1].set_title('Weights: layer 2  neuron 2')
# ax1[1].plot(model.network[2].weight.detach()[1,:].flatten(),'.')

# plt.tight_layout()
plt.show()