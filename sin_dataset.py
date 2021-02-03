import torch
import numpy as np

class SinDataset(torch.utils.data.IterableDataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, size, samples, batch_size = 8, min_freq = 45, max_freq = 55):
    'Initialization'
    data, labels = self.sin_data(size=size,
     samples=samples,
     min_freq = min_freq,
     max_freq = max_freq)
    self.labels = labels
    self.data = data
    self.size = size
    self.samples = samples
    self.batch_size = batch_size
    self.position = 0    

  def __len__(self):
    'Denotes the total number of samples'
    return self.size

  def __getitem__(self, index):
    'Generates one sample of data'       
    return self.data[index], self.labels[index]

  def __iter__(self):
    return SinDatasetIterator(self.data,self.labels,
      self.size,self.batch_size,self.samples)

  def sin_data(self,
    samples = 100, # number of samples
    size = 10000, # number of size
    T = 0.0001, # sampling time
    min_amp = 1, max_amp = 1, # maximum amplitude
    min_freq = 45, max_freq = 55, # maximum frequency
    phase = 0,
    snr = 10):

    probs = torch.zeros(size,3)
    if type(phase) == int or float:
      r = torch.rand(size,2)
      p = torch.zeros(size,1) + phase
      probs = torch.cat((r,p),dim=1)
    else:
      probs = torch.rand(size,3)

    t_scale = torch.tensor([max_amp - min_amp, max_freq - min_freq, 1]).repeat(size,1)
    t_bias = torch.tensor([min_amp, min_freq, 0]).repeat(size,1)
    labels = probs * t_scale + t_bias
    
    self.time = torch.linspace(0, samples*T, samples)
    time = self.time.view(1,-1)
    phases = labels[:,1:2]*2*np.pi @ time + labels[:,2:3].repeat(1,samples)*2*np.pi
    amplitudes = labels[:,:1].repeat(1,samples)

    signals = amplitudes * torch.sin(phases)

    if snr and snr > 0:
      noises = torch.rand(size,samples) * amplitudes/snr
      signals += noises

    return signals, labels
      
  def saveCsv(self, prefix):
    import numpy as np        
    
    data_file = open(prefix + '-data.csv', 'wb')
    data_np = self.data.numpy()    
    np.savetxt(data_file,data_np)
    data_file.close()

    labels_file = open(prefix + '-labels.csv', 'wb')
    labels_np = self.labels.numpy()
    np.savetxt(labels_file,labels_np)
    labels_file.close()

  def loadCsv(self,prefix):
    import numpy as np        
    
    data_file = open(prefix + '-data.csv', 'rb')    
    data_np = np.loadtxt(data_file)
    data_file.close()

    labels_file = open(prefix + '-labels.csv', 'rb')
    labels_np = np.loadtxt(labels_file)    
    labels_file.close()

    data = torch.from_numpy(data_np)
    labels = torch.from_numpy(labels_np)
    return data, labels

class SinDatasetIterator():
  'Iterator for SinDataset'
  def __init__(self, data, labels, size, batch_size, samples):
    self.labels = labels
    self.data = data
    self.size = size
    self.samples = samples
    self.batch_size = batch_size
    self.position = 0

  def __next__(self):    
    if (self.position + self.batch_size) >= self.size:
      raise StopIteration

    labels = self.labels[self.position:self.position+self.batch_size,:]
    data = self.data[self.position:self.position+self.batch_size,:]    
    self.position += self.batch_size  
    return data, labels