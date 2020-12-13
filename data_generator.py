from __future__ import print_function, division
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def sin_data(samples = 100, # number of samples
	size = 10000, # number of size
	T = 0.0001, # sampling time
	min_amp = 0.8, max_amp = 1.2, # maximum amplitude
	min_freq = 45, max_freq = 55, # maximum frequency
	snr = None):

	probs = torch.rand(size,3)
	t_scale = torch.tensor([max_amp - min_amp, max_freq - min_freq, 1]).repeat(size,1)
	t_bias = torch.tensor([min_amp, min_freq, 0]).repeat(size,1)
	labels = probs * t_scale + t_bias
	
	time = torch.linspace(0, samples*T, samples).view(1,-1)
	phases = labels[:,1:2]*2*np.pi @ time + labels[:,2:3].repeat(1,samples)*2*np.pi
	amplitudes = labels[:,:1].repeat(1,samples)

	signals = amplitudes * torch.sin(phases)

	if snr and snr > 0:
		noises = torch.rand(size,samples) * amplitudes/snr
		signals += noises

	return signals, labels