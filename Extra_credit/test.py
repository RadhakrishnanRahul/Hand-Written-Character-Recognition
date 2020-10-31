# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 15:30:02 2019

@author: sragh
We have used Convolutional Neural Network for this project. The following packages and libraries
have to be installed:
    1) pytorch
    2) sklearn
    3) PIL
    4) numpy
    5) pickle
These libraries and packages are preinstalled in latest versions of anaconda
"""

import sys
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import skimage.filters as filt

''' Python functions to load and save pickle objects '''
def load_pkl(fname):
	with open(fname,'rb') as f:
		return pickle.load(f)

def save_pkl(fname,obj):
	with open(fname,'wb') as f:
		pickle.dump(obj,f)

''' Load Data and labels '''
fname1= sys.argv[1]
fname2 = sys.argv[2]
data  = load_pkl(fname1)

''' convert to numpy array '''
data = np.array([np.array(i)/1.00 for i in data])



''' Preprocessing the input images '''
for i in range(data.shape[0]):
    temp = data[i]
    temp = resize(temp,[50,50])
    temp = rgb2gray(temp)
    data[i] = temp
    
classes = ['a','b','c','d','h','i','j','k']

''' Define the Neural Network '''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 5) # First Convolution Layer
        self.pool = nn.MaxPool2d(2, 2) # Max pool 
        self.conv2 = nn.Conv2d(4, 16, 5) # Second Convolution Layer
        self.fc1 = nn.Linear(16 * 9 * 9, 120) # Fully connected Layer1
        self.fc2 = nn.Linear(120, 84) # Fully connected Layer2
        self.fc3 = nn.Linear(84, 8) # Fully connected Layer3

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) #relu activation 
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 9 * 9)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
''' Instantiate NN '''
net = Net()

''' Load the trained parameters'''
net.load_state_dict(torch.load('trained_model.pt'))

''' Loaders to iterate through images. The iterators will return a pytorch tensor '''
ImageLoader = torch.utils.data.DataLoader(data, batch_size=1,
                                              shuffle=False, num_workers=0)

''' Predict the outputs '''
predictions = []
with torch.no_grad():
    for images in ImageLoader:
        images = images.view(1,1,50,50)
        outputs = net(images.float())
        _, predicted = torch.max(outputs, 1)
        outputs = outputs.numpy()
        outputs[0] = outputs[0] - min(outputs[0])/ (max(outputs[0]) - min(outputs[0]))
        predicted = np.argmax(outputs, 1)
        if(outputs[0][predicted] < 0.35):
            predicted = -1
        else:
            predicted = predicted + 1
        predictions.append(int(predicted))
    
      
predictions = np.array(predictions)
predictions = predictions.flatten()
np.save(fname2, predictions)

    
















