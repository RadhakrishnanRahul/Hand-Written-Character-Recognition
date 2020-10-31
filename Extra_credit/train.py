# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 19:23:58 2019

@author: sragh
"""

    
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
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
      
def load_pkl(fname):
	with open(fname,'rb') as f:
		return pickle.load(f)

def save_pkl(fname,obj):
	with open(fname,'wb') as f:
		pickle.dump(obj,f)
max_x = 0
max_y= 0
''' Hyperparameters '''
im_h = 50
im_w = 50
batch_size = 6
epochs = 6
learning_rate = 0.1
c1 = 4 #Number of feature maps in the first convolution layer
c2 = 16 #Number of feature maps in the second convolution layer
''' The remaining hyperparameters are defined in other parts of the programcode '''
''' load the images '''
data = load_pkl('train_data.pkl')
''' convert to numpy array '''
data = np.array([np.array(i)/1.00 for i in data])
'''load the labels'''
labels = np.load('finalLabelsTrain.npy')
# Convert labels to Binary representation
lb = preprocessing.LabelBinarizer()
lb.fit([1,2,3,4,5,6,7,8])

''' Remove rotated and bad images from the training set '''
mask = np.ones(len(data),dtype = bool)
mask[[i for i in range(960,1040)]] = False
mask[[i for i in range(2000,2080)]] = False
mask[[i for i in range(5371,5440)]] = False
mask[[1609,3330,4326,5722,5724,5725,5734]] = False
data= data[mask == True]
labels = labels[mask == True]
classes = ['a','b','c','d','h','i','j','k']    
''' Resize and convert to grayscale '''
for i in range(data.shape[0]):
    temp = data[i]
    temp = resize(temp,[im_h,im_w])
    temp = rgb2gray(temp)
    data[i] = temp


#train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

''' Define the Neural Network '''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, c1, 5) # First Convolution Layer
        self.pool = nn.MaxPool2d(2, 2) # Max pool 
        self.conv2 = nn.Conv2d(c1, c2, 5) # Second Convolution Layer
        self.fc1 = nn.Linear(c2 * 9 * 9, 120) # Fully connected Layer1
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
''' Loss criterion and optimizer '''
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

#for train_index, test_index in kf.split(data):
train_data, test_data, train_labels, test_labels = train_test_split(data,labels, test_size=0.25, random_state=42)
#train_data,test_data = data[train_index],data[test_index]
#train_labels, test_labels = labels[train_index], labels[test_index]
data_size = len(train_data)
data_size -= data_size%batch_size
train_data = train_data[:data_size]
train_labels = train_labels[:data_size]
''' Instantiate Data loaders for images and labels '''

ImageLoader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                          shuffle=False, num_workers=0)
LabelLoader = torch.utils.data.DataLoader(train_labels, batch_size=batch_size,
                                          shuffle=False, num_workers=0)
optimizer.zero_grad()


''' Train the Neural Network '''
print('Starting the training process')
for epoch in range(epochs):  # loop over the dataset multiple times
    print('epoch:' + str(epoch))
    labeliter = iter(LabelLoader)
    running_loss = 0.0
    for i, inputs in enumerate(ImageLoader, 0):
        inputs = inputs.view(batch_size,1,im_h,im_w)
        label = labeliter.next()
        label = lb.transform(label)
        label = torch.from_numpy(label)
        ''' zero the parameter gradients '''
        optimizer.zero_grad()

        ''' forward + backward + optimize '''
        ''' We are using pytorch's autograd to save gradients of each parameter '''
        outputs = net(inputs.float())
        loss = criterion(outputs,label.float())
        loss.backward()
        optimizer.step()



print('Finished Training:')

''' Calculate error on training data '''
ImageLoader = torch.utils.data.DataLoader(train_data, batch_size=1,
                                          shuffle=False, num_workers=0)
LabelLoader = torch.utils.data.DataLoader(train_labels, batch_size=1,
                                          shuffle=False, num_workers=0)
correct = 0
total = 0

class_correct = list(0. for i in range(8))
class_total = list(0. for i in range(8))
with torch.no_grad():
    labeliter = iter(LabelLoader)
    for images in ImageLoader:
        images = images.view(1,1,im_h,im_w)
        outputs = net(images.float())
        _, predicted = torch.max(outputs, 1)
        label = labeliter.next()
        label = int(label)
        total += 1
        label = label-1
        if ((predicted) == label):
            correct += 1
            class_correct[label] += 1
        class_total[label] += 1

print('Accuracy of the network on the training images: %d %%' % (
    100 * correct / total))
for i in range(8):
    print('Accuracy of '+classes[i]+' is: ')
    print(class_correct[i]/class_total[i])

''' Cross Validate '''

ImageLoader = torch.utils.data.DataLoader(test_data, batch_size=1,
                                          shuffle=False, num_workers=0)
LabelLoader = torch.utils.data.DataLoader(test_labels, batch_size=1,
                                          shuffle=False, num_workers=0)
correct = 0
total = 0

class_correct = list(0. for i in range(8))
class_total = list(0. for i in range(8))
predictions = []
with torch.no_grad():
    labeliter = iter(LabelLoader)
    for images in ImageLoader:
        images = images.view(1,1,im_h,im_w)
        outputs = net(images.float())
        _, predicted = torch.max(outputs, 1)
        label = labeliter.next()
        label = int(label)
        predictions.append(int(predicted) + 1)
        total += 1
        label = label-1
        if ((predicted) == label):
            correct += 1
            class_correct[label] += 1
        class_total[label] += 1

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
for i in range(8):
    print('Accuracy of '+classes[i]+' is: ')
    print(class_correct[i]/class_total[i])



''' Save the trained network weights '''
torch.save(net.state_dict(),'trained_model.pt')


