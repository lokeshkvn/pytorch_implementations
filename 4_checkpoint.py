#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 14:11:21 2020

@author: lokeshkvn
"""


# Imports

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

   
# device setup 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters setup

input_channel = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5
load_model = True


# CNN 
class CNN(nn.Module):
    def __init__(self, input_channels = 1, num_classes = 10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size = (3,3), stride= (1,1), padding=(1,1))      # same convolution
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2),stride = (2,2))         
        self.conv2 = nn.Conv2d(8, 16, kernel_size = (3,3), stride= (1,1), padding=(1,1))      # same convolution
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2),stride = (2,2))
        self.fc1 = nn.Linear(16*7*7, num_classes)
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)
        
        return x

model = CNN()
x = torch.randn(64,1,28,28)
print(model(x).shape)



def save_checkpoint(state, filename = "my_ckpt.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    

def load_checkpoint(state):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
 
# Load data

train_dataset = datasets.MNIST(root ='dataset/', train= True, transform= transforms.ToTensor(), download=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root ='dataset/', train= False, transform= transforms.ToTensor(), download=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
 
          
# Initialize network

model = CNN().to(device)  


# Loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = learning_rate)

if load_model:
    load_checkpoint(torch.load("my_ckpt.pth.tar"))

# Train Network

for epochs in range(num_epochs):
    losses = []
    
    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    if epochs == 2:    
        save_checkpoint(checkpoint)
    for batch_idx, (data, targets) in enumerate(train_dataloader):
        data = data.to(device =  device)
        targets = targets.to(device = device)
        
        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient descent
        optimizer.step()
        
    print(f' Loss at {epochs}/{num_epochs} is {loss.item()}')

# Check Accuracy and test

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on testing data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device= device)
            y = y.to(device = device)
            
            
            scores = model(x)
            _ , predictions = scores.max(1)
            
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            # acc = float(num_correct)/float(num_samples)*100
            
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
    
    model.train()
    # return acc

check_accuracy(train_dataloader, model)
check_accuracy(test_dataloader, model)
                    