#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 20:50:50 2020

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


# Fully connected neural network

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = NN(784, 10)
x = torch.randn(64, 784)

print(model(x).shape)


# device setup 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters setup

input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1


# Load data

train_dataset = datasets.MNIST(root ='dataset/', train= True, transform= transforms.ToTensor(), download=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root ='dataset/', train= False, transform= transforms.ToTensor(), download=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
 
          
# Initialize network

model = NN(input_size = input_size, num_classes=num_classes).to(device)  


# Loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = learning_rate)


# Train Network

for epochs in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_dataloader):
        data = data.to(device =  device)
        targets = targets.to(device = device)
        
        # print(data.shape)
        
        data = data.reshape(data.shape[0], -1)
        
        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient descent
        optimizer.step()
        

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
            
            x = x.reshape(x.shape[0], -1)
            
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
                    