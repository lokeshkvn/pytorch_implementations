#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 00:05:55 2020

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

input_size = 28
sequences_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

# RNN

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # N x Time_seq x features
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self,x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward Prop
        out, _ = self.lstm(x,(h0,c0))
        # out = out.reshape(out.shape[0],-1)
        out = self.fc(out[:,-1,:])
        return out
        
        
        
        
x = torch.randn(64,28,28)        
model = RNN(input_size, hidden_size, num_layers, num_classes)
print(model(x).shape)




# Load data

train_dataset = datasets.MNIST(root ='dataset/', train= True, transform= transforms.ToTensor(), download=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root ='dataset/', train= False, transform= transforms.ToTensor(), download=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
 
          
# Initialize network

model = RNN(input_size, hidden_size, num_layers, num_classes)


# Loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = learning_rate)


# Train Network

for epochs in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_dataloader):
        data = data.to(device =  device).squeeze(1)
        targets = targets.to(device = device)
        
        # print(data.shape)
        
        
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
            x = x.to(device= device).squeeze(1)
            y = y.to(device = device)
            
            # x = x.reshape(x.shape[0], -1)
            
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
                    