#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 19:23:56 2020

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
import torchvision
   
# device setup 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters setup

input_channel = 3
num_classes = 10
learning_rate = 0.001
batch_size = 1024
num_epochs = 5
load_model = True


# Load pretrained model



class Identity(nn.Module):
    def __init__(self):
        super(Identity,self).__init__()
        
    def forward(self,x):
        return x

model = torchvision.models.vgg16(pretrained = True)

for param in model.parameters():
    param.requires_grad = False

model.avgpool = Identity()
model.classifier= nn.Sequential(nn.Linear(512, 100), nn.ReLU(),
                   nn.Linear(100, 10))

print(model)

def save_checkpoint(state, filename = "my_ckpt.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    

def load_checkpoint(state):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
 
# Load data

train_dataset = datasets.CIFAR10(root ='dataset/', train= True, transform= transforms.ToTensor(), download=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.CIFAR10(root ='dataset/', train= False, transform= transforms.ToTensor(), download=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
 
          

# Loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = learning_rate)

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
        
    print(f"Cost at epoch {epochs} is {sum(losses)/len(losses):.5f}")

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
                    