#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 01:01:06 2020

@author: lokeshkvn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



# CNN 
class CNN(nn.Module):
    def __init__(self, input_channels = 1, num_classes = 10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size = (3,3), stride= (1,1), padding=(1,1))      # same convolution
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2),stride = (2,2))         
        self.conv2 = nn.Conv2d(8, 16, kernel_size = (3,3), stride= (1,1), padding=(1,1))      # same convolution
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2),stride = (2,2))
        self.fc1 = nn.Linear(16*7*7, num_classes)
        
        self.initialize_weights()
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)
        
        return x
    
    def initialize_weights(self):
        for m in self.modules():
            print(m)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight,)
                nn.init.constant_(m.bias, 0)
                
                
model = CNN()
# x = torch.randn(64,1,28,28)
# print(model(x).shape)