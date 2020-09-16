#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 19:06:24 2020

@author: lokeshkvn
"""

################################### Indexing ############

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor =  torch.tensor([[1,2,3],[4,5,6]], 
                          dtype = torch.float32, 
                          device = device,
                          requires_grad = True);

print(my_tensor.shape)
print(my_tensor.device)
print(my_tensor.dtype)
print(my_tensor.requires_grad)


############# Other intialization methods

x = torch.empty(size = (3,3))

x = torch.zeros((3,3))

x = torch.rand((3,3))

x = torch.ones((3,3))

x = torch.eye(5,5)  # Identity matrix

x = torch.arange(start=0, end =5, step=1)

x = torch.linspace(start= 0.1, end = 1, steps=20)

x =  torch.empty(size= (1,5)).normal_(mean=0, std=1)

x =  torch.empty(size= (1,5)).uniform_(0, 1)

x = torch.diag(torch.ones(3))

print(x)

############### Initialization and covertion 

tensor = torch.arange(0, 4)

print(tensor.bool())

print(tensor.short())

print(tensor.long())

print(tensor.half())

print(tensor.float())

print(tensor.double())

################  NUmpy to Torch tensor

import numpy as np

np_array = np.zeros((3,3))

tensor = torch.from_numpy(np_array)

np_array_back = tensor.numpy()


################ Maths for tensor

x = torch.tensor([1,2,3])

y = torch.tensor([9,78,20])

z1 = torch.empty((3))

torch.add(x,y,out=z1)

z = x + y


###### substraction

z  = x - y 

########## Division

z = torch.true_divide(x, y)


######### Inplace operations


t = torch.zeros(3)
t.add_(x)

t += x   # != (t  = t +x)

######## Exponent

z = x.pow(2)

z = x **2

##### Campare

z = x > 1

Z = x < 1

###### Matrix multiplication

x1 =  torch.rand((2,3))

x2 =  torch.rand((3,5))

x3 =  torch.mm(x1,x2)

x3 = x1.mm(x2)


########## Matric exponent

matrix_exp = torch.rand(4,4)

z = matrix_exp.matrix_power(3)

############ Element wise 

z = x*y

######## dot product

z = torch.dot(x, y)

####### Batch Matrix multi

batch = 32

n = 10
m = 20
p = 30

layer1 = torch.rand((batch,n,m))
layer2 = torch.rand((batch,m,p))

out  = torch.bmm(layer1, layer2)  # (batch,n,p)


######## Broadcasting

x1 = torch.rand((5,5))
x2 = torch.rand((1,5))

z = x1 - x2

z = x1 ** x2

######### other operations

x = torch.rand((3,3))

sum_x = torch.sum(x, dim = 1)

values, indices = torch.max(x, dim = 0)

values, indices = torch.min(x, dim = 0)

abs_X = torch.abs(x)

argmax_X = torch.argmax(x, dim=0)

argmin_X = torch.argmin(x, dim=1)

mean_x = torch.mean(x.float(), dim= 1)

eq_X = torch.eq(x,y)

sorted, indices = torch.sort(x, dim= 0, descending=True)

########### Clamp

z = torch.clamp (x, min = 0.3, max= 0.7)

x = torch.tensor([1,0,111,1,0], dtype = torch.bool)

z = torch.any(x)

z = torch.all(x)



#################################    Indexing


batch_size = 16

features = 25

x = torch.rand((batch_size, features))

print(x[0].shape)

print(x[:,0].shape)

print(x[2,:10])

x [0,0 ] =100

########### fancy indexing

x = torch.arange(10)

indixes = [1,4,6]

print(x[indixes])

x = torch.rand((2,5))

rows = [1,0]
cols = [4,1]
 
print(x[rows,cols])


########### Advanced indexing

X = torch.arange(10)

print(X[(X > 2) & (X < 8)])

print(X[X.remainder(2) == 0])


####### Useful

print(torch.where( X >5, X, X*5))

print( torch.tensor([0,0,1,1,2,3,4]).unique())

print(x.ndimension())

print(X.numel())


################################### Reshape


x = torch.arange(0, 9)

x_3x3 = x.view(3,3)
print(x_3x3)

x_3x3 = x.reshape(3,3)


y = x_3x3.t()

print(y.contiguous().view(9))


########## concat

x1 = torch.rand((2,5))

x2 = torch.rand((2,5))

print(torch.cat((x1,x2),dim=0))

print(torch.cat((x1,x2),dim=1))

z = x1.view(-1)


x = torch.rand((batch_size, n, m))

z = x.view(batch_size, -1)


z = x.permute(0,2,1)  ## numpy transpose

x = torch.arange(10)

print(x.unsqueeze(0).shape)  ## new dimension 
print(x.unsqueeze(1).shape)

x = torch.arange(10).unsqueeze(0).unsqueeze(1)

x = x.squeeze(1)

