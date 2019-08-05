import torch
import numpy as np
from my_model import Encoder
import torch.nn as nn

# from torch_cluster import graclus_cluster
#
# row = torch.tensor([0, 1, 1, 2,3,4,5])
# col = torch.tensor([1, 0, 2, 1,4,5,1])
# cluster = graclus_cluster(row, col)
# print(cluster)
a = torch.zeros(1, 100, 100)
model = Encoder(100)

# # print(model.forward(a).size())
# x = np.zeros((2, 3))
# x[1][1] = 1
# x[1][2] = 1
# print(x)
# #y = np.random.rand(5, 6)
# #print(y)
# print("---")
# print(x.sum(axis=0))
# x = torch.zeros((2, 3))
# x[1][1] = 1
# x[1][2] = 1
#
# a = [x, x, x]
#
# print(torch.cat(tuple(a), 0))
# m = nn.Softmax(dim=0)
# a = torch.zeros((1, 3))
# b = torch.ones((1, 3))
# b[0][1] = 3
# a[0][0] = 1
# a[0][1] = 1
# a[0][2] = 2
# i, j = torch.max(a, dim=1)
# print(i)
# print(j)
#
#
# print(b[0][1] * a)
#
# z = nn.Parameter(torch.rand(3, 3))
# w = nn.Parameter(torch.zeros(1, 1))
# y = torch.tensor(3)
#
# print((z[1][1] * w * y)/1)
import numpy as np

# x = np.empty([3,2], dtype = int)
# x = torch.Tensor([[1, 2], [3, 4], [5, 6]])
# print(x)
#
# import torch.nn.functional as F
# y = F.pad(x,(0,0,0,0),value=0)
# print(y)
# A = np.arange(95,99).reshape(2,2)
# print(A)

import torch
import torch.nn as nn

loss = nn.CrossEntropyLoss()

# input, NxC=2x3

input = torch.randn((1, 2), requires_grad=True)
print(input.size())
print(input)
softmax = nn.Softmax(dim=1)
output = softmax(input)
print(output.size())
print(output)
