# import torch
#
# class_num = 2
# batch_size = 1
# label = torch.LongTensor(1, 1)
# label[0][0] = 1
#
# one_hot = torch.zeros(batch_size, class_num).scatter_(1, label, 1)
# print(one_hot.size())
# print(one_hot)
import torch
import torch.nn as nn

loss = nn.CrossEntropyLoss()
# input, NxC=2x3
input = torch.randn(1, 2, requires_grad=True)
# target, N
target = torch.empty(1, dtype=torch.long)
target[0] = 1
print(input.size())
print(input)
print(target.size())
print(target)
output = loss(input, target)
output.backward()

print(output.data.item())
