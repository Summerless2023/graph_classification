# import torch
#
# x1 = input = torch.randn(1, 3, requires_grad=True).cuda()
# print(x1)
# x, y = torch.max(x1, dim=1)
# print(x)
# print(y)
#
# z, d = torch.max(x1, dim=0)
# print(z)
# print(d)
import random

a = [1, 2, 3, 4]

random.shuffle(a)
print(a)
