import torch

class_num = 2
batch_size = 1
label = torch.LongTensor(1, 1)
label[0][0] = 1

one_hot = torch.zeros(batch_size, class_num).scatter_(1, label, 1)
print(one_hot.size())
print(one_hot)