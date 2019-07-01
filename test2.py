import torch

class_num = 10
batch_size = 4
label = torch.LongTensor(batch_size, 1).random_() % class_num
print(label)
print(label.size())

one_hot = torch.zeros(batch_size, class_num).scatter_(1, label, 1)
print(one_hot)
print(one_hot.size())
