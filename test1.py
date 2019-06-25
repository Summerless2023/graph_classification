from my_model import GraphClassifier
import torch

model = GraphClassifier(3, 4)

x1 = input = torch.randn(3, 3, requires_grad=True).cuda()
x2 = input = torch.randn(4, 4, requires_grad=True).cuda()
adj1 = input = torch.randn(3, 3, requires_grad=True).cuda()
adj2 = input = torch.randn(4, 4, requires_grad=True).cuda()

output = model.forward(x1, x2, adj1, adj2)

print(output.size())
