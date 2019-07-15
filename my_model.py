import torch
import torch.nn as nn


class Encoder(nn.Module):
	def __init__(self, node_num):
		super(Encoder, self).__init__()
		self.fc1 = nn.Linear(node_num, 256)
		self.fc2 = nn.Linear(256, 128)
		self.fc3 = nn.Linear(128, 64)
		self.relu = nn.ReLU(True)

	# self.sigmod = nn.Sigmoid()

	def forward(self, x):
		x = self.fc1(x).cuda()
		x = self.fc2(x).cuda()
		x = self.fc3(x).cuda()
		# x = self.sigmod(x).cuda()
		x = self.relu(x)
		return x


class GraphClassifier(nn.Module):
	def __init__(self, node_num1, node_num2):
		super(GraphClassifier, self).__init__()
		self.layer1 = Encoder(node_num1).cuda()
		self.layer2 = Encoder(node_num2).cuda()
		self.W = nn.Parameter(torch.rand(1, 1), requires_grad=True).cuda()
		nn.init.normal_(self.W)
		self.alpha1 = nn.Parameter(torch.rand(node_num1, node_num1), requires_grad=True).cuda()
		nn.init.normal_(self.alpha1)
		self.alpha2 = nn.Parameter(torch.rand(node_num2, node_num2), requires_grad=True).cuda()
		nn.init.normal_(self.alpha2)
		self.classfier = nn.Linear((node_num1 + node_num2) * 64, 2).cuda()
		self.softmax = nn.Softmax(dim=1).cuda()

	def forward(self, x1, x2, adj1, adj2):
		x1 = self.layer1(x1).cuda()
		x2 = self.layer2(x2).cuda()
		new1 = self.self_attention(x1, adj1).cuda()
		new2 = self.self_attention(x2, adj2).cuda()
		new_fea = torch.cat((new1, new2)).view(1, -1).cuda()
		x = self.classfier(new_fea).cuda()
		# print("x.size1 = ", x.size())
		#print("x = ", x)
		# x = self.softmax(x).cuda()
		# print("x.size2 = ", x.size())
		# print("x = ", x)
		return x

	def self_attention(self, x, adj):
		new = []
		for i in range(len(adj[0])):  # 0 - n
			tmp = torch.zeros(1, 64).float().cuda()  # 1*n
			degree1 = adj.sum(dim=1).cuda()  # 1*n
			if degree1[i] != 0:
				for j in range(len(adj[0])):
					if adj[i][j] == 1:
						tmp += self.alpha1[i][j] * self.W * x[j]
				tmp /= degree1[i] * 1.0
				tmp += x[i]
			new.append(tmp)
		new = torch.cat(tuple(new), 0).cuda()
		return new
