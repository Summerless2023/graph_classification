import torch
import torch.nn as nn


class Encoder(nn.Module):
	def __init__(self, node_num):
		super(Encoder, self).__init__()
		self.fc1 = nn.Sequential(nn.Linear(node_num, 256), nn.BatchNorm1d(256), nn.ReLU(True))
		self.fc2 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(True))
		self.fc3 = nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(True))

	def forward(self, x):
		x = self.fc1(x)
		x = self.fc2(x)
		x = self.fc3(x)
		return x


class GraphClassifier(nn.Module):
	def __init__(self, node_num1, node_num2):
		super(GraphClassifier, self).__init__()
		self.layer1 = Encoder(node_num1)
		self.layer2 = Encoder(node_num2)
		self.W = nn.Parameter(torch.zeros(1, 1))
		self.alpha1 = nn.Parameter(torch.rand(node_num1, node_num1))
		self.alpha2 = nn.Parameter(torch.rand(node_num2, node_num2))
		self.classfier = nn.Linear((node_num1 + node_num2) * 64, 2)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x1, x2, adj1, adj2):
		x1 = self.layer1(x1)
		x2 = self.layer2(x2)
		new1 = self.self_attention(x1,adj1)
		print("new1 = ",new1)
		print("new1.size()=", new1.size())
		new2 = self.self_attention(x2, adj2)
		print("new2 = ", new2)
		print("new2.size()=", new2.size())
		x = self.classfier(torch.cat((new1, new2), 0))
		print("x.classfier.size():", x.size())
		x = self.softmax(x)
		print("x.size()", x.size())
		return x

	def self_attention(self, x, adj):
		new = []
		for i in range(len(adj[0])):  # 0 - n
			tmp = torch.zeros(1, 64).float().cuda()  # 1*n
			degree1 = adj.sum(dim=1)  # 1*n
			for j in range(len(adj[0])):
				if adj[i][j] == 1:
					print(tmp.size())
					# print("---", self.alpha1[i][j].size(), self.W.size(), x[j].size())
					tmp += self.alpha1[i][j] * self.W * x[j]
			tmp /= degree1[i] * 1.0
			tmp += x[i]
			new.append(tmp)
		new = torch.cat(tuple(new), 0)
		return new