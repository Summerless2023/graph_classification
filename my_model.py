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
		# --------1-------------#
		new1 = []
		for i in range(len(adj1[0])):
			tmp = torch.zeros(len(adj1[0])).float()
			degree1 = adj1.sum(dim=1)
			for j in range(len(adj1[0])):
				if adj1[i][j] == 1:
					print("---", self.alpha1[i][j].size(), self.W.size(), x1[j].size())
					tmp += self.alpha1[i][j] * self.W * x1[j]
			tmp /= degree1[i] * 1.0
			new1.append(tmp)
		new1 = torch.cat(tuple(new1), 0)
		# --------2-------------#
		new2 = []
		for i in range(len(adj2[0])):
			tmp = torch.zeros(len(adj2[0])).float()
			degree2 = adj2.sum(axis=1)
			for j in range(len(adj2[0])):
				if adj2[i][j] == 1:
					tmp += self.alpha2[i][j] * self.W * x2[j]
			tmp /= degree2[i] * 1.0
			new2.append(tmp)
		new2 = torch.cat(tuple(new2), 0)

		x = self.classfier(torch.cat((new1, new2), 0))
		x = self.softmax(x)
		print("x.size()", x.size())
		return x
