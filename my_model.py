import torch
import torch.nn as nn
import time


class Encoder(nn.Module):
	def __init__(self, node_num):
		super(Encoder, self).__init__()
		self.fc1 = nn.Sequential(nn.Linear(node_num, 256), nn.ReLU(True))
		self.fc2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(True))
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
		self.alpha1 = nn.Parameter(torch.rand(node_num1, node_num1), requires_grad=True)
		self.alpha2 = nn.Parameter(torch.rand(node_num2, node_num2), requires_grad=True)
		self.classfier = nn.Linear((node_num1 + node_num2) * 64, 2).cuda()
		self.softmax = nn.Softmax(dim=1).cuda()

	def forward(self, x1, x2, adj1, adj2):
		# print("1.encoder !            ", time.asctime(time.localtime(time.time())))
		x1 = self.layer1(x1)
		x2 = self.layer2(x2)
		# print("1.encoder finished !   ", time.asctime(time.localtime(time.time())))
		x1 = self.self_attention(x1, adj1, self.alpha1)
		x2 = self.self_attention(x2, adj2, self.alpha2)
		# print("2.attention finished ! ", time.asctime(time.localtime(time.time())))
		new_fea = torch.cat((x1, x2)).view(1, -1)
		x = self.classfier(new_fea)
		# x = self.softmax(x)
		# print("3.forward finished !   ", time.asctime(time.localtime(time.time())))
		return x

	def self_attention(self, x, adj, alpha):
		new = []
		len_adj = len(adj[0])
		degree = adj.sum(dim=1)
		for i in range(len_adj):  # 0 - n
			tmp = torch.zeros(1, 64, requires_grad=False).float().cuda()  # 1*n
			if degree[i] != 0:
				for j in range(len_adj):
					if adj[i][j] == 1:
						tmp += alpha[i][j] * x[j]
				# tmp /= degree[i] * 1.0
				tmp = torch.div(tmp, degree[i])
				# tmp += x[i]
				tmp = torch.add(tmp, x[i])
			new.append(tmp)
		new = torch.cat(tuple(new), 0)
		return new
