import os
import torch
import random
import numpy as np
from torch import nn, optim
from load_data import init_data
from my_model import GraphClassifier
from utils import handle_graph
import torch.nn.init as init

# 相关配置
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
datadir = "./data"
dataname = "DD"
batch_size = 36
learning_rate = 0.001
num_epoches = 2
class_num = 2
index = int(1178 * 0.7)
# index = 3


if __name__ == '__main__':
	my_graphs, max_node_num1, max_node_num2 = init_data(datadir, dataname)
	random.shuffle(my_graphs)
	print("数据处理完成")
	model = GraphClassifier(max_node_num1, max_node_num2)
	model = model.cuda()
	print('model:', model)
	crossentropy = torch.nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=learning_rate)
	print('开始训练')
	for epoch in range(num_epoches):
		for i in range(len(my_graphs)):
			if i > index:
				break
			torch.cuda.empty_cache()
			input1, adj1 = handle_graph(my_graphs[i].graph1, max_node_num1)
			input2, adj2 = handle_graph(my_graphs[i].graph2, max_node_num2)
			label = my_graphs[i].label
			input1 = torch.from_numpy(input1).float().cuda()
			# print('input1.size = ', input1.size())
			input2 = torch.from_numpy(input2).float().cuda()
			# print('input2.size = ', input2.size())
			adj1 = np.pad(adj1, ((0, max_node_num1 - len(adj1[0])), (0, max_node_num1 - len(adj1[0]))),
			              'constant', constant_values=((0, 0), (0, 0)))
			adj1 = torch.from_numpy(adj1).cuda()
			# print('adj1.size = ', adj1.size())
			adj2 = np.pad(adj2, ((0, max_node_num2 - len(adj2[0])), (0, max_node_num2 - len(adj2[0]))),
			              'constant', constant_values=((0, 0), (0, 0)))
			adj2 = torch.from_numpy(adj2).cuda()
			# print('adj2.size = ', adj2.size())
			output = model.forward(input1, input2, adj1, adj2).cuda()
			# tmp = np.zeros((1, 1), dtype=np.int)
			# tmp[0][0] = label
			# tmp_label = torch.from_numpy(tmp)
			# one_hot = torch.zeros(batch_size, class_num).scatter_(1, tmp_label, 1).cuda()
			# loss = criterion(output, one_hot)
			tmp = np.zeros(1, dtype=np.int)
			tmp[0] = label
			loss = crossentropy(output, torch.from_numpy(tmp).cuda())
			print_loss = loss.data.item()
			print(print_loss)
			if i % batch_size == 0:
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
	print("训练结束")
	acc_num = 0
	test_count = 0
	for i in range(len(my_graphs)):
		if i < index:
			continue
		test_count += 1
		torch.cuda.empty_cache()
		input1, adj1 = handle_graph(my_graphs[i].graph1, max_node_num1)
		input2, adj2 = handle_graph(my_graphs[i].graph2, max_node_num2)
		label = my_graphs[i].label
		input1 = torch.from_numpy(input1).float().cuda()
		# print('input1.size = ', input1.size())
		input2 = torch.from_numpy(input2).float().cuda()
		# print('input2.size = ', input2.size())
		adj1 = np.pad(adj1, ((0, max_node_num1 - len(adj1[0])), (0, max_node_num1 - len(adj1[0]))),
		              'constant', constant_values=((0, 0), (0, 0)))
		adj1 = torch.from_numpy(adj1).cuda()
		# print('adj1.size = ', adj1.size())
		adj2 = np.pad(adj2, ((0, max_node_num2 - len(adj2[0])), (0, max_node_num2 - len(adj2[0]))),
		              'constant', constant_values=((0, 0), (0, 0)))
		adj2 = torch.from_numpy(adj2).cuda()
		# print('adj2.size = ', adj2.size())
		output = model.forward(input1, input2, adj1, adj2).cuda()
		_, pre = torch.max(output, dim=1)
		tmp = np.zeros((1, 1), dtype=np.int)
		tmp[0][0] = label
		tmp_label = torch.from_numpy(tmp)
		pre = pre.cuda()
		tmp_label = tmp_label.cuda()
		if pre[0] == tmp_label[0][0]:
			acc_num += 1
	print("accuracy : ", acc_num, "of ", test_count)
