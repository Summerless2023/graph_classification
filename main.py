import os
import torch
from torch import nn, optim
from load_data import init_data
from load_data import load_to_dataset
from my_model import GraphClassifier
from utils import handle_graph

# 相关配置
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
datadir = "./data"
dataname = "DD"
batch_size = 1
learning_rate = 0.01
num_epoches = 1
class_num = 2
index = int(1178 * 0.7)

if __name__ == '__main__':
	my_graphs, max_node_num1, max_node_num2 = init_data(datadir, dataname)
	print("数据处理完成")
	model = GraphClassifier(max_node_num1, max_node_num2)
	model = model.cuda()
	print('model:', model)
	criterion = nn.MSELoss()
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
			input1 = torch.from_numpy(input1).cuda()
			print('input1.size = ', input1.size())
			input2 = torch.from_numpy(input2).cuda()
			print('input2.size = ', input2.size())
			adj1 = torch.from_numpy(adj1).cuda()
			print('adj1.size = ', adj1.size())
			adj2 = torch.from_numpy(adj2).cuda()
			print('adj2.size = ', adj2.size())
			print('label = ', label)
			output = model.forward(input1, input2, adj1, adj2).cuda()
			cu_label = torch.zeros(batch_size, class_num).scatter_(1, label, 1).cuda()
			loss = criterion(output, cu_label)
			print_loss = loss.data.item()
			print("loss : ", print_loss)
			if i % 5 == 0:
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
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
			input1 = torch.from_numpy(input1).cuda()
			print('input1.size = ', input1.size())
			input2 = torch.from_numpy(input2).cuda()
			print('input2.size = ', input2.size())
			adj1 = torch.from_numpy(adj1).cuda()
			print('adj1.size = ', adj1.size())
			adj2 = torch.from_numpy(adj2).cuda()
			print('adj2.size = ', adj2.size())
			print('label.size = ', label.size())
			print('label = ', label)
			output = model.forward(input1, input2, adj1, adj2).cuda()
			_, pre = torch.max(output, dim=1)
			pre = pre.cuda()
			label = label.cuda()
			if pre[0] == label[0]:
				acc_num += 1
		print("accuracy : ", acc_num, "of ", test_count)
