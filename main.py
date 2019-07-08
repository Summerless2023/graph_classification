import os
import torch
from torch import nn, optim
from load_data import init_data
from load_data import load_to_dataset
from my_model import GraphClassifier

# 相关配置
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
datadir = "./data"
dataname = "DD"
batch_size = 1
learning_rate = 0.01
num_epoches = 1
class_num = 2
index = 15

if __name__ == '__main__':
	my_graphs, max_node_num1, max_node_num2 = init_data(datadir, dataname)
	train_data, test_data = load_to_dataset(my_graphs, index)
	print("数据处理完成")
	model = GraphClassifier(max_node_num1, max_node_num2)
	model = model.cuda()
	print('model:', model)
	criterion = nn.MSELoss()
	optimizer = optim.SGD(model.parameters(), lr=learning_rate)
	for epoch in range(num_epoches):
		print("第%d轮训练" % epoch)

		for i, tmp in enumerate(train_data):
			torch.cuda.empty_cache()
			print(i)
			input1, input2, adj1, adj2, label = tmp
			del tmp
			output = model.forward(input1[0], input2[0], adj1[0], adj2[0]).cuda()
			cu_label = torch.zeros(batch_size, class_num).scatter_(1, label, 1).cuda()
			print("*" * 40)
			print("label = ", label)
			print('output.size = ', output.size())
			print('output = ', output)
			print('cu_label.size = ', cu_label.size())
			print('cu_label ', cu_label)
			loss = criterion(output, cu_label)
			print_loss = loss.data.item()
			print("loss : ", print_loss)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		acc_num = 0
		for i, tmp in enumerate(test_data):
			input1, input2, adj1, adj2, label = tmp
			output = model.forward(input1[0], input2[0], adj1[0], adj2[0]).cuda()
			_, pre = torch.max(output, dim=1)
			pre = pre.cuda()
			label = label.cuda()
			if pre[0] == label[0]:
				acc_num += 1
		print("accuracy : ", acc_num, "of ", len(test_data))
