import os
import torch
from torch import nn, optim
from load_data import init_data
from load_data import load_to_dataset
from my_model import GraphClassifier

# 相关配置
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
datadir = "./data"
dataname = "DD"
batch_size = 1
learning_rate = 0.01
num_epoches = 200

if __name__ == '__main__':
	my_graphs, max_node_num1, max_node_num2 = init_data(datadir, dataname)
	data = load_to_dataset(my_graphs)
	print("数据处理完成")
	model = GraphClassifier(max_node_num1, max_node_num2)
	model = model.cuda()
	print('model:', model)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=learning_rate)
	for epoch in range(num_epoches):
		print("第%d轮训练" % epoch)
		for i, tmp in enumerate(data):
			torch.cuda.empty_cache()
			print(i)
			input1, input2, adj1, adj2, label = tmp
			del tmp
			output = model.forward(input1[0], input2[0], adj1[0], adj2[0])
			print('output:',output)