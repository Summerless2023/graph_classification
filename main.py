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
num_epoches = 200

if __name__ == '__main__':
	my_graphs, max_node_num1, max_node_num2 = init_data(datadir, dataname)
	data = load_to_dataset(my_graphs)
	print("数据处理完成")
	model = GraphClassifier(max_node_num1, max_node_num2)
	model = model.cuda()
	print("moved model to gpu")

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
			_, pre_label = torch.max(output, 1)
			loss = criterion(pre_label, label)
			print_loss = loss.data.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			print(print_loss)
	# 验证部分
	model.eval()
	eval_loss = 0
	eval_acc = 0
	for tmp in data:
		img, label = tmp
		img = img.view(img.size(0), -1)
		if torch.cuda.is_available():
			img = img.cuda()
			label = label.cuda()

		out = model(img)
		loss = criterion(out, label)
		eval_loss += loss.data.item() * label.size(0)
		_, pred = torch.max(out, 1)
		num_correct = (pred == label).sum()
		eval_acc += num_correct.item()
	print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
		eval_loss / (len(data)),
		eval_acc / (len(data))
	))
