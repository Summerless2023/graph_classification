import os
import re
from Entity import My_graph
import networkx as nx
import numpy as np
from torch.utils.data import DataLoader
from Entity import GraphDataset


# 上采样
def get_max_nodenum(my_graphs):
	max_node_num1 = 0
	max_node_num2 = 0
	for graph in my_graphs:
		max_node_num1 = max(max_node_num1, graph.graph1.number_of_nodes())
		max_node_num2 = max(max_node_num2, graph.graph2.number_of_nodes())
	print("max_node_num1 =", max_node_num1)
	print("max_node_num2 =", max_node_num2)
	return max_node_num1, max_node_num2


# 处理数据集 返回一个networkx 的 graph对象列表
def read_graphfile(datadir, dataname, max_nodes=None):
	prefix = os.path.join(datadir, dataname, dataname)
	filename_graph_indic = prefix + '_graph_indicator.txt'
	# index of graphs that a given node belongs to
	graph_indic = {}
	with open(filename_graph_indic) as f:
		i = 1
		for line in f:
			line = line.strip("\n")
			graph_indic[i] = int(line)
			i += 1

	filename_nodes = prefix + '_node_labels.txt'
	node_labels = []
	try:
		with open(filename_nodes) as f:
			for line in f:
				line = line.strip("\n")
				node_labels += [int(line) - 1]
		num_unique_node_labels = max(node_labels) + 1
	except IOError:
		print('No node labels')

	filename_node_attrs = prefix + '_node_attributes.txt'
	node_attrs = []
	try:
		with open(filename_node_attrs) as f:
			for line in f:
				line = line.strip("\s\n")
				attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
				node_attrs.append(np.array(attrs))
	except IOError:
		print('No node attributes')

	label_has_zero = False
	filename_graphs = prefix + '_graph_labels.txt'
	graph_labels = []

	# assume that all graph labels appear in the dataset
	# (set of labels don't have to be consecutive)
	label_vals = []
	with open(filename_graphs) as f:
		for line in f:
			line = line.strip("\n")
			val = int(line)
			# if val == 0:
			#    label_has_zero = True
			if val not in label_vals:
				label_vals.append(val)
			graph_labels.append(val)
	# graph_labels = np.array(graph_labels)
	label_map_to_int = {val: i for i, val in enumerate(label_vals)}
	graph_labels = np.array([label_map_to_int[l] for l in graph_labels])
	# if label_has_zero:
	#    graph_labels += 1

	filename_adj = prefix + '_A.txt'
	adj_list = {i: [] for i in range(1, len(graph_labels) + 1)}
	index_graph = {i: [] for i in range(1, len(graph_labels) + 1)}
	num_edges = 0
	with open(filename_adj) as f:
		for line in f:
			line = line.strip("\n").split(",")
			e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
			adj_list[graph_indic[e0]].append((e0, e1))
			index_graph[graph_indic[e0]] += [e0, e1]
			num_edges += 1
	for k in index_graph.keys():
		index_graph[k] = [u - 1 for u in set(index_graph[k])]

	graphs = []
	for i in range(1, 1 + len(adj_list)):
		# indexed from 1 here
		G = nx.from_edgelist(adj_list[i])
		if max_nodes is not None and G.number_of_nodes() > max_nodes:
			continue

		# add features and labels
		G.graph['label'] = graph_labels[i - 1]
		for u in G.nodes():
			if len(node_labels) > 0:
				node_label_one_hot = [0] * num_unique_node_labels
				node_label = node_labels[u - 1]
				node_label_one_hot[node_label] = 1
				G.node[u]['label'] = node_label_one_hot
			if len(node_attrs) > 0:
				G.node[u]['feat'] = node_attrs[u - 1]
		if len(node_attrs) > 0:
			G.graph['feat_dim'] = node_attrs[0].shape[0]

		# relabeling
		mapping = {}
		it = 0
		if float(nx.__version__) < 2.0:
			for n in G.nodes():
				mapping[n] = it
				it += 1
		else:
			for n in G.nodes:
				mapping[n] = it
				it += 1

		# indexed from 0
		graphs.append(nx.relabel_nodes(G, mapping))
	return graphs


def init_data(datadir, dataname):
	graphs = read_graphfile(datadir, dataname)
	my_graphs = []
	count = 0
	print("total graph number: ", len(graphs))
	for graph in graphs:
		# if count > 5:
		# 	break
		# else:
		# 	count += 1
		my_graph = My_graph(graph, graph.graph['label'])
		my_graphs.append(my_graph)
	maxn1, maxn2 = get_max_nodenum(my_graphs)
	return my_graphs, maxn1, maxn2


def load_to_dataset(my_graphs):
	max_node_num1, max_node_num2 = get_max_nodenum(my_graphs)
	lap_mats1 = []
	lap_mats2 = []
	adjs1 = []
	adjs2 = []
	labels = []
	_ = 0
	for graph in my_graphs:
		print("正在处理第%d个图" % _)
		_ += 1
		graph.cal_lap(max_node_num1, max_node_num2)
		lap_mats1.append(graph.nor_lap_mat1)
		lap_mats2.append(graph.nor_lap_mat2)
		adjs1.append(graph.adj1)
		adjs2.append(graph.adj2)
		labels.append(graph.label)
	data = DataLoader(dataset=GraphDataset(lap_mats1, lap_mats2, adjs1, adjs2, labels), batch_size=1, shuffle=True)
	return data
