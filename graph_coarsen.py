import networkx as nx
import torch
from torch_cluster import graclus_cluster
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def get_coarsen_graphs(graph):
	graphs = []
	graphs.append(graph)
	# 第一次粗化
	fir_graph = coarsen(graph)
	graphs.append(fir_graph)
	sec_graph = coarsen(fir_graph)
	graphs.append(sec_graph)
	return graphs


def coarsen(graph):
	# 传入一个networkx图对象，返回他的粗化结果networkx图对象
	# print("粗化前节点数量:", graph.number_of_nodes())
	row = []
	col = []
	for i, j in graph.edges():
		row.append(i)
		col.append(j)
		row.append(j)
		col.append(i)
	cluster = graclus_cluster(torch.tensor(row), torch.tensor(col)).numpy()
	edge_list = []
	for i in range(len(row)):
		if cluster[row[i]] != cluster[col[i]] and edge_list.count((cluster[row[i]], cluster[col[i]])) == 0:
			edge_list.append((cluster[row[i]], cluster[col[i]]))
	del row
	del col
	coarsen_graph = nx.from_edgelist(edge_list)
	# print("粗化后节点数量",coarsen_graph.number_of_nodes())
	return coarsen_graph
