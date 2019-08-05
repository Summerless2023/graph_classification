import math
import networkx as nx
import numpy as np
import os
import torch


def handle_graph(graph, max_node_num):
	node_num = graph.number_of_nodes()
	adj_mat = np.zeros([node_num, node_num], dtype=np.int)
	degree_mat = np.zeros([node_num, node_num], dtype=np.int)
	nor_lap_mat = np.zeros([max_node_num, max_node_num], dtype=np.float)
	# adj_mat = torch.zeros([node_num, node_num]).int().cuda()
	# degree_mat = torch.zeros([node_num, node_num]).int().cuda()
	# nor_lap_mat = torch.zeros([max_node_num, max_node_num]).int().cuda()

	# 计算图的邻接矩阵
	for s, t in graph.edges():
		adj_mat[s][t] = 1
		adj_mat[t][s] = 1
	# 计算图度矩阵-对角矩阵
	__tmp = adj_mat.sum(axis=0)
	for i in range(len(__tmp)):
		degree_mat[i][i] = __tmp[i]

	# 计算图的归一化拉普拉斯矩阵
	for i in range(node_num):
		for j in range(node_num):
			if i == j and degree_mat[i][i] != 0:
				nor_lap_mat[i][j] = 1
			elif i != j and adj_mat[i][j] == 1:
				nor_lap_mat[i][j] = -(1.0 / math.sqrt(degree_mat[i][i] * degree_mat[j][j]))
			else:
				nor_lap_mat[i][j] = 0

	return nor_lap_mat, adj_mat


def reshape_graph(graph):
	new_edges = []
	mapping = {}
	count = 0
	for s, t in graph.edges():
		if s not in mapping:
			mapping[s] = count
			count += 1
		new_s = mapping[s]

		if t not in mapping:
			mapping[t] = count
			count += 1
		new_t = mapping[t]
		if new_edges.count((new_s, new_t)) == 0:
			new_edges.append((new_s, new_t))

	return nx.from_edgelist(new_edges)
