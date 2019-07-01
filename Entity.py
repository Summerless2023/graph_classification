import numpy as np
import torch
from torch.utils.data import Dataset
from graph_coarsen import coarsen
from utils import handle_graph
from utils import reshape_graph
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class My_graph(object):
	def __init__(self, networkx_work, label):
		self.graph1 = networkx_work
		self.graph2 = reshape_graph(coarsen(self.graph1))
		self.label = label
		self.max_node_number1 = 1
		self.max_node_number2 = 1
		self.nor_lap_mat1 = None
		self.nor_lap_mat2 = None
		self.adj1 = None
		self.adj2 = None

	def cal_lap(self, n1, n2):
		self.max_node_number1 = n1
		self.max_node_number2 = n2
		self.nor_lap_mat1, self.adj1 = handle_graph(self.graph1, self.max_node_number1)
		self.nor_lap_mat2, self.adj2 = handle_graph(self.graph2, self.max_node_number2)


class GraphDataset(Dataset):
	def __init__(self, laps1, laps2, adjs1, adjs2, labels):
		self.lap_mats1 = laps1
		self.lap_mats2 = laps2
		self.adjs1 = adjs1
		self.adjs2 = adjs2
		self.labels = labels

	def __getitem__(self, index):
		label = self.labels[index]
		tmp_label = np.zeros(1, dtype=np.int)
		tmp_label[0] = label
		return (torch.from_numpy(self.lap_mats1[index]).float().cuda(),
		        torch.from_numpy(self.lap_mats2[index]).float().cuda(),
		        torch.from_numpy(self.adjs1[index]).cuda(),
		        torch.from_numpy(self.adjs2[index]).cuda(),
		        torch.from_numpy(tmp_label).float().cuda())

	def __len__(self):
		return len(self.lap_mats1)
