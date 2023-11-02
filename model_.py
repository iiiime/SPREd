import torch
import torch.nn as nn
import numpy as np


class Network(nn.Module):
	"""simplest multi-output NN
	"""
	def __init__(self, hidden_unit, n_features, dropout):
		super(Network, self).__init__()
		self.n_tf = n_features
		self.idx_tfg = n_features
		self.idx_tf = [i for i in range(n_features)]
		self.conv_tf = nn.Sequential(
			nn.Conv2d(1,16,(5,1)),
			nn.BatchNorm2d(16),
			nn.LeakyReLU(inplace=False),
			nn.Conv2d(16,1,(1,1)),
			nn.Dropout(0.3),
		)

		self.conv_tfg = nn.Sequential(
			nn.Conv2d(1,16,(5,1)),
			nn.BatchNorm2d(16),
			nn.LeakyReLU(inplace=False),
			nn.Conv2d(16,1,(1,1)),
		)

		self.dense = nn.Sequential(
			nn.BatchNorm1d(self.n_tf + 1),
			nn.LeakyReLU(inplace=False),
			nn.Linear(self.n_tf + 1, 128),
			nn.BatchNorm1d(128),
			nn.LeakyReLU(inplace=False),
			nn.Dropout(0.5),
		)

		self.output = nn.Sequential(
			nn.Linear(128, 1)
			# nn.Sigmoid()
		)


	def forward(self, x):
		out_tf = x[:,:,:, self.idx_tf]
		out_tfg = x[:,:,:, self.idx_tfg].unsqueeze(-1)
		out_tf = self.conv_tf(out_tf)
		out_tf = out_tf.view(out_tf.size(0), -1)
		out_tfg = self.conv_tfg(out_tfg)
		out_tfg = out_tfg.view(out_tfg.size(0), -1)
		out = torch.cat((out_tf, out_tfg), 1)

		out = self.dense(out)
		return self.output(out)
