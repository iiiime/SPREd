import torch
import torch.nn as nn
import numpy as np

idx = [100] + [i for i in range(100, 1, -1)]
idx = np.cumsum(idx)
idx2 = [i for i in range(5151) if i not in idx]

class Network(nn.Module):
	def __init__(self, in_channels, hidden_unit, n_features, dropout):
		super(Network, self).__init__()
		self.conv = nn.Conv2d(1,1,kernel_size=(5,1))
		self.dropout = nn.Dropout(p=dropout)
		self.bn = nn.BatchNorm1d(5151)
		self.relu = nn.ReLU(inplace=False)
		self.fc1 = nn.Linear(5151, 128)
		self.bn1 = nn.BatchNorm1d(128)
		self.relu1 = nn.ReLU(inplace=False)
		self.fc2 = nn.Linear(128, 100)

		self.fc = nn.Linear(5151, 100)
		self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		out = self.conv(x)
		out = out.view(out.size(0), -1)
		out = self.bn(out)
		out = self.relu(out)

		"""
		out1 = out[:, idx]
		out2 = out[:, idx2]
		out2 = self.dropout(out2)
		out = torch.cat((out1, out2), 1)
		"""
		out = self.fc(out)
		#out = self.sigmoid(out)

		return out
