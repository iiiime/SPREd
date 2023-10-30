import torch
import torch.nn as nn
import numpy as np

idx = [100] + [i for i in range(100, 1, -1)]
idx = np.cumsum(idx)
idx2 = [i for i in range(5151) if i not in idx]

class Network(nn.Module):
	"""multi-label nn model
	"""
	def __init__(self, in_channels, hidden_unit, n_features, dropout):
		super(Network, self).__init__()
		self.conv = nn.Conv2d(1,1,kernel_size=(5,1))
		self.dropout = nn.Dropout(p=dropout)
		self.bn = nn.BatchNorm1d(in_channels)
		self.fc1 = nn.Linear(in_channels, hidden_unit)
		self.relu = nn.ReLU(inplace=False)
		self.bn2 = nn.BatchNorm1d(hidden_unit)
		self.relu2 = nn.ReLU(inplace=False)
		self.fc2 = nn.Linear(hidden_unit, hidden_unit)
		self.bn3 = nn.BatchNorm1d(hidden_unit)
		self.relu3 = nn.ReLU(inplace=False)
		self.fc3 = nn.Linear(hidden_unit, n_features)
		self.fc = nn.Linear(in_channels, n_features)
		self.dropout2 = nn.Dropout(p=0.3)


	def forward(self, x):
		out = self.conv(x)
		out = out.view(out.size(0), -1)
		out = self.bn(out)
		out = self.relu(out)

		out1 = out[:, idx]
		out2 = out[:, idx2]
		out2 = self.dropout(out2)
		out = torch.cat((out1, out2), 1)

		out = self.fc(out)
		return out
