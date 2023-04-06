import torch
import torch.nn as nn
import numpy as np

idx = [100] + [i for i in range(100, 1, -1)]
idx = np.cumsum(idx)
idx2 = [i for i in range(5151) if i not in idx]

class Network(nn.Module):
	"""simplest multi-output NN
	"""
	def __init__(self, in_channels, hidden_unit, n_features, dropout):
		super(Network, self).__init__()
		"""
		a = [-3. + 3/4 * i for i in range(9)]
		b = [(a[i] + a[(i+1)])/2 for i in range(len(a) - 1)]
		kernel = [[b[i] * b[j] for i in range(len(b))] for j in range(len(b))]
		self.conv = nn.Conv2d(1, 2, 8, 8, bias=False)
		with torch.no_grad():
			self.conv.weight = nn.Parameter(torch.DoubleTensor([[kernel],[torch.rand(8,8)]]))
		self.conv.weight.data = torch.Tensor([[kernel],[torch.rand(8,8)]])"""
		#self.conv = nn.Conv2d(1,1,8,8)
		self.conv = nn.Conv2d(1,1,kernel_size=(5,1))
		#torch.nn.init.uniform_(self.conv.weight, -10, 10)
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
		#print(x.shape)
		out = self.conv(x)
		#print(self.conv.weight.shape)
		#out = self.dropout(out)
		#print(out.shape)
		out = out.view(out.size(0), -1)
		#print(out.shape)
		out = self.bn(out)
		out = self.relu(out)

		out1 = out[:, idx]
		out2 = out[:, idx2]
		out2 = self.dropout(out2)
		out = torch.cat((out1, out2), 1)

		#out = self.fc1(out)
		#out = self.bn2(out)
		#out = self.relu2(out)
		#out = self.dropout2(out)
		#out = self.fc2(out)
		#out = self.bn3(out)
		#out = self.relu3(out)

		#out = self.fc3(out)
		out = self.fc(out)
		return out
