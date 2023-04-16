import torch
import torch.nn as nn
import numpy as np

idx = [100] + [i for i in range(100, 1, -1)]
idx = np.cumsum(idx)
idx_ = [i for i in range(5151) if i not in idx]

class Network(nn.Module):
	def __init__(self, in_channels, hidden_unit, n_features, dropout):
		super(Network, self).__init__()
		self.conv_tftf = nn.Conv2d(1,16,kernel_size=(5,1))
		self.conv_tftf_ = nn.Conv2d(16,1,kernel_size=(1,1))
		self.conv_tfg = nn.Conv2d(1,16,kernel_size=(5,1))
		self.conv_tfg_ = nn.Conv2d(16,1,kernel_size=(1,1))
		self.dropout = nn.Dropout(p=dropout)
		self.dropout_ = nn.Dropout(p=dropout)
		self.bn = nn.BatchNorm1d(5151)
		self.bn_tftf = nn.BatchNorm2d(16)
		self.bn_tfg = nn.BatchNorm2d(16)
		self.bn_tftf_ = nn.BatchNorm2d(1)
		self.bn_tfg_ = nn.BatchNorm2d(1)
		self.relu_tftf = nn.LeakyReLU(inplace=False)
		self.relu_tfg = nn.LeakyReLU(inplace=False)
		self.relu = nn.LeakyReLU(inplace=False)
		self.bn1 = nn.BatchNorm1d(100)
		self.relu1 = nn.LeakyReLU(inplace=False)
		self.fc2 = nn.Linear(200, 100)

		self.fc = nn.Linear(5151, 100)
		self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		out_tftf = x[:,:,:, idx_]
		out_tftf = self.conv_tftf(out_tftf)
		out_tftf = self.bn_tftf(out_tftf)
		out_tftf = self.relu_tftf(out_tftf)
		out_tftf = self.conv_tftf_(out_tftf)
		out_tftf = self.bn_tftf_(out_tftf)
		out_tftf = out_tftf.view(out_tftf.size(0), -1)
		out_tftf = self.dropout(out_tftf)

		out_tfg = x[:,:,:, idx]
		out_tfg = self.conv_tfg(out_tfg)
		out_tfg = self.bn_tfg(out_tfg)
		out_tfg = self.relu_tfg(out_tfg)
		out_tfg = self.conv_tfg_(out_tfg)
		out_tfg = self.bn_tfg_(out_tfg)
		out_tfg = out_tfg.view(out_tfg.size(0), -1)
		out = torch.cat((out_tftf, out_tfg), 1)
		
		out = self.fc(out)
		out = self.bn1(out)

		out = self.dropout_(out)
		inv = x[:, :, 3, idx]
		inv = inv.view(inv.size(0), -1)
		out += inv
		out = self.relu1(out)

		return out
