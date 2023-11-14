import os
import sys
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import logging
import argparse
import sklearn.metrics
import matplotlib.pyplot as plt
import random


import utils
from model_ import Network
from torch.autograd import Variable

n_features = 100
n_bins = 8

a = [-3. + 3/4 * i for i in range(9)]
b = [(a[i] + a[(i+1)])/2 for i in range(len(a) - 1)]
kernel = [[[[b[i] * b[j] for i in range(len(b))] for j in range(len(b))]]]
opt_kernel = torch.FloatTensor(kernel).cuda()

if __name__ == '__main__':
	parser = argparse.ArgumentParser("regression")
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--learning_rate', type=float, default=0.0002)
	parser.add_argument('--momentum', type=float, default=0.9)
	parser.add_argument('--weight_decay', type=float, default=5e-4)
	parser.add_argument('--gpu', type=int, default=0)
	parser.add_argument('--epochs', type=int, default=200)
	parser.add_argument('--in_channels', type=int, default=256)
	parser.add_argument('--hidden_unit', type=int, default=128)
	parser.add_argument('--split', type=float, default=0.8)
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--save', type=str, default='EXP', help='experiment name')
	parser.add_argument('--pos_weight', type=int, default=9)
	parser.add_argument('--reg_l', type=float, default=0.05, help='coefficient of l2 regularization')
	parser.add_argument('--saved_weights', type=str, default='./model_weights_150.pth')
	parser.add_argument('--is_plot', type=bool, default=False, help='if to plot auroc and auprc or not')

	args = parser.parse_args()
	args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
	utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

	log_format = '%(asctime)s %(message)s'
	logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S: %p')
	fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
	fh.setFormatter(logging.Formatter(log_format))
	logging.getLogger().addHandler(fh)


class RegressionDataset(torch.utils.data.Dataset):
	def __init__(self):
		data = torch.from_numpy(np.load('./hist0.npy')).type(torch.FloatTensor)
		#idx = random.sample(range(5000), 5)
		#self.data = torch.flatten(data, 1, 2).unsqueeze(1)
		self.data = data.unsqueeze(1)
		self.labels = torch.from_numpy(np.loadtxt('./label0.csv', delimiter=",", dtype=np.float32))

	def __len__(self):
		return self.labels.shape[0]

	def __getitem__(self, idx):
		return self.data[idx], self.labels[idx]


def acc(loader, model):
	correct = 0.
	total = 0.

	for data in loader:
		x, y = data
		x = Variable(x, requires_grad=True).cuda()
		y = Variable(y).cuda()
		output = model(x)
		output = torch.round(torch.sigmoid(output))
		batch = y.size(0) * y.size(1)
		total += batch
		
		output = output.detach().cpu().numpy()
		y = y.detach().cpu().numpy()
		xor_list = np.logical_xor(output, y)
		count = batch - np.count_nonzero(xor_list)
		and_list = np.logical_and(output, y)
		pos_list = np.logical_or(xor_list, and_list)
		count_neg = batch - np.count_nonzero(pos_list)
		correct += (count_neg/9 + (count - count_neg))
		
		
	correct *= 5.


	return 100 * correct / total




def main(args):
	if not torch.cuda.is_available():
		print('no gpu device available')
		sys.exit(1)


	np.random.seed(args.seed)
	torch.cuda.set_device(args.gpu)
	cudnn.benchmarks = True
	torch.manual_seed(args.seed)
	cudnn.enable = True
	torch.cuda.manual_seed(args.seed)
	logging.info('gpu device = %d' % args.gpu)
	logging.info('args = %s', args)


	model = torch.load(args.saved_weights)['weights']

	pos_weight = torch.FloatTensor([args.pos_weight]*n_features)
	criterion = nn.BCEWithLogitsLoss(pos_weight= pos_weight).cuda()
	optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)


	data = RegressionDataset()
	num_train = len(data)
	indices = list(range(num_train))
	split = int(np.floor(args.split * num_train))


	valid_queue = torch.utils.data.DataLoader(
			data, batch_size=args.batch_size, shuffle=False, pin_memory=True)

	valid_loss = 0
	valid_overall = 0


	#test

	model.eval()
	roc = []
	prc = []
	with torch.no_grad():
		for i, d in enumerate(valid_queue):
			x, y = d
			x = Variable(x).cuda()
			y = Variable(y).cuda()

			output = model(x).cuda()
			loss = criterion(output, y)

			predict = torch.flatten(torch.sigmoid(output)).cpu()
			label = torch.flatten(y).cpu()
			idx = torch.randperm(n_features)


			fpr, tpr, _ = sklearn.metrics.roc_curve(label, predict, pos_label=1)
			roc.append(sklearn.metrics.auc(fpr, tpr))
			prc.append(sklearn.metrics.average_precision_score(label, predict, pos_label=1))

			if(args.is_plot):
				plt.plot(fpr, tpr)
				plt.xlabel('fpr')
				plt.ylabel('tpr')
				plt.savefig('./figs/roc_%d.png' % i)
				plt.close()

				precision, recall, _ = sklearn.metrics.precision_recall_curve(label, predict, pos_label=1)
				plt.plot(recall, precision)
				plt.xlabel('recall')
				plt.ylabel('precision')
				plt.savefig('./figs/prc_%d.png' % i)
				plt.close()
				np.savetxt('test_predict_%d.csv' % i, predict, delimiter=",")
				np.savetxt('test_labels_%d.csv' % i, label, delimiter=",")
			

			valid_loss += float(loss)
			valid_overall += 1

		valid_loss /= valid_overall
		print('validation loss %f', valid_loss)
		logging.info('valid_loss %f', valid_loss)
		valid_acc = acc(valid_queue, model)
		logging.info('valid_acc %f', valid_acc)
		np.save(os.path.join(args.save, 'roc'), roc)
		np.save(os.path.join(args.save, 'prc'), prc)
		logging.info('roc %f', np.mean(roc))
		logging.info('prc %f', np.mean(prc))



if __name__ == '__main__':
	main(args)
