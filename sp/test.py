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
from model import Network
from torch.autograd import Variable


if __name__ == '__main__':
	parser = argparse.ArgumentParser("regression")
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--learning_rate', type=float, default=0.0002)
	parser.add_argument('--momentum', type=float, default=0.9)
	parser.add_argument('--weight_decay', type=float, default=5e-4)
	parser.add_argument('--gpu', type=int, default=0)
	parser.add_argument('--epochs', type=int, default=200)
	parser.add_argument('--hidden_unit', type=int, default=128)
	parser.add_argument('--split', type=float, default=0.8)
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--save', type=str, default='EXP', help='experiment name')
	parser.add_argument('--pos_weight', type=int, default=9)
	parser.add_argument('--saved_weights', type=str, default='./model_weights_150.pth')
	parser.add_argument('--is_plot', type=bool, default=False, help='if to plot auroc and auprc or not')
	parser.add_argument('--n_tf', type=int, default=245)
	parser.add_argument('--data', type=str, default='./hist0.npy', help='data')
	parser.add_argument('--label', type=str, default='./label0.csv', help='labels')

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
		data = torch.from_numpy(np.load(args.data)).type(torch.FloatTensor)
		self.data = data.unsqueeze(1)
		self.labels = torch.from_numpy(np.loadtxt(args.label, delimiter=",", dtype=np.float32)).unsqueeze(-1)

	def __len__(self):
		return self.labels.shape[0]

	def __getitem__(self, idx):
		return self.data[idx], self.labels[idx]


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

	pos_weight = torch.FloatTensor([args.pos_weight] * 1)
	criterion = nn.BCEWithLogitsLoss(pos_weight= pos_weight).cuda()
	optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)


	data = RegressionDataset()

	valid_queue = torch.utils.data.DataLoader(
			data, batch_size=args.batch_size, shuffle=False, pin_memory=True)

	valid_loss = 0
	valid_overall = 0
	predict = []

	model.eval()
	with torch.no_grad():
		for i, d in enumerate(valid_queue):
			x, y = d
			x = Variable(x).cuda()
			y = Variable(y).cuda()

			output = model(x).cuda()
			loss = criterion(output, y)

			predict.extend(torch.flatten(torch.sigmoid(output)).cpu())

			valid_loss += float(loss)
			valid_overall += 1

		valid_loss /= valid_overall
		print('validation loss %f', valid_loss)
		logging.info('valid_loss %f', valid_loss)

	np.save(os.path.join(args.save, 'predict.npy'), np.array(predict))
	label = np.loadtxt(args.label, delimiter=',')
	roc = []
	prc = []
	for i in range(int(label.shape[0]/args.n_tf)):
		p = predict[i*args.n_tf:(i+1)*args.n_tf]
		l = label[i*args.n_tf:(i+1)*args.n_tf]
		fpr, tpr, _ = sklearn.metrics.roc_curve(l, p, pos_label=1)
		roc.append(sklearn.metrics.auc(fpr, tpr))
		prc.append(sklearn.metrics.average_precision_score(l, p, pos_label=1))

	logging.info('roc %f', np.mean(roc))
	logging.info('prc %f', np.mean(prc))
	np.save(os.path.join(args.save, 'roc'), roc)
	np.save(os.path.join(args.save, 'prc'), prc)


if __name__ == '__main__':
	main(args)
