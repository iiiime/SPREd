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

import utils
from model_ import Network
from torch.autograd import Variable

n_features = 1
n_bins = 8


if __name__ == '__main__':
	parser = argparse.ArgumentParser("regression")
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--learning_rate', type=float, default=2e-4)
	parser.add_argument('--momentum', type=float, default=0.9)
	parser.add_argument('--weight_decay', type=float, default=5e-4)
	parser.add_argument('--gpu', type=int, default=0)
	parser.add_argument('--epochs', type=int, default=300)
	parser.add_argument('--in_channels', type=int, default=256)
	parser.add_argument('--hidden_unit', type=int, default=128)
	parser.add_argument('--split', type=float, default=0.8)
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--save', type=str, default='EXP', help='experiment name')
	parser.add_argument('--pos_weight', type=int, default=9)
	parser.add_argument('--reg_l', type=float, default=0.05, help='coefficient of l2 regularization')
	parser.add_argument('--dropout', type=float, default=0.3, help='dropout coefficient')

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
		data = []
		labels = []
		for i in range(1,6):
			data.append(torch.from_numpy(np.load('./hist%d.npy' % i)).type(torch.FloatTensor))
			labels.append(torch.from_numpy(np.loadtxt('./label%d.csv' % i, delimiter=',', dtype=np.float32)))
		self.data = torch.cat((data), 0).unsqueeze(1)
		self.labels = torch.cat((labels), 0).unsqueeze(-1)

	def __len__(self):
		return self.labels.shape[0]

	def __getitem__(self, idx):
		return self.data[idx], self.labels[idx]



"""plot and save a single plot"""
def pl(x, y, x_label, y_label, n):
	fig = plt.figure()
	plt.plot(x, y)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.savefig(n)


"""plot and save training curves"""
def pl_train_curve(train, valid, train_label, valid_label, y_label, n):
	fig = plt.figure()
	epoch = np.arange(len(train))
	if len(train) < 100:
		epoch *= 10 # set const
	plt.plot(epoch, train, label=train_label)
	plt.plot(epoch, valid, label=valid_label)
	plt.xlabel('epochs')
	plt.ylabel(y_label)
	plt.legend()
	plt.savefig(os.path.join(args.save, n))


def main(args):
	if not torch.cuda.is_available():
		print('no gpu device availabel')
		sys.exit(1)

	np.random.seed(args.seed)
	torch.cuda.set_device(args.gpu)
	cudnn.benchmarks = True
	torch.manual_seed(args.seed)
	cudnn.enable = True
	torch.cuda.manual_seed(args.seed)
	logging.info('gpu device = %d' % args.gpu)
	logging.info('args = %s', args)

	#in_channels = int(n_features + n_features * (n_features + 1) / 2)
	threshold = 0.5
	in_channels = 5151
	model = Network(in_channels, args.hidden_unit, n_features, args.dropout).cuda()

	pos_weight = torch.FloatTensor([args.pos_weight]*n_features)
	criterion = nn.BCEWithLogitsLoss(pos_weight= pos_weight).cuda()
	optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)

	data = RegressionDataset()
	num_train = len(data)
	indices = list(range(num_train))
	split = int(np.floor(args.split * num_train))

	train_data, valid_data = torch.utils.data.random_split(data, [split, num_train - split])
	train_queue = torch.utils.data.DataLoader(
			train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
	valid_queue = torch.utils.data.DataLoader(
			valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)

	_train_loss = []
	_valid_loss = [] 
	_train_acc = []
	_valid_acc = []
	_train_auroc = []
	_valid_auroc = []
	_train_auprc = []
	_valid_auprc = []
	valid_auroc = 0
	valid_auprc = 0
	for epoch in range(1, args.epochs + 1):
		model.train()
		tp, tn, fp, fn = 0, 0, 0, 0
		train_loss = 0
		train_overall = 0
		valid_loss = 0
		valid_overall = 0

		predict = []
		label = []
		for x, y in train_queue:
			x = Variable(x, requires_grad=True).cuda()
			y = Variable(y).cuda()

			optimizer.zero_grad()
			model.zero_grad()
			output = model(x)

			loss = criterion(output, y)
			loss.backward() # calculate gradient
			optimizer.step() # update params

			output = torch.sigmoid(output).detach().cpu()
			y = y.detach().cpu()

			binaryOutputs = torch.where(output >= threshold, torch.tensor(1.), torch.tensor(0.))
			tp += ((binaryOutputs == 1) & (y == 1)).sum().item()
			tn += ((binaryOutputs == 0) & (y == 0)).sum().item()
			fp += ((binaryOutputs == 1) & (y == 0)).sum().item()
			fn += ((binaryOutputs == 0) & (y == 1)).sum().item()

			train_loss += float(loss)
			train_overall += 1

			# global aupr auroc calculation
			if epoch % 10 == 0:
				predict.extend(torch.flatten(output))
				label.extend(torch.flatten(y))

		if epoch % 1 == 0:
			acc = 100 * (tp + tn) / (tp + tn + fp + fn)
			recall = 100 * tp / (tp + fn)
			train_loss /= train_overall
			_train_loss.append(train_loss)
			_train_acc.append(acc)
			print('epoch %03d - training loss %f' % (epoch, train_loss))
			print('epoch %03d - training acc %f' % (epoch, acc))
			print('epoch %03d - training recall %f' % (epoch, recall))
			logging.info('epoch %d', epoch)
			logging.info('train_loss %f', train_loss)
			logging.info('train_acc %f', acc)
			logging.info('train_recall %f', recall)

		if epoch % 10 == 0:
			fpr, tpr, thresholds = sklearn.metrics.roc_curve(label, predict, pos_label=1)
			pl(fpr, tpr, 'fpr', 'tpr', os.path.join(args.save, 'train_roc_epoch_%d' % epoch))
			print('train auroc')
			train_auroc = sklearn.metrics.auc(fpr, tpr)
			_train_auroc.append(train_auroc)
			print(train_auroc)
			logging.info('train_auroc %f', train_auroc)

			train_auprc = sklearn.metrics.average_precision_score(label, predict, pos_label=1)
			print('train auprc')
			print(train_auprc)
			_train_auprc.append(train_auprc)
			logging.info('train_auprc %f', train_auprc)


		#test
		predict = []
		label = []
		model.eval()
		testtp, testtn, testfp, testfn = 0, 0, 0, 0
		with torch.no_grad():
			for x, y in valid_queue:
				x = Variable(x).cuda()
				y = Variable(y).cuda()

				output = model(x).cuda()
				loss = criterion(output, y)

				output = torch.sigmoid(output).cpu()
				y = y.cpu()

				valid_loss += float(loss)
				valid_overall += 1

				binaryOutputs = torch.where(output >= threshold, torch.tensor(1.), torch.tensor(0.))
				testtp += ((binaryOutputs == 1) & (y == 1)).sum().item()
				testtn += ((binaryOutputs == 0) & (y == 0)).sum().item()
				testfp += ((binaryOutputs == 1) & (y == 0)).sum().item()
				testfn += ((binaryOutputs == 0) & (y == 1)).sum().item()

				if epoch % 10 == 0:
					predict.extend(torch.flatten(output))
					label.extend(torch.flatten(y))

			if epoch % 1 == 0:
				accuracy = 100 * (testtp + testtn) / (testtp + testtn + testfp + testfn)
				recall = 100 * testtp / (testtp + testfn)
				valid_loss /= valid_overall
				_valid_loss.append(valid_loss)
				_valid_acc.append(accuracy)
				print('epoch %03d - validation loss %f' % (epoch, valid_loss))
				logging.info('valid_loss %f', valid_loss)
				logging.info('valid_acc %f', accuracy)
				logging.info('valid_recall %f', recall)


			if epoch % 10 == 0:
				fpr, tpr, thresholds = sklearn.metrics.roc_curve(label, predict, pos_label=1)
				valid_auprc = sklearn.metrics.average_precision_score(label, predict, pos_label=1)
				pl(fpr, tpr, 'fpr', 'tpr', os.path.join(args.save, 'valid roc_%d' % epoch))
				print('test auc:')
				valid_auroc = sklearn.metrics.auc(fpr, tpr)
				print(valid_auroc)
				_valid_auroc.append(valid_auroc)
				logging.info('valid_auroc %f', valid_auroc)
				print('test precision recall:')
				print(valid_auprc)
				_valid_auprc.append(valid_auprc)
				logging.info('valid_auprc %f', valid_auprc)

		if epoch % 20 == 0:
			print('saving model ...')
			state = {
				'weights': model,
				'epoch': epoch,
			}
			torch.save(state, os.path.join(args.save, 'model_weights_%d.pth' % epoch))

	pl_train_curve(_train_loss, _valid_loss, 'train loss', 'valid loss', 'BCEloss', 'loss curve.png')
	pl_train_curve(_train_acc, _valid_acc, 'train acc', 'valid acc', 'accuracy', 'acc curve.png')
	pl_train_curve(_train_auroc, _valid_auroc, 'train auroc', 'valid auroc', 'auroc', 'auroc curve.png')
	pl_train_curve(_train_auprc, _valid_auprc, 'train auprc', 'valid auprc', 'auprc', 'auprc curve.png')

	logging.info('valid_auroc %f', valid_auroc)
	logging.info('valid_auprc %f', valid_auprc)


if __name__ == '__main__':
	main(args)
