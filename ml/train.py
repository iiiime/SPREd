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

n_features = 100
n_bins = 8


if __name__ == '__main__':
	parser = argparse.ArgumentParser("regression")
	parser.add_argument('--batch_size', type=int, default=32)
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
	parser.add_argument('--dropout', type=float, default=0.3, help='dropout coefficient')

	args = parser.parse_args()
	args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
	utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

	log_format = '%(asctime)s %(message)s'
	logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S: %p')
	fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
	fh.setFormatter(logging.Formatter(log_format))
	logging.getLogger().addHandler(fh)


class Dataset(torch.utils.data.Dataset):
	def __init__(self):
		data = []
		labels = []
		for i in range(1,6):
			data.append(torch.from_numpy(np.load('./hist%d.npy' % i)).type(torch.FloatTensor))
			labels.append(torch.from_numpy(np.loadtxt('./label%d.csv' % i, delimiter=',', dtype=np.float32)))
		self.data = torch.cat((data), 0).unsqueeze(1)
		self.labels = torch.cat((labels), 0)

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
		epoch *= 20 # set const
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
	#in_channels = 65 # for sub dataset
	#in_channels = 5148 # conv=5,4
	in_channels = 5151
	#in_channels = 5144
	model = Network(in_channels, args.hidden_unit, n_features, args.dropout).cuda()

	pos_weight = torch.FloatTensor([args.pos_weight]*n_features)
	criterion = nn.BCEWithLogitsLoss(pos_weight= pos_weight).cuda()
	optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)

	data = Dataset()
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
	for epoch in range(args.epochs):
		model.train()
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

			# adding L1 norm
			#l1_lambda = 0.0001
			#l1_norm = sum(p.abs().sum() for p in model.parameters())
			#loss = loss + l1_lambda * l1_norm

			loss.backward() # calculate gradient
			optimizer.step() # update params

			train_loss += float(loss)
			train_overall += 1

			# TODO: add function to concat and calculate roc and prc
			if epoch % 20 == 0 or epoch == args.epochs - 1:
				concat_predict = torch.flatten(output).detach().cpu()
				concat_label = torch.flatten(y).detach().cpu()
				predict = np.concatenate((predict, concat_predict))
				label = np.concatenate((label, concat_label))

		if epoch % 20 == 0 or epoch == args.epochs - 1:
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

			#TODO: include prc plot
			

		#print(train_loss)
		#print(train_overall)
		train_loss /= train_overall
		_train_loss.append(train_loss)
		print('epoch %03d - training loss %f' % (epoch, train_loss))
		logging.info('epoch %d', epoch)
		logging.info('train_loss %f', train_loss)
		train_acc = utils.acc(train_queue, model)
		logging.info('train_acc %f', train_acc)
		_train_acc.append(train_acc)

		#test
		predict = []
		label = []
		model.eval()
		with torch.no_grad():
			for x, y in valid_queue:
				x = Variable(x).cuda()
				y = Variable(y).cuda()

				output = model(x).cuda()
				loss = criterion(output, y)
				#if epoch == args.epochs-1:
					#loss = criterion(output, y)
				if epoch % 20 == 0 or epoch == args.epochs-1:
					#print(output.shape)
					concat_predict = torch.flatten(output).cpu()
					concat_label = torch.flatten(y).cpu()
					#print(concat_label.shape)
					predict = np.concatenate((predict, concat_predict))
					label = np.concatenate((label, concat_label))
					#print(np.array(label).shape)
					#print(concat_label.shape)
					
					#print('prediction is: ', output)
					#print('label is: ', y)

					#print(concat_label)
					#print(concat_predict)
					#print(np.array(label).shape)

				valid_loss += float(loss)
				valid_overall += 1
			if epoch % 20 == 0 or epoch == args.epochs - 1:
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

			valid_loss /= valid_overall
			_valid_loss.append(valid_loss)
			print('epoch %03d - validation loss %f' % (epoch, valid_loss))
			logging.info('valid_loss %f', valid_loss)
			valid_acc = utils.acc(valid_queue, model)
			logging.info('valid_acc %f', valid_acc)
			_valid_acc.append(valid_acc)
			#np.savetxt('predict.csv', predict, delimiter=",")
			#np.savetxt('labels.csv', label, delimiter=",")

		if epoch % 50 == 0:
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
