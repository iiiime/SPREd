#
# generate matrix of correlation, pearson correlation, spearman correlation, precision matrix, and mutual info from SERGIO simulated matrix
#


import numpy as np
import scipy as sp
import random
import argparse
import time
import os
import sys
import glob
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, entropy
from sklearn.preprocessing import PowerTransformer, StandardScaler
from pandas import DataFrame as df
from SERGIO.sergio import sergio


parser = argparse.ArgumentParser()
parser.add_argument('--file_id', type=int, default=0, help='file id of hist, cov, and label')
parser.add_argument('--save', type=str, default='./dataset/data/', help='directory to save input files for neural network')
parser.add_argument('--data_dir', type=str, default='./dataset/', help='directory of SERGIO simulated data files')
parser.add_argument('--n_cond', type=int, default=50, help='number of conditions in expression matrix; number of cells in SERGIO simulations')
parser.add_argument('--n_bins', type=int, default=8, help='bin number of histograms to calculate mutual information ')
parser.add_argument('--n_samples', type=int, default=10, help='number of data points for nn models')
parser.add_argument('--n_mrs', type=int, default=5, help='number of master regulators')
parser.add_argument('--n_features', type=int, default=100, help='number of tfs')
parser.add_argument('--n_genes', type=int, default=100, help='number of target genes')

args = parser.parse_args()


def _mi_score(x, y, n_bins):
	H_X = entropy(np.histogram(x, n_bins)[0], base=2)
	H_Y = entropy(np.histogram(y, n_bins)[0], base=2)
	H_XY = entropy(np.histogram2d(x, y, n_bins)[0].flatten(), base=2)
	return H_X + H_Y - H_XY


def mi_score(x, n_bins=6):
	n = x.shape[0]
	out = np.zeros((n, n))
	for i in range(n):
		for j in range(i, n):
			out[i, j] = _mi_score(x[i], x[j], n_bins)
	return out


def gen_label(fname):
	""" generate label of samples for SPREd from the interaction file for SERGIO simulation

	Parameters
	----------
	fname : str
		directory and file name of the interaction file

	Returns
	-------
	list
		labels of the dataset
	"""
	d = {}
	f = pd.read_csv(fname, header=None)
	a = f.values
	for i in a:
		if i[1] not in d:
			d[i[1]] = []
		d[i[1]].append(i[0])

	label = []
	for key in d:
		l = []
		value = d[key]
		if key > args.n_features + args.n_mrs - 1:
			l = [1 if i in value else 0 for i in range(args.n_mrs, args.n_mrs + args.n_features)]
		label = np.concatenate((label, l))
	return  label


def gen_input(expr, idx, n_features):
	"""generate features of genes from SERGIO simulated expression matrix

	Parameters
	----------
	expr : numpy.array
		expression matrix

	idx : int
		target gene id

	n_features : int
		the number of TF genes

	Returns
	-------
	numpy.array
		list of 2d histograms
	"""
	out = []
	cov = []
	spearman = []
	pearson = []
	pm = []
	mi = []
	x = expr
	y = x[idx]
	x = x[args.n_mrs:args.n_mrs+args.n_features]

	x = np.concatenate((x, [y]))
	x = x[:, :args.n_cond]


	cov = np.cov(x)
	l = len(cov[0])
	spearman = spearmanr(x, axis=1)[0]
	pearson = np.corrcoef(x)
	pm = np.linalg.inv(np.array(cov) + 0.001 * np.identity(l))
	mi = mi_score(x, args.n_bins)
	
	for i in range(args.n_features):
		samples = []
		samples.append(cov[i])
		samples.append(spearman[i])
		samples.append(pearson[i])
		samples.append(pm[i])
		samples.append(mi[i])
		out.append(samples)
	return out


#main
def main():
	epsilon = 1e-50
	hist = []
	label = []
	n_genes = args.n_mrs + args.n_features + args.n_genes
	for n in range(args.file_id * args.n_samples, args.file_id * args.n_samples + args.n_samples):
		start = time.time()
		path = args.data_dir + 'De-noised_%dG_%dT_1cPerT_DS%d/' % (n_genes, args.n_cond, n)
		if not os.path.exists(path):
			print('path does not exist')
			sys.exit(1)

		fname = path + 'l3_GRN.csv'
		label.append(gen_label(fname))
		inter = path + 'Interaction_cID.txt'

		expr = np.loadtxt(path + 'simulated_noNoise.txt', delimiter='\t')
		mask = np.asarray([(len(np.unique(expr[:, i])) > 1) for i in range(expr.shape[1])], dtype=bool)
		expr[:, mask] = PowerTransformer(method='box-cox', standardize=True).fit_transform(expr[:, mask] + epsilon)
		scaler = StandardScaler(copy=False, with_mean=True, with_std=True)
		expr = scaler.fit_transform(expr)
		expr = expr.T

		with open(inter, 'r') as f:
			for line in f:
				idx = int(float(line.rstrip('\n').split(',')[0]))
				if idx < args.n_features + args.n_mrs:
					continue
				out = gen_input(expr, idx, args.n_features)
				hist.extend(out)

		end = time.time()
		print("time elapsed:")
		print(end - start)

	label = [i for s in label for i in s]
	if not os.path.exists(args.save):
		os.mkdir(args.save)
	np.save(args.save + 'hist%d' % args.file_id, hist)
	np.savetxt(args.save + 'label%d.csv' % args.file_id, label, delimiter=',')


if __name__ == '__main__':
	main()
