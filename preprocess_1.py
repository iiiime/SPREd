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

# TODO: interaction file has redundent ,,, at the end of lines
# generate nn input from existing SERGIO expression matrix


parser = argparse.ArgumentParser()
parser.add_argument('--file_id', type=int, default=0, help='file id of hist, cov, and label')
parser.add_argument('--save', type=str, default='100_feature_50_cond_3_5_mr_5input_coxbox/', help='directory to save input files for neural network')
parser.add_argument('--data_dir', type=str, default='./dataset/100_feature_100_cond_3_5_mr_5input/', help='directory of SERGIO simulated data files')
parser.add_argument('--n_cond', type=int, default=50, help='number of conditions in expression matrix; number of cells in SERGIO simulations')
parser.add_argument('--n_bins', type=int, default=8, help='bin number of 2d histograms for mi calculation')
parser.add_argument('--n_samples', type=int, default=10, help='number of data points for nn models')
parser.add_argument('--n_mrs', type=int, default=5, help='number of master regulators')
parser.add_argument('--n_features', type=int, default=100, help='number of tfs')
parser.add_argument('--n_genes', type=int, default=100, help='number of target genes')
parser.add_argument('--dropout', type=float, default=0, help='probability of dropout')

args = parser.parse_args()

np.set_printoptions(threshold=sys.maxsize)

n_features = args.n_features
mr = np.arange(args.n_mrs)


"""
def shannon_entropy(x):
	norm = x / float(np.sum(x))
	norm = norm[np.nonzero(norm)]
	return -sum(norm * np.log2(norm))
"""


def _mi_score(x, y, n_bins):
	H_X = entropy(np.histogram(x, n_bins)[0], base=2)
	H_Y = entropy(np.histogram(y, n_bins)[0], base=2)
	H_XY = entropy(np.histogram2d(x, y, n_bins)[0].flatten(), base=2)
	return H_X + H_Y - H_XY


def mi_score(x, n_bins=6): # x: matrix like
	n = x.shape[0]
	out = np.zeros((n, n))
	for i in range(n):
		for j in range(i, n):
			out[i, j] = _mi_score(x[i], x[j], n_bins)
	return out


def gen_edge():
	"""generate 2-layer bipartite graph between regulators and target genes
	"""
	res = []
	n_reg = np.arange(3, 8) # number of regulators for each gene
	prob_d = [0.1, 0.15, 0.5, 0.15, 0.1] # probability distribution of the number of regulators
	#prob_d = [0.25, 0.5, 0.25]
	# master regulator ids: 0-9 -> 0-39 -> 0-4
	# TF ids: 10-109 -> 40 ->139 -> 
	for i in range(args.n_features):
		res.append(random.sample(range(args.n_mrs), 3))
	for i in range(args.n_features, args.n_features+args.n_genes):
		draw = np.random.choice(n_reg, 1, p=prob_d)
		res.append(random.sample(range(args.n_mrs, args.n_mrs+args.n_features), draw[0]))
	return res


def gen_grn(graph):
	"""returns an adjacency list from gen_edge
	"""
	res = []
	idx = args.n_mrs # number of master regulators
	for i in graph:
		for j in i:
			res.append([j, idx])
		idx += 1
	return sorted(res)	


def gen_mr_pr(mr, bins, r):
	res = []
	for i in range(len(mr)):
		expr = [mr[i]]
		for j in range(bins):
			expr_range = random.choice(r)
			expr.append(random.uniform(*expr_range))
		res.append(expr)
	return res


def gen_target(fname):
	res = []
	d = {}
	f = pd.read_csv(fname, header=None)
	a = f.values
	for i in a:
		if i[1] not in d:
			d[i[1]] = []
		d[i[1]].append(i[0])

	label = []
	for key in d:
		temp = []
		temp.append("{:.1f}".format(key))
		value = d[key]
		l = []
		if key > args.n_features + args.n_mrs - 1:
			l = [1 if i in value else 0 for i in range(args.n_mrs, args.n_mrs + args.n_features)]
		label = np.concatenate((label, l))
		temp.append("{:.1f}".format(len(value)))
		for i in value:
			temp.append("{:.1f}".format(i))

		for i in range(len(value)): # max contribution
			k = random.uniform(1.0, 5.0)
			if random.random() < 0.2:
				k = -k
			temp.append(k)

		for i in range(len(value)): # hill coefficient
			n = 1.0 if random.random() > 0.1 else 2.0
			temp.append(n)
		res.append(temp)

	return res, label


#TODO: set number_sc and average over columns
def gen_expr(interaction, regs):
	"""generate single-cell expression matrix using SERGIO"""
	sim = sergio(number_genes=200, number_bins=args.n_cond, number_sc=1, noise_params=1, decays=0.8, sampling_state=15, noise_type='dpd')
	sim.build_graph(interaction, regs, shared_coop_state=2)
	sim.simulate()
	expr = sim.getExpressions()
	expr_clean = np.concatenate(expr, axis = 1)
	return expr_clean


def gen_input(expr, idx, n_features):
	#TODO: add boolean flag to decide equal bin size or customized bin width
	"""generate 2d histograms of genes from SERGIO simulated expression matrix

	Parameters
	----------
	expr : numpy.array
		expression matrix

	idx : int
		target gene id

	n_features : int
		the number of TF genes

	n_bins : int
		the number of bins of 2d histograms

	Returns
	-------
	numpy.array
		list of 2d histograms

	list
		list of covariance

	list
		list of correlation
	"""
	#TODO: also try log normalization
	#r = [[-3., 3.], [-3., 3.]] # range of the histogram
	#r = [[-2.5, 2.5], [-2.5, 2.5]]
	#bins = [[-3., -1.1, -0.7, -0.3, 0, 0.3, 0.7, 1.1, 3.], [-3., -1.1, -0.7, -0.3, 0, 0.3, 0.7, 1.1, 3.]]
	out = []
	cov = []
	spearman = []
	pearson = []
	pm = []
	mi = []
	x = expr
	y = x[idx]
	x = x[args.n_mrs:args.n_mrs+args.n_features] #n_target gene

	x = np.concatenate((x, [y]))
	x = x[:, :args.n_cond]


	cov = np.cov(x)
	l = len(cov[0]) # length of the first dimension of covariance matrix
	spearman = spearmanr(x, axis=1)[0]
	pearson = np.corrcoef(x)
	pm = np.linalg.inv(np.array(cov) + 0.001 * np.identity(l)) # precision matrix
	mi = mi_score(x, args.n_bins)
	
	for i in range(args.n_features):
		out.append(cov[i])
		out.append(spearman[i])
		out.append(pearson[i])
		out.append(pm[i])
		out.append(mi[i])
	#print(np.array(out).shape)
	return out


#main
def main():
	# dataset file direction from previous SERGIO simulations
	epsilon = 1e-50
	hist = []
	label = [] # need reshape
	cov = []
	#corr = []
	n_genes = args.n_mrs + args.n_features + args.n_genes # total number of genes
	for n in range(args.file_id * args.n_samples, args.file_id * args.n_samples + args.n_samples):
		start = time.time()
		path = args.data_dir + 'De-noised_%dG_100T_1cPerT_DS%d/' % (n_genes, n)
		if not os.path.exists(path):
			print('path does not exist')
			sys.exit(1)

		fname = path + 'bipartite_GRN.csv'
		inter, l = gen_target(fname)
		#print(np.array(l.reshape(-1, 199)))
		label.append(l)
		inter = path + 'Interaction_cID_%d.txt' % n
		regs = path + 'Regs_cID_%d.txt' % n

		expr = np.loadtxt(path + 'simulated_noNoise_%d.txt' % n, delimiter='\t')
		# cox box transformation
		mask = np.asarray([(len(np.unique(expr[:, i])) > 1) for i in range(expr.shape[1])], dtype=bool)
		expr[:, mask] = PowerTransformer(method='box-cox', standardize=True).fit_transform(expr[:, mask] + epsilon)
		# z norm
		scaler = StandardScaler(copy=False, with_mean=True, with_std=True)
		expr = scaler.fit_transform(expr)
		expr = expr.T


		with open(inter, 'r') as f:
			for line in f:
				idx = int(float(line.rstrip('\n').split(',')[0]))
				if idx < n_features + args.n_mrs:
					continue
				out = gen_input(expr, idx, n_features)
				hist.append(out)
				#print(idx)

		end = time.time()
		print("time elapsed:")
		print(end - start)

	if not os.path.exists(args.save):
		os.mkdir(args.save)
	np.save(args.save + 'hist%d' % args.file_id, hist)
	np.savetxt(args.save + 'label%d.csv' % args.file_id, label, delimiter=',')


if __name__ == '__main__':
	main()
