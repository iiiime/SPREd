#----------
# synthetic expression generation using SERGIO
#----------

import numpy as np
import scipy as sp
import random
import argparse
import time
import os
import sys
import pandas as pd
from scipy.stats import pearsonr, spearmanr, entropy
from sklearn.preprocessing import PowerTransformer, StandardScaler
from pandas import DataFrame as df
from SERGIO.sergio import sergio


parser = argparse.ArgumentParser()
parser.add_argument('--file_id', type=int, default=0, help='file id of hist, cov, and label')
parser.add_argument('--data_dir', type=str, default='./dataset/', help='directory of SERGIO input files and simulated data files')
parser.add_argument('--n_cond', type=int, default=50, help='number of conditions in expression matrix; number of cells in SERGIO simulations')
parser.add_argument('--n_bins', type=int, default=8, help='bin number of 2d histograms')
parser.add_argument('--n_samples', type=int, default=10, help='number of data points per dataset for nn models')
parser.add_argument('--n_mrs', type=int, default=5, help='number of master regulators')
parser.add_argument('--n_genes', type=int, default=100, help='number of target genes')
parser.add_argument('--n_features', type=int, default=100, help='number of transcription factors (features)')


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


def gen_edge():
	"""generate a 3-layer graph between regulators and target genes
	"""
	res = []
	n_reg = np.arange(3, 8)
	for i in range(args.n_features):
		res.append(random.sample(range(args.n_mrs), 3))
	for i in range(args.n_features, args.n_features+args.n_genes):
		draw = np.random.choice(n_reg, 1)
		res.append(random.sample(range(args.n_mrs, args.n_mrs+args.n_features), draw[0]))
	return res


def gen_grn(graph):
	"""returns an adjacency list from gen_edge
	"""
	res = []
	idx = args.n_mrs
	for i in graph:
		for j in i:
			res.append([j, idx])
		idx += 1
	return sorted(res)	


def gen_mr_pr(n_mrs, bins, r):
	""" generate the production rates of master regulators by uniform sampling
	"""
	mr = np.arange(n_mrs)
	res = []
	for i in range(n_mrs):
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

		for i in range(len(value)):
			k = random.uniform(1.0, 5.0)
			if random.random() < 0.2:
				k = -k
			temp.append(k)

		for i in range(len(value)):
			n = 1.0 if random.random() < 0.1 else 2.0
			temp.append(n)
		res.append(temp)

	return res, label


def gen_expr(interaction, regs):
	"""generate single-cell expression matrix using SERGIO"""
	sim = sergio(number_genes=(args.n_mrs+args.n_features+args.n_genes), number_bins=args.n_cond, number_sc=1, noise_params=1, decays=0.8, sampling_state=15, noise_type='dpd')
	sim.build_graph(interaction, regs, shared_coop_state=2)
	sim.simulate()
	expr = sim.getExpressions()
	expr_clean = np.concatenate(expr, axis = 1)
	return expr_clean


#main
def main():
	r0 = [(0.2, 0.5), (0.7, 1.0)]
	r1 = [(0.0, 2.0), (2.0, 4.0)]
	r2 = [(0.0, 1.0), (3.0, 4.0)]
	ranges = np.concatenate(([r0], [r1], [r2]))
	n_genes = args.n_mrs + args.n_features + args.n_genes
	for n in range(args.file_id * args.n_samples, args.file_id * args.n_samples + args.n_samples):
		start = time.time()
		if not os.path.exists(args.data_dir):
			os.mkdir(args.data_dir)
		path = args.data_dir + 'De-noised_%dG_%dT_1cPerT_DS%d/' % (n_genes, args.n_cond, n)
		if not os.path.exists(path):
			os.mkdir(path)

		graph = gen_edge()
		grn = gen_grn(graph)
		fname = path + 'l3_GRN.csv'
		np.savetxt(fname, grn, fmt='%s', delimiter=',')
		inter, l = gen_target(fname)
		inter = pd.DataFrame(inter)
		r = random.choice(ranges)
		reg = pd.DataFrame(gen_mr_pr(args.n_mrs, args.n_cond, r))
		reg.to_csv(path + 'Regs_cID.txt', header=False, index=False)
		inter.to_csv(path + 'Interaction_cID.txt', header=False, index=False)

		inter = path + 'Interaction_cID.txt'
		regs = path + 'Regs_cID.txt'
		expr = np.array(gen_expr(inter, regs)).T
		np.savetxt(path + 'simulated_noNoise.txt', expr, delimiter='\t')

		end = time.time()
		print("time elapsed:")
		print(end - start)



if __name__ == '__main__':
	main()
