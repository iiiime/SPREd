import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
import sys
import pandas as pd
import os
from scipy.stats import spearmanr, entropy, rankdata, wilcoxon
from sklearn.preprocessing import PowerTransformer, StandardScaler


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


def gen_input(tf_id, tfs, gene_names, genes, gene_pairs, save):
	n_genes = len(tf_id) + 1
	upper_idx = np.triu_indices(n_genes)
	res = []
	labels = []
	c = 0
	for n, g in enumerate(gene_names):
		print(g)
		if g not in np.array(gene_pairs):
			continue
		out = []
		if g not in tf_id:
			e = [] # expression matrix with 261 tfs
			l = []

			for tf in tf_id:
				e.append(tfs[tf])
				l.append(1 if (g, tf) in gene_pairs or (tf, g) in gene_pairs else 0)
			if l.count(1) == 0:
				print('error')
				continue
			e.append(genes[g])
			e = np.array(e)
			np.savetxt(save + '/expr%d.txt' % c, e, delimiter='\t')
			np.savetxt(save + '/label%d.txt' % c, l, delimiter='\t')

			cov = np.cov(e)
			spearman = spearmanr(e, axis=1)[0]
			pearson = np.corrcoef(e)
			pm = np.linalg.inv(np.array(cov) + 0.001 * np.identity(n_genes))
			mi = mi_score(e, 6)

			out.append(cov[upper_idx])
			out.append(spearman[upper_idx])
			out.append(pearson[upper_idx])
			out.append(pm[upper_idx])
			out.append(mi[upper_idx])
			res.append(out)
			labels.append(l)
			#print(l)
			c += 1
	return res, labels



def gen_input_1(tf_id, tfs, gene_names, genes, gene_pairs, save):
	n_genes = len(tf_id) + 1
	upper_idx = np.triu_indices(n_genes)
	res = []
	labels = []
	c = 0
	for n, g in enumerate(gene_names):
		print(g)
		if g not in np.array(gene_pairs):
			continue
		out = []
		if g not in tf_id:
			e = [] # expression matrix with 261 tfs
			l = []

			for tf in tf_id:
				e.append(tfs[tf])
				l.append(1 if (g, tf) in gene_pairs or (tf, g) in gene_pairs else 0)
			if l.count(1) == 0:
				print('error')
				continue
			e.append(genes[g])
			e = np.array(e)
			#np.savetxt(save + '/expr%d.txt' % c, e, delimiter='\t')
			#np.savetxt(save + '/label%d.txt' % c, l, delimiter='\t')

			expr = e.T
			mask = np.any(expr < 0, axis=0)
			if np.any(mask):
				expr[:, mask] -= (np.min(expr[:, mask], axis=0) - 1e-2)
			mask = np.asarray([(len(np.unique(expr[:, i])) > 1) for i in range(expr.shape[1])], dtype=bool)
			expr[:, mask] = PowerTransformer(method='box-cox', standardize=True).fit_transform(expr[:, mask] + epsilon)
			# z norm
			scaler = StandardScaler(copy=False, with_mean=True, with_std=True)
			expr = scaler.fit_transform(expr)
			e = expr.T
			
			cov = np.cov(e)
			spearman = spearmanr(e, axis=1)[0]
			pearson = np.corrcoef(e)
			pm = np.linalg.inv(np.array(cov) + 0.001 * np.identity(n_genes))
			mi = mi_score(e, 6)

			for i in range(len(tf_id)):
				samples = []
				samples.append(cov[i])
				samples.append(spearman[i])
				samples.append(pearson[i])
				samples.append(pm[i])
				samples.append(mi[i])
				res.append(samples)
			labels.append(l)
			labels = [i for s in labels for i in s]
			#print(l)
			c += 1
	return res, labels


