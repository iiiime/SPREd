import numpy as np
import sklearn.metrics
from sklearn.preprocessing import PowerTransformer, StandardScaler
from portia.correction import apply_correction
from portia.la import *
import os
import sys


def correct(dataset, n_tf, lambda1=0.8, lambda2=0.05, normalize=True, verbose=True):
	""" return covariance matrix after shrinkage and corrected precision matrix
	"""
	epsilon = 1e-50
	tf_idx = np.arange(n_tf)

	_X = dataset
	n_samples = _X.shape[0]
	n_genes = _X.shape[1]
	if verbose:
		print(f'Gene expression matrix of shape ({n_samples}, {n_genes})')

	if normalize:
		quantiles = np.quantile(_X, 0.5, axis=0)[np.newaxis, :]
		quantiles[quantiles == 0] = 1
		_X = _X / quantiles

	if n_samples >= n_genes:
		_P = 1. - scipy.spatial.distance.cdist(_X, _X, metric='correlation')
		theta = 0.8
		counts = np.zeros(n_samples)
		idx = np.where(_P > theta)
		np.add.at(counts, idx[0], 1)
		np.add.at(counts, idx[1], 1)
		weights = 1.0 / (1. + np.asarray(counts, dtype=float))
	else:
		weights = np.ones(n_samples) / n_samples


	mask = np.asarray([(len(np.unique(_X[:, i])) > 1) for i in range(n_genes)], dtype=bool)

	if np.sum(mask) > 0:
		power_transform = PowerTransformer(method='box-cox', standardize=True)
		_X_transformed = _X
		_X_transformed[:, mask] = power_transform.fit_transform(_X[:, mask] + epsilon)
	else:
		_X_transformed = _X + epsilon


	scaler = StandardScaler(copy=False, with_mean=True, with_std=True)
	_X_transformed = scaler.fit_transform(_X_transformed)

	_S = np.cov(_X_transformed.T, ddof=1, aweights=weights)
	_S_bar = lambda1 * np.eye(n_genes) + (1. - lambda1) * _S
	
	if (tf_idx is not None) and (len(tf_idx) < 0.5 * n_genes):
		_Theta = partial_inv(_S_bar, tf_idx)
	else:
		_Theta = np.linalg.inv(_S_bar)

	_M = np.abs(_Theta)
	if tf_idx is not None:
		mask = np.zeros((n_genes, n_genes))
		mask[tf_idx, :] = 1
		np.fill_diagonal(mask, 0)
		_M *= mask
		_M[tf_idx, :] = apply_correction(_M[tf_idx, :], method='rcw')
	else:
		_M = apply_correction(_M, method='rcw')

	np.fill_diagonal(_M, 0)

	beta = all_linear_regressions(_X_transformed, _lambda=lambda2)
	beta = np.abs(beta)
	np.fill_diagonal(beta, 0)
	div = np.maximum(beta, beta.T)
	div_mask = (div > 0)
	beta[div_mask] /= div[div_mask]
	_M = _M * beta

	_M_bar = _M
	_M_bar *= np.std(_M_bar, axis=1)[:, np.newaxis]
	_range = _M_bar.max() - _M_bar.min()
	if _range > 0:
		_M_bar = (_M_bar - _M_bar.min()) / _range # pmi in range 0,1


	return _S_bar, _M_bar
