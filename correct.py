import numpy as np
import sklearn.metrics
from sklearn.preprocessing import PowerTransformer, StandardScaler
from portia.correction import apply_correction
from portia.la import *
import os
import sys


def transform(dataset, n_tf, normalize=True, verbose=True):
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

	mask = np.asarray([(len(np.unique(_X[:, i])) > 1) for i in range(n_genes)], dtype=bool)

	if np.sum(mask) > 0:
		power_transform = PowerTransformer(method='box-cox', standardize=True)
		_X_transformed = _X
		_X_transformed[:, mask] = power_transform.fit_transform(_X[:, mask] + epsilon)
	else:
		_X_transformed = _X + epsilon

	scaler = StandardScaler(copy=False, with_mean=True, with_std=True)
	_X_transformed = scaler.fit_transform(_X_transformed)

	return _X_transformed


def correct(transformed, n_tf, lambda1=0.8, lambda2=0.05):
	tf_idx = np.arange(n_tf)
	_X_transformed = transformed
	n_samples = _X_transformed.shape[0]
	n_genes = _X_transformed.shape[1]
	weights = np.ones(n_samples) / n_samples

	_S = np.cov(_X_transformed.T, ddof=1, aweights=weights)
	_S_bar = lambda1 * np.eye(n_genes) + (1. - lambda1) * _S
	
	_Theta = np.linalg.inv(_S_bar)

	_M = np.abs(_Theta)
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
