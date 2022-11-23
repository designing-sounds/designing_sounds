#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf

from gpflow import kernels, inducing_variables
from gpflow.base import TensorLike
from gpflow.config import default_jitter
from gpflow.utilities import Dispatcher
from src.gpflow_samp.gpflow_sampling import covariances, kernels as kernels_ext
from src.gpflow_samp.gpflow_sampling.utils import swap_axes, move_axis
from src.gpflow_samp.gpflow_sampling.bases import kernel as kernel_basis
from src.gpflow_samp.gpflow_sampling.bases.core import AbstractBasis
from src.gpflow_samp.gpflow_sampling.sampling.core import DenseSampler
from src.gpflow_samp.gpflow_sampling.inducing_variables import InducingImages


# ==============================================
#                                  exact_updates
# ==============================================

exact = Dispatcher("exact_updates")


@exact.register(kernels.Kernel, TensorLike, TensorLike, TensorLike)
def _exact_fallback(kern: kernels.Kernel,
                    Z: TensorLike,
                    u: TensorLike,
                    f: TensorLike,
                    *,
                    L : TensorLike = None,
                    diag: TensorLike = None,
                    basis: AbstractBasis = None,
                    **kwargs):
  """
  Return pathwise updates of a prior processes $f$ subject to the
  condition $p(f | u) = N(f | u, diag)$ on $f = f(Z)$.
  """
  u_shape = tuple(u.shape)
  f_shape = tuple(f.shape)
  assert u_shape[-1] == 1, "Recieved multiple output features"
  assert u_shape == f_shape[-len(u_shape):],  "Incompatible shapes detected"
  if basis is None:  # finite-dimensional basis used to express the update
    basis = kernel_basis(kern, centers=Z)

  # Prepare diagonal term
  if diag is None:
    diag = default_jitter()
  if isinstance(diag, float):
    diag = tf.convert_to_tensor(diag, dtype=f.dtype)
  diag = tf.expand_dims(diag, axis=-1)  # [M, 1] or [1, 1] or [1]

  # Compute error term and matrix square root $Cov(u, u)^{1/2}$
  err = u - f  # [S, M, 1]
  err -= tf.sqrt(diag) * tf.random.normal(err.shape, dtype=err.dtype)
  if L is None:
    if isinstance(Z, inducing_variables.InducingVariables):
      K = covariances.Kuu(Z, kern, jitter=0.0)
    else:
      K = kern(Z, full_cov=True)
    K = tf.linalg.set_diag(K, tf.linalg.diag_part(K) + diag[..., 0])
    L = tf.linalg.cholesky(K)

  # Solve for $Cov(u, u)^{-1}(u - f(Z))$
  weights = tf.linalg.adjoint(tf.linalg.cholesky_solve(L, err))
  return DenseSampler(basis=basis, weights=weights, **kwargs)