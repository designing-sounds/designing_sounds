#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf

from warnings import warn
from gpflow.base import TensorLike
from gpflow.config import default_jitter
from gpflow.utilities import Dispatcher
from gpflow import kernels, inducing_variables
from src.gpflow_samp.gpflow_sampling import covariances, kernels as kernels_ext
from src.gpflow_samp.gpflow_sampling.bases import kernel as kernel_basis
from src.gpflow_samp.gpflow_sampling.bases.core import AbstractBasis
from src.gpflow_samp.gpflow_sampling.sampling.core import DenseSampler
from src.gpflow_samp.gpflow_sampling.inducing_variables import InducingImages
from src.gpflow_samp.gpflow_sampling.utils import get_default_preconditioner


# ==============================================
#                                     cg_updates
# ==============================================

cg = Dispatcher("cg_updates")


@cg.register(kernels.Kernel, TensorLike, TensorLike, TensorLike)
def _cg_fallback(kern: kernels.Kernel,
                 Z: TensorLike,
                 u: TensorLike,
                 f: TensorLike,
                 *,
                 diag: TensorLike = None,
                 basis: AbstractBasis = None,
                 preconditioner: tf.linalg.LinearOperator = "default",
                 tol: float = 1e-3,
                 max_iter: int = 100,
                 **kwargs):
  """
  Return pathwise updates of a prior processes $f$ subject to the
  condition $p(f | u) = N(f | u, diag)$ on $f = f(Z)$.
  """
  u_shape = tuple(u.shape)
  f_shape = tuple(f.shape)
  assert u_shape[-1] == 1, "Recieved multiple output features"
  assert u_shape == f_shape[-len(u_shape):], "Incompatible shapes detected"
  if basis is None:  # finite-dimensional basis used to express the update
    basis = kernel_basis(kern, centers=Z)

  if diag is None:
    diag = default_jitter()

  # Prepare linear system for CG solver
  if isinstance(Z, inducing_variables.InducingVariables):
    Kff = covariances.Kuu(Z, kern, jitter=0.0)
  else:
    Kff = kern(Z, full_cov=True)
  Kuu = tf.linalg.set_diag(Kff, tf.linalg.diag_part(Kff) + diag)
  operator = tf.linalg.LinearOperatorFullMatrix(Kuu,
                                                is_non_singular=True,
                                                is_self_adjoint=True,
                                                is_positive_definite=True,
                                                is_square=True)

  if preconditioner == "default":
    preconditioner = get_default_preconditioner(Kff, diag=diag)

  # Compute error term and CG initializer
  err = tf.linalg.adjoint(u - f)  # [S, 1, M]
  err -= (diag ** 0.5) * tf.random.normal(err.shape, dtype=err.dtype)
  if preconditioner is None:
    initializer = None
  else:
    initializer = preconditioner.matvec(err)

  # Approximately solve for $Cov(u, u)^{-1}(u - f(Z))$ using linear CG
  res = tf.linalg.experimental.conjugate_gradient(operator=operator,
                                                  rhs=err,
                                                  preconditioner=preconditioner,
                                                  x=initializer,
                                                  tol=tol,
                                                  max_iter=max_iter)

  weights = res.x
  if tf.math.count_nonzero(tf.math.is_nan(weights)):
    warn("One or more update weights returned by CG are NaN")

  return DenseSampler(basis=basis, weights=weights, **kwargs)