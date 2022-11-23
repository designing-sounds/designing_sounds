#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
from typing import Union
from gpflow import kernels as gpflow_kernels
from gpflow.base import TensorType
from gpflow.utilities import Dispatcher
from gpflow.inducing_variables import InducingVariables
from src.gpflow_samp.gpflow_sampling import kernels
from src.gpflow_samp.gpflow_sampling.bases import fourier_bases
from src.gpflow_samp.gpflow_sampling.bases.core import KernelBasis


# ---- Exports
__all__ = (
  'kernel_basis',
  'fourier_basis',
)

kernel_basis = Dispatcher("kernel_basis")
fourier_basis = Dispatcher("fourier_basis")


# ==============================================
#                                       dispatch
# ==============================================
@kernel_basis.register(gpflow_kernels.Kernel)
def _kernel_fallback(kern: gpflow_kernels.Kernel,
                     centers: Union[TensorType, InducingVariables],
                     **kwargs):
  return KernelBasis(kernel=kern, centers=centers, **kwargs)


@fourier_basis.register(gpflow_kernels.Periodic)
def _fourier_stationary(kern: gpflow_kernels.Stationary, **kwargs):
  return fourier_bases.Dense(kernel=kern, **kwargs)

@fourier_basis.register(gpflow_kernels.Stationary)
def _fourier_stationary(kern: gpflow_kernels.Stationary, **kwargs):
  return fourier_bases.Dense(kernel=kern, **kwargs)
