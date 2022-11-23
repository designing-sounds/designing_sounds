#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf

from typing import Any, List, Callable
from gpflow.config import default_float
from gpflow.kernels import Kernel, MultioutputKernel
from gpflow.utilities import Dispatcher
from src.gpflow_samp.gpflow_sampling.bases import fourier as fourier_basis
from src.gpflow_samp.gpflow_sampling.sampling.core import DenseSampler
from src.gpflow_samp.gpflow_sampling.kernels import Conv2d, DepthwiseConv2d


# ---- Exports


# ==============================================
#                                 fourier_priors
# ==============================================
def random_fourier(kernel: Kernel,
                    sample_shape: List,
                    num_bases: int,
                    basis: Callable = None,
                    dtype: Any = None,
                    name: str = None,
                    **kwargs):

  if dtype is None:
    dtype = default_float()

  if basis is None:
    basis = fourier_basis(kernel, num_bases=num_bases)

  weights = tf.random.normal(list(sample_shape) + [1, num_bases], dtype=dtype)
  return DenseSampler(weights=weights, basis=basis, name=name, **kwargs)

