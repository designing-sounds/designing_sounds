#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf

from typing import List, Callable
from gpflow import inducing_variables
from gpflow.base import TensorLike
from gpflow.utilities import Dispatcher
from gpflow.kernels import Kernel, MultioutputKernel, LinearCoregionalization
from src.gpflow_samp.gpflow_sampling.sampling.updates import exact as exact_update
from src.gpflow_samp.gpflow_sampling.sampling.core import AbstractSampler, CompositeSampler
from src.gpflow_samp.gpflow_sampling.kernels import Conv2d
from src.gpflow_samp.gpflow_sampling.inducing_variables import InducingImages

# ---- Exports
# ==============================================
#                             decoupled_samplers
# ==============================================
def decoupled(kern: Kernel,
                        prior: AbstractSampler,
                        Z: TensorLike,
                        u: TensorLike,
                        *,
                        mean_function: Callable = None,
                        update_rule: Callable = exact_update,
                        join_rule: Callable = sum,
                        **kwargs):

  f = prior(Z, sample_axis=None)  # [S, M, L]
  update = update_rule(kern, Z, u, f, **kwargs)
  return CompositeSampler(samplers=[prior, update],
                          join_rule=join_rule,
                          mean_function=mean_function)