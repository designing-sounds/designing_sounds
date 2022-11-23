import numpy as np
import tensorflow as tf

from gpflow.config import default_float as floatx
from gpflow.kernels import Matern52, Periodic, SquaredExponential
from gpflow_sampling.models import PathwiseGPR
import matplotlib.pyplot as plt
tf.random.set_seed(1)

kernel = Matern52()
freq = 200
lengthscale = 0.1
#kernel.period.assign((1 / freq) if freq != 0 else 1)
#kernel.base_kernel.lengthscales.assign(lengthscale)
noise2 = 1e-3  # measurement noise variance

xmin = 0.15  # range over which we observe
xmax = 0.50  # the behavior of a function $f$
X = tf.convert_to_tensor(np.linspace(xmin, xmax, 3)[:, None])
y = tf.convert_to_tensor(np.array([[-1], [2], [3]]), dtype=floatx())

model = PathwiseGPR(data=(X, y), kernel=kernel, noise_variance=noise2)
Xnew = np.linspace(0, 1, 1024)[:, None]

paths = model.generate_paths(num_samples=2, num_bases=1024)  # returned paths are deterministic!
f_orig = paths(Xnew)
print(f"shape(f) = {f_orig.shape}")

model.set_paths(paths)

fig, ax = plt.subplots(figsize=(7, 3))

f_plot = tf.squeeze(model.predict_f_samples(Xnew))


# Plot some sample paths
ax.scatter(X, y)
ax.plot(Xnew, f_plot[0], alpha=0.5, linewidth=0.5, color='tab:blue')