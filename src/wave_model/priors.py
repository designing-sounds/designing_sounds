import numpy as np

from scipy.special import ive


class Prior:
    def __init__(self, approx_dim: int):
        self.weights = None
        self.approx_dim = approx_dim

    def update(self, _, __):
        pass

    def resample(self):
        if self.weights is None:
            return
        self.weights = np.asarray(np.random.randn(*self.weights.shape), dtype=np.float32)

    def phi(self, _, __, ___, ____, _____, _______):
        return np.array([])

    def prior(self, x, freqs, sds, lengthscales, sds_squared, lengthscales_squared):
        temp = self.phi(x, freqs, sds, lengthscales, sds_squared, lengthscales_squared) @ self.weights[:, :, None]
        return np.sum(temp[:, :, 0], axis=0)

    def covariance_matrix(self, x, freqs, sds, lengthscales, sds_squared, lengthscales_squared):
        temp = np.zeros((len(freqs), x.shape[0], x.shape[1]), dtype=np.float32)
        for i, freq in enumerate(freqs):
            temp[i] = self.kernel(x, freq, sds[i], lengthscales[i], sds_squared[i], lengthscales_squared[i])
        return np.exp(-0.5 * temp)

    def kernel(self, _, __, ___, ____, _____, ______):
        return 0


class SquaredExpPrior(Prior):
    def __init__(self, approx_dim):
        Prior.__init__(self, approx_dim)
        self.squared_w = np.asarray(np.random.randn(self.approx_dim), dtype=np.float32)
        self.bias = np.asarray(np.random.uniform(0, 2 * np.pi), dtype=np.float32)

    def update(self, _, __):
        pass

    def resample(self):
        super().resample()
        self.squared_w = np.asarray(np.random.randn(self.approx_dim), dtype=np.float32)
        self.bias = np.asarray(np.random.uniform(0, 2 * np.pi), dtype=np.float32)

    def phi(self, x, _, sds, lengthscales, __, ___):
        return sds[:, None, None] * np.sqrt(2 / len(self.squared_w)) * np.cos(
            ((1 / lengthscales)[:, None, None] * (x[:, None] @ self.squared_w[None, :])[None, :, :]) + self.bias)

    def kernel(self, x, _, sd, lengthscale, __, ___):
        return squared_exponential(x, sd, lengthscale)


class PeriodicPrior(Prior):
    def __init__(self, approx_dim):
        Prior.__init__(self, approx_dim)
        self.calc = None
        self.nums = 2 * np.pi * np.arange(self.approx_dim // 2, dtype=np.float32)

    def update(self, lengthscale, sd):
        self.calc = np.zeros((sd.size, 1, self.approx_dim), dtype=np.float32)
        lengthscale = np.power(lengthscale, -2)
        for k in range(self.approx_dim // 2):
            if self.approx_dim == 0:
                num = 1
            else:
                num = 2
            self.calc[:, 0, k] = sd * np.sqrt(num * ive(k, lengthscale))
        self.calc[:, :, self.approx_dim // 2:] = self.calc[:, :, :self.approx_dim // 2]

    def phi(self, x, freqs, _, __, ___, ____):
        vals = freqs[:, None, None] * (x[:, None] @ self.nums[None, :])
        residue = np.empty((freqs.size, x.size, self.approx_dim), dtype=np.float32)
        residue[:, :, :self.approx_dim // 2] = np.cos(vals)
        residue[:, :, self.approx_dim // 2:] = np.sin(vals)
        return residue * self.calc

    def kernel(self, x, freq, sd, lengthscale, _, __):
        return squared_exponential(2 * np.sin(np.pi * x * freq), sd, lengthscale)


class MultPrior(Prior):
    def __init__(self, approx_dim):
        Prior.__init__(self, approx_dim)
        self.periodic = PeriodicPrior(self.approx_dim)
        self.squared = SquaredExpPrior(self.approx_dim)

    def resample(self):
        super().resample()
        self.squared.resample()
        self.periodic.resample()

    def update(self, lengthscale, sd):
        self.periodic.update(lengthscale, sd)

    def phi(self, x, freqs, sds, lengthscales, sds_squared, lengthscales_squared):
        return self.periodic.phi(x, freqs, sds, lengthscales, 0, 0) * self.squared.phi(x, freqs, sds_squared,
                                                                                       lengthscales_squared, 0, 0)

    def kernel(self, x, freq, sd, lengthscale, sd_squared, l_squared):
        return self.periodic.kernel(x, freq, sd, lengthscale, 0, 0) * self.squared.kernel(x, freq,
                                                                                          sd_squared, l_squared, 0, 0)


def squared_exponential(x, sd, lengthscale):
    return sd ** 2 * np.exp(-0.5 * np.square(x / lengthscale))
