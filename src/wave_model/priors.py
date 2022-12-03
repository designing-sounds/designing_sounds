import numpy as np

from scipy.special import ive


class Prior:
    def __init__(self, d: int):
        self.weights = None
        self.d = d

    def update(self, a, b):
        pass

    def resample(self):
        if self.weights is None:
            return
        self.weights = np.asarray(np.random.randn(*self.weights.shape), dtype=np.float32)

    def z(self, x, freqs, sds, lengthscales):
        return None

    def prior(self, x, freqs, sds, lengthscales):
        return np.sum((self.z(x, freqs, sds, lengthscales) @ self.weights[:, :, None])[:, :, 0], axis=0)

    def covariance_matrix(self, x1, x2, freqs, sds, lengthscales):
        x = x1[:, None] - x2
        temp = np.zeros((len(freqs), len(x1), len(x2)), dtype=np.float32)
        for i, freq in enumerate(freqs):
            temp[i] = self.kernel(x, freq, sds[i], lengthscales[i])
        return np.exp(-0.5 * temp)

    def kernel(self, x, freq, sd, l):
        return None


class SquaredExpPrior(Prior):
    def __init__(self, d):
        Prior.__init__(self, d)
        self.w = np.asarray(np.random.randn(self.d), dtype=np.float32)
        self.b = np.asarray(np.random.uniform(0, 2 * np.pi), dtype=np.float32)

    def update(self, a, b):
        pass

    def resample(self):
        super().resample()
        self.w = np.asarray(np.random.randn(self.d), dtype=np.float32)
        self.b = np.asarray(np.random.uniform(0, 2 * np.pi), dtype=np.float32)

    def z(self, x, freqs, sds, lengthscales):
        return sds[:, None, None] * np.sqrt(2 / len(self.w)) * np.cos(
            ((1 / lengthscales)[:, None, None] * (x[:, None] @ self.w[None, :])[None, :, :]) + self.b)

    def kernel(self, x, _, sd, l):
        return squared_exponential(x, sd, l)


class PeriodicPrior(Prior):
    def __init__(self, d):
        Prior.__init__(self, d)
        self.calc = None
        self.ds = 2 * np.pi * np.arange(self.d // 2, dtype=np.float32)

    def update(self, lengthscale, sd):
        self.calc = np.zeros((sd.size, 1, self.d), dtype=np.float32)
        l = np.power(lengthscale, -2)
        for k in range(self.d // 2):
            if self.d == 0:
                num = 1
            else:
                num = 2
            self.calc[:, 0, k] = sd * np.sqrt(num * ive(k, l))
        self.calc[:, :, self.d // 2:] = self.calc[:, :, :self.d // 2]

    def z(self, x, freqs, sds, lengthscales):
        vals = freqs[:, None, None] * (x[:, None] @ self.ds[None, :])
        residue = np.empty((freqs.size, x.size, self.d), dtype=np.float32)
        residue[:, :, :self.d // 2] = np.cos(vals)
        residue[:, :, self.d // 2:] = np.sin(vals)
        return residue * self.calc

    def kernel(self, x, freq, sd, l):
        return squared_exponential(2 * np.sin(np.pi * x * freq), sd, l)


class MultPrior(Prior):
    def __init__(self, d):
        Prior.__init__(self, d)
        self.periodic = PeriodicPrior(self.d)
        self.squared = SquaredExpPrior(self.d)

    def resample(self):
        super().resample()
        self.squared.resample()
        self.periodic.resample()

    def update(self, lengthscale, sd):
        self.periodic.update(lengthscale, sd)

    def z(self, x, freqs, sds, lengthscales):
        return self.periodic.z(x, freqs, sds, lengthscales) * self.squared.z(x, freqs, sds, lengthscales)

    def kernel(self, x, freq, sd, l):
        return self.periodic.kernel(x, freq, sd, l) * self.squared.kernel(x, freq, sd, l)


def squared_exponential(x, sd, l):
    return sd ** 2 * np.exp(-0.5 * np.square(x / l))
