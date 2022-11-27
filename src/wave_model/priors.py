import numpy as np

from scipy.special import iv


class SquaredExpPrior:
    def __init__(self, d: int, nothing):
        self.w = np.asarray(np.random.randn(d), dtype=np.float32)
        self.b = np.asarray(np.random.uniform(0, 2 * np.pi), dtype=np.float32)
        self.weights = np.asarray(np.random.randn(d), dtype=np.float32)
        self.d = d

    def update(self, a, b):
        pass

    def resample(self):
        self.weights = np.asarray(np.random.randn(self.d), dtype=np.float32)

    def z(self, x, sds, lengthscales):
        return sds * np.sqrt(2 / len(self.w)) * np.cos((x[:, None] / lengthscales @ self.w[None, :]) + self.b)

    def prior(self, x, _, sds, lengthscales):
        return self.z(x, sds, lengthscales) @ self.weights

    def covariance_matrix(self, x1, x2, freqs, sds, lengthscales):
        x = x1 - x2
        temp = np.zeros((len(freqs), len(x1), len(x2)), dtype=np.float32)
        for i, freq in enumerate(freqs):
            temp[i] = self.covariance(x, 0, sds[i], lengthscales[i])
        return np.exp(-0.5 * temp)

    def covariance(self, x, _, sd, l):
        return sd ** 2 * np.exp(-0.5 * np.square(x / l))


class PeriodicPrior:
    def __init__(self, d: int, parameter_size: int):
        self.d = d
        self.cos_weights = np.asarray(np.random.randn(d), dtype=np.float32)
        self.sin_weights = np.asarray(np.random.randn(d), dtype=np.float32)
        self.calc = np.zeros((self.d, parameter_size, 1), dtype=np.float32)

    def resample(self):
        self.cos_weights = np.asarray(np.random.randn(self.d), dtype=np.float32)
        self.sin_weights = np.asarray(np.random.randn(self.d), dtype=np.float32)

    def update(self, lengthscale, sd):
        l = np.power(lengthscale, -2)
        for k in range(self.d):
            if self.d == 0:
                num = 1
            else:
                num = 2
            self.calc[k, :, 0] = np.square(sd) * np.sqrt(num * iv(k, l) / np.exp(l))

    def prior(self, x, freqs, sds, lengthscales):
        ds = 2 * np.pi * np.asarray(np.arange(self.d), dtype=np.float32)
        vals = ds[:, None, None] * (x[:, None] @ freqs[None, :])[None, :, :]
        return (np.cos(vals) @ self.calc)[:, :, 0].T @ self.cos_weights + (np.sin(vals) @ self.calc)[:, :,
                                                                          0].T @ self.sin_weights

    def covariance_matrix(self, x1, x2, freqs, sds, lengthscales):
        x = difference_matrix(x1, x2)
        temp = np.zeros((len(freqs), len(x1), len(x2)), dtype=np.float32)
        for i, freq in enumerate(freqs):
            temp[i] = self.covariance(x, freq, sds[i], lengthscales[i])
        return temp

    def covariance(self, x, freq, sd, l):
        return self.squared_exponential(np.sin(2 * np.pi * x * freq), sd, l)

    def squared_exponential(self, x, sd, l):
        return sd ** 2 * np.exp(-0.5 * np.square(x / l))


def difference_matrix(x1, x2):
    return x1[:, None] - x2
