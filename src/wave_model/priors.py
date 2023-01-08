import numpy as np

from scipy.special import ive
from typing import Union


class Prior:
    def __init__(self, approx_dim: int):
        self.weights = None
        self.approx_dim = approx_dim

    def update(self, _, __, ___) -> None:
        pass

    def resample(self) -> None:
        if self.weights is None:
            return
        self.weights = np.asarray(np.random.randn(*self.weights.shape), dtype=np.float32)

    def phi(self, x: np.ndarray, freqs: np.ndarray, sds: np.ndarray, lengthscales: np.ndarray,
            sds_squared: np.ndarray, lengthscales_squared: np.ndarray) -> np.ndarray:
        return np.array([])

    def prior(self, x: np.ndarray, freqs: np.ndarray, sds: np.ndarray, lengthscales: np.ndarray,
              sds_squared: np.ndarray, lengthscales_squared: np.ndarray) -> np.ndarray:
        temp = self.phi(x, freqs, sds, lengthscales, sds_squared, lengthscales_squared) @ self.weights[:, :, None]
        return np.sum(temp[:, :, 0], axis=0)

    def covariance_matrix(self, x_1: np.ndarray, x_2: np.ndarray, freqs: np.ndarray, sds: np.ndarray,
                          lengthscales: np.ndarray, sds_squared: np.ndarray,
                          lengthscales_squared: np.ndarray) -> np.ndarray:
        temp = np.zeros((len(freqs), len(x_1), len(x_2)), dtype=np.float32)
        for i, freq in enumerate(freqs):
            x = x_1[:, None] - x_2
            temp[i] = self.kernel(x, freq, sds[i], lengthscales[i], sds_squared[i], lengthscales_squared[i])
        return temp

    def kernel(self, x: Union[float, np.ndarray], freq: float, sd: float, lengthscale: float, sd_squared: float,
               l_squared: float) -> Union[float, np.ndarray]:
        return 0


class SquaredExpPrior(Prior):
    def __init__(self, approx_dim: int):
        Prior.__init__(self, approx_dim)
        self.squared_w = np.asarray(np.random.randn(self.approx_dim), dtype=np.float32)
        self.bias = np.asarray(np.random.uniform(0, 2 * np.pi), dtype=np.float32)

    def resample(self) -> None:
        super().resample()
        self.squared_w = np.asarray(np.random.randn(self.approx_dim), dtype=np.float32)
        self.bias = np.asarray(np.random.uniform(0, 2 * np.pi), dtype=np.float32)

    def phi(self, x: np.ndarray, _, sds: np.ndarray, lengthscales: np.ndarray, __, ___) -> np.ndarray:
        return sds[:, None, None] * np.sqrt(2 / len(self.squared_w)) * np.cos(
            ((1 / lengthscales)[:, None, None] * (x[:, None] @ self.squared_w[None, :])[None, :, :]) + self.bias)

    def kernel(self, x: Union[float, np.ndarray], _, sd: float, lengthscale: float, __,
               ___) -> Union[float, np.ndarray]:
        return squared_exponential(x, sd, lengthscale)


def to_period(x: np.ndarray, period: float) -> np.ndarray:
    return x - period * np.array(np.floor(x / period), int)


class PeriodicPrior(Prior):
    def __init__(self, approx_dim: int):
        Prior.__init__(self, approx_dim)
        self.temp = None
        self.calc = None
        self.nums = 2 * np.pi * np.arange(self.approx_dim // 2, dtype=np.float32)

    def update(self, freqs: np.ndarray, lengthscales: np.ndarray, sds: np.ndarray) -> None:
        self.calc = np.zeros((sds.size, 1, self.approx_dim), dtype=np.float32)
        lengthscales = np.power(lengthscales, -2)
        for k in range(self.approx_dim // 2):
            if self.approx_dim == 0:
                num = 1
            else:
                num = 2
            self.calc[:, 0, k] = sds * np.sqrt(num * ive(k, lengthscales))
        self.calc[:, :, self.approx_dim // 2:] = self.calc[:, :, :self.approx_dim // 2]
        self.temp = freqs[:, None, None] * self.nums[None, :]

    def covariance_matrix(self, x_1: np.ndarray, x_2: np.ndarray, freqs: np.ndarray, sds: np.ndarray,
                          lengthscales: np.ndarray, sds_squared: np.ndarray,
                          lengthscales_squared: np.ndarray) -> np.ndarray:
        temp = np.zeros((len(freqs), len(x_1), len(x_2)), dtype=np.float32)
        for i, freq in enumerate(freqs):
            period = 1 / freq
            x = to_period(x_1, period)[:, None] - to_period(x_2, period)
            temp[i] = self.kernel(x, freq, sds[i], lengthscales[i], sds_squared[i], lengthscales_squared[i])
        return temp

    def phi(self, x: np.ndarray, freqs: np.ndarray, _, __, ___, ____) -> np.ndarray:
        vals = x[:, None] @ self.temp
        residue = np.empty((freqs.size, x.size, self.approx_dim), dtype=np.float32)
        residue[:, :, :self.approx_dim // 2] = np.cos(vals)
        residue[:, :, self.approx_dim // 2:] = np.sin(vals)
        return residue * self.calc

    def kernel(self, x: Union[float, np.ndarray], freq: float, sd: float, lengthscale: float, _,
               __) -> Union[float, np.ndarray]:
        return squared_exponential(np.sin(np.pi * x * freq), sd, lengthscale)


class MultPrior(Prior):
    def __init__(self, approx_dim: int):
        Prior.__init__(self, approx_dim)
        self.periodic = PeriodicPrior(self.approx_dim)
        self.squared = SquaredExpPrior(self.approx_dim)

    def resample(self) -> None:
        super().resample()
        self.squared.resample()
        self.periodic.resample()

    def update(self, freqs: np.ndarray, lengthscales: np.ndarray, sds: np.ndarray) -> None:
        self.periodic.update(freqs, lengthscales, sds)

    def phi(self, x: np.ndarray, freqs: np.ndarray, sds: np.ndarray, lengthscales: np.ndarray,
            sds_squared: np.ndarray, lengthscales_squared: np.ndarray) -> np.ndarray:
        return self.periodic.phi(x, freqs, sds, lengthscales, np.array([]),
                                 np.array([])) * self.squared.phi(x, freqs, sds_squared, lengthscales_squared,
                                                                  np.array([]), np.array([]))
    def kernel(self, x: Union[float, np.ndarray], freq: float, sd: float, lengthscale: float, sd_squared: float,
               l_squared: float) -> Union[float, np.ndarray]:
        return self.periodic.kernel(x, freq, sd, lengthscale, np.array([]),
                                    np.array([])) * self.squared.kernel(x, freq, sd_squared, l_squared, np.array([]),
                                                                        np.array([]))


def squared_exponential(x: Union[float, np.ndarray], sd: float, lengthscale: float) -> None:
    return sd ** 2 * np.exp(-0.5 * np.square(x / lengthscale))
