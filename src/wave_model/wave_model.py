import threading
import typing
import numpy as np

from numpy.linalg import inv
from scipy.special import iv


class PowerSpectrum:
    def __init__(self, max_power_spectrum: int, max_harmonics: int):
        self.max_harmonics = max_harmonics
        self.freqs = np.zeros(max_power_spectrum * self.max_harmonics, dtype=np.float32)
        self.lengthscales = np.ones(max_power_spectrum * self.max_harmonics, dtype=np.float32)
        self.sds = np.ones(max_power_spectrum * self.max_harmonics, dtype=np.float32)

    def update_harmonic(self, harmonic_index, mean: int, std: float,
                        num_harmonics: int, lengthscale: float) -> None:
        idx = harmonic_index * self.max_harmonics
        self.freqs[idx:idx + self.max_harmonics] = np.zeros(self.max_harmonics)
        self.lengthscales[idx:idx + self.max_harmonics] = np.ones(self.max_harmonics)
        for i in range(num_harmonics):
            self.freqs[idx + i] = mean * (i + 1)
            self.lengthscales[idx + i] = lengthscale
            self.sds[idx + i] = std


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

    def z(self, x):
        return np.sqrt(2 / len(self.w)) * np.cos(x[:, None] @ self.w[None, :] + self.b)

    def prior(self, x, a):
        return self.z(x) @ self.weights

    def covariance_matrix(self, x1, x2, freqs, sds, lengthscales):
        x = x1 - x2
        temp = np.zeros((len(freqs), len(x1), len(x2)), dtype=np.float32)
        for i, freq in enumerate(freqs):
            temp[i] = self.covariance(x, 0, 1, 1)
        return np.exp(-0.5 * temp)

    def covariance(self, x, freq, sd, l):
        return sd ** 2 * np.exp(-0.5 * np.square(x / l))


class PeriodicPrior:
    def __init__(self, d: int, parameter_size: int):
        self.d = d
        self.weights = np.asarray(np.random.randn(2 * d), dtype=np.float32)
        self.calc = np.zeros((self.d, parameter_size))

    def resample(self):
        self.weights = np.asarray(np.random.randn(2 * self.d), dtype=np.float32)

    def update(self, lengthscale, sd):
        l = np.power(lengthscale, -2)
        for k in range(self.d):
            if self.d == 0:
                num = 1
            else:
                num = 2
            self.calc[k] = np.square(sd) * np.sqrt(num * iv(k, l) / np.exp(l))

    def z(self, x, freq):
        result = np.zeros((len(x), 2 * self.d), dtype=np.float32)

        for k in range(self.d):
            val = 2 * x[:, None] @ freq[None, :] * np.pi * k
            result[:, k] = np.cos(val) @ self.calc[k]
            result[:, self.d + k] = np.sin(val) @ self.calc[k]
        return result

    def prior(self, x, freq):
        return self.z(x, freq) @ self.weights

    def covariance_matrix(self, x1, x2, freqs, sds, lengthscales):
        x = x1 - x2
        temp = np.zeros((len(freqs), len(x1), len(x2)), dtype=np.float32)
        for i, freq in enumerate(freqs):
            temp[i] = self.covariance(x, freq, sds[i], lengthscales[i])
        return temp

    def covariance(self, x, freq, sd, l):
        return self.squared_exponential(np.sin(2 * np.pi * x * freq), sd, l)

    def squared_exponential(self, x, sd, l):
        return sd ** 2 * np.exp(-0.5 * np.square(x / l))


class SoundModel:
    def __init__(self, max_power_spectrums: int, max_freq: int, max_harmonics: int):
        self.inv = None
        self.max_freq = max_freq
        self.max_power_spectrum = max_power_spectrums
        self.max_harmonics = max_harmonics
        self.phases = None
        self.__power_spectrum = PowerSpectrum(self.max_power_spectrum, self.max_harmonics)
        self.lock = threading.Lock()
        self.prior = PeriodicPrior(100, self.__power_spectrum.freqs.size)
        self.x_train = None
        self.y_train = None

    def get_power_spectrum_histogram(self, idx: int,
                                     samples: int) -> typing.List[typing.Tuple[float, float]]:
        self.lock.acquire()
        x = np.linspace(0, 1, samples)
        k = np.zeros(len(x))
        idx *= self.max_harmonics
        max_freq = np.max(self.__power_spectrum.freqs[idx:idx + self.max_harmonics])
        for i in range(self.max_harmonics):
            k += self.prior.covariance(x, self.__power_spectrum.freqs[idx + i], self.__power_spectrum.sds[idx + i], self.__power_spectrum.lengthscales[idx + i])

        freqs = np.fft.fftfreq(len(x), 1 / samples)
        b = freqs < max_freq * 10
        freqs = freqs[b]
        yf = np.abs(np.fft.fft(k)[1:len(k)//2])
        self.lock.release()
        return list(zip(freqs, yf)), np.max(freqs), np.max(yf)

    def remove_power_spectrum(self, index, num_power_spectrums):
        for i in range(index, num_power_spectrums - 1):
            idx = i * self.max_harmonics
            next_idx = (i + 1) * self.max_harmonics
            self.__power_spectrum.freqs[idx:idx + self.max_harmonics] = self.__power_spectrum.freqs[next_idx:next_idx + self.max_harmonics]
            self.__power_spectrum.lengthscales[idx:idx + self.max_harmonics] = self.__power_spectrum.lengthscales[next_idx:next_idx + self.max_harmonics]
            self.__power_spectrum.sds[idx:idx + self.max_harmonics] = self.__power_spectrum.sds[next_idx:next_idx + self.max_harmonics]

        idx = (num_power_spectrums - 1) * self.max_harmonics
        self.__power_spectrum.freqs[idx:idx + self.max_harmonics] = np.zeros(self.max_harmonics, dtype=np.float32)
        self.__power_spectrum.lengthscales[idx:idx + self.max_harmonics] = np.ones(self.max_harmonics, dtype=np.float32)
        self.__power_spectrum.sds[idx:idx + self.max_harmonics] = np.zeros(self.max_harmonics, dtype=np.float32)
        self.prior.update(self.__power_spectrum.lengthscales, self.__power_spectrum.sds)

    def get_sum_all_power_spectrum_histogram(self) -> typing.List[typing.Tuple[float, float]]:
        self.lock.acquire()
        freqs = self.__power_spectrum.freqs.flatten()
        freqs = freqs[np.nonzero(freqs)]
        max_range = max(1000, freqs.max() + 100) if len(freqs) > 0 else 1000
        histogram, bin_edges = np.histogram(freqs, self.max_freq // 2, range=(0.1, max_range))
        self.lock.release()
        return list(zip(bin_edges, histogram))

    def interpolate_points(self, points: typing.List[typing.Tuple[float, float]]):
        self.lock.acquire()
        X, Y = [x for (x, _) in points], [y for (_, y) in points]
        if not X:
            X, Y = [0], [0]
        self.x_train, self.y_train = np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
        try:
            self.inv = inv(self.matrix_covariance(self.x_train, self.x_train))
        except:
            self.inv = None
        self.lock.release()

    def update_power_spectrum(self, harmonic_index: int, mean: int, std: float,
                              num_harmonics: int, lengthscale: float) -> None:
        with self.lock:
            self.__power_spectrum.update_harmonic(harmonic_index, mean, std, num_harmonics, lengthscale)
            self.prior.update(self.__power_spectrum.lengthscales, self.__power_spectrum.sds)

    def model_sound(self, sample_rate: int, chunk_duration: float, start_time: float) -> np.ndarray:
        x = np.linspace(start_time, start_time + chunk_duration, int(chunk_duration * sample_rate), endpoint=False,
                        dtype=np.float32)

        self.lock.acquire()
        sound = self.prior.prior(x, self.__power_spectrum.freqs)
        if not (self.inv is None or self.x_train is None or self.y_train is None):
            sound += self.update(x)
        self.lock.release()

        return sound

    def update_prior(self):
        self.prior.resample()

    def update(self, x_test):
        return self.matrix_covariance(x_test, self.x_train) @ self.inv @ (self.y_train - self.prior.prior(self.x_train, self.__power_spectrum.freqs))

    def matrix_covariance(self, x1, x2):
        return np.sum(self.prior.covariance_matrix(x1[:, None], x2, self.__power_spectrum.freqs, self.__power_spectrum.sds, self.__power_spectrum.lengthscales), axis=0)
