import threading
import typing
import numpy as np

from math import sqrt, log, log2, log10, cos, sin, tan, ceil, floor, fabs, factorial, exp
from numpy.linalg import inv


class PowerSpectrum:
    def __init__(self, max_harmonics: int, max_samples_per_harmonic: int):
        self.max_samples_per_harmonic = max_samples_per_harmonic
        self.freqs = np.zeros(max_harmonics * 5, dtype=np.float32)
        self.lengthscales = np.ones(max_harmonics * 5, dtype=np.float32)
        self.functions = {'sqrt': sqrt, 'pow': pow, 'log': log, 'log2': log2, 'log10': log10, 'cos': cos, 'sin': sin,
                          'tan': tan, 'ceil': ceil, 'floor': floor, 'abs': fabs, 'factorial': factorial, 'exp': exp}

    def update_harmonic(self, harmonic_index, mean: int, std: float,
                        num_harmonics: int, decay_function: str) -> None:
        idx = harmonic_index * 5
        self.freqs[idx:idx + 5] = np.zeros(5)
        self.lengthscales[idx:idx + 5] = np.ones(5)
        for i in range(num_harmonics):
            self.freqs[idx + i] = mean * (i + 1)
            self.lengthscales[idx + i] = std


class Prior:
    def __init__(self, d: int):
        self.w = np.asarray(np.random.randn(d), dtype=np.float32)
        self.b = np.asarray(np.random.uniform(0, 2 * np.pi), dtype=np.float32)
        self.weights = np.asarray(np.random.randn(d), dtype=np.float32)

    def z(self, x):
        return np.sqrt(2 / len(self.w)) * np.cos(x[:, None] @ self.w[None, :] + self.b)

    def prior(self, x):
        return self.z(x) @ self.weights


class SoundModel:
    def __init__(self, max_harmonics: int, max_samples_per_harmonic: int, max_freq: int):
        self.inv = None
        self.max_freq = max_freq
        self.max_harmonics = max_harmonics
        self.max_samples_per_harmonic = max_samples_per_harmonic
        self.amps = np.asarray(np.random.randn(self.max_samples_per_harmonic * self.max_harmonics), dtype=np.float32)
        self.phases = None
        self.__power_spectrum = PowerSpectrum(self.max_harmonics, self.max_samples_per_harmonic)
        self.samples_per_harmonic = np.zeros(self.max_harmonics)
        self.lock = threading.Lock()
        prior = Prior(100)
        self.prior = prior.prior
        self.x_train = None
        self.y_train = None

    def get_power_spectrum_histogram(self, harmonic_index: int,
                                     _num_bins: int) -> typing.List[typing.Tuple[float, float]]:
        self.lock.acquire()
        freqs = self.__power_spectrum.freqs[harmonic_index * 5: harmonic_index * 5 + 5]
        freqs = freqs[np.nonzero(freqs)]
        max_range = max(1000, freqs.max() + 100) if len(freqs) > 0 else 1000
        histogram, bin_edges = np.histogram(freqs, self.max_freq // 2, range=(0.1, max_range))
        self.lock.release()
        return list(zip(bin_edges, histogram))

    def remove_power_spectrum(self, index, num_power_spectrums):
        for i in range(index, num_power_spectrums - 1):
            self.__power_spectrum.freqs[i] = self.__power_spectrum.freqs[i + 1]
            self.samples_per_harmonic[i] = self.samples_per_harmonic[i + 1]

        self.__power_spectrum.freqs[num_power_spectrums - 1] = np.zeros(self.max_samples_per_harmonic)
        self.samples_per_harmonic[num_power_spectrums - 1] = 0

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

    def update_power_spectrum(self, harmonic_index: int, mean: int, std: float, num_harmonic_samples: int,
                              num_harmonics: int, decay_function: str) -> None:
        with self.lock:
            self.__power_spectrum.update_harmonic(harmonic_index, mean, std, num_harmonics, None)
            self.samples_per_harmonic[harmonic_index] = num_harmonic_samples

    def model_sound(self, sample_rate: int, chunk_duration: float, start_time: float) -> np.ndarray:
        x = np.linspace(start_time, start_time + chunk_duration, int(chunk_duration * sample_rate), endpoint=False,
                        dtype=np.float32)

        self.lock.acquire()
        if self.inv is None or self.x_train is None or self.y_train is None:
            sound = self.prior(x)
        else:
            sound = self.prior(x) + self.update(x)
        self.lock.release()

        return sound

    def update(self, x_test):
        return self.matrix_covariance(x_test, self.x_train) @ self.inv @ (self.y_train - self.prior(self.x_train))

    def covariance(self, x1, x2):
        temp2 = (x1 - x2)
        freqs = self.__power_spectrum.freqs
        temp = np.zeros((len(freqs), len(x1), len(x2)), dtype=np.float32)
        for i, freq in enumerate(freqs):
            temp[i] = np.square(temp2) / self.__power_spectrum.lengthscales[i]
        return np.exp(-0.5 * temp)

    def matrix_covariance(self, x1, x2):
        cov = np.sum(self.covariance(x1[:, None], x2), axis=0)
        return cov
