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


class SoundModel:
    def __init__(self, max_harmonics: int, max_samples_per_harmonic: int, max_freq: int):
        self.max_freq = max_freq
        self.max_harmonics = max_harmonics
        self.max_samples_per_harmonic = max_samples_per_harmonic
        self.amps = np.asarray(np.random.randn(self.max_samples_per_harmonic * self.max_harmonics), dtype=np.float32)
        self.phases = None
        self.__power_spectrum = PowerSpectrum(self.max_harmonics, self.max_samples_per_harmonic)
        self.samples_per_harmonic = np.zeros(self.max_harmonics)
        self.lock = threading.Lock()

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
        self.X = np.array(X, dtype=np.float32)
        try:
            self.mat = inv(self.matrix_covariance(self.X, self.X)) @ np.array(Y, dtype=np.float32).T
        except:
            self.mat = None
        self.lock.release()

    def update_power_spectrum(self, harmonic_index: int, mean: int, std: float, num_harmonic_samples: int,
                              num_harmonics: int, decay_function: str) -> None:
        with self.lock:
            self.__power_spectrum.update_harmonic(harmonic_index, mean, std, num_harmonics, None)
            self.samples_per_harmonic[harmonic_index] = num_harmonic_samples

    def model_sound(self, sample_rate: int, chunk_duration: float, start_time: float) -> np.ndarray:
        x = np.linspace(start_time, start_time + chunk_duration, int(chunk_duration * sample_rate), endpoint=False, dtype=np.float32)

        self.lock.acquire()
        if self.mat is None:
            sound = np.array([0])
        else:
            sound = self.matrix_covariance(x, self.X) @ self.mat
        self.lock.release()

        return sound

    def covariance(self, x1, x2):
        temp2 = (x1 - x2)
        freqs = self.__power_spectrum.freqs
        temp = np.zeros((len(freqs), len(x1), len(x2)), dtype=np.float32)
        for i, freq in enumerate(freqs):
            temp[i] = np.square(np.sin(np.pi * temp2 * freq)) / self.__power_spectrum.lengthscales[i]
        return np.exp(-0.5 * temp)

    def matrix_covariance(self, x1, x2):
        cov = np.sum(self.covariance(x1[:, None], x2), axis=0)
        return cov
