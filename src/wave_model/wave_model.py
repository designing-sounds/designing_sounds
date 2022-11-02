import threading
import typing

import numpy as np

from numpy.linalg import inv


class PowerSpectrum:
    def __init__(self, max_harmonics: int, max_samples_per_harmonic: int):
        self.max_samples_per_harmonic = max_samples_per_harmonic
        self.harmonics = np.zeros((max_harmonics, self.max_samples_per_harmonic), dtype=np.float32)

    def update_harmonic(self, harmonic_index, mean: int, std: float, num_harmonic_samples: int) -> None:
        freqs = np.random.randn(num_harmonic_samples) * std + mean
        self.harmonics[harmonic_index] = np.zeros(self.max_samples_per_harmonic)
        self.harmonics[harmonic_index, :num_harmonic_samples] = freqs


class SoundModel:
    def __init__(self, max_harmonics: int, max_samples_per_harmonic: int, max_freq: int):
        self.max_freq = max_freq
        self.max_harmonics = max_harmonics
        self.max_samples_per_harmonic = max_samples_per_harmonic
        self.amps = np.asarray(np.random.randn(self.max_samples_per_harmonic * self.max_harmonics), dtype=np.float32)
        self.phases = None
        self.power_spectrum = PowerSpectrum(self.max_harmonics, self.max_samples_per_harmonic)
        self.samples_per_harmonic = np.zeros(self.max_harmonics)
        self.lock = threading.Lock()
        self.kern = None
        self.m = None
        self.X = np.array([0])
        self.Y = None
        self.interpolate_points([])

    def get_power_spectrum_histogram(self, harmonic_index: int, num_bins: int) -> typing.List[typing.Tuple[float, float]]:
        self.lock.acquire()
        freqs = self.power_spectrum.harmonics[harmonic_index]
        freqs = freqs[np.nonzero(freqs)]
        histogram, bin_edges = np.histogram(freqs, self.max_freq // 2, range=(0.1, 1000))
        self.lock.release()
        return list(zip(bin_edges, histogram))

    def get_sum_all_power_spectrum_histogram(self) -> typing.List[typing.Tuple[float, float]]:
        self.lock.acquire()
        freqs = self.power_spectrum.harmonics.flatten()
        freqs = freqs[np.nonzero(freqs)]
        histogram, bin_edges = np.histogram(freqs, self.max_freq // 2, range=(0.1, 1000))
        self.lock.release()
        return list(zip(bin_edges, histogram))

    def interpolate_points(self, points: typing.List[typing.Tuple[float, float]]):
        self.lock.acquire()
        X, Y = [x for (x, _) in points], [y for (_, y) in points]
        self.X, self.Y = np.array(X), np.array(Y)
        self.lock.release()

    def update_power_spectrum(self, harmonic_index: int, mean: int, std: float, num_harmonic_samples: int) -> None:
        self.lock.acquire()
        self.power_spectrum.update_harmonic(harmonic_index, mean, std, num_harmonic_samples)
        self.samples_per_harmonic[harmonic_index] = num_harmonic_samples
        self.lock.release()

    def model_sound(self, sample_rate: int, chunk_duration: float, start_time: float) -> np.ndarray:
        if self.X.size == 0:
            return np.array([])
        x = np.linspace(start_time, start_time + chunk_duration, int(chunk_duration * sample_rate), endpoint=False)

        self.lock.acquire()
        sound = self.matrix_covariance(x, self.X, 440) @ inv(self.matrix_covariance(self.X, self.X, 440)) @ self.Y.T
        sound = np.asarray(sound, dtype=np.float32)
        self.lock.release()

        return sound

    def covariance(self, x1, x2, period, lengthscale):
        return np.exp(-0.5 * np.square(np.sin(np.pi * (x1 - x2) / period)) / lengthscale)

    def matrix_covariance(self, x1, x2, freq):
        n = len(x1)
        m = len(x2)
        res = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                res[i, j] = self.covariance(x1[i], x2[j], 1 / freq, 1)
        return res
