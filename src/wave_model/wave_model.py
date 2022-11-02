import threading
import typing

import numpy as np

from numpy.linalg import inv


class PowerSpectrum:
    def __init__(self, max_harmonics: int, max_samples_per_harmonic: int):
        self.max_samples_per_harmonic = max_samples_per_harmonic
        self.harmonics = []

        for i in range(max_harmonics):
            self.harmonics.append(None)
            self.update_harmonic(i, 0, 0, max_samples_per_harmonic)

    def update_harmonic(self, harmonic_index, mean: int, std: float, num_harmonic_samples: int) -> None:
        # new_freqs = np.random.randn(num_harmonic_samples) * std + mean
        # freqs = np.zeros(self.max_samples_per_harmonic)
        # freqs[:num_harmonic_samples] = new_freqs
        # kern_sum = None
        # for freq in freqs:
        #     kern = gpflow.kernels.Periodic(base_kernel=gpflow.kernels.SquaredExponential())
        #     kern.period.assign(1 / freq) if freq != 0 else kern.period.assign(0.01)
        #     kern.base_kernel.lengthscales.assign(0.5)
        #     sqexp = gpflow.kernels.SquaredExponential()
        #     sqexp.lengthscales.assign(7.0)
        #     kern = kern * sqexp
        #
        #     if not kern_sum:
        #         kern_sum = kern
        #     kern_sum += kern
        # assert kern_sum
        # self.harmonics[harmonic_index] = kern_sum
        pass


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

    def get_power_spectrum_histogram(self, harmonic_index: int, num_bins: int) -> typing.List[
        typing.Tuple[float, float]]:
        return [(0, 0)]

    def get_sum_all_power_spectrum_histogram(self) -> typing.List[typing.Tuple[float, float]]:
        return [(0, 0)]

    def interpolate_points(self, points: typing.List[typing.Tuple[float, float]]):
        self.lock.acquire()
        X, Y = [x for (x, _) in points], [y for (_, y) in points]
        self.X, self.Y = np.array(X), np.array(Y)
        self.lock.release()

    def update_power_spectrum(self, harmonic_index: int, mean: int, std: float, num_harmonic_samples: int) -> None:
        self.lock.acquire()
        self.power_spectrum.update_harmonic(harmonic_index, mean, std, num_harmonic_samples)
        self.lock.release()

    def model_sound(self, sample_rate: int, chunk_duration: float, start_time: float) -> np.ndarray:
        if self.X.size == 0:
            return np.array([])
        x = np.linspace(start_time, start_time + chunk_duration, int(chunk_duration * sample_rate), endpoint=False)

        self.lock.acquire()
        # predict_stats = self.m.predict_f(x[:, None])
        # sound, _ = [d.numpy() for d in predict_stats]
        # sound = np.asarray(sound.flatten(), dtype=np.float32)
        sound = self.matrix_covariance(x, self.X) @ inv(self.matrix_covariance(self.X, self.X)) @ self.Y.T
        sound = np.asarray(sound, dtype=np.float32)
        self.lock.release()

        return sound

    def covariance(self, x1, x2, period, lengthscale):
        return np.exp(-0.5 * np.square(np.sin(np.pi * (x1 - x2) / period)) / lengthscale)

    def matrix_covariance(self, x1, x2):
        n = len(x1)
        m = len(x2)
        res = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                res[i, j] = self.covariance(x1[i], x2[j], 1 / 440, 1)
        return res
