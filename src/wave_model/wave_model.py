import threading
import typing
import numpy as np

from numpy.linalg import inv
from scipy.special import iv

import tensorflow as tf
from gpflow import kernels
from gpflow.config import default_float as floatx
from src.gpflow_samp.gpflow_sampling.sampling import priors, updates, decoupled
import gpflow.kernels


class PowerSpectrum:
    def __init__(self, max_power_spectrum: int, max_harmonics: int):
        self.kernels = []
        self.num_kernels_per_spectrum = np.zeros(max_power_spectrum, dtype=int)
        self.max_harmonics = max_harmonics

    def update_harmonic(self, harmonic_index, mean: float, std: float,
                        num_harmonics: int, lengthscale: float) -> None:
        idx = np.sum(self.num_kernels_per_spectrum[:self.max_harmonics * harmonic_index])
        for i in range(num_harmonics):
            if idx + i >= len(self.kernels):
                base_kernel = kernels.SquaredExponential()
                base_kernel.variance.assign(std)
                base_kernel.lengthscales.assign(lengthscale)
                kernel = kernels.Periodic(base_kernel=base_kernel)
                kernel.period.assign(1 / mean)
                self.kernels.append(kernel)
            else:
                kernel = self.kernels[idx + i]
                kernel.base_kernel.variance.assign(std)
                kernel.period.assign(1 / mean)
                kernel.base_kernel.lengthscales.assign(lengthscale)

        for i in range(num_harmonics, self.max_harmonics):
            if idx + i < len(self.kernels):
                self.kernels.pop(idx + i)
        self.num_kernels_per_spectrum[self.max_harmonics * harmonic_index] = num_harmonics



class SquaredExpPrior:
    def __init__(self, d: int):
        self.w = np.asarray(np.random.randn(d), dtype=np.float32)
        self.b = np.asarray(np.random.uniform(0, 2 * np.pi), dtype=np.float32)
        self.weights = np.asarray(np.random.randn(d), dtype=np.float32)

    def update(self, a, b):
        pass

    def z(self, x):
        return np.sqrt(2 / len(self.w)) * np.cos(x[:, None] @ self.w[None, :] + self.b)

    def prior(self, x, a):
        return self.z(x) @ self.weights

class SoundModel:
    def __init__(self, max_power_spectrums: int, max_freq: int, max_harmonics: int):
        self.inv = None
        self.max_freq = max_freq
        self.max_power_spectrum = max_power_spectrums
        self.max_harmonics = max_harmonics
        self.phases = None
        self.__power_spectrum = PowerSpectrum(self.max_power_spectrum, self.max_harmonics)
        self.lock = threading.Lock()
        self.x_train = None
        self.y_train = None
        self.kernel = None
        self.posterior = None
        self.prior = None

    def get_power_spectrum_histogram(self, harmonic_index: int,
                                     _num_bins: int) -> typing.List[typing.Tuple[float, float]]:
        return [(0, 0)]

    def remove_power_spectrum(self, index, num_power_spectrums):
        for i in range(index, num_power_spectrums - 1):
            self.__power_spectrum.freqs[i] = self.__power_spectrum.freqs[i + 1]

        self.__power_spectrum.freqs[num_power_spectrums - 1] = np.zeros(self.max_samples_per_harmonic)
        self.kernel = self.sum_kernels()
        self.prior = priors.random_fourier(self.kernel, sample_shape=[1], num_bases=1024)

    def get_sum_all_power_spectrum_histogram(self) -> typing.List[typing.Tuple[float, float]]:
        return [(0, 0)]

    def interpolate_points(self, points: typing.List[typing.Tuple[float, float]]):
        self.lock.acquire()
        X, Y = [x for (x, _) in points], [y for (_, y) in points]
        X = np.array(X)[:, None]
        Y = np.array(Y)[:, None][None, :]
        if len(X) != 0:
            self.posterior = decoupled(self.kernel, self.prior, tf.convert_to_tensor(X), tf.convert_to_tensor(Y))
        self.lock.release()

    def update_power_spectrum(self, harmonic_index: int, mean: int, std: float,
                              num_harmonics: int, lengthscale: float) -> None:
        with self.lock:
            self.__power_spectrum.update_harmonic(harmonic_index, mean, std, num_harmonics, lengthscale)
            self.kernel = self.sum_kernels()
            self.prior = priors.random_fourier(self.kernel, sample_shape=[1], num_bases=1024)

    def model_sound(self, sample_rate: int, chunk_duration: float, start_time: float) -> np.ndarray:
        x = tf.convert_to_tensor(np.linspace(start_time, start_time + chunk_duration, int(chunk_duration * sample_rate), endpoint=False)[:, None])

        self.lock.acquire()
        if self.posterior is None:
            sound = self.prior(x).numpy().flatten()
        else:
            print(self.kernel.period)
            sound = self.posterior(x).numpy().flatten()
        self.lock.release()
        return sound

    def sum_kernels(self):
        return self.__power_spectrum.kernels[0]

