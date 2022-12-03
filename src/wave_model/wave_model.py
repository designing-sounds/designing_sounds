import threading
import typing
from typing import Tuple, List, Any, Union

import numpy as np
from numpy import ndarray

from numpy.linalg import inv

from src.wave_model.power_spectrum import PowerSpectrum
from src.wave_model.priors import PeriodicPrior, SquaredExpPrior, MultPrior


class SoundModel:
    def __init__(self, max_power_spectrums: int, max_freq: int, max_harmonics: int):
        self.inv = None
        self.max_freq = max_freq
        self.max_power_spectrum = max_power_spectrums
        self.max_harmonics = max_harmonics
        self.prior = PeriodicPrior(50)
        self.__power_spectrum = PowerSpectrum(self.max_power_spectrum, self.max_harmonics, self.prior)
        self.lock = threading.Lock()
        self.x_train = None
        self.y_train = None
        self.noise = 0
        self.variance = 0

    def get_power_spectrum_histogram(self, idx: int, samples: int) -> Tuple[
        List[Tuple[Any, Any]], Union[ndarray, int, float, complex], Union[ndarray, int, float, complex]]:
        with self.lock:
            x = np.linspace(0, 1, samples)
            k = np.zeros(len(x))
            num_kernels = self.__power_spectrum.num_kernels_per_spectrum[idx]
            idx = np.sum(self.__power_spectrum.num_kernels_per_spectrum[:idx])
            for i in range(num_kernels):
                k += self.prior.kernel(x, self.__power_spectrum.freqs[idx + i], self.__power_spectrum.sds[idx + i],
                                       self.__power_spectrum.lengthscales[idx + i])

            freqs = np.fft.fftfreq(samples, 1 / samples)
            freqs = [0] + freqs[1:samples // 2]
            yf = [0] + np.abs(np.fft.fft(k)[1:samples // 2])

        return list(zip(freqs, yf)), np.max(yf)

    def remove_power_spectrum(self, index):
        idx = np.sum(self.__power_spectrum.num_kernels_per_spectrum[:index])
        num_kernels = self.__power_spectrum.num_kernels_per_spectrum[index]
        indices = np.arange(idx, idx + num_kernels)
        self.__power_spectrum.freqs = np.delete(self.__power_spectrum.freqs, indices)
        self.__power_spectrum.lengthscales = np.delete(self.__power_spectrum.lengthscales, indices)
        self.__power_spectrum.sds = np.delete(self.__power_spectrum.sds, indices)
        self.prior.update(self.__power_spectrum.lengthscales, self.__power_spectrum.sds)

    def get_sum_all_power_spectrum_histogram(self, samples: int) -> typing.List[typing.Tuple[float, float]]:
        with self.lock:
            x = np.linspace(0, 1, samples)
            k = np.zeros(len(x))
            for i in range(len(self.__power_spectrum.freqs)):
                k += self.prior.kernel(x, self.__power_spectrum.freqs[i], self.__power_spectrum.sds[i],
                                       self.__power_spectrum.lengthscales[i])

            freqs = np.fft.fftfreq(samples, 1 / samples)
            freqs = [0] + freqs[1:samples // 2]
            yf = [0] + np.abs(np.fft.fft(k)[1:samples // 2])

        return list(zip(freqs, yf)), np.max(yf)

    def interpolate_points(self, points: typing.List[typing.Tuple[float, float]]):
        with self.lock:
            self.x_train = np.array([x for (x, _) in points], dtype=np.float32)
            self.y_train = np.array([y for (_, y) in points], dtype=np.float32)
            self.inv = None
            if self.x_train.size != 0:
                try:
                    self.inv = inv(self.matrix_covariance(self.x_train, self.x_train) + self.variance * np.eye(len(self.x_train)))
                except:
                    pass

    def update_power_spectrum(self, harmonic_index: int, mean: int, std: float,
                              num_harmonics: int, lengthscale: float) -> None:
        with self.lock:
            self.__power_spectrum.update_harmonic(harmonic_index, mean, std, num_harmonics, lengthscale)
            self.prior.update(self.__power_spectrum.lengthscales, self.__power_spectrum.sds)

    def model_sound(self, sample_rate: int, chunk_duration: float, start_time: float) -> np.ndarray:
        x = np.linspace(start_time, start_time + chunk_duration, int(chunk_duration * sample_rate), endpoint=False,
                        dtype=np.float32)

        self.lock.acquire()
        sound = self.prior.prior(x, self.__power_spectrum.freqs, self.__power_spectrum.sds,
                                 self.__power_spectrum.lengthscales)
        if not (self.inv is None or self.x_train is None or self.y_train is None):
            sound += self.update(x)
        self.lock.release()

        return sound

    def update_prior(self):
        self.prior.resample()
        self.update_noise()

    def update_noise(self):
        self.noise = np.random.normal(0, np.sqrt(self.variance))

    def update(self, x_test):
        return self.matrix_covariance(x_test, self.x_train) @ self.inv @ (self.y_train - self.noise
                                                                          - self.prior.prior(self.x_train,
                                                                                             self.__power_spectrum.freqs,
                                                                                             self.__power_spectrum.sds,
                                                                                             self.__power_spectrum.lengthscales))

    def matrix_covariance(self, x1, x2):
        return np.sum(self.prior.covariance_matrix(x1, x2, self.__power_spectrum.freqs, self.__power_spectrum.sds,
                                                   self.__power_spectrum.lengthscales), axis=0)
