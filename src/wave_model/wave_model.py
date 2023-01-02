import threading
import typing
from typing import Tuple, List, Any, Union

import numpy as np
from numpy import ndarray

from numpy.linalg import inv

from src.wave_model.power_spectrum import PowerSpectrum
from src.wave_model.priors import MultPrior
from src.wave_model.priors import PeriodicPrior


class SoundModel:
    def __init__(self, max_harmonics_per_spectrum: int):
        self.inv = None
        self.max_harmonics_per_spectrum = max_harmonics_per_spectrum
        self.prior = MultPrior(50)
        self.__power_spectrum = PowerSpectrum(self.max_harmonics_per_spectrum, self.prior)
        self.lock = threading.Lock()
        self.x_train = None
        self.y_train = None
        self.noise = 0
        self.variance = 0

    def set_periodic_prior(self):
        old_weights = self.prior.weights
        self.prior = PeriodicPrior(50)
        self.prior.weights = old_weights
        self.prior.update(self.__power_spectrum.get_freqs(), self.__power_spectrum.get_periodic_lengthscales(),
                          self.__power_spectrum.get_periodic_sds())

    def set_mult_prior(self):
        old_weights = self.prior.weights
        self.prior = MultPrior(50)
        self.prior.weights = old_weights
        self.prior.update(self.__power_spectrum.get_freqs(), self.__power_spectrum.get_periodic_lengthscales(),
                          self.__power_spectrum.get_periodic_sds())

    def get_power_spectrum_histogram(self, idx: int, samples: int) -> Tuple[List[Tuple[Any, Any]],
                                                                            Union[ndarray, int, float, complex]]:
        with self.lock:
            x = np.linspace(0, 1, samples)
            k = np.zeros(len(x))
            num_harmonics = self.__power_spectrum.num_harmonics_per_spectrum[idx]
            idx = np.sum(self.__power_spectrum.num_harmonics_per_spectrum[:idx])
            for i in range(num_harmonics):
                k += self.prior.kernel(x, self.__power_spectrum.get_freqs()[idx + i],
                                       self.__power_spectrum.get_periodic_sds()[idx + i],
                                       self.__power_spectrum.get_periodic_lengthscales()[idx + i],
                                       self.__power_spectrum.get_squared_sds()[idx + i],
                                       self.__power_spectrum.get_squared_lengthscales()[idx + i])

            freqs = np.fft.fftfreq(samples, 1 / samples)
            freqs = [0] + freqs[1:samples // 2]
            fft = [0] + np.abs(np.fft.fft(k)[1:samples // 2])

        return list(zip(freqs, fft)), np.max(fft)

    def remove_power_spectrum(self, index):
        num_kernels = self.__power_spectrum.num_harmonics_per_spectrum[index]
        self.__power_spectrum.delete_harmonics(index, 0, num_kernels)
        self.__power_spectrum.num_harmonics_per_spectrum[index] = 0
        self.prior.update(self.__power_spectrum.get_freqs(), self.__power_spectrum.get_periodic_lengthscales(),
                          self.__power_spectrum.get_periodic_sds())

    def get_sum_all_power_spectrum_histogram(self, samples: int) -> Tuple[List[Tuple[Any, Any]],
                                                                          Union[ndarray, int, float, complex]]:
        with self.lock:
            x = np.linspace(0, 1, samples)
            k = np.zeros(len(x))
            for i, freq in enumerate(self.__power_spectrum.get_freqs()):
                k += self.prior.kernel(x, freq, self.__power_spectrum.get_periodic_sds()[i],
                                       self.__power_spectrum.get_periodic_lengthscales()[i],
                                       self.__power_spectrum.get_squared_sds()[i],
                                       self.__power_spectrum.get_squared_lengthscales()[i])

            freqs = np.fft.fftfreq(samples, 1 / samples)
            freqs = [0] + freqs[1:samples // 2]
            fft = [0] + np.abs(np.fft.fft(k)[1:samples // 2])

        return list(zip(freqs, fft)), np.max(fft)

    def interpolate_points(self, points: typing.List[typing.Tuple[float, float]]):
        with self.lock:
            self.x_train = np.array([x for (x, _) in points], dtype=np.float32)
            self.y_train = np.array([y for (_, y) in points], dtype=np.float32)
            self.update_noise()
            self.inv = None
            if self.x_train.size != 0:
                try:
                    self.inv = inv(
                        self.matrix_covariance(self.x_train, self.x_train) + self.variance * np.eye(len(self.x_train)))
                except:
                    pass

    def update_power_spectrum(self, power_spectrum_index: int, mean: int, periodic_sd: float,
                              periodic_lengthscale: float, squared_sd: float, squared_lengthscale: float,
                              curr_harmonic_index: int, ) -> None:
        with self.lock:
            self.__power_spectrum.update_harmonic(power_spectrum_index, mean, periodic_sd,
                                                  periodic_lengthscale, squared_sd, squared_lengthscale,
                                                  curr_harmonic_index)
            self.prior.update(self.__power_spectrum.get_freqs(), self.__power_spectrum.get_periodic_lengthscales(),
                              self.__power_spectrum.get_periodic_sds())

    def clear_all_power_spectrums(self) -> None:
        with self.lock:
            self.__power_spectrum.clear_all()

    def get_power_spectrum(self, sound):
        with self.lock:
            k = sound
            samples = k.size
            freqs = np.fft.fftfreq(samples, 1 / samples)
            freqs = [0] + freqs[1:samples // 2]
            fft = [0] + np.abs(np.fft.fft(k)[1:samples // 2])
        return list(zip(freqs, fft))

    def model_sound(self, sample_rate: int, chunk_duration: float, start_time: float) -> np.ndarray:
        x = np.linspace(start_time, start_time + chunk_duration, int(chunk_duration * sample_rate), endpoint=False,
                        dtype=np.float32)

        with self.lock:
            sound = self.prior.prior(x, self.__power_spectrum.get_freqs(), self.__power_spectrum.get_periodic_sds(),
                                     self.__power_spectrum.get_periodic_lengthscales(),
                                     self.__power_spectrum.get_squared_sds(),
                                     self.__power_spectrum.get_squared_lengthscales())
            if not (self.inv is None or self.x_train is None or self.y_train is None):
                sound += self.update(x)

        return sound

    def update_prior(self):
        self.prior.resample()
        self.update_noise()

    def update_noise(self):
        self.noise = np.random.normal(0, np.sqrt(self.variance), size=self.y_train.shape)

    def update(self, x_test):
        prior = self.prior.prior(self.x_train, self.__power_spectrum.get_freqs(),
                                 self.__power_spectrum.get_periodic_sds(),
                                 self.__power_spectrum.get_periodic_lengthscales(),
                                 self.__power_spectrum.get_squared_sds(),
                                 self.__power_spectrum.get_squared_lengthscales())
        return self.matrix_covariance(x_test, self.x_train) @ self.inv @ (self.y_train - self.noise - prior)

    def matrix_covariance(self, x_1, x_2):
        return np.sum(
            self.prior.covariance_matrix(x_1[:, None] - x_2, self.__power_spectrum.get_freqs(),
                                         self.__power_spectrum.get_periodic_sds(),
                                         self.__power_spectrum.get_periodic_lengthscales(),
                                         self.__power_spectrum.get_squared_sds(),
                                         self.__power_spectrum.get_squared_lengthscales()), axis=0)
