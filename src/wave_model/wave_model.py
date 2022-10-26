import threading
import typing
from statistics import NormalDist

import numpy as np


class PowerSpectrum:
    def __init__(self, max_harmonics: int, max_samples_per_harmonic: int):
        self.max_samples_per_harmonic = max_samples_per_harmonic
        self.harmonics = np.zeros((max_harmonics, self.max_samples_per_harmonic), dtype=np.float32)

    def update_harmonic(self, harmonic_index, mean: int, std: float, num_harmonic_samples: int) -> None:
        freqs = np.random.randn(num_harmonic_samples) * std + mean
        self.harmonics[harmonic_index] = np.zeros(self.max_samples_per_harmonic)
        self.harmonics[harmonic_index, :num_harmonic_samples] = freqs


class SoundModel:
    def __init__(self, max_harmonics: int, max_samples_per_harmonic: int):
        self.amps = None
        self.phases = None
        self.max_harmonics = max_harmonics
        self.max_samples_per_harmonic = max_samples_per_harmonic
        self.power_spectrum = PowerSpectrum(self.max_harmonics, self.max_samples_per_harmonic)
        self.samples_per_harmonic = np.zeros(self.max_harmonics)
        self.lock = threading.Lock()

    @staticmethod
    def get_normal_distribution_points(mean: float, std: float, num_samples: int) -> typing.List[
        typing.Tuple[float, float]]:
        if std == 0:
            y_vals = np.linspace(0, num_samples, num_samples)
            x_vals = np.repeat(mean, num_samples)
        else:
            x_vals = np.linspace(mean - 4 * std, mean + 4 * std, num_samples)
            find_normal = np.vectorize(lambda x: NormalDist(mu=mean, sigma=std).pdf(x))
            y_vals = find_normal(x_vals)
        return list(zip(x_vals, y_vals))

    def interpolate_points(self, points: typing.List[typing.Tuple[float, float]]):
        pass

    def update_power_spectrum(self, harmonic_index: int, mean: int, std: float, num_harmonic_samples: int) -> None:
        self.lock.acquire()
        self.power_spectrum.update_harmonic(harmonic_index, mean, std, num_harmonic_samples)
        self.samples_per_harmonic[harmonic_index] = num_harmonic_samples
        self.amps = np.asarray(np.random.randn(self.max_samples_per_harmonic * self.max_harmonics), dtype=np.float32)
        # self.phases = np.asarray(np.random.randn(self.max_harmonics, self.max_samples_per_harmonic), dtype=np.float32)
        self.lock.release()

    def model_sound(self, sample_rate: int, chunk_duration: float, start_time: float) -> np.ndarray:
        x = np.linspace(start_time, start_time + chunk_duration, int(chunk_duration * sample_rate), endpoint=False,
                        dtype=np.float32)

        self.lock.acquire()

        freqs = self.power_spectrum.harmonics.flatten()
        sins = np.sin(x[:, None] * 2 * np.pi * freqs)

        sound = (sins @ self.amps) / np.sum(self.samples_per_harmonic)
        self.lock.release()

        return sound
