import threading
import typing
from statistics import NormalDist

import numpy as np


class PowerSpectrum:
    def __init__(self, max_harmonics: int, max_samples: int):
        self.max_samples = max_samples
        self.harmonics = np.zeros((max_harmonics, self.max_samples), dtype=np.float32)

    def update_harmonic(self, index, mean: int, std: float, num_samples: int) -> None:
        freqs = np.random.randn(1, num_samples) * std + mean
        self.harmonics[index] = np.zeros(self.max_samples)
        self.harmonics[index].put(np.arange(0, freqs.size), freqs)


class SoundModel:
    def __init__(self, max_harmonics: int, max_samples: int):
        self.amps = None
        self.phases = None
        self.max_harmonics = max_harmonics
        self.max_samples = max_samples
        self.power_spectrum = PowerSpectrum(self.max_harmonics, self.max_samples)
        self.actual_sizes = np.zeros(self.max_harmonics)
        self.lock = threading.Lock()

    @staticmethod
    def get_normal_distribution_points(mean: float, std: float, num_samples: int) -> typing.List[typing.Tuple[float, float]]:
        x_vals = np.linspace(mean - 4 * std, mean + 4 * std, num_samples)
        find_normal = np.vectorize(lambda x: NormalDist(mu=mean, sigma=std).pdf(x) if std != 0 else num_samples)
        return list(zip(x_vals, find_normal(x_vals)))

    def update_power_spectrum(self, index: int, mean: int, std: float, num_samples: int) -> None:
        self.lock.acquire()
        self.power_spectrum.update_harmonic(index, mean, std, num_samples)
        self.actual_sizes[index] = num_samples
        self.amps = np.asarray(np.random.randn(self.max_samples * self.max_harmonics), dtype=np.float32)
        self.phases = np.asarray(np.random.randn(self.max_harmonics, self.max_samples), dtype=np.float32)
        self.lock.release()

    def model_sound(self, sample_rate: int, chunk_duration: float, start_time: float) -> np.ndarray:
        x = np.linspace(start_time, start_time + chunk_duration, int(chunk_duration * sample_rate), endpoint=False, dtype=np.float32)

        self.lock.acquire()
        sins = np.sin(x[:, None, None] * 2 * np.pi * self.power_spectrum.harmonics)
        sins = sins.reshape(-1, self.max_samples * self.max_harmonics)

        sound = (sins @ self.amps) / np.sum(self.actual_sizes)
        self.lock.release()

        return sound
