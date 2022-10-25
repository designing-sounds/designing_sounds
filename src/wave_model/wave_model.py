import threading
from statistics import NormalDist

import numpy as np


class PowerSpectrum:
    def __init__(self):
        self.harmonics = None
        self.total_size = 0

    def add_harmonic(self, mean, std, num_samples):
        freqs = np.random.randn(1, num_samples) * std + mean
        if self.harmonics is None:
            self.harmonics = np.array(freqs, dtype=object)
        else:
            self.harmonics = np.append(self.harmonics, freqs, axis=0)

    def update_harmonic(self, index, mean: int, std: float, num_samples: int):
        freqs = np.random.randn(1, num_samples) * std + mean
        self.harmonics[index] = freqs


class SoundModel:
    def __init__(self):
        self.amps = None
        self.phases = None
        self.power_spectrum = PowerSpectrum()
        self.max_samples = 0
        self.lock = threading.Lock()

    @staticmethod
    def get_normal_distribution_points(mean: float, std: float, num_samples: int) -> []:
        x_vals = np.linspace(mean - 4 * std, mean + 4 * std, num_samples)
        find_normal = np.vectorize(lambda x: NormalDist(mu=mean, sigma=std).pdf(x))
        return list(zip(x_vals, find_normal(x_vals)))

    def add_to_power_spectrum(self, mean, std, num_samples):
        self.lock.acquire()
        self.power_spectrum.add_harmonic(mean, std, num_samples)
        self.lock.release()

    def update_power_spectrum(self, index, mean: int, std: float, num_samples: int) -> None:
        self.lock.acquire()
        self.power_spectrum.update_harmonic(index, mean, std, num_samples)
        self.lock.release()

    def model_sound(self, sample_rate: int, chunk_duration: float, start_time: float) -> np.ndarray:
        x = np.linspace(start_time, start_time + chunk_duration, int(chunk_duration * sample_rate), endpoint=False)

        self.lock.acquire()
        freqs = np.asarray(self.power_spectrum.harmonics.flatten(), dtype=float)
        self.lock.release()

        if self.max_samples != freqs.size:
            self.max_samples = freqs.size
            self.amps = np.random.randn(self.max_samples)
            self.phases = np.random.randn(self.max_samples)

        sins = np.sin(x[:, None] * 2 * np.pi * freqs)

        sound = (sins @ self.amps).astype(np.float32) / self.max_samples

        return sound
