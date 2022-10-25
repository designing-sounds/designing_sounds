import typing
from statistics import NormalDist
import threading
import numpy as np


class Harmonic:
    def __init__(self):
        self.freqs = np.array([])

    def update_freq(self, mean: int, std: float, num_samples: int) -> None:
        self.freqs = np.random.randn(1, num_samples) * std + mean


class PowerSpectrum:
    def __init__(self):
        self.harmonics = []
        self.total_size = 0

    def add_harmonic(self) -> None:
        self.harmonics.append(Harmonic())

    def update_harmonic(self, index: int, mean: int, std: float, num_samples: int):
        self.total_size -= self.harmonics[index].freqs.size
        self.harmonics[index].update_freq(mean, std, num_samples)
        self.total_size += num_samples

    def get_flatten_freqs(self) -> np.ndarray:
        flatten_freqs = np.empty(self.total_size)
        i = 0
        for harmonic in self.harmonics:
            flatten_freqs.put(np.arange(i, i + harmonic.freqs.size), harmonic.freqs)
            i += harmonic.freqs.size
        return flatten_freqs


class SoundModel:
    def __init__(self):
        self.amps = None
        self.phases = None
        self.power_spectrum = PowerSpectrum()
        self.lock = threading.Lock()

    @staticmethod
    def get_normal_distribution_points(mean: float, std: float, num_samples: int) -> typing.List[typing.Tuple[float, float]]:
        x_vals = np.linspace(mean - 3 * std, mean + 3 * std, num_samples)
        find_normal = np.vectorize(lambda x: NormalDist(mu=mean, sigma=std).pdf(x))
        return list(zip(x_vals, find_normal(x_vals)))

    def add_to_power_spectrum(self) -> None:
        self.lock.acquire()
        self.power_spectrum.add_harmonic()
        self.lock.release()

    def update_power_spectrum(self, index: int, mean: int, std: float, num_samples: int) -> None:
        self.lock.acquire()
        self.power_spectrum.update_harmonic(index, mean, std, num_samples)
        self.amps = np.random.randn(self.power_spectrum.total_size)
        self.phases = np.random.randn(self.power_spectrum.total_size)
        self.lock.release()

    def model_sound(self, sample_rate: int, chunk_duration: float, start_time: float) -> np.ndarray:
        x = np.linspace(start_time, start_time + chunk_duration, int(chunk_duration * sample_rate), endpoint=False)

        self.lock.acquire()
        freqs = self.power_spectrum.get_flatten_freqs()

        sins = np.sin(x[:, None] * 2 * np.pi * freqs + self.phases)

        sound = (sins @ self.amps).astype(np.float32) / freqs.size
        self.lock.release()

        return sound
