from statistics import NormalDist

import numpy as np


class PowerSpectrum:
    def __init__(self):
        self.freqs = None

    def add_element(self, mean: float, std: float, num_samples: float) -> None:
        new_freqs = np.random.randn(1, num_samples) * std + mean

        if self.freqs is None:
            self.freqs = new_freqs
        else:
            self.freqs = np.append(self.freqs, new_freqs, 0)

    @staticmethod
    def get_normal_distribution_points(mean: float, std: float, num_samples: int) -> (
            np.array, np.array):
        x_vals = np.linspace(mean - 3 * std, mean + 3 * std, num_samples)
        find_normal = np.vectorize(lambda x: NormalDist(mu=mean, sigma=std).pdf(x))
        return x_vals, find_normal(x_vals)


class SoundModel:
    def __init__(self):
        self.power_spectrum = PowerSpectrum()
        self.length = 250
        self.amps = np.random.randn(self.length)

    def update_power_spectrum(self, powers: np.array, num_samples=250) -> None:
        spectrum = PowerSpectrum()
        for power in powers:
            spectrum.add_element(power[0], power[1], num_samples)
        self.power_spectrum = spectrum

    def model_sound(self, sample_rate, chunk_duration: float, start_time: float):
        x = np.linspace(start_time, start_time + chunk_duration, int(chunk_duration * sample_rate), endpoint=False)
        sins = np.sin(x[:, None, None] * 2 * np.pi * self.power_spectrum.freqs)
        sins = sins.reshape(-1, self.length)

        sound = (sins @ self.amps).astype(np.float32)

        return sound / 50
