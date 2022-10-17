import numpy as np


class SinWave:
    def __init__(self, freq: float, amp: float):
        self.freq = freq
        self.amp = amp

    def get_array(self, sample_rate: int, duration: float = 1) -> np.array:
        points = np.linspace(0, duration, int(duration * sample_rate), endpoint=False, dtype=np.float32)
        return self.amp * np.sin(self.freq * points * 2 * np.pi)


class PowerSpectrum:
    def __init__(self):
        self.freqs = None

    def add_element(self, mean: float, std: float, num_samples: float) -> None:
        new_freqs = np.random.randn(num_samples, 1) * std + mean
        if self.freqs is None:
            self.freqs = new_freqs
        else:
            self.freqs = np.append(self.freqs, new_freqs, 1)


class SoundModel:
    def __init__(self, sample_rate: int):
        self.sound = np.array([])
        self.sample_rate = sample_rate
        self.power_spectrum = PowerSpectrum()
        means = 440 * np.arange(1, 5)
        self.power_spectrum.add_element(mean=means[0], std=3, num_samples=250)
        self.power_spectrum.add_element(mean=means[1], std=3, num_samples=250)
        self.power_spectrum.add_element(mean=means[2], std=3, num_samples=250)
        self.power_spectrum.add_element(mean=means[3], std=3, num_samples=250)

    def model_sound(self, duration: float):
        length = 1000
        amps = np.random.randn(length)
        x = np.linspace(0, duration, int(duration * self.sample_rate), endpoint=False)
        weights = 1.0 / np.arange(1, 5) ** 2.0
        sins = np.sin(x[:, None, None] * 2 * np.pi * self.power_spectrum.freqs) * weights[None, None, :]
        sins = sins.reshape(-1, length)

        self.sound = (sins @ amps).astype(np.float32)

    def normalize_sound(self, amp: float):
        if self.sound is None:
            normalized = 1
        else:
            normalized = self.sound.max()

        self.sound = self.sound * amp / normalized

    def get_sound(self):
        return self.sound
