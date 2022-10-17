import numpy as np


class SoundModel:
    def __init__(self, sample_rate):
        self.sound = np.array([])
        self.sample_rate = sample_rate
        self.sin_wave = SinWave(0, 0)

    def model_sound(self, freq, amp, duration):
        self.sin_wave.freq = freq
        self.sin_wave.amp = amp / 100.0
        self.sound = self.sin_wave.get_array(self.sample_rate, duration)

    def get_sound(self):
        return self.sound


class SinWave:
    def __init__(self, freq: float, amp: float):
        self.freq = freq
        self.amp = amp

    def get_array(self, sample_rate: int, duration: float = 1) -> np.array:
        points = np.linspace(0, duration, int(duration * sample_rate), endpoint=False, dtype=np.float32)
        return self.amp * np.sin(self.freq * points * 2 * np.pi)


class PowerSpectrum:
    def __init__(self, freqs: np.ndarray, means: np.ndarray, stds: np.ndarray):
        self.means = means
        self.stds = stds
        self.freqs = freqs

    def add_freq(self, new_freqs: np.ndarray, new_means: np.ndarray, new_stds: np.ndarray) -> None:
        assert (new_freqs.ndim, new_means, new_stds == 1)
        self.freqs = np.append(self.freqs, np.array([new_freqs]), 0)
        self.means = np.append(self.means, np.array([new_means]))
        self.stds = np.append(self.stds, np.array([new_stds]))
