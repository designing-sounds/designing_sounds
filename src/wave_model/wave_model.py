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
        new_freqs = np.random.randn(1, num_samples) * std + mean

        if self.freqs is None:
            self.freqs = new_freqs
        else:
            self.freqs = np.append(self.freqs, new_freqs, 0)


class SoundModel:
    def __init__(self, sample_rate: int):
        self.sound = np.array([])
        self.sample_rate = sample_rate
        self.duration = 1
        self.power_spectrum = PowerSpectrum()
        self.update_power_spectrum(np.reshape(np.array([440, 3, 250]), (1, -1)))

    def update_power_spectrum(self, powers: np.array) -> None:
        spectrum = PowerSpectrum()
        for power in powers:
            spectrum.add_element(power[0], power[1], power[2])
        self.power_spectrum = spectrum

    def model_sound(self, duration: float):
        self.duration = duration
        length = self.power_spectrum.freqs.shape[0] * self.power_spectrum.freqs.shape[1]
        amps = np.random.randn(length)
        x = np.linspace(0, duration, int(duration * self.sample_rate), endpoint=False)

        sins = np.sin(x[:, None, None] * 2 * np.pi * self.power_spectrum.freqs)
        sins = sins.reshape(-1, length)

        self.sound = (sins @ amps).astype(np.float32)

    def normalize_sound(self, amp: float):
        if self.sound is None:
            normalized = 1
        else:
            normalized = self.sound.max()

        self.sound = self.sound * amp / normalized

    def get_sound(self) -> np.array:
        return self.sound

    def reshape(self, chunk_duration: float) -> np.array:
        samples_per_chunk = (chunk_duration / self.duration) * len(self.sound)
        num_chunks = len(self.sound) / samples_per_chunk
        assert(samples_per_chunk.is_integer() and num_chunks.is_integer())
        return np.reshape(self.sound, (int(num_chunks), int(samples_per_chunk)))
