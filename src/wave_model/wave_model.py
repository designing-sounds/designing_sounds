import threading
import typing

from math import sqrt, log, log2, log10, cos, sin, tan, ceil, floor, fabs, factorial, exp
import numpy as np


class Peak:
    def __init__(self, mean: float, sd: float, power: float, max_samples_per_harmonic: int):
        self.max_samples = max_samples_per_harmonic
        self.freqs = None
        self.mean = mean
        self.power = power
        self.update_peak(mean, sd)

    def update_peak(self, mean: float, sd: float):
        self.freqs = np.random.randn(self.max_samples) * sd + mean
        self.freqs = np.asarray(self.freqs, dtype=np.float32)


class PowerSpectrum:
    def __init__(self, max_samples_per_harmonic: int):
        self.max_samples_per_harmonic = max_samples_per_harmonic
        self.harmonics = []
        self.functions = {'sqrt': sqrt, 'pow': pow, 'log': log, 'log2': log2, 'log10': log10, 'cos': cos, 'sin': sin,
                          'tan': tan, 'ceil': ceil, 'floor': floor, 'abs': fabs, 'factorial': factorial, 'exp': exp}

    def update_harmonic(self, harmonic_index, mean: int, std: float, num_harmonic_samples: int,
                        num_harmonics: int, decay_function: str) -> None:
        decay_ratios = []
        for x in range(num_harmonics):
            self.functions['x'] = x + 1
            decay_factor = 0
            try:
                decay_factor = eval(decay_function, {'__builtins__': self.functions})
            except:
                decay_factor = 0
            finally:
                decay_ratios.append(decay_factor)

        peaks = []
        for i in range(num_harmonics):
            sample_size = int(num_harmonic_samples * decay_ratios[i])
            peaks.append(Peak(mean * (i + 1), std, sample_size, self.max_samples_per_harmonic))

        if len(self.harmonics) > harmonic_index:
            self.harmonics[harmonic_index] = peaks
        else:
            self.harmonics.append(peaks)


class SoundModel:
    def __init__(self, max_samples_per_harmonic: int, max_freq: int):
        self.amps = None
        self.powers = None
        self.freqs = None
        self.max_freq = max_freq
        self.max_samples_per_harmonic = max_samples_per_harmonic
        self.phases = None
        self.__power_spectrum = PowerSpectrum(self.max_samples_per_harmonic)
        self.total_freqs = 0
        self.lock = threading.Lock()

    def get_power_spectrum_histogram(self, harmonic_index: int,
                                     _num_bins: int) -> typing.List[typing.Tuple[float, float]]:
        with self.lock:
            peaks = self.__power_spectrum.harmonics[harmonic_index]
            freqs = np.zeros(len(peaks) * self.max_samples_per_harmonic)
            max_freq = 0
            for i, peak in enumerate(peaks):
                idx = i * self.max_samples_per_harmonic
                freqs[idx: idx + self.max_samples_per_harmonic] = peak.freqs
                if peak.mean > max_freq:
                    max_freq = peak.mean
            max_range = max(1000, max_freq + 100)
            histogram, bin_edges = np.histogram(freqs, bins='sqrt')
        return list(zip(bin_edges, histogram)), max_range

    def remove_power_spectrum(self, index):
        with self.lock:
            self.total_freqs -= self.max_samples_per_harmonic * len(self.__power_spectrum.harmonics[index])
            self.__power_spectrum.harmonics.pop(index)
            self.update_freqs_and_powers()

    def update_freqs_and_powers(self):
        freqs = np.zeros(self.total_freqs, dtype=np.float32)
        powers = np.zeros(self.total_freqs // self.max_samples_per_harmonic, dtype=np.float32)
        idx = 0

        for peaks in self.__power_spectrum.harmonics:
            for peak in peaks:
                freqs[idx: idx + self.max_samples_per_harmonic] = peak.freqs
                # 100 below refers to the max value of samples slider
                powers[idx // self.max_samples_per_harmonic] = peak.power / 100
                idx += self.max_samples_per_harmonic

        self.freqs, self.powers = freqs, powers

    def get_sum_all_power_spectrum_histogram(self) -> typing.List[typing.Tuple[float, float]]:
        with self.lock:
            freqs = self.freqs
            max_range = max(1000, freqs.max() + 100) if len(freqs) > 0 else 1000
            histogram, bin_edges = np.histogram(freqs, self.max_freq // 2, range=(0.1, max_range))
        return list(zip(bin_edges, histogram))

    def interpolate_points(self, points: typing.List[typing.Tuple[float, float]]):
        with self.lock:
            if points:
                x, y = (np.array([i for i, _ in points]), np.array([j for _, j in points]))
                self.amps, _, _, _ = np.linalg.lstsq(self.calculate_sins(x), y, rcond=None)
            else:
                self.amps = np.random.randn(self.max_samples_per_harmonic)
                self.amps = self.amps / (self.amps.max()) / 2
            self.amps = np.asarray(self.amps, dtype=np.float32)

    def update_power_spectrum(self, harmonic_index: int, mean: int, std: float, num_harmonic_samples: int,
                              num_harmonics: int, decay_function: str) -> None:
        with self.lock:
            if len(self.__power_spectrum.harmonics) > harmonic_index:
                self.total_freqs -= self.max_samples_per_harmonic * len(self.__power_spectrum.harmonics[harmonic_index])
            self.__power_spectrum.update_harmonic(harmonic_index, mean, std, num_harmonic_samples, num_harmonics,
                                                  decay_function)
            self.total_freqs += self.max_samples_per_harmonic * num_harmonics
            self.update_freqs_and_powers()

    def calculate_sins(self, x):
        powers = np.zeros((self.total_freqs, self.max_samples_per_harmonic), dtype=np.float32)
        for i in range(self.max_samples_per_harmonic):
            powers[i::self.max_samples_per_harmonic, i] = self.powers
        sins = np.sin((x[:, None]) * 2 * np.pi * self.freqs) @ powers
        return sins

    def model_sound(self, sample_rate: int, chunk_duration: float, start_time: float) -> np.ndarray:
        x = np.linspace(start_time, start_time + chunk_duration, int(chunk_duration * sample_rate), endpoint=False,
                        dtype=np.float32)

        with self.lock:
            sound = self.calculate_sins(x) @ self.amps
        sound[sound > 1] = 1
        sound[sound < -1] = -1
        return sound
