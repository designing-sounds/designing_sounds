import numpy as np


class PowerSpectrum:
    def __init__(self, max_power_spectrum: int, max_harmonics: int):
        self.max_harmonics = max_harmonics
        self.freqs = np.array([0], dtype=np.float32)
        self.lengthscales = np.array([0], dtype=np.float32)
        self.sds = np.array([0], dtype=np.float32)
        self.num_kernels_per_spectrum = np.zeros(max_power_spectrum, dtype=int)

    def update_harmonic(self, harmonic_index, mean: float, std: float,
                        num_harmonics: int, lengthscale: float) -> None:
        idx = np.sum(self.num_kernels_per_spectrum[:harmonic_index])
        for i in range(num_harmonics):
            if idx + i >= len(self.freqs):
                self.freqs = np.append(self.freqs, np.float32(mean))
                self.lengthscales = np.append(self.lengthscales, np.float32(std))
                self.sds = np.append(self.sds, np.float32(std))
            else:
                self.freqs[idx + i] = mean
                self.lengthscales[idx + i] = lengthscale
                self.sds[idx + i] = std
        for i in range(num_harmonics, self.max_harmonics):
            if idx + i < len(self.freqs):
                self.freqs = np.delete(self.freqs, idx + i)
                self.lengthscales = np.delete(self.lengthscales, idx + i)
                self.sds = np.delete(self.sds, idx + i)
        self.num_kernels_per_spectrum[harmonic_index] = num_harmonics