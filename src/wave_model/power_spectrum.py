import numpy as np


class PowerSpectrum:
    def __init__(self, max_power_spectrum: int, max_harmonics: int):
        self.max_harmonics = max_harmonics
        self.freqs = np.empty(0, dtype=np.float32)
        self.lengthscales = np.empty(0, dtype=np.float32)
        self.sds = np.empty(0, dtype=np.float32)
        self.num_kernels_per_spectrum = np.zeros(max_power_spectrum, dtype=int)

    def update_harmonic(self, power_spectrum_index, mean: float, std: float,
                        num_harmonics: int, lengthscale: float) -> None:
        id = np.sum(self.num_kernels_per_spectrum[:power_spectrum_index]) + num_harmonics - 1
        if num_harmonics > self.num_kernels_per_spectrum[power_spectrum_index]: # We need to add
            self.freqs = np.insert(self.freqs, id, np.float32(mean))
            self.lengthscales = np.insert(self.lengthscales, id, np.float32(std))
            self.sds = np.insert(self.sds, id, np.float32(std))
        elif num_harmonics < self.num_kernels_per_spectrum[power_spectrum_index]: # We need to delete
            self.freqs = np.delete(self.freqs, id)
            self.lengthscales = np.delete(self.lengthscales, id)
            self.sds = np.delete(self.sds, id)
        for i in range(np.sum(self.num_kernels_per_spectrum[:power_spectrum_index]), id - 2):
            self.freqs[i] = mean
            self.lengthscales[i] = lengthscale
            self.sds[i] = std
        self.num_kernels_per_spectrum[power_spectrum_index] = num_harmonics