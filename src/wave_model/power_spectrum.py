import numpy as np


class PowerSpectrum:
    def __init__(self, max_power_spectrum: int, max_harmonics: int, prior):
        self.max_harmonics = max_harmonics
        self.freqs = np.empty(0, dtype=np.float32)
        self.lengthscales = np.empty(0, dtype=np.float32)
        self.sds = np.empty(0, dtype=np.float32)
        self.num_kernels_per_spectrum = np.zeros(max_power_spectrum, dtype=int)
        self.prior = prior

    def update_harmonic(self, harmonic_index, mean: float, std: float,
                        num_harmonics: int, lengthscale: float) -> None:
        idx = np.sum(self.num_kernels_per_spectrum[:harmonic_index])
        cur_num_harmonics = self.num_kernels_per_spectrum[harmonic_index]
        d = self.prior.d
        for i in range(cur_num_harmonics, num_harmonics):
            self.freqs = np.insert(self.freqs, idx + i, np.float32(mean * (i + 1)))
            self.lengthscales = np.insert(self.lengthscales, idx + i,  np.float32(std))
            self.sds = np.insert(self.sds, idx + i, np.float32(std))
            if self.prior.weights is None:
                self.prior.weights = np.asarray(np.random.randn(1, d), dtype=np.float32)
            else:
                self.prior.weights = np.insert(self.prior.weights, idx + i, np.random.randn(d), axis=0)

        for i in reversed(range(num_harmonics, cur_num_harmonics)):
            self.freqs = np.delete(self.freqs, idx + i)
            self.lengthscales = np.delete(self.lengthscales, idx + i)
            self.sds = np.delete(self.sds, idx + i)
            self.prior.weights = np.delete(self.prior.weights, idx + i, axis=0)

        for i in range(num_harmonics):
            self.freqs[idx + i] = mean * (i + 1)
            self.lengthscales[idx + i] = lengthscale
            self.sds[idx + i] = std
        self.num_kernels_per_spectrum[harmonic_index] = num_harmonics