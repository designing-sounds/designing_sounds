import numpy as np

from src.wave_model.priors import Prior


class PowerSpectrum:
    def __init__(self, max_power_spectrum: int, max_harmonics: int, prior: Prior):
        self.max_harmonics = max_harmonics
        self.freqs = np.empty(0, dtype=np.float32)
        self.periodic_lengthscales = np.empty(0, dtype=np.float32)
        self.periodic_sds = np.empty(0, dtype=np.float32)
        self.squared_lengthscales = np.empty(0, dtype=np.float32)
        self.squared_sds = np.empty(0, dtype=np.float32)
        self.max_power_spectrum = max_power_spectrum
        self.num_kernels_per_spectrum = np.zeros(max_power_spectrum, dtype=int)
        self.prior = prior

    def clear_all(self):
        self.freqs = np.empty(0, dtype=np.float32)
        self.periodic_lengthscales = np.empty(0, dtype=np.float32)
        self.periodic_sds = np.empty(0, dtype=np.float32)
        self.squared_lengthscales = np.empty(0, dtype=np.float32)
        self.squared_sds = np.empty(0, dtype=np.float32)
        self.num_kernels_per_spectrum = np.zeros(self.max_power_spectrum, dtype=int)
        self.prior.weights = None

    def update_harmonic(self, harmonic_index, mean: float, periodic_sd: float, periodic_lengthscale: float,
                        squared_sd: float, squared_lengthscale: float,
                        num_harmonics: int) -> None:
        idx = np.sum(self.num_kernels_per_spectrum[:harmonic_index])
        cur_num_harmonics = self.num_kernels_per_spectrum[harmonic_index]
        approx_dim = self.prior.approx_dim
        for i in range(cur_num_harmonics, num_harmonics):
            self.freqs = np.insert(self.freqs, idx + i, np.float32(mean * (i + 1)))
            self.periodic_sds = np.insert(self.periodic_sds, idx + i, np.float32(periodic_sd))
            self.periodic_lengthscales = np.insert(self.periodic_lengthscales, idx + i, np.float32(periodic_sd))
            self.squared_sds = np.insert(self.squared_sds, idx + i, np.float32(squared_sd))
            self.squared_lengthscales = np.insert(self.squared_lengthscales, idx + i, np.float32(squared_lengthscale))
            if self.prior.weights is None:
                self.prior.weights = np.asarray(np.random.randn(1, approx_dim), dtype=np.float32)
            else:
                self.prior.weights = np.insert(self.prior.weights, idx + i, np.random.randn(approx_dim), axis=0)

        self.delete_harmonics(harmonic_index, num_harmonics, cur_num_harmonics)

        for i in range(num_harmonics):
            self.freqs[idx + i] = mean * (i + 1)
            self.periodic_sds[idx + i] = periodic_sd
            self.periodic_lengthscales[idx + i] = periodic_lengthscale
            self.squared_sds[idx + i] = squared_sd
            self.squared_lengthscales[idx + i] = squared_lengthscale
        self.num_kernels_per_spectrum[harmonic_index] = num_harmonics

    def delete_harmonics(self, power_spec_idx: int, first_harmonic: int, last_harmonic: int):
        idx = np.sum(self.num_kernels_per_spectrum[:power_spec_idx])
        for i in reversed(range(first_harmonic, last_harmonic)):
            self.freqs = np.delete(self.freqs, idx + i)
            self.periodic_sds = np.delete(self.periodic_sds, idx + i)
            self.periodic_lengthscales = np.delete(self.periodic_lengthscales, idx + i)
            self.squared_sds = np.delete(self.squared_sds, idx + i)
            self.squared_lengthscales = np.delete(self.squared_lengthscales, idx + i)
            self.prior.weights = np.delete(self.prior.weights, idx + i, axis=0)
        self.num_kernels_per_spectrum[power_spec_idx] = first_harmonic
