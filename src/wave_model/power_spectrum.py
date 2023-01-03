import numpy as np

from src.wave_model.priors import Prior
from src.wave_model.priors import MultPrior
from src.wave_model.priors import PeriodicPrior


class PowerSpectrum:
    def __init__(self, max_harmonics_per_spectrum: int):
        self.max_harmonics_per_spectrum = max_harmonics_per_spectrum
        self.stats = np.empty((0, 5), dtype=np.float32)
        self.num_harmonics_per_spectrum = np.zeros(self.max_harmonics_per_spectrum, dtype=int)
        approx_num = 50
        self.priors = [MultPrior(approx_num), PeriodicPrior(approx_num)]
        self.prior = self.priors[0]

    def change_kernel(self, idx: int):
        old_weights = self.prior.weights
        self.prior = self.priors[idx]
        self.prior.weights = old_weights
        self.prior.update(self.get_freqs(), self.get_periodic_lengthscales(), self.get_periodic_sds())

    def get_freqs(self):
        return self.stats[:, 0]

    def get_periodic_sds(self):
        return self.stats[:, 1]

    def get_periodic_lengthscales(self):
        return self.stats[:, 2]

    def get_squared_sds(self):
        return self.stats[:, 3]

    def get_squared_lengthscales(self):
        return self.stats[:, 4]

    def clear_all(self):
        self.stats = np.empty((0, 5), dtype=np.float32)
        self.num_harmonics_per_spectrum = np.zeros(self.max_harmonics_per_spectrum, dtype=int)
        self.prior.weights = None

    def update_harmonic(self, power_spectrum_index, mean: float, periodic_sd: float, periodic_lengthscale: float,
                        squared_sd: float, squared_lengthscale: float,
                        curr_harmonic_index: int) -> None:
        idx = np.sum(self.num_harmonics_per_spectrum[:power_spectrum_index])
        curr_num_harmonics = self.num_harmonics_per_spectrum[power_spectrum_index]
        approx_dim = self.prior.approx_dim
        for i in range(curr_num_harmonics, curr_harmonic_index):
            self.stats = np.insert(self.stats, idx + i,
                                   np.array([mean * (i + 1), periodic_sd, periodic_lengthscale, squared_sd,
                                             squared_lengthscale]), axis=0)
            if self.prior.weights is None:
                self.prior.weights = np.asarray(np.random.randn(1, approx_dim), dtype=np.float32)
            else:
                self.prior.weights = np.insert(self.prior.weights, idx + i, np.random.randn(approx_dim), axis=0)

        self.delete_harmonics(power_spectrum_index, curr_harmonic_index, curr_num_harmonics)

        for i in range(curr_harmonic_index):
            self.stats[idx + i] = np.array([mean * (i + 1), periodic_sd, periodic_lengthscale, squared_sd,
                                            squared_lengthscale])

        self.num_harmonics_per_spectrum[power_spectrum_index] = curr_harmonic_index

    def delete_harmonics(self, power_spec_idx: int, first_harmonic: int, last_harmonic: int):
        idx = np.sum(self.num_harmonics_per_spectrum[:power_spec_idx])
        for i in reversed(range(first_harmonic, last_harmonic)):
            self.stats = np.delete(self.stats, idx + i, axis=0)
            self.prior.weights = np.delete(self.prior.weights, idx + i, axis=0)
        self.num_harmonics_per_spectrum[power_spec_idx] = first_harmonic
