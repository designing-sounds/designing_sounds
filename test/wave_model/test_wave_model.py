from src.wave_model.wave_model import PowerSpectrum, SoundModel
import numpy as np
import unittest

if __name__ == '__main__':
    unittest.main()


class TestPowerSpectrum(unittest.TestCase):
    def setUp(self):
        self.spectrum = PowerSpectrum(10, 100)
        self.tolerance = 1e-6

    def test_update_harmonics(self):
        self.spectrum.update_harmonic(0, 1000, 1, 10, 1, "1")
        self.spectrum.update_harmonic(1, 2000, 3, 20, 1, "1")

        assert self.spectrum.harmonics[0].nonzero()[0].size == 10 and self.spectrum.harmonics[1].nonzero()[0].size == 20


class TestSoundModel(unittest.TestCase):

    def setUp(self):
        self.max_harmonics = 1000
        self.max_samples_per_harmonic = 4
        self.sound_model = SoundModel(self.max_harmonics, self.max_samples_per_harmonic, self.max_samples_per_harmonic)
        self.tolerance = 1e-9

    def test_model_chunk_sound(self):
        samples = self.max_samples_per_harmonic
        num_harmonics = self.max_power_spectrum
        sample_rate = samples * num_harmonics
        test = np.zeros((num_harmonics, samples))
        self.sound_model.update_power_spectrum(0, 1000, 1, samples, 1, "1")
        for i in range(num_harmonics):
            test[i] = self.sound_model.model_sound(sample_rate, samples / sample_rate, i * samples / sample_rate)

        test = test.flatten()
        expected = self.sound_model.model_sound(sample_rate, 1, 0)
        np.testing.assert_allclose(expected, test, self.tolerance)
