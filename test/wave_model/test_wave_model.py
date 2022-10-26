from src.wave_model.wave_model import PowerSpectrum, SoundModel
import numpy as np
import unittest


class TestPowerSpectrum(unittest.TestCase):
    def setUp(self):
        self.spectrum = PowerSpectrum(10, 100)
        self.tolerance = 1e-6

    def test_update_harmonics(self):
        self.spectrum.update_harmonic(0, 1000, 1, 10)
        self.spectrum.update_harmonic(1, 2000, 3, 20)

        assert self.spectrum.harmonics[0].nonzero()[0].size == 10 and self.spectrum.harmonics[1].nonzero()[0].size == 20


class TestSoundModel(unittest.TestCase):

    def setUp(self):
        self.sound_model = SoundModel(10, 100)
        self.tolerance = 1e-5

    def test_get_normal_distribution_points(self):
        vals = self.sound_model.get_normal_distribution_points(1, 2, 3)
        expected = np.array([[-7, 6.6915112882443E-5], [1, 0.19947114020072], [9, 6.6915112882443E-5]])

        np.testing.assert_allclose(vals, expected, self.tolerance)

    def test_model_chunk_sound(self):
        self.sound_model.update_power_spectrum(0, 1000, 1, 100)
        sample_rate = 44100
        test = np.zeros((10, 4410))
        for i in range(10):
            test[i] = self.sound_model.model_sound(sample_rate, 0.1, i * 0.1)

        test = test.flatten()
        expected = self.sound_model.model_sound(sample_rate, 1, 0)

        np.testing.assert_allclose(expected, test, self.tolerance)


if __name__ == '__main__':
    unittest.main()
