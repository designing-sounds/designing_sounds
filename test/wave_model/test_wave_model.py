from src.wave_model.wave_model import PowerSpectrum, SoundModel
import numpy as np
import unittest

class TestPowerSpectrum(unittest.TestCase):
    def setUp(self):
        self.spectrum = PowerSpectrum()
        self.tolerance = 1e-6

    def test_adds_freqs(self):
        self.spectrum.add_element(1000, 1, 10)
        self.spectrum.add_element(2000, 3, 10)
        assert self.spectrum.freqs.shape == (2, 10)

    def test_get_normal_distribution_points(self):
        vals = self.spectrum.get_normal_distribution_points(1, 2, 3)
        expected = np.array([[-5, 0.00221592422], [1, 0.1994711402007], [7, 0.00221592422]])

        np.testing.assert_allclose(vals, expected, self.tolerance)


class TestSoundModel(unittest.TestCase):

    def setUp(self):
        self.sound_model = SoundModel()

        self.tolerance = 1e-5

    def test_model_chunk_sound(self):
        self.sound_model.update_power_spectrum(np.reshape(np.array([1000, 1]), (1, -1)))
        sample_rate = 44100
        test = np.zeros((10, 4410))
        for i in range(10):
            test[i] = self.sound_model.model_sound(sample_rate, 0.1, i * 0.1)

        test = test.flatten()
        expected = self.sound_model.model_sound(sample_rate, 1, 0)

        np.testing.assert_allclose(expected, test, self.tolerance)




if __name__ == '__main__':
    unittest.main()
