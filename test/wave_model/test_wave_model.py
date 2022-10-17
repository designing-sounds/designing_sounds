from src.wave_model.wave_model import SinWave, PowerSpectrum
import numpy as np
import unittest


class TestSinWave(unittest.TestCase):
    def setUp(self):
        self.wave = SinWave(freq=1, amp=1)
        self.tolerance = 1e-6

    def test_calculate_sine_correctly(self):
        expected_arr = np.array([0, 0.9510565163, 0.5877852523, -0.5877852523, -0.9510565163])
        np.testing.assert_allclose(self.wave.get_array(5), expected_arr, self.tolerance)

    def test_altering_amplitude_calculates_sine_correctly(self):
        self.wave.amp = 3
        expected_arr = np.array([0, 2.853169549, 1.763355757, -1.763355757, -2.853169549])
        np.testing.assert_allclose(self.wave.get_array(5), expected_arr, self.tolerance)

    def test_altering_frequency_calculates_sine_correctly(self):
        self.wave.freq = 2
        expected_arr = np.array([0, 0.5877852523, -0.9510565163, 0.9510565163, -0.5877852523])
        np.testing.assert_allclose(self.wave.get_array(5), expected_arr, self.tolerance)

    def test_altering_time_constraint_calculates_sine_correctly(self):
        expected_arr = np.array([0, 0.5877852523, 0.9510565163, 0.9510565163, 0.5877852523])
        np.testing.assert_allclose(self.wave.get_array(10, 0.5), expected_arr, self.tolerance)


class TestPowerSpectrum(unittest.TestCase):
    def setUp(self):
        self.spectrum = PowerSpectrum(freqs=np.array([np.array([2, 5, 1]), np.array([8, 10, 1])]),
                                      means=np.array([200, 300]), stds=np.array([1, 3]))

    def test_adds_freqs(self):
        new_freqs = np.array([23, 47, 15])
        new_means = np.array([500, 1000, 17000])
        new_stds = np.array([1, 2, 3])
        self.spectrum.add_freq(new_freqs, new_means, new_stds)
        np.testing.assert_array_equal(self.spectrum.freqs, np.array([np.array([2, 5, 1]), np.array([8, 10, 1]),
                                      np.array([23, 47, 15])]))
        np.testing.assert_array_equal(self.spectrum.means, np.array([200, 300, 500, 1000, 17000]))
        np.testing.assert_array_equal(self.spectrum.stds, np.array([1, 3, 1, 2, 3]))

if __name__ == '__main__':
    unittest.main()
