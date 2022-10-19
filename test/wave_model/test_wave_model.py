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
        self.spectrum = PowerSpectrum()

    def test_adds_freqs(self):
        self.spectrum.add_element(1000, 1, 10)
        self.spectrum.add_element(2000, 3, 10)
        assert self.spectrum.freqs.shape == (2,10)

if __name__ == '__main__':
    unittest.main()
