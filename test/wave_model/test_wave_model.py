from src.wave_model.wave_model import SinWave
import numpy as np
import unittest


class TestSinWave(unittest.TestCase):
    def setUp(self):
        self.wave = SinWave(freq=1, amp=1)
        self.tolerance = 1e-16

    def test_calculate_sine_correctly(self):
        expected_arr = np.array([0, 1.0, 1.2246467991473532e-16, -1.0, -2.4492935982947064e-16])
        np.testing.assert_allclose(self.wave.get_array(5), expected_arr, self.tolerance)

    def test_altering_amplitude_calculates_sine_correctly(self):
        self.wave.amp = 3
        expected_arr = np.array([0, 3.0, 3.6739403974420594e-16, -3.0, -7.347880794884119e-16])
        np.testing.assert_allclose(self.wave.get_array(5), expected_arr, self.tolerance)

    def test_altering_frequency_calculates_sine_correctly(self):
        self.wave.freq = 2
        expected_arr = np.array([0, 0.9749279121818236, -0.433883739117558, -0.7818314824680299, 0.7818314824680296,
                                 0.43388373911755845, -0.9749279121818235, -4.898587196589413e-16])
        np.testing.assert_allclose(self.wave.get_array(8), expected_arr, self.tolerance)


if __name__ == '__main__':
    unittest.main()
