from src.wavemodel.wave_model import SinWave
import numpy as np
import unittest


class TestSinWave(unittest.TestCase):
    def setUp(self):
        self.wave = SinWave()
        self.tolerance = 1e-07

    def test_calculate_sine_correctly(self):
        expected_arr = np.array([0, -0.4303012352, 0.7768532196, -0.9722068419, 0.9783405476])
        np.testing.assert_allclose(self.wave.get_array(5), expected_arr, self.tolerance)

    def test_altering_amplitude_calculates_sine_correctly(self):
        self.wave.set_amp(3)
        expected_arr = np.array([0, -1.2909037056, 2.3305596588, -2.9166205257, 2.9350216428])
        np.testing.assert_allclose(self.wave.get_array(5), expected_arr, self.tolerance)

    def test_altering_frequency_calculates_sine_correctly(self):
        self.wave.set_freq(5)
        expected_arr = np.array([0, -0.7940605467, -0.9653214123, -0.3794588189, 0.5040219251])
        np.testing.assert_allclose(self.wave.get_array(5), expected_arr, self.tolerance)

    def test_altering_phase_calculates_sine_correctly(self):
        self.wave.set_phase(0.5)
        expected_arr = np.array([0.5, 0.0696987829, 1.2768532196, -0.4722068419, 1.4783405476])
        np.testing.assert_allclose(self.wave.get_array(5), expected_arr, self.tolerance)


if __name__ == '__main__':
    unittest.main()
