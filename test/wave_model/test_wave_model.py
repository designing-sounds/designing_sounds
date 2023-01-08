from src.wave_model.wave_model import SoundModel
import numpy as np
import unittest

if __name__ == '__main__':
    unittest.main()


class TestSoundModel(unittest.TestCase):

    def setUp(self):
        self.max_harmonics = 1000
        self.max_samples_per_harmonic = 4
        self.max_power_spectrums = 4
        self.sound_model = SoundModel(self.max_samples_per_harmonic, self.max_power_spectrums)
        self.tolerance = 3 * 1e-4

    def test_model_chunk_sound(self):
        samples = self.max_samples_per_harmonic
        num_harmonics = self.max_harmonics
        sample_rate = samples * num_harmonics
        test = np.zeros((num_harmonics, samples))
        self.sound_model.update_power_spectrum(0, 1000, 1, 1, 1, 1, 1)
        for i in range(num_harmonics):
            test[i] = self.sound_model.model_sound(sample_rate, samples / sample_rate, i * samples / sample_rate)

        test = test.flatten()
        expected = self.sound_model.model_sound(sample_rate, 1, 0)
        np.testing.assert_allclose(expected, test, self.tolerance)
