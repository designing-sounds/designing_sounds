from src.wave_model.priors import PeriodicPrior, SquaredExpPrior, MultPrior
import numpy as np
import unittest

if __name__ == '__main__':
    unittest.main()


class TestPriors(unittest.TestCase):

    def setUp(self):
        self.tolerance = 3 * 1e-5
        self.approx_dim = 50

    def test_periodic_kernel(self):
        assert round(PeriodicPrior(50).kernel(0.497, 100, 3, 1.2, 0, 0), 9) == 7.170423242

    def test_periodic_covariance_matrix(self):
        expected = np.array([[[14.45854572, 2.82116777], [14.45854572, 2.821167767]]])
        np.testing.assert_allclose(expected,
                                   PeriodicPrior(50).covariance_matrix(np.array([1, -3]), np.array([0.19, 5.12]), [442],
                                                                       [5], [0.06], [0], [0]), self.tolerance)

    def test_squared_exp_kernel(self):
        assert round(SquaredExpPrior(50).kernel(0.497, 100, 3, 1.2, 0, 0), 9) == 8.260272386

    def test_squared_exp_covariance_matrix(self):
        expected = [[[6.650516e-39, 0.000000e+00], [0.000000e+00, 0.000000e+00]]]
        np.testing.assert_allclose(expected,
                                   SquaredExpPrior(50).covariance_matrix(np.array([1, -3]), np.array([0.19, 5.12]),
                                                                         [442],
                                                                         [5], [0.06], [0], [0]), self.tolerance)

    def test_mult_kernel(self):
        assert round(MultPrior(50).kernel(0.497, 100, 3, 1.2, 2, 3), 9) == 28.290790322

    def test_mult_covariance_matrix(self):
        expected = [[[8.976230e-34, 0.000000e+00], [7.908304e-34, 0.000000e+00]]]
        np.testing.assert_allclose(expected,
                                   MultPrior(50).covariance_matrix(np.array([3, -3]), np.array([0.19, 5.12]),
                                                                   [12],
                                                                   [5], [0.06], [6], [3]), self.tolerance)
