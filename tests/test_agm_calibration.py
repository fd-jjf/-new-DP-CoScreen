"""
Unit tests for Analytic Gaussian Mechanism (AGM) calibration and L2 sensitivity.
"""

import math
import unittest
import numpy as np

from dp_coscreen.privacy import (
    agm_equation,
    calibrate_agm,
    compute_l2_sensitivity,
    sample_symmetric_gaussian_noise,
)


class TestAGMPrivacy(unittest.TestCase):
    def test_l2_sensitivity(self):
        d, n = 94, 1000
        delta_2 = compute_l2_sensitivity(d, n)
        expected = math.sqrt(2.0) * (d + 1) / n
        self.assertAlmostEqual(delta_2, expected, places=7)

    def test_agm_root_satisfaction(self):
        epsilons = [0.1, 0.5, 1.0, 2.0, 5.0]
        delta = 1e-5
        delta_2 = 0.15

        for eps in epsilons:
            sigma = calibrate_agm(eps, delta, delta_2)
            v = sigma / delta_2
            residual = agm_equation(v, eps, delta)
            self.assertAlmostEqual(residual, 0.0, places=9, msg=f"Failed for eps={eps}")

    def test_agm_tighter_than_classical(self):
        eps = 1.0
        delta = 1e-5
        delta_2 = 0.2
        
        sigma_agm = calibrate_agm(eps, delta, delta_2)
        sigma_classical = delta_2 * math.sqrt(2.0 * math.log(1.25 / delta)) / eps
        
        # AGM is provably strictly tighter than classical Gaussian bound
        self.assertLess(sigma_agm, sigma_classical)

    def test_symmetric_noise_shape_and_properties(self):
        dim = 10
        sigma = 0.5
        rng = np.random.default_rng(42)
        E = sample_symmetric_gaussian_noise(dim, sigma, rng)
        
        # Symmetry check
        np.testing.assert_allclose(E, E.T, atol=1e-12)
        self.assertEqual(E.shape, (dim, dim))


if __name__ == "__main__":
    unittest.main()
