"""
Unit tests for Schur-complement orthogonal feature screening.
Verifies that redundant collinear attributes are suppressed.
"""

import unittest
import numpy as np

from dp_coscreen.schur import project_psd_cone, schur_orthogonal_screening


class TestSchurOrthogonal(unittest.TestCase):
    def test_collinear_feature_suppression(self):
        """
        Construct a synthetic covariance where feature 0 and feature 1 are nearly identical (collinear),
        both strongly correlated with target. Schur screening must pick one of them, and then reject
        the duplicate in favor of complementary features.
        """
        rng = np.random.default_rng(789)
        n = 1000
        # True informative signals
        f0 = rng.normal(size=n)
        f1 = f0 + rng.normal(scale=0.01, size=n)  # Nearly identical duplicate of f0
        f2 = rng.normal(size=n)  # Independent informative signal
        f3 = rng.normal(scale=0.1, size=n)  # Low-signal noise

        # Target depends on f0 and f2
        y = 2.0 * f0 + 3.0 * f2 + rng.normal(scale=0.5, size=n)
        
        X = np.column_stack([f0, f1, f2, f3])
        A = (X.T @ X) / n
        q = (X.T @ y) / n
        
        A_psd = project_psd_cone(A)
        selected = schur_orthogonal_screening(A_psd, q, K=2, rho=1e-3)
        
        # Must select f0 (or f1) AND f2! Must NOT select both f0 and f1!
        self.assertIn(2, selected)
        self.assertTrue((0 in selected and 1 not in selected) or (1 in selected and 0 not in selected),
                        msg=f"Failed to eliminate collinear duplicate: selected={selected}")

    def test_selection_budget_exactness(self):
        d = 20
        K = 7
        rng = np.random.default_rng(101)
        A = rng.normal(size=(d, d))
        A = A.T @ A
        q = rng.normal(size=d)
        
        selected = schur_orthogonal_screening(A, q, K=K, rho=1e-3)
        self.assertEqual(len(selected), K)
        self.assertEqual(len(set(selected)), K)  # All distinct


if __name__ == "__main__":
    unittest.main()
