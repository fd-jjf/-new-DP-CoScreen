"""
Unit tests for PSD cone projection and numerical stabilization.
"""

import unittest
import numpy as np

from dp_coscreen.schur import project_psd_cone


class TestPSDProjection(unittest.TestCase):
    def test_indefinite_matrix_projection(self):
        # Create a symmetric indefinite matrix with negative eigenvalues
        rng = np.random.default_rng(123)
        d = 15
        M = rng.normal(size=(d, d))
        A = (M + M.T) / 2.0
        # Force some strongly negative eigenvalues
        A[0, 0] -= 50.0
        
        eigvals_before = np.linalg.eigvalsh(A)
        self.assertLess(np.min(eigvals_before), 0.0)
        
        A_proj = project_psd_cone(A)
        eigvals_after = np.linalg.eigvalsh(A_proj)
        
        # Check all eigenvalues are >= 0 (within float precision)
        self.assertGreaterEqual(np.min(eigvals_after), -1e-12)
        # Check symmetry
        np.testing.assert_allclose(A_proj, A_proj.T, atol=1e-12)

    def test_already_psd_matrix_invariance(self):
        rng = np.random.default_rng(456)
        d = 10
        M = rng.normal(size=(50, d))
        A_true_psd = M.T @ M
        
        A_proj = project_psd_cone(A_true_psd)
        np.testing.assert_allclose(A_proj, A_true_psd, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
