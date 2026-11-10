"""
Integration test for the full DP-CoScreen collaborative screening pipeline.
"""

import unittest
import numpy as np

from dp_coscreen import DPCoScreen, evaluate_classifier, partition_dirichlet_noniid


class TestEndToEndPipeline(unittest.TestCase):
    def test_full_federated_pipeline(self):
        # 1. Generate synthetic non-IID federation with representative sample size
        rng = np.random.default_rng(2026)
        n_samples = 15000
        n_features = 25
        
        # Ground truth signals
        X = rng.uniform(0.0, 1.0, size=(n_samples, n_features))
        # Binary target strongly governed by features 0 and 1
        logits = 5.0 * X[:, 0] - 4.0 * X[:, 1] - 0.5
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.uniform(0.0, 1.0, size=n_samples) < probs).astype(int)
        
        # Partition into train/test
        train_idx = np.arange(12000)
        test_idx = np.arange(12000, 15000)
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Partition train into M=5 edge clients non-IID
        client_datasets = partition_dirichlet_noniid(X_train, y_train, n_clients=5, alpha=0.5, seed=42)
        
        # 2. Run DP-CoScreen
        screener = DPCoScreen(K=5, epsilon=1.0, delta=1e-5, rho=1e-3, seed=42)
        screener.fit(client_datasets)
        
        selected = screener.selected_indices_
        self.assertIsNotNone(selected)
        self.assertEqual(len(selected), 5)
        
        # Check that top informative features are among selected
        self.assertTrue(0 in selected or 1 in selected)
        
        # 3. Downstream evaluation
        metrics = evaluate_classifier(X_train, y_train, X_test, y_test, selected_indices=selected)
        self.assertGreater(metrics["macro_f1"], 0.60)
        self.assertGreater(metrics["accuracy"], 0.60)


if __name__ == "__main__":
    unittest.main()
