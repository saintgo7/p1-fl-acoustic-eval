import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import mimii_adapter


class P1NormalizationModesTest(unittest.TestCase):
    def setUp(self):
        # Shape: N, n_mels, T.  Two sites have intentionally different offsets.
        site0 = np.array(
            [
                [[1.0, 2.0, 3.0], [10.0, 11.0, 12.0]],
                [[2.0, 3.0, 4.0], [11.0, 12.0, 13.0]],
                [[3.0, 4.0, 5.0], [12.0, 13.0, 14.0]],
            ],
            dtype=np.float32,
        )
        site1 = np.array(
            [
                [[101.0, 102.0, 103.0], [210.0, 211.0, 212.0]],
                [[102.0, 103.0, 104.0], [211.0, 212.0, 213.0]],
                [[103.0, 104.0, 105.0], [212.0, 213.0, 214.0]],
            ],
            dtype=np.float32,
        )
        self.features = np.concatenate([site0, site1], axis=0)
        self.splits = [
            (np.array([0, 1], dtype=int), np.array([2], dtype=int)),
            (np.array([3, 4], dtype=int), np.array([5], dtype=int)),
        ]

    def test_federated_train_matches_central_train_statistics(self):
        central = mimii_adapter._normalize_features_by_mode(
            self.features, self.splits, mode="central"
        )
        federated = mimii_adapter._normalize_features_by_mode(
            self.features, self.splits, mode="federated_train"
        )
        np.testing.assert_allclose(central, federated, rtol=1e-6, atol=1e-6)

    def test_local_site_normalizes_each_sites_train_split_independently(self):
        normalized = mimii_adapter._normalize_features_by_mode(
            self.features, self.splits, mode="local_site"
        )
        for train_idx, _ in self.splits:
            train = normalized[train_idx]
            np.testing.assert_allclose(
                train.mean(axis=(0, 2)),
                np.zeros(self.features.shape[1], dtype=np.float32),
                atol=1e-6,
            )
            np.testing.assert_allclose(
                train.std(axis=(0, 2)),
                np.ones(self.features.shape[1], dtype=np.float32),
                atol=1e-5,
            )

    def test_invalid_normalization_mode_fails_fast(self):
        with self.assertRaises(ValueError):
            mimii_adapter._normalize_features_by_mode(
                self.features, self.splits, mode="global_magic"
            )


if __name__ == "__main__":
    unittest.main()
