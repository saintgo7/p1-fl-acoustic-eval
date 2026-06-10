import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
import sys


def _load_fl_train():
    module_path = Path(__file__).with_name("fl_train.py")
    sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location("fl_train", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class P1EvidenceLoggingTest(unittest.TestCase):
    def test_clustered_fl_writes_evidence_bundle(self):
        fl_train = _load_fl_train()
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = {
                "name": "test_p1_evidence_bundle",
                "algorithm": "clustered_fl",
                "alpha": 0.5,
                "num_sites": 3,
                "rounds": 2,
                "local_epochs": 1,
                "seed": 7,
                "machine_type": "synthetic",
                "n_mels": 8,
                "n_frames": 4,
                "bottleneck": 8,
                "n_normal": 8,
                "use_wandb": False,
                "log_evidence": True,
                "evidence_output_dir": tmpdir,
            }
            metrics = fl_train.train_federated(cfg)
            self.assertIn("auroc", metrics)

            bundle_root = Path(tmpdir) / cfg["name"]
            self.assertTrue(bundle_root.exists())

            expected = {
                "reconstruction_errors.csv",
                "site_auroc.csv",
                "cluster_assignments.csv",
                "cluster_stability.csv",
                "manifest.json",
            }
            self.assertTrue(expected.issubset({p.name for p in bundle_root.iterdir()}))

            with (bundle_root / "site_auroc.csv").open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertGreaterEqual(len(rows), 1)
            self.assertIn("auroc", rows[0])
            self.assertIn("partial_auroc_fpr_0_1", rows[0])

            with (bundle_root / "reconstruction_errors.csv").open(encoding="utf-8") as fh:
                recon_rows = list(csv.DictReader(fh))
            self.assertGreater(len(recon_rows), 0)
            self.assertIn("sample_path_hash", recon_rows[0])
            self.assertIn("predicted_anomaly", recon_rows[0])

            with (bundle_root / "cluster_stability.csv").open(encoding="utf-8") as fh:
                stability_rows = list(csv.DictReader(fh))
            self.assertGreater(len(stability_rows), 0)
            self.assertIn("stability_metric", stability_rows[0])


if __name__ == "__main__":
    unittest.main()
