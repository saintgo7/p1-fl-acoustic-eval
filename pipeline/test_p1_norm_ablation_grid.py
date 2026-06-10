import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import gen_normalization_ablation_backlog as grid


class P1NormalizationAblationGridTest(unittest.TestCase):
    def test_smoke_profile_has_expected_axes_and_unique_names(self):
        configs = grid.all_configs(profile="smoke")
        self.assertEqual(len(configs), 72)
        self.assertEqual(
            {c["normalization_mode"] for c in configs},
            {"central", "federated_train", "local_site"},
        )
        self.assertEqual({c["experiment_group"] for c in configs}, {"p1_normalization_ablation"})
        self.assertTrue(all(c["use_wandb"] is True for c in configs))
        self.assertTrue(all(c["wandb_project"] == "abada-night" for c in configs))
        names = [c["name"] for c in configs]
        self.assertEqual(len(names), len(set(names)))

    def test_paper_defense_profile_count(self):
        configs = grid.all_configs(profile="paper-defense")
        # 3 algorithms x 3 normalization modes x 3 machines x 3 dB x 3 alphas x 5 seeds
        self.assertEqual(len(configs), 1215)

    def test_shards_partition_without_overlap(self):
        full = grid.all_configs(profile="smoke")
        shard0 = grid.all_configs(profile="smoke", shard=(0, 2))
        shard1 = grid.all_configs(profile="smoke", shard=(1, 2))
        names0 = {c["name"] for c in shard0}
        names1 = {c["name"] for c in shard1}
        self.assertFalse(names0 & names1)
        self.assertEqual(names0 | names1, {c["name"] for c in full})

    def test_generate_writes_manifest_and_configs(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "backlog"
            configs = grid.generate(out_dir=str(out_dir), profile="smoke")
            self.assertEqual(len(configs), 72)
            self.assertEqual(
                len([p for p in out_dir.glob("*.json") if not p.name.startswith("_")]),
                72,
            )
            manifest = json.loads((out_dir / "_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["total_jobs"], 72)
            self.assertEqual(manifest["gpu_safety"]["n1_forbidden_gpus"], [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
