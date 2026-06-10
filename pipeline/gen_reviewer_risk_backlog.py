"""Generate P1 reviewer-risk reduction experiment grids.

Profiles:
  e2_baseline_anchors
    - centralized_pooled: 4 machines x 3 SNR x 10 seeds = 120
    - local_only: 4 machines x 3 SNR x 2 alpha x 10 seeds = 240
    - total 360

  e3_fedprox_mu
    - valve/-6dB and fan/6dB x 4 mu x 2 alpha x 5 seeds = 80

  e3_fedprox_mu_scoped
    - {valve, slider} x {-6dB, 6dB} x 5 mu x 2 alpha x 10 seeds = 400
    - includes mu=0 to verify the FedAvg-limit sanity check

  e4_backbone_sensitivity
    - 4 algorithms x {valve, slider} x {-6dB, 6dB} x 2 alpha x 5 seeds = 160
    - uses LiteConvAutoencoder to test whether condition-vs-algorithm ordering
      is tied to the original ConvAutoencoder capacity point

The grids are deliberately small relative to the completed 5,760-run sweep.
They are meant to reduce reviewer risk, not to replace the main result.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "analysis_outputs" / "job_grids" / "p1_reviewer_risk_20260607"

MACHINES = ["fan", "pump", "slider", "valve"]
DB_LEVELS = ["-6dB", "0dB", "6dB"]
BASELINE_SEEDS = list(range(10))
LOCAL_ONLY_ALPHAS = [0.05, 100.0]
FEDPROX_MUS = [0.001, 0.01, 0.1, 1.0]
FEDPROX_ALPHAS = [0.1, 1.0]
FEDPROX_STRESS_CELLS = [("valve", "-6dB"), ("fan", "6dB")]
FEDPROX_SEEDS = list(range(5))
FEDPROX_SCOPED_MUS = [0.0, 0.001, 0.01, 0.1, 1.0]
FEDPROX_SCOPED_ALPHAS = [0.05, 100.0]
FEDPROX_SCOPED_CELLS = [("valve", "-6dB"), ("valve", "6dB"), ("slider", "-6dB"), ("slider", "6dB")]
FEDPROX_SCOPED_SEEDS = list(range(10))
BACKBONE_ALGORITHMS = ["fedavg", "fedprox", "clustered_fl", "personalized"]
BACKBONE_ALPHAS = [0.05, 100.0]
BACKBONE_CELLS = [("valve", "-6dB"), ("valve", "6dB"), ("slider", "-6dB"), ("slider", "6dB")]
BACKBONE_SEEDS = list(range(5))


def alpha_slug(alpha: float | str) -> str:
    if isinstance(alpha, str):
        return alpha
    return str(alpha).replace(".", "p")


def mu_slug(mu: float) -> str:
    return str(mu).replace(".", "p")


def base_config(name: str, algorithm: str, machine: str, db: str, seed: int, alpha: float) -> dict:
    return {
        "name": name,
        "paper_id": "P1",
        "experiment_family": "p1_reviewer_risk_reduction",
        "algorithm": algorithm,
        "alpha": alpha,
        "machine_type": machine,
        "db_level": db,
        "seed": seed,
        "num_sites": 10,
        "rounds": 30,
        "local_epochs": 2,
        "n_mels": 128,
        "n_frames": 320,
        "bottleneck": 1024,
        "lr": 0.001,
        "normalization_mode": "central",
        "use_wandb": True,
        "wandb_project": "abada-night",
        "wandb_group": "p1_reviewer_risk_20260607",
        "log_evidence": False,
    }


def centralized_configs() -> list[dict]:
    configs = []
    for machine, db, seed in itertools.product(MACHINES, DB_LEVELS, BASELINE_SEEDS):
        name = f"p1_anchor_centralized_pooled_{machine}_{db}_s{seed}"
        cfg = base_config(name, "centralized_pooled", machine, db, seed, alpha=1.0)
        cfg["baseline_alpha_scope"] = "not_applicable_for_pooled_training"
        cfg["wandb_tags"] = ["P1", "reviewer-risk", "E2", "centralized-pooled"]
        configs.append(cfg)
    return configs


def local_only_configs() -> list[dict]:
    configs = []
    for machine, db, alpha, seed in itertools.product(
        MACHINES, DB_LEVELS, LOCAL_ONLY_ALPHAS, BASELINE_SEEDS
    ):
        name = f"p1_anchor_local_only_a{alpha_slug(alpha)}_{machine}_{db}_s{seed}"
        cfg = base_config(name, "local_only", machine, db, seed, alpha=alpha)
        cfg["wandb_tags"] = ["P1", "reviewer-risk", "E2", "local-only", f"alpha:{alpha_slug(alpha)}"]
        configs.append(cfg)
    return configs


def fedprox_mu_configs() -> list[dict]:
    configs = []
    for (machine, db), mu, alpha, seed in itertools.product(
        FEDPROX_STRESS_CELLS, FEDPROX_MUS, FEDPROX_ALPHAS, FEDPROX_SEEDS
    ):
        name = f"p1_fedprox_mu{mu_slug(mu)}_a{alpha_slug(alpha)}_{machine}_{db}_s{seed}"
        cfg = base_config(name, "fedprox", machine, db, seed, alpha=alpha)
        cfg["fedprox_mu"] = mu
        cfg["wandb_tags"] = [
            "P1",
            "reviewer-risk",
            "E3",
            "fedprox-mu",
            f"mu:{mu_slug(mu)}",
            f"alpha:{alpha_slug(alpha)}",
        ]
        configs.append(cfg)
    return configs


def fedprox_mu_scoped_configs() -> list[dict]:
    configs = []
    for (machine, db), mu, alpha, seed in itertools.product(
        FEDPROX_SCOPED_CELLS, FEDPROX_SCOPED_MUS, FEDPROX_SCOPED_ALPHAS, FEDPROX_SCOPED_SEEDS
    ):
        name = f"p1_fedprox_scoped_mu{mu_slug(mu)}_a{alpha_slug(alpha)}_{machine}_{db}_s{seed}"
        cfg = base_config(name, "fedprox", machine, db, seed, alpha=alpha)
        cfg["fedprox_mu"] = mu
        cfg["wandb_group"] = "p1_fedprox_mu_scoped_20260607"
        cfg["wandb_tags"] = [
            "P1",
            "reviewer-risk",
            "E3",
            "fedprox-mu-scoped",
            f"mu:{mu_slug(mu)}",
            f"alpha:{alpha_slug(alpha)}",
        ]
        configs.append(cfg)
    return configs


def backbone_sensitivity_configs() -> list[dict]:
    configs = []
    for algorithm, (machine, db), alpha, seed in itertools.product(
        BACKBONE_ALGORITHMS, BACKBONE_CELLS, BACKBONE_ALPHAS, BACKBONE_SEEDS
    ):
        name = (
            f"p1_backbone_lite_{algorithm}_a{alpha_slug(alpha)}_"
            f"{machine}_{db}_s{seed}"
        )
        cfg = base_config(name, algorithm, machine, db, seed, alpha=alpha)
        cfg["model_family"] = "LiteConvAutoencoder"
        cfg["bottleneck"] = 256
        cfg["wandb_group"] = "p1_backbone_sensitivity_20260607"
        cfg["wandb_tags"] = [
            "P1",
            "reviewer-risk",
            "E4",
            "backbone-sensitivity",
            "model:LiteConvAutoencoder",
            f"alpha:{alpha_slug(alpha)}",
        ]
        configs.append(cfg)
    return configs


def configs_for_profile(profile: str) -> list[dict]:
    if profile == "e2_baseline_anchors":
        return centralized_configs() + local_only_configs()
    if profile == "e3_fedprox_mu":
        return fedprox_mu_configs()
    if profile == "e3_fedprox_mu_scoped":
        return fedprox_mu_scoped_configs()
    if profile == "e4_backbone_sensitivity":
        return backbone_sensitivity_configs()
    if profile == "all":
        return centralized_configs() + local_only_configs() + fedprox_mu_configs()
    raise ValueError("profile must be e2_baseline_anchors, e3_fedprox_mu, e3_fedprox_mu_scoped, e4_backbone_sensitivity, or all")


def apply_shard(configs: list[dict], shard: tuple[int, int] | None) -> list[dict]:
    if shard is None:
        return configs
    idx, total = shard
    if total <= 0 or idx < 0 or idx >= total:
        raise ValueError("shard must satisfy 0 <= index < total")
    return [cfg for i, cfg in enumerate(configs) if i % total == idx]


def manifest(configs: list[dict], profile: str, shard: tuple[int, int] | None) -> dict:
    def counts(key: str) -> dict[str, int]:
        return dict(sorted(Counter(str(c.get(key, "")) for c in configs).items()))

    return {
        "profile": profile,
        "shard": shard,
        "total_jobs": len(configs),
        "axis_counts": {
            "algorithm": counts("algorithm"),
            "machine_type": counts("machine_type"),
            "db_level": counts("db_level"),
            "alpha": counts("alpha"),
            "seed": counts("seed"),
            "fedprox_mu": counts("fedprox_mu"),
            "model_family": counts("model_family"),
        },
        "purpose": [
            "E2 closes the centralized/local training-anchor reviewer gap.",
            "E3 is conditional and probes whether FedProx underperformance is mu-specific.",
            "E3 scoped includes mu=0 to verify the FedAvg-limit sanity check in high/low condition corners.",
            "E4 tests whether the condition-stratified conclusion survives a second compact ConvAE capacity point.",
        ],
    }


def write_grid(out_dir: Path, configs: list[dict], profile: str, shard: tuple[int, int] | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for cfg in configs:
        (out_dir / f"{cfg['name']}.json").write_text(
            json.dumps(cfg, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (out_dir / "_manifest.json").write_text(
        json.dumps(manifest(configs, profile, shard), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate P1 reviewer-risk reduction grids")
    p.add_argument(
        "--profile",
        default="e2_baseline_anchors",
        choices=[
            "e2_baseline_anchors",
            "e3_fedprox_mu",
            "e3_fedprox_mu_scoped",
            "e4_backbone_sensitivity",
            "all",
        ],
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--shard-index", type=int)
    p.add_argument("--shard-total", type=int)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    shard = None
    if args.shard_index is not None or args.shard_total is not None:
        if args.shard_index is None or args.shard_total is None:
            raise SystemExit("Both --shard-index and --shard-total are required for sharding.")
        shard = (args.shard_index, args.shard_total)
    configs = apply_shard(configs_for_profile(args.profile), shard)
    print(json.dumps(manifest(configs, args.profile, shard), indent=2, sort_keys=True))
    if not args.dry_run:
        write_grid(args.out_dir, configs, args.profile, shard)
        print(f"Wrote {len(configs)} configs to {args.out_dir}")


if __name__ == "__main__":
    main()
