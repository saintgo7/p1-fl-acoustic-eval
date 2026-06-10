"""Generate P1 normalization-ablation backlog configs.

This sweep is intentionally smaller than the original 5,760-run grid.  It
targets the main publication risk: whether the current centralized per-mel
train statistic can be replaced by privacy-preserving or site-local
normalization without changing the paper's interpretation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


DATA_ROOT = "~/abada-night/data/mimii"
NORMALIZATION_MODES = ["central", "federated_train", "local_site"]

PROFILES = {
    "smoke": {
        "algorithms": ["fedavg", "personalized"],
        "normalization_modes": NORMALIZATION_MODES,
        "machines": ["valve", "slider"],
        "db_levels": ["-6dB", "6dB"],
        "alphas": [0.05],
        "seeds": [0, 1, 2],
    },
    "paper-defense": {
        "algorithms": ["fedavg", "clustered_fl", "personalized"],
        "normalization_modes": NORMALIZATION_MODES,
        "machines": ["valve", "slider", "fan"],
        "db_levels": ["-6dB", "0dB", "6dB"],
        "alphas": [0.05, 0.5, 100.0],
        "seeds": list(range(5)),
    },
    "full": {
        "algorithms": ["fedavg", "fedprox", "clustered_fl", "personalized"],
        "normalization_modes": NORMALIZATION_MODES + ["none"],
        "machines": ["fan", "pump", "slider", "valve"],
        "db_levels": ["-6dB", "0dB", "6dB"],
        "alphas": [0.05, 0.1, 0.5, 1.0, 10.0, 100.0],
        "seeds": list(range(10)),
    },
}

FIXED = {
    "num_sites": 10,
    "rounds": 30,
    "local_epochs": 2,
    "data_root": DATA_ROOT,
    "use_wandb": True,
    "wandb_project": "abada-night",
    "experiment_group": "p1_normalization_ablation",
    "log_evidence": True,
    "evidence_output_dir": "~/abada-night/p1_evidence",
}


def alpha_slug(alpha: float) -> str:
    """Convert a numeric alpha into the existing filename slug style."""
    return f"a{alpha:.4g}".replace(".", "p")


def db_slug(db_level: str) -> str:
    """Make a dB level safe and compact for filenames."""
    return db_level.replace("-", "m").replace("+", "p")


def make_config(
    algorithm: str,
    normalization_mode: str,
    alpha: float,
    machine_type: str,
    db_level: str,
    seed: int,
) -> dict:
    """Build one deterministic P1 normalization-ablation config."""
    name = (
        f"p1_norm_{normalization_mode}_{algorithm}_{alpha_slug(alpha)}_"
        f"{machine_type}_{db_slug(db_level)}_s{seed}"
    )
    return {
        **FIXED,
        "name": name,
        "algorithm": algorithm,
        "normalization_mode": normalization_mode,
        "alpha": alpha,
        "machine_type": machine_type,
        "db_level": db_level,
        "seed": seed,
        "wandb_group": "p1_normalization_ablation",
        "wandb_tags": [
            "p1",
            "normalization_ablation",
            f"norm:{normalization_mode}",
            f"machine:{machine_type}",
            f"db:{db_level}",
        ],
    }


def _profile_axes(profile: str) -> dict:
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {sorted(PROFILES)}, got {profile!r}")
    return PROFILES[profile]


def all_configs(profile: str = "paper-defense", shard: tuple[int, int] | None = None) -> list[dict]:
    """Return deterministic configs for a profile, optionally sharded by index mod n."""
    axes = _profile_axes(profile)
    combos = itertools.product(
        axes["algorithms"],
        axes["normalization_modes"],
        axes["alphas"],
        axes["machines"],
        axes["db_levels"],
        axes["seeds"],
    )
    cfgs = [make_config(*combo) for combo in combos]
    if shard:
        i, n = shard
        if n <= 0 or i < 0 or i >= n:
            raise ValueError(f"invalid shard {i}/{n}")
        cfgs = [cfg for idx, cfg in enumerate(cfgs) if idx % n == i]
    return cfgs


def _axis_counts(configs: list[dict], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(c[key]) for c in configs).items()))


def build_manifest(configs: list[dict], profile: str, shard: tuple[int, int] | None) -> dict:
    """Build a non-job manifest; p1_worker skips underscore-prefixed JSON files."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_group": "p1_normalization_ablation",
        "profile": profile,
        "shard": f"{shard[0]}/{shard[1]}" if shard else None,
        "total_jobs": len(configs),
        "axes": {
            "algorithm": _axis_counts(configs, "algorithm"),
            "normalization_mode": _axis_counts(configs, "normalization_mode"),
            "machine_type": _axis_counts(configs, "machine_type"),
            "db_level": _axis_counts(configs, "db_level"),
            "alpha": _axis_counts(configs, "alpha"),
            "seed": _axis_counts(configs, "seed"),
        },
        "gpu_safety": {
            "preferred_hosts": ["master", "n3", "wku02"],
            "n1_forbidden_gpus": [0, 1, 2, 3],
            "n1_allowed_gpus": [4, 5, 6, 7],
        },
        "notes": [
            "central reproduces the current simulation preprocessing.",
            "federated_train should be numerically equivalent to central but uses site-local sufficient statistics.",
            "local_site tests deployable per-site normalization under feature shift.",
            "Do not launch on n1 GPU 0-3.",
        ],
    }


def print_summary(configs: list[dict], minutes_per_job: float, num_gpus: int) -> None:
    total_jobs = len(configs)
    total_minutes = total_jobs * minutes_per_job
    print(f"Total jobs       : {total_jobs}")
    print(f"Per-job estimate : {minutes_per_job} min")
    print(f"GPUs available   : {num_gpus}")
    print(f"Total GPU-hours  : {total_minutes / 60.0:.1f} h")
    print(f"Wall-clock est.  : {total_minutes / max(num_gpus, 1) / 60.0:.2f} h")


def generate(
    out_dir: str,
    profile: str = "paper-defense",
    dry_run: bool = False,
    minutes_per_job: float = 8.0,
    num_gpus: int = 16,
    shard: tuple[int, int] | None = None,
) -> list[dict]:
    """Write configs and a manifest, or just return configs in dry-run mode."""
    configs = all_configs(profile=profile, shard=shard)
    print_summary(configs, minutes_per_job, num_gpus)
    if dry_run:
        print("[dry-run] No files written.")
        return configs

    out = Path(os.path.expanduser(out_dir))
    out.mkdir(parents=True, exist_ok=True)
    for cfg in configs:
        with (out / f"{cfg['name']}.json").open("w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)

    manifest = build_manifest(configs, profile=profile, shard=shard)
    with (out / "_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Written {len(configs)} JSON configs plus _manifest.json to: {out}")
    return configs


def parse_shard(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    left, right = value.split("/", 1)
    return int(left), int(right)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P1 normalization-ablation backlog generator")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="paper-defense")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--out",
        default=os.path.expanduser("~/abada-night/p1_norm_ablation/backlog"),
        help="Output backlog directory.",
    )
    parser.add_argument("--minutes-per-job", type=float, default=8.0)
    parser.add_argument("--num-gpus", type=int, default=16)
    parser.add_argument("--shard", default=None, metavar="i/n")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(
        out_dir=args.out,
        profile=args.profile,
        dry_run=args.dry_run,
        minutes_per_job=args.minutes_per_job,
        num_gpus=args.num_gpus,
        shard=parse_shard(args.shard),
    )
