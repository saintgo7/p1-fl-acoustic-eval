#!/usr/bin/env python3
# P1 E5 그리드(DCASE 2022 ToyCar/ToyTrain 2차 데이터셋) run config 생성기
"""Generate the scoped E5 second-dataset grid.

Grid: 2 machines x 3 sections x 4 algorithms x alpha {0.05, 100.0} x 10 seeds
    = 480 runs (mirrors the E3/E4 reviewer-risk profile scale).
Source domain only; section is the condition axis replacing MIMII's SNR.
"""

import argparse
import json
import os

MACHINES = ("ToyCar", "ToyTrain")
SECTIONS = ("00", "01", "02")
ALGORITHMS = ("fedavg", "fedprox", "clustered_fl", "personalized")
ALPHAS = (0.05, 100.0)
SEEDS = tuple(range(10))


def alpha_key(alpha: float) -> str:
    return ("a%g" % alpha).replace(".", "p")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate P1 E5 DCASE-2022 run configs.")
    parser.add_argument("--out", default=os.path.expanduser("~/abada-night/backlog"),
                        help="Output directory for JSON config files.")
    parser.add_argument("--data-root", default="~/abada-night/data/dcase2022")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--num-sites", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    count = 0
    for machine in MACHINES:
        for section in SECTIONS:
            for algorithm in ALGORITHMS:
                for alpha in ALPHAS:
                    for seed in SEEDS:
                        name = (f"p1e5_{algorithm}_{alpha_key(alpha)}_"
                                f"{machine.lower()}_sec{section}_s{seed}")
                        config = {
                            "name": name,
                            "dataset": "dcase2022",
                            "algorithm": algorithm,
                            "alpha": alpha,
                            "machine_type": machine,
                            "section": section,
                            "seed": seed,
                            "num_sites": args.num_sites,
                            "rounds": args.rounds,
                            "local_epochs": args.local_epochs,
                            "data_root": args.data_root,
                        }
                        path = os.path.join(args.out, name + ".json")
                        with open(path, "w", encoding="utf-8") as fh:
                            json.dump(config, fh, indent=2)
                        count += 1
    print(f"wrote {count} configs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
