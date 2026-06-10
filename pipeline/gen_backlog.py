# P1 비IID 연합학습 이상탐지 실험 전체 스윕을 backlog JSON 파일로 생성하는 스크립트
import argparse
import itertools
import json
import os


# 최대 의미있는 스윕: 4 algo × 6 alpha × 4 machine × 20 seed × 3 dB = 5760 잡
ALGORITHMS = ["fedavg", "fedprox", "clustered_fl", "personalized"]
ALPHAS = [0.05, 0.1, 0.5, 1.0, 10.0, 100.0]
MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
SEEDS = list(range(20))
DB_LEVELS = ["6dB", "0dB", "-6dB"]
DATA_ROOT = "~/abada-night/data/mimii"

FIXED = {
    "num_sites": 10,
    "rounds": 30,
    "local_epochs": 2,
    "data_root": DATA_ROOT,
}


def alpha_slug(alpha: float) -> str:
    """알파 값을 파일명용 슬러그로 변환한다."""
    return f"a{alpha:.4g}".replace(".", "p")


def make_config(algorithm: str, alpha: float, machine_type: str, seed: int, db: str) -> dict:
    """단일 실험 설정 딕셔너리를 생성한다."""
    name = f"p1_{algorithm}_{alpha_slug(alpha)}_{machine_type}_{db}_s{seed}"
    return {
        "name": name,
        "algorithm": algorithm,
        "alpha": alpha,
        "machine_type": machine_type,
        "db_level": db,
        "seed": seed,
        **FIXED,
    }


def all_configs(db_filter=None, shard=None) -> list:
    """전체 스윕 설정 목록 (순서 결정론적). db_filter=허용 dB 리스트, shard=(i,n) 노드 분할."""
    combos = itertools.product(ALGORITHMS, ALPHAS, MACHINE_TYPES, SEEDS, DB_LEVELS)
    cfgs = [make_config(*c) for c in combos]
    if db_filter:
        cfgs = [c for c in cfgs if c["db_level"] in db_filter]
    if shard:
        i, n = shard
        cfgs = [c for idx, c in enumerate(cfgs) if idx % n == i]
    return cfgs


def print_summary(configs: list, minutes_per_job: float, num_gpus: int) -> None:
    """총 잡 수, GPU-시간, 벽시계 예측을 출력한다."""
    total_jobs = len(configs)
    total_minutes = total_jobs * minutes_per_job
    total_gpu_hours = total_minutes / 60.0
    wall_hours = total_minutes / num_gpus / 60.0
    print(f"Total jobs       : {total_jobs}")
    print(f"Per-job estimate : {minutes_per_job} min")
    print(f"GPUs available   : {num_gpus}")
    print(f"Total GPU-hours  : {total_gpu_hours:.1f} h")
    print(f"Wall-clock est.  : {wall_hours:.2f} h  ({total_jobs} × {minutes_per_job} min / {num_gpus} GPUs)")


def generate(out_dir: str, dry_run: bool = False,
             minutes_per_job: float = 8.0, num_gpus: int = 20,
             db_filter=None, shard=None) -> None:
    """스윕 전체를 JSON 파일로 생성하거나 dry-run 요약만 출력한다."""
    configs = all_configs(db_filter=db_filter, shard=shard)
    print_summary(configs, minutes_per_job, num_gpus)
    if dry_run:
        print("[dry-run] No files written.")
        return
    os.makedirs(out_dir, exist_ok=True)
    for cfg in configs:
        path = os.path.join(out_dir, f"{cfg['name']}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
    written = len(configs)
    print(f"Written {written} JSON files to: {out_dir}")


def parse_args() -> argparse.Namespace:
    """CLI 인수를 파싱한다."""
    parser = argparse.ArgumentParser(
        description="P1 federated-learning sweep backlog generator"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print summary only; do not write files."
    )
    parser.add_argument(
        "--out", default=os.path.expanduser("~/abada-night/backlog"),
        help="Output directory for JSON config files (default: ~/abada-night/backlog)."
    )
    parser.add_argument(
        "--minutes-per-job", type=float, default=8.0,
        metavar="MINUTES",
        help="Estimated minutes per job (default: 8)."
    )
    parser.add_argument(
        "--num-gpus", type=int, default=20,
        metavar="N",
        help="Number of GPUs for wall-clock estimate (default: 20)."
    )
    parser.add_argument(
        "--db-levels", default=None,
        help="콤마구분 dB 필터 (예: '-6dB,0dB'). 미지정시 전체 3 dB."
    )
    parser.add_argument(
        "--shard", default=None, metavar="i/n",
        help="노드 분할: i/n (예: 0/2 = 절반). 미지정시 전체."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    db_filter = args.db_levels.split(",") if args.db_levels else None
    shard = tuple(int(x) for x in args.shard.split("/")) if args.shard else None
    generate(
        out_dir=args.out,
        dry_run=args.dry_run,
        minutes_per_job=args.minutes_per_job,
        num_gpus=args.num_gpus,
        db_filter=db_filter,
        shard=shard,
    )
