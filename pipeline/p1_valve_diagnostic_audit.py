"""No-GPU P1 valve diagnostic audit.

The script scans MIMII WAV paths only. It does not extract audio features or
train models. Its purpose is to determine whether valve difficulty could be
partly explained by site partition imbalance or anomaly assignment gaps.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

import data_loader as dl
import mimii_adapter


DEFAULT_ALPHAS = [0.05, 0.5, 100.0]
DEFAULT_DB_LEVELS = ["-6dB", "0dB", "6dB"]
DEFAULT_SEEDS = list(range(5))


def collect_file_index(root: str, machine_type: str, db_level: str) -> tuple[list[int], list[dict]]:
    """Collect labels/meta from MIMII directory structure without feature extraction."""
    machine_root = Path(os.path.expanduser(root)) / db_level / machine_type
    if not machine_root.exists():
        raise FileNotFoundError(f"missing MIMII directory: {machine_root}")

    labels: list[int] = []
    meta: list[dict] = []
    for machine_id_dir in sorted(machine_root.glob("id_*")):
        if not machine_id_dir.is_dir():
            continue
        for split, label in (("normal", 0), ("abnormal", 1)):
            for wav_path in sorted((machine_id_dir / split).glob("*.wav")):
                labels.append(label)
                meta.append({
                    "machine_type": machine_type,
                    "machine_id": machine_id_dir.name,
                    "db_level": db_level,
                    "split": split,
                    "wav_path": str(wav_path),
                })
    if not labels:
        raise RuntimeError(f"no WAV files found under {machine_root}")
    return labels, meta


def _normal_dummy_features(labels: list[int]) -> np.ndarray:
    return np.zeros((len(labels), 1, 1), dtype=np.float32)


def _gini(values: list[int]) -> float:
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0 or arr.sum() == 0:
        return 0.0
    diff = np.abs(arr[:, None] - arr[None, :]).sum()
    return float(diff / (2 * len(arr) * arr.sum()))


def partition_audit_rows(
    labels: list[int],
    meta: list[dict],
    machine_type: str,
    db_level: str,
    alpha: float,
    seed: int,
    num_sites: int,
) -> tuple[list[dict], dict]:
    """Return per-site rows and one summary row for a partition setting."""
    labels_arr = np.array(labels, dtype=np.int8)
    site_normal_idx = dl.make_dirichlet_partition(
        features=_normal_dummy_features(labels),
        labels=labels_arr,
        meta=meta,
        num_sites=num_sites,
        alpha=alpha,
        seed=seed,
    )
    splits = [mimii_adapter._split_site(idx, seed) for idx in site_normal_idx]
    site_mids = [set(meta[i]["machine_id"] for i in idx) for idx in site_normal_idx]
    anomaly_idx = np.where(labels_arr == 1)[0]
    site_anom_idx = mimii_adapter._assign_anomalies(meta, anomaly_idx, site_mids, seed)

    rows = []
    for site, idx in enumerate(site_normal_idx):
        train_idx, test_idx = splits[site]
        anomaly_test = site_anom_idx[site]
        mids = sorted(site_mids[site])
        rows.append({
            "machine_type": machine_type,
            "db_level": db_level,
            "alpha": alpha,
            "seed": seed,
            "site": site,
            "machine_ids": ";".join(mids),
            "normal_total_count": int(len(idx)),
            "normal_train_count": int(len(train_idx)),
            "normal_test_count": int(len(test_idx)),
            "anomaly_test_count": int(len(anomaly_test)),
            "empty_train": int(len(train_idx) == 0),
            "empty_normal_test": int(len(test_idx) == 0),
            "empty_anomaly_test": int(len(anomaly_test) == 0),
        })

    train_counts = [int(r["normal_train_count"]) for r in rows]
    anom_counts = [int(r["anomaly_test_count"]) for r in rows]
    summary = {
        "machine_type": machine_type,
        "db_level": db_level,
        "alpha": alpha,
        "seed": seed,
        "num_sites": num_sites,
        "total_normal": int((labels_arr == 0).sum()),
        "total_anomaly": int((labels_arr == 1).sum()),
        "assigned_anomaly": int(sum(anom_counts)),
        "unassigned_anomaly": int((labels_arr == 1).sum() - sum(anom_counts)),
        "empty_train_sites": int(sum(r["empty_train"] for r in rows)),
        "empty_anomaly_sites": int(sum(r["empty_anomaly_test"] for r in rows)),
        "min_train_count": int(min(train_counts) if train_counts else 0),
        "max_train_count": int(max(train_counts) if train_counts else 0),
        "train_count_gini": _gini(train_counts),
        "anomaly_count_gini": _gini(anom_counts),
    }
    return rows, summary


def run_audit(
    data_root: str,
    out_dir: str,
    machine_type: str = "valve",
    db_levels: list[str] | None = None,
    alphas: list[float] | None = None,
    seeds: list[int] | None = None,
    num_sites: int = 10,
) -> tuple[list[dict], list[dict]]:
    site_rows: list[dict] = []
    summary_rows: list[dict] = []
    for db_level in db_levels or DEFAULT_DB_LEVELS:
        labels, meta = collect_file_index(data_root, machine_type, db_level)
        for alpha in alphas or DEFAULT_ALPHAS:
            for seed in seeds or DEFAULT_SEEDS:
                rows, summary = partition_audit_rows(
                    labels, meta, machine_type, db_level, alpha, seed, num_sites
                )
                site_rows.extend(rows)
                summary_rows.append(summary)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "p1_valve_partition_site_counts.csv", site_rows)
    _write_csv(out / "p1_valve_partition_summary.csv", summary_rows)
    _write_report(out / "p1_valve_partition_audit.md", summary_rows)
    return site_rows, summary_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, summary_rows: list[dict]) -> None:
    total = len(summary_rows)
    worst_empty_train = max((int(r["empty_train_sites"]) for r in summary_rows), default=0)
    worst_empty_anom = max((int(r["empty_anomaly_sites"]) for r in summary_rows), default=0)
    max_unassigned = max((int(r["unassigned_anomaly"]) for r in summary_rows), default=0)
    max_train_gini = max((float(r["train_count_gini"]) for r in summary_rows), default=0.0)
    max_anom_gini = max((float(r["anomaly_count_gini"]) for r in summary_rows), default=0.0)
    lines = [
        "# P1 Valve Partition Audit",
        "",
        f"Partition settings inspected: {total}",
        f"Maximum empty-train sites in one setting: {worst_empty_train}",
        f"Maximum empty-anomaly sites in one setting: {worst_empty_anom}",
        f"Maximum unassigned anomaly files in one setting: {max_unassigned}",
        f"Maximum train-count Gini: {max_train_gini:.4f}",
        f"Maximum anomaly-count Gini: {max_anom_gini:.4f}",
        "",
        "This audit uses WAV file metadata and the existing P1 Dirichlet partitioner only; it does not use GPU or extract audio features.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_csv_floats(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x]


def _parse_csv_ints(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x]


def main() -> None:
    parser = argparse.ArgumentParser(description="P1 valve partition diagnostic audit")
    parser.add_argument("--data-root", default="~/abada-night/data/mimii")
    parser.add_argument("--out-dir", default="analysis_outputs")
    parser.add_argument("--machine-type", default="valve")
    parser.add_argument("--db-levels", default=",".join(DEFAULT_DB_LEVELS))
    parser.add_argument("--alphas", default=",".join(str(a) for a in DEFAULT_ALPHAS))
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--num-sites", type=int, default=10)
    args = parser.parse_args()
    _, summary = run_audit(
        data_root=args.data_root,
        out_dir=args.out_dir,
        machine_type=args.machine_type,
        db_levels=[x for x in args.db_levels.split(",") if x],
        alphas=_parse_csv_floats(args.alphas),
        seeds=_parse_csv_ints(args.seeds),
        num_sites=args.num_sites,
    )
    print(f"partition_settings={len(summary)}")
    print(f"wrote={args.out_dir}")


if __name__ == "__main__":
    main()
