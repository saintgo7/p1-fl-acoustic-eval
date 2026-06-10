"""Analyze P1 E2 centralized/local-only reviewer-risk anchors.

The E2 worker run stores completed config JSON files in ``done/`` and writes
AUROC to worker logs. This script joins the generated job-grid configs with
those log lines, then compares the new non-federated anchors against the
existing P1 main federated sweep on matched machine/SNR/seed units where
possible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRIDS = [
    ROOT / "analysis_outputs" / "job_grids" / "p1_reviewer_risk_20260607_e2_shard0",
    ROOT / "analysis_outputs" / "job_grids" / "p1_reviewer_risk_20260607_e2_shard1",
]
DEFAULT_REMOTE = ROOT / "analysis_outputs" / "p1_reviewer_risk_e2_remote"
DEFAULT_MAIN = ROOT / "analysis_outputs" / "p1_results_aggregate.csv"
DEFAULT_OUT = ROOT / "analysis_outputs" / "p1_e2_baseline_anchors"

DONE_RE = re.compile(
    r"\[(?P<worker>[^/\]]+)/gpu(?P<gpu>[^\]]+)\]\s+done\s+"
    r"(?P<file>\S*p1_anchor_\S+\.json)\s+auroc=(?P<auroc>[0-9.eE+-]+)"
)
WORKER_PREFIX_RE = re.compile(r"^norm_g\d+_")

ALGORITHM_LABELS = {
    "centralized_pooled": "Centralized pooled",
    "local_only": "Local only",
    "personalized": "Personalized FL",
    "clustered_fl": "Clustered FL",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
}


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * stdev(values) / math.sqrt(len(values))


def fmt(value: float, digits: int = 4) -> str:
    return "nan" if math.isnan(value) else f"{value:.{digits}f}"


def load_job_grid(paths: list[Path]) -> dict[str, dict]:
    configs: dict[str, dict] = {}
    for root in paths:
        for path in sorted(root.glob("*.json")):
            if path.name.startswith("_"):
                continue
            with path.open(encoding="utf-8") as fh:
                cfg = json.load(fh)
            configs[path.name] = cfg
    return configs


def load_remote_done(remote_root: Path) -> set[str]:
    done: set[str] = set()
    if not remote_root.exists():
        return done
    for path in sorted(remote_root.glob("*/done/*.json")):
        if path.name.startswith(("FAILED_", "DUP_", "_", ".")):
            continue
        done.add(canonical_config_name(path.name))
    return done


def canonical_config_name(file_name: str) -> str:
    return WORKER_PREFIX_RE.sub("", Path(file_name).name)


def load_log_metrics(remote_root: Path) -> dict[str, dict]:
    metrics: dict[str, dict] = {}
    if not remote_root.exists():
        return metrics
    for log_path in sorted(remote_root.glob("*/logs/p1_norm_g*.log")):
        host = log_path.parts[-3]
        with log_path.open(encoding="utf-8", errors="replace") as fh:
            for line_no, line in enumerate(fh, start=1):
                match = DONE_RE.search(line)
                if not match:
                    continue
                row = match.groupdict()
                file_name = canonical_config_name(row["file"])
                row["host"] = host
                row["log_file"] = str(log_path.relative_to(remote_root))
                row["line_no"] = line_no
                row["auroc"] = float(row["auroc"])
                metrics[file_name] = row
    return metrics


def alpha_slug(value: object) -> str:
    if value in ("", None):
        return "pooled"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    return ("a" + f"{val:g}".replace(".", "p")).replace("-", "m")


def join_anchor_rows(configs: dict[str, dict], done: set[str], metrics: dict[str, dict]) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    missing: list[str] = []
    for file_name, cfg in sorted(configs.items()):
        metric = metrics.get(file_name)
        if metric is None:
            if file_name in done:
                missing.append(file_name)
            continue
        alpha = "" if cfg["algorithm"] == "centralized_pooled" else float(cfg["alpha"])
        rows.append(
            {
                "host": metric["host"],
                "worker": metric["worker"],
                "gpu": metric["gpu"],
                "config_file": file_name,
                "name": cfg["name"],
                "algorithm": cfg["algorithm"],
                "algorithm_label": ALGORITHM_LABELS.get(cfg["algorithm"], cfg["algorithm"]),
                "alpha": alpha,
                "alpha_slug": alpha_slug(alpha),
                "machine_type": cfg["machine_type"],
                "db_level": cfg["db_level"],
                "seed": int(cfg["seed"]),
                "auroc": float(metric["auroc"]),
                "equivalent_epochs": int(cfg["rounds"]) * int(cfg["local_epochs"]),
                "log_file": metric["log_file"],
                "log_line": metric["line_no"],
            }
        )
    return rows, missing


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_main_rows(path: Path) -> list[dict]:
    rows = []
    for row in read_csv_rows(path):
        rows.append(
            {
                "algorithm": row["algorithm"],
                "algorithm_label": ALGORITHM_LABELS.get(row["algorithm"], row["algorithm"]),
                "alpha": row["alpha"],
                "alpha_slug": row["alpha"],
                "machine_type": row["machine_type"],
                "db_level": row["db_level"],
                "seed": int(row["seed"]),
                "auroc": float(row["auroc"]),
                "source": "main_federated_sweep",
            }
        )
    return rows


def group_summary(rows: list[dict], keys: list[str]) -> list[dict]:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(float(row["auroc"]))
    out: list[dict] = []
    for key, vals in sorted(grouped.items()):
        rec = {k: v for k, v in zip(keys, key)}
        rec.update(
            {
                "n": len(vals),
                "auroc_mean": mean(vals),
                "auroc_std": stdev(vals),
                "auroc_ci95": ci95(vals),
                "auroc_min": min(vals),
                "auroc_max": max(vals),
            }
        )
        out.append(rec)
    return out


def paired_anchor_deltas(anchor_rows: list[dict], main_rows: list[dict]) -> list[dict]:
    main_index: dict[tuple, dict[str, float]] = defaultdict(dict)
    for row in main_rows:
        key = (row["alpha_slug"], row["machine_type"], row["db_level"], row["seed"])
        main_index[key][row["algorithm"]] = float(row["auroc"])

    deltas: list[dict] = []
    for anchor in anchor_rows:
        if anchor["algorithm"] != "local_only":
            continue
        key = (anchor["alpha_slug"], anchor["machine_type"], anchor["db_level"], anchor["seed"])
        for algorithm, main_auroc in sorted(main_index.get(key, {}).items()):
            deltas.append(
                {
                    "anchor": "local_only",
                    "comparator": algorithm,
                    "comparator_label": ALGORITHM_LABELS.get(algorithm, algorithm),
                    "alpha_slug": anchor["alpha_slug"],
                    "machine_type": anchor["machine_type"],
                    "db_level": anchor["db_level"],
                    "seed": anchor["seed"],
                    "anchor_auroc": anchor["auroc"],
                    "comparator_auroc": main_auroc,
                    "delta_anchor_minus_comparator": float(anchor["auroc"]) - main_auroc,
                }
            )
    return deltas


def summarize_deltas(rows: list[dict], keys: list[str]) -> list[dict]:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(float(row["delta_anchor_minus_comparator"]))
    out: list[dict] = []
    for key, vals in sorted(grouped.items()):
        rec = {k: v for k, v in zip(keys, key)}
        rec.update(
            {
                "n_pairs": len(vals),
                "delta_mean": mean(vals),
                "delta_std": stdev(vals),
                "delta_ci95": ci95(vals),
                "delta_min": min(vals),
                "delta_max": max(vals),
            }
        )
        out.append(rec)
    return out


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_anchor_tex(path: Path, rows: list[dict]) -> None:
    order = ["centralized_pooled", "local_only"]
    rows = sorted(rows, key=lambda r: (order.index(r["algorithm"]) if r["algorithm"] in order else 9, str(r["alpha_slug"])))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Reviewer-risk E2 non-federated training anchors. Centralized pooled trains one model on pooled normal data; local only trains one model per site. Both use the same 60-equivalent-epoch update budget as the main federated sweep.}",
        r"\label{tab:p1-e2-training-anchors}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Anchor & $\alpha$ & $n$ & Mean & 95\% CI & Min--max \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['algorithm_label']} & {row['alpha_slug']} & {row['n']} & "
            f"{fmt(row['auroc_mean'])} & $\\pm${fmt(row['auroc_ci95'])} & "
            f"{fmt(row['auroc_min'])}--{fmt(row['auroc_max'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_delta_tex(path: Path, rows: list[dict]) -> None:
    rows = [row for row in rows if row["alpha_slug"] in {"a0p05", "a100"}]
    rows = sorted(rows, key=lambda r: (r["alpha_slug"], r["comparator"]))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Paired local-only anchor deltas against federated algorithms on matched machine-type, SNR, Dirichlet-alpha, and seed units. Positive values mean local-only training exceeded the federated comparator.}",
        r"\label{tab:p1-e2-local-anchor-deltas}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}llrrr@{}}",
        r"\toprule",
        r"$\alpha$ & Comparator & $n$ & Mean $\Delta$ & 95\% CI \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['alpha_slug']} & {row['comparator_label']} & {row['n_pairs']} & "
            f"{fmt(row['delta_mean'])} & $\\pm${fmt(row['delta_ci95'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(path: Path, anchor_rows: list[dict], missing_metrics: list[str], summary: list[dict], deltas: list[dict]) -> None:
    lines = [
        "# P1 E2 Baseline Anchor Analysis",
        "",
        f"- Completed anchor rows with AUROC: {len(anchor_rows)}",
        f"- Completed done files missing log AUROC: {len(missing_metrics)}",
        "",
        "## Anchor Overall Summary",
        "",
        "| anchor | alpha | n | mean AUROC | 95% CI | min | max |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['algorithm_label']} | {row['alpha_slug']} | {row['n']} | "
            f"{fmt(row['auroc_mean'])} | +/- {fmt(row['auroc_ci95'])} | "
            f"{fmt(row['auroc_min'])} | {fmt(row['auroc_max'])} |"
        )
    lines.extend(
        [
            "",
            "## Local-Only Paired Delta Summary",
            "",
            "| alpha | comparator | n | mean delta | 95% CI | min | max |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in deltas:
        lines.append(
            f"| {row['alpha_slug']} | {row['comparator_label']} | {row['n_pairs']} | "
            f"{fmt(row['delta_mean'])} | +/- {fmt(row['delta_ci95'])} | "
            f"{fmt(row['delta_min'])} | {fmt(row['delta_max'])} |"
        )
    if missing_metrics:
        lines.extend(["", "## Missing Log Metrics", ""])
        lines.extend(f"- `{name}`" for name in missing_metrics[:50])
        if len(missing_metrics) > 50:
            lines.append(f"- ... {len(missing_metrics) - 50} more")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", action="append", type=Path, default=None, help="Job-grid directory. May be repeated.")
    parser.add_argument("--remote-root", type=Path, default=DEFAULT_REMOTE)
    parser.add_argument("--main-results", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--require-complete", action="store_true", help="Fail unless all 360 E2 jobs have AUROC metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid_paths = args.grid if args.grid else DEFAULT_GRIDS
    configs = load_job_grid(grid_paths)
    done = load_remote_done(args.remote_root)
    metrics = load_log_metrics(args.remote_root)
    anchor_rows, missing_metrics = join_anchor_rows(configs, done, metrics)
    if args.require_complete and len(anchor_rows) != len(configs):
        raise SystemExit(f"Expected {len(configs)} complete E2 rows, found {len(anchor_rows)}.")

    main_rows = load_main_rows(args.main_results)
    anchor_summary = group_summary(anchor_rows, ["algorithm", "algorithm_label", "alpha_slug"])
    anchor_by_condition = group_summary(anchor_rows, ["algorithm", "algorithm_label", "alpha_slug", "machine_type", "db_level"])
    delta_rows = paired_anchor_deltas(anchor_rows, main_rows)
    delta_summary = summarize_deltas(delta_rows, ["alpha_slug", "comparator", "comparator_label"])
    delta_by_condition = summarize_deltas(delta_rows, ["alpha_slug", "comparator", "comparator_label", "machine_type", "db_level"])

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "p1_e2_anchor_raw.csv", anchor_rows)
    write_csv(args.out / "p1_e2_anchor_summary.csv", anchor_summary)
    write_csv(args.out / "p1_e2_anchor_by_condition.csv", anchor_by_condition)
    write_csv(args.out / "p1_e2_local_vs_federated_paired_deltas.csv", delta_rows)
    write_csv(args.out / "p1_e2_local_vs_federated_delta_summary.csv", delta_summary)
    write_csv(args.out / "p1_e2_local_vs_federated_delta_by_condition.csv", delta_by_condition)
    write_anchor_tex(args.out / "p1_e2_anchor_summary.tex", anchor_summary)
    write_delta_tex(args.out / "p1_e2_local_vs_federated_delta_summary.tex", delta_summary)
    write_report(args.out / "P1_E2_BASELINE_ANCHORS_2026-06-07.md", anchor_rows, missing_metrics, anchor_summary, delta_summary)
    print(f"Wrote {args.out.relative_to(ROOT)} with {len(anchor_rows)} / {len(configs)} E2 rows.")


if __name__ == "__main__":
    main()
