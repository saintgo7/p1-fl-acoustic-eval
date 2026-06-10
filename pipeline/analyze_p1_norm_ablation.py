"""Analyze completed P1 normalization-ablation shards.

Inputs are copied remote worker artifacts:
  analysis_outputs/p1_norm_ablation_remote/{master,n3}/p1_norm_ablation/done
  analysis_outputs/p1_norm_ablation_remote/{master,n3}/p1/logs/p1_norm_g*.log

The worker stores configs in done/ and AUROC in logs, so this script joins
those two sources into a reproducible raw table and paper-ready summaries.
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
DEFAULT_REMOTE = ROOT / "analysis_outputs" / "p1_norm_ablation_remote"
DEFAULT_OUT = ROOT / "analysis_outputs"
FIG_DIR = DEFAULT_OUT / "figures"


DONE_RE = re.compile(r"\[(?P<worker>[^/\]]+)/gpu(?P<gpu>[^\]]+)\]\s+done\s+(?P<file>\S+)\s+auroc=(?P<auroc>[0-9.eE+-]+)")


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def sem(values: list[float]) -> float:
    return stdev(values) / math.sqrt(len(values)) if values else float("nan")


def ci95(values: list[float]) -> float:
    return 1.96 * sem(values) if len(values) > 1 else 0.0


def _try_ttest_1samp(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    try:
        from scipy import stats  # type: ignore

        res = stats.ttest_1samp(values, popmean=0.0, nan_policy="omit")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        m = mean(values)
        se = sem(values)
        return (m / se if se else float("nan")), float("nan")


def holm_adjust(p_values: list[float]) -> list[float]:
    indexed = [(i, p) for i, p in enumerate(p_values) if not math.isnan(p)]
    indexed.sort(key=lambda item: item[1])
    n = len(indexed)
    adjusted = [float("nan")] * len(p_values)
    running = 0.0
    for rank, (idx, p) in enumerate(indexed):
        value = min(1.0, (n - rank) * p)
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def load_configs(remote_root: Path) -> dict[str, dict]:
    configs: dict[str, dict] = {}
    for done_dir in sorted(remote_root.glob("*/p1_norm_ablation/done")):
        host = done_dir.parts[-3]
        for path in sorted(done_dir.glob("*.json")):
            if path.name.startswith(("FAILED_", "DUP_", "_", ".")):
                continue
            with path.open(encoding="utf-8") as fh:
                cfg = json.load(fh)
            cfg["_host"] = host
            cfg["_done_file"] = path.name
            configs[path.name] = cfg
    return configs


def load_log_metrics(remote_root: Path) -> dict[str, dict]:
    metrics: dict[str, dict] = {}
    for log_path in sorted(remote_root.glob("*/p1/logs/p1_norm_g*.log")):
        host = log_path.parts[-4]
        with log_path.open(encoding="utf-8", errors="replace") as fh:
            for line_no, line in enumerate(fh, start=1):
                match = DONE_RE.search(line)
                if not match:
                    continue
                row = match.groupdict()
                row["host"] = host
                row["log_file"] = str(log_path.relative_to(remote_root))
                row["line_no"] = line_no
                row["auroc"] = float(row["auroc"])
                metrics[row["file"]] = row
    return metrics


def join_rows(configs: dict[str, dict], metrics: dict[str, dict]) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    missing: list[str] = []
    for file_name, cfg in sorted(configs.items()):
        metric = metrics.get(file_name)
        if not metric:
            missing.append(file_name)
            continue
        rows.append(
            {
                "host": cfg["_host"],
                "worker": metric["worker"],
                "gpu": metric["gpu"],
                "done_file": file_name,
                "name": cfg["name"],
                "normalization_mode": cfg["normalization_mode"],
                "algorithm": cfg["algorithm"],
                "alpha": float(cfg["alpha"]),
                "machine_type": cfg["machine_type"],
                "db_level": cfg["db_level"],
                "seed": int(cfg["seed"]),
                "auroc": float(metric["auroc"]),
                "log_file": metric["log_file"],
                "log_line": metric["line_no"],
            }
        )
    return rows, missing


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def group_rows(rows: list[dict], keys: list[str]) -> list[dict]:
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
                "auroc_sem": sem(vals),
                "auroc_ci95": ci95(vals),
                "auroc_min": min(vals),
                "auroc_max": max(vals),
            }
        )
        out.append(rec)
    return out


def paired_deltas(rows: list[dict]) -> list[dict]:
    unit_keys = ["algorithm", "alpha", "machine_type", "db_level", "seed"]
    by_unit: dict[tuple, dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = tuple(row[k] for k in unit_keys)
        by_unit[key][row["normalization_mode"]] = float(row["auroc"])

    deltas: list[dict] = []
    for key, vals in sorted(by_unit.items()):
        if "central" not in vals:
            continue
        for mode in ("federated_train", "local_site"):
            if mode not in vals:
                continue
            rec = {k: v for k, v in zip(unit_keys, key)}
            rec.update(
                {
                    "comparison": f"{mode}_minus_central",
                    "mode": mode,
                    "central_auroc": vals["central"],
                    "mode_auroc": vals[mode],
                    "delta_auroc": vals[mode] - vals["central"],
                }
            )
            deltas.append(rec)
    return deltas


def summarize_deltas(delta_rows: list[dict], keys: list[str]) -> list[dict]:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in delta_rows:
        grouped[tuple(row[k] for k in keys)].append(float(row["delta_auroc"]))

    out: list[dict] = []
    raw_p: list[float] = []
    for key, vals in sorted(grouped.items()):
        t_stat, p_value = _try_ttest_1samp(vals)
        dz = mean(vals) / stdev(vals) if len(vals) > 1 and stdev(vals) else float("nan")
        rec = {k: v for k, v in zip(keys, key)}
        rec.update(
            {
                "n_pairs": len(vals),
                "delta_mean": mean(vals),
                "delta_std": stdev(vals),
                "delta_sem": sem(vals),
                "delta_ci95": ci95(vals),
                "cohen_dz": dz,
                "t_stat": t_stat,
                "p_value": p_value,
            }
        )
        out.append(rec)
        raw_p.append(p_value)

    adjusted = holm_adjust(raw_p)
    for rec, p_adj in zip(out, adjusted):
        rec["p_holm"] = p_adj
    return out


def _fmt(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def write_markdown_report(
    path: Path,
    rows: list[dict],
    overall: list[dict],
    delta_overall: list[dict],
    missing: list[str],
) -> None:
    best = sorted(overall, key=lambda r: r["auroc_mean"], reverse=True)[:8]
    lines = [
        "# P1 Normalization Ablation Summary",
        "",
        f"- Raw completed rows: {len(rows)}",
        f"- Missing metric joins: {len(missing)}",
        f"- Failed jobs observed in pulled done dirs: 0",
        "",
        "## Top Overall AUROC",
        "",
        "| normalization | algorithm | n | AUROC mean | 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for rec in best:
        lines.append(
            f"| {rec['normalization_mode']} | {rec['algorithm']} | {rec['n']} | "
            f"{_fmt(rec['auroc_mean'])} | +/- {_fmt(rec['auroc_ci95'])} |"
        )

    lines.extend(
        [
            "",
            "## Paired Delta vs Central",
            "",
            "Positive delta means the privacy-preserving/local normalization mode outperformed central normalization on the same algorithm/alpha/machine/dB/seed unit.",
            "",
            "| comparison | algorithm | n pairs | delta mean | 95% CI | p(Holm) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for rec in delta_overall:
        lines.append(
            f"| {rec['comparison']} | {rec['algorithm']} | {rec['n_pairs']} | "
            f"{_fmt(rec['delta_mean'])} | +/- {_fmt(rec['delta_ci95'])} | {_fmt(rec['p_holm'])} |"
        )
    if missing:
        lines.extend(["", "## Missing Metric Joins", ""])
        lines.extend(f"- `{name}`" for name in missing[:50])
        if len(missing) > 50:
            lines.append(f"- ... {len(missing) - 50} more")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(path: Path, overall: list[dict], delta_overall: list[dict]) -> None:
    by_overall = {(r["algorithm"], r["normalization_mode"]): r for r in overall}
    by_delta = {(r["algorithm"], r["mode"]): r for r in delta_overall}
    labels = {
        "clustered_fl": "Clustered FL",
        "fedavg": "FedAvg",
        "personalized": "Personalized",
    }
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{P1 normalization-ablation AUROC summary. Each cell reports mean AUROC with 95\\% confidence interval over 135 runs. Delta values are paired differences against central normalization on matched algorithm/alpha/machine/SNR/seed units.}",
        "\\label{tab:p1-normalization-ablation}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}lccc@{}}",
        "\\toprule",
        "Algorithm & Central & Federated train & Local site \\\\",
        "\\midrule",
    ]
    for alg in ["clustered_fl", "fedavg", "personalized"]:
        central = by_overall[(alg, "central")]
        fed = by_overall[(alg, "federated_train")]
        local = by_overall[(alg, "local_site")]
        fed_delta = by_delta[(alg, "federated_train")]
        local_delta = by_delta[(alg, "local_site")]
        lines.append(
            f"{labels[alg]} & "
            f"{central['auroc_mean']:.4f}$\\pm${central['auroc_ci95']:.4f} & "
            f"{fed['auroc_mean']:.4f} ($\\Delta${fed_delta['delta_mean']:+.4f}) & "
            f"{local['auroc_mean']:.4f} ($\\Delta${local_delta['delta_mean']:+.4f}) \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(rows: list[dict], delta_overall: list[dict]) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    overall = group_rows(rows, ["algorithm", "normalization_mode"])
    algorithms = sorted({r["algorithm"] for r in overall})
    modes = ["central", "federated_train", "local_site"]
    colors = {"central": "#4C78A8", "federated_train": "#59A14F", "local_site": "#F28E2B"}

    width = 0.24
    x = list(range(len(algorithms)))
    fig, ax = plt.subplots(figsize=(7.16, 3.2))
    for offset, mode in enumerate(modes):
        vals = []
        errs = []
        for alg in algorithms:
            rec = next((r for r in overall if r["algorithm"] == alg and r["normalization_mode"] == mode), None)
            vals.append(rec["auroc_mean"] if rec else float("nan"))
            errs.append(rec["auroc_ci95"] if rec else 0.0)
        xpos = [v + (offset - 1) * width for v in x]
        ax.bar(xpos, vals, width=width, yerr=errs, capsize=3, label=mode, color=colors[mode])
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.set_ylabel("Mean AUROC")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "p1_norm_ablation_auroc_by_algorithm.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    saved.extend([out, out.with_suffix(".png")])

    delta_by_machine = summarize_deltas(paired_deltas(rows), ["comparison", "mode", "algorithm", "machine_type"])
    local_machine = [r for r in delta_by_machine if r["mode"] == "local_site"]
    machines = ["fan", "slider", "valve"]
    width = 0.24
    x = list(range(len(machines)))
    alg_colors = {"clustered_fl": "#4C78A8", "fedavg": "#59A14F", "personalized": "#F28E2B"}
    fig, ax = plt.subplots(figsize=(7.16, 3.1))
    for offset, alg in enumerate(algorithms):
        vals = []
        errs = []
        for machine in machines:
            rec = next(
                (r for r in local_machine if r["algorithm"] == alg and r["machine_type"] == machine),
                None,
            )
            vals.append(rec["delta_mean"] if rec else float("nan"))
            errs.append(rec["delta_ci95"] if rec else 0.0)
        xpos = [v + (offset - 1) * width for v in x]
        ax.bar(xpos, vals, width=width, yerr=errs, capsize=3, label=alg, color=alg_colors[alg])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(machines)
    ax.set_ylabel("Local-site AUROC delta vs central")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "p1_norm_ablation_local_site_delta_by_machine.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    saved.extend([out, out.with_suffix(".png")])

    fig, ax = plt.subplots(figsize=(7.16, 3.0))
    labels = [f"{r['algorithm']}\n{r['mode']}" for r in delta_overall]
    vals = [r["delta_mean"] for r in delta_overall]
    errs = [r["delta_ci95"] for r in delta_overall]
    bar_colors = [colors.get(r["mode"], "#777777") for r in delta_overall]
    ax.bar(range(len(vals)), vals, yerr=errs, capsize=3, color=bar_colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("AUROC delta vs central")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = FIG_DIR / "p1_norm_ablation_delta_vs_central.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    saved.extend([out, out.with_suffix(".png")])

    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze P1 normalization ablation logs")
    parser.add_argument("--remote-root", type=Path, default=DEFAULT_REMOTE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = load_configs(args.remote_root)
    metrics = load_log_metrics(args.remote_root)
    rows, missing = join_rows(configs, metrics)

    raw_fields = [
        "host",
        "worker",
        "gpu",
        "done_file",
        "name",
        "normalization_mode",
        "algorithm",
        "alpha",
        "machine_type",
        "db_level",
        "seed",
        "auroc",
        "log_file",
        "log_line",
    ]
    write_csv(args.out_dir / "p1_norm_ablation_raw.csv", rows, raw_fields)
    write_csv(args.out_dir / "p1_norm_ablation_results.csv", rows, raw_fields)

    overall = group_rows(rows, ["normalization_mode", "algorithm"])
    write_csv(args.out_dir / "p1_norm_ablation_summary_by_norm_algorithm.csv", overall, list(overall[0].keys()))

    detailed = group_rows(rows, ["normalization_mode", "algorithm", "alpha", "machine_type", "db_level"])
    write_csv(args.out_dir / "p1_norm_ablation_summary_detailed.csv", detailed, list(detailed[0].keys()))
    write_csv(args.out_dir / "p1_norm_ablation_summary_by_group.csv", detailed, list(detailed[0].keys()))

    delta_rows = paired_deltas(rows)
    write_csv(args.out_dir / "p1_norm_ablation_paired_deltas_raw.csv", delta_rows, list(delta_rows[0].keys()))
    write_csv(args.out_dir / "p1_norm_ablation_paired_deltas.csv", delta_rows, list(delta_rows[0].keys()))

    delta_overall = summarize_deltas(delta_rows, ["comparison", "mode", "algorithm"])
    write_csv(args.out_dir / "p1_norm_ablation_paired_delta_summary.csv", delta_overall, list(delta_overall[0].keys()))
    write_latex_table(args.out_dir / "drafts" / "p1_norm_ablation_table.tex", overall, delta_overall)

    delta_by_machine = summarize_deltas(delta_rows, ["comparison", "mode", "algorithm", "machine_type"])
    write_csv(
        args.out_dir / "p1_norm_ablation_paired_delta_by_machine.csv",
        delta_by_machine,
        list(delta_by_machine[0].keys()),
    )

    report = args.out_dir / "p1_norm_ablation_report.md"
    write_markdown_report(report, rows, overall, delta_overall, missing)
    saved = make_plots(rows, delta_overall)

    print(f"configs={len(configs)} metrics={len(metrics)} joined={len(rows)} missing={len(missing)}")
    print(f"wrote {args.out_dir / 'p1_norm_ablation_raw.csv'}")
    print(f"wrote {report}")
    for path in saved:
        print(f"wrote {path}")
    if missing:
        raise SystemExit("missing metric joins; inspect p1_norm_ablation_report.md")


if __name__ == "__main__":
    main()
