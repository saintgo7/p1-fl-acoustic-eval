"""Analyze P1 E3/E4 reviewer-risk reduction experiments.

E3 tests whether FedProx tuning changes the condition-dominance claim.
E4 tests whether the same claim survives a second compact autoencoder
backbone. The outputs are intentionally manuscript-facing: run tables,
effect-size summaries, decision-rule reports, and two compact figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols


ROOT = Path(__file__).resolve().parents[1]
E3_GRID = ROOT / "analysis_outputs" / "job_grids" / "p1_reviewer_risk_20260607_e3_scoped"
E4_GRID = ROOT / "analysis_outputs" / "job_grids" / "p1_reviewer_risk_20260607_e4_backbone"
E3_REMOTE = ROOT / "analysis_outputs" / "p1_reviewer_risk_e3_remote"
E4_REMOTE = ROOT / "analysis_outputs" / "p1_reviewer_risk_e4_remote"
MAIN_SWEEP = ROOT / "analysis_outputs" / "p1_results_aggregate.csv"
OUT_DIR = ROOT / "analysis_outputs" / "p1_e3_e4_reviewer_risk"
GLOBAL_FIG_DIR = ROOT / "analysis_outputs" / "figures"

DONE_RE = re.compile(
    r"\[(?P<worker>[^/\]]+)/gpu(?P<gpu>[^\]]+)\]\s+done\s+"
    r"(?P<file>\S*\.json)\s+auroc=(?P<auroc>[0-9.eE+-]+)"
)
WORKER_PREFIX_RE = re.compile(r"^norm_g\d+_")

ALGORITHM_LABELS = {
    "clustered_fl": "Clustered FL",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "personalized": "Personalized FL",
}


def canonical_config_name(name: str) -> str:
    return WORKER_PREFIX_RE.sub("", Path(name).name)


def slug_float(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def fmt(value: float, digits: int = 4) -> str:
    if value is None or math.isnan(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def mean_ci(values: Iterable[float]) -> tuple[float, float]:
    vals = list(float(v) for v in values)
    if not vals:
        return float("nan"), float("nan")
    if len(vals) < 2:
        return statistics.mean(vals), 0.0
    ci = stats.t.ppf(0.975, len(vals) - 1) * statistics.stdev(vals) / math.sqrt(len(vals))
    return statistics.mean(vals), float(ci)


def load_grid(path: Path) -> dict[str, dict]:
    configs: dict[str, dict] = {}
    for cfg_path in sorted(path.glob("*.json")):
        if cfg_path.name.startswith("_"):
            continue
        with cfg_path.open(encoding="utf-8") as fh:
            configs[cfg_path.name] = json.load(fh)
    return configs


def load_log_metrics(remote_root: Path) -> dict[str, dict]:
    metrics: dict[str, dict] = {}
    for log_path in sorted(remote_root.glob("*/logs/p1_norm_g*.log")):
        host = log_path.parts[-3]
        with log_path.open(encoding="utf-8", errors="replace") as fh:
            for line_no, line in enumerate(fh, start=1):
                match = DONE_RE.search(line)
                if not match:
                    continue
                row = match.groupdict()
                name = canonical_config_name(row["file"])
                metrics[name] = {
                    "host": host,
                    "worker": row["worker"],
                    "gpu": row["gpu"],
                    "log_file": str(log_path.relative_to(remote_root)),
                    "log_line": line_no,
                    "auroc": float(row["auroc"]),
                }
    return metrics


def join_rows(configs: dict[str, dict], metrics: dict[str, dict], experiment: str) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict] = []
    missing: list[str] = []
    for file_name, cfg in sorted(configs.items()):
        metric = metrics.get(file_name)
        if metric is None:
            missing.append(file_name)
            continue
        rows.append(
            {
                "experiment": experiment,
                "config_file": file_name,
                "name": cfg["name"],
                "algorithm": cfg["algorithm"],
                "algorithm_label": ALGORITHM_LABELS.get(cfg["algorithm"], cfg["algorithm"]),
                "alpha": float(cfg["alpha"]),
                "machine_type": cfg["machine_type"],
                "db_level": cfg["db_level"],
                "seed": int(cfg["seed"]),
                "fedprox_mu": float(cfg.get("fedprox_mu", 0.0)),
                "model_family": cfg.get("model_family", "ConvAutoencoder"),
                "bottleneck": int(cfg["bottleneck"]),
                "auroc": float(metric["auroc"]),
                "host": metric["host"],
                "worker": metric["worker"],
                "gpu": metric["gpu"],
                "log_file": metric["log_file"],
                "log_line": int(metric["log_line"]),
            }
        )
    return pd.DataFrame(rows), missing


def write_csv(path: Path, rows: Iterable[dict] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
        return
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as fh:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def effect_sizes(df: pd.DataFrame, formula: str, label_map: dict[str, str]) -> pd.DataFrame:
    model = ols(formula, data=df).fit()
    table = sm.stats.anova_lm(model, typ=2)
    residual_ss = float(table.loc["Residual", "sum_sq"])
    rows = []
    for term, label in label_map.items():
        ss = float(table.loc[term, "sum_sq"])
        eta = ss / (ss + residual_ss) if ss + residual_ss else float("nan")
        rows.append(
            {
                "factor": label,
                "term": term,
                "df": int(table.loc[term, "df"]),
                "sum_sq": ss,
                "f_stat": float(table.loc[term, "F"]),
                "p_value": float(table.loc[term, "PR(>F)"]),
                "partial_eta2": eta,
                "magnitude": eta_magnitude(eta),
            }
        )
    out = pd.DataFrame(rows).sort_values("partial_eta2", ascending=False)
    return out.reset_index(drop=True)


def eta_magnitude(eta: float) -> str:
    if math.isnan(float(eta)):
        return "nan"
    if eta < 0.06:
        return "small"
    if eta < 0.14:
        return "medium"
    return "large"


def grouped_summary(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in df.groupby(keys, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        mean, ci = mean_ci(group["auroc"])
        row = {name: value for name, value in zip(keys, key)}
        row.update(
            {
                "n": int(len(group)),
                "auroc_mean": mean,
                "auroc_ci95": ci,
                "auroc_std": float(group["auroc"].std(ddof=1)) if len(group) > 1 else 0.0,
                "auroc_min": float(group["auroc"].min()),
                "auroc_max": float(group["auroc"].max()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def load_main_subset() -> pd.DataFrame:
    df = pd.read_csv(MAIN_SWEEP)
    return df[
        df["machine_type"].isin(["valve", "slider"])
        & df["db_level"].isin(["-6dB", "6dB"])
        & df["alpha"].isin(["a0p05", "a100"])
        & df["seed"].between(0, 4)
    ].copy()


def normalize_main_subset(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    alpha_map = {"a0p05": 0.05, "a100": 100.0}
    work["alpha"] = work["alpha"].map(alpha_map).astype(float)
    work["algorithm_label"] = work["algorithm"].map(ALGORITHM_LABELS).fillna(work["algorithm"])
    work["model_family"] = "ConvAutoencoder"
    work["bottleneck"] = 1024
    return work


def plot_e3_mu(e3: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = grouped_summary(e3, ["fedprox_mu", "machine_type", "db_level"])
    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    styles = {"-6dB": "--", "6dB": "-"}
    for (machine, db), group in summary.groupby(["machine_type", "db_level"]):
        group = group.sort_values("fedprox_mu")
        ax.errorbar(
            group["fedprox_mu"],
            group["auroc_mean"],
            yerr=group["auroc_ci95"],
            marker="o",
            linestyle=styles.get(db, "-"),
            linewidth=1.4,
            capsize=2.5,
            label=f"{machine}, {db}",
        )
    ax.set_xscale("symlog", linthresh=0.001)
    ax.set_xlabel("FedProx $\\mu$")
    ax.set_ylabel("AUROC")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    GLOBAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, GLOBAL_FIG_DIR / path.name)
    plt.close(fig)


def plot_e4_scatter(main: pd.DataFrame, e4: pd.DataFrame, path: Path) -> tuple[float, float]:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["algorithm", "alpha", "machine_type", "db_level"]
    x = grouped_summary(main, keys).rename(columns={"auroc_mean": "dense_auroc"})
    y = grouped_summary(e4, keys).rename(columns={"auroc_mean": "lite_auroc"})
    merged = x[keys + ["dense_auroc"]].merge(y[keys + ["lite_auroc"]], on=keys, how="inner")
    rho, p_value = stats.spearmanr(merged["dense_auroc"], merged["lite_auroc"])
    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    colors = {"valve": "#1f77b4", "slider": "#d62728"}
    markers = {"-6dB": "s", "6dB": "o"}
    for _, row in merged.iterrows():
        ax.scatter(
            row["dense_auroc"],
            row["lite_auroc"],
            color=colors.get(row["machine_type"], "#333333"),
            marker=markers.get(row["db_level"], "o"),
            s=42,
            alpha=0.85,
        )
    lo = min(float(merged["dense_auroc"].min()), float(merged["lite_auroc"].min()))
    hi = max(float(merged["dense_auroc"].max()), float(merged["lite_auroc"].max()))
    pad = 0.02
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#555555", linewidth=1, alpha=0.6)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Original ConvAE AUROC")
    ax.set_ylabel("LiteConvAE AUROC")
    ax.set_title(f"Condition-rank stability ($\\rho$={rho:.2f})")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    GLOBAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, GLOBAL_FIG_DIR / path.name)
    plt.close(fig)
    return float(rho), float(p_value)


def write_tex_tables(
    e3_mu: pd.DataFrame,
    e3_effects: pd.DataFrame,
    e3_best_mu_effects: pd.DataFrame,
    e4_effects: pd.DataFrame,
    e4_scatter: dict,
) -> None:
    tex = OUT_DIR / "tex"
    tex.mkdir(parents=True, exist_ok=True)
    with (tex / "p1_e3_mu_summary.tex").open("w", encoding="utf-8") as fh:
        fh.write("% Auto-generated by analyze_p1_e3_e4_reviewer_risk.py\n")
        fh.write("\\begin{tabular}{l l r r r}\n\\toprule\n")
        fh.write("Machine & SNR & $\\mu$ & AUROC & 95\\% CI \\\\\n\\midrule\n")
        rows = e3_mu.sort_values(["machine_type", "db_level", "fedprox_mu"])
        for _, r in rows.iterrows():
            fh.write(
                f"{r['machine_type']} & {r['db_level']} & {r['fedprox_mu']:g} & "
                f"{r['auroc_mean']:.3f} & $\\pm$ {r['auroc_ci95']:.3f} \\\\\n"
            )
        fh.write("\\bottomrule\n\\end{tabular}\n")

    with (tex / "p1_e3_e4_effect_sizes.tex").open("w", encoding="utf-8") as fh:
        fh.write("% Auto-generated by analyze_p1_e3_e4_reviewer_risk.py\n")
        fh.write("\\begin{table}[t]\n\\centering\n")
        fh.write("\\caption{Scoped E3/E4 sensitivity effect-size summary.}\n")
        fh.write("\\label{tab:p1-e3-e4-effect-sizes}\n")
        fh.write("\\scriptsize\n")
        fh.write("\\setlength{\\tabcolsep}{3pt}\n")
        fh.write("\\resizebox{\\linewidth}{!}{%\n")
        fh.write("\\begin{tabular}{l l r r l}\n\\toprule\n")
        fh.write("Experiment & Factor & df & partial $\\eta^2$ & magnitude \\\\\n\\midrule\n")
        for label, frame in [
            ("E3 FedProx-$\\mu$", e3_effects),
            ("E3 best-$\\mu$", e3_best_mu_effects),
            ("E4 LiteConvAE", e4_effects),
        ]:
            for _, r in frame.iterrows():
                fh.write(
                    f"{label} & {r['factor']} & {int(r['df'])} & "
                    f"{r['partial_eta2']:.3f} & {r['magnitude']} \\\\\n"
                )
        fh.write("\\midrule\n")
        fh.write(
            f"E4 rank check & Spearman $\\rho$ & -- & {e4_scatter['spearman_rho']:.3f} & "
            f"p={e4_scatter['spearman_p']:.3g} \\\\\n"
        )
        fh.write("\\bottomrule\n\\end{tabular}%\n}\n")
        fh.write("\\end{table}\n")


def write_report(
    e3: pd.DataFrame,
    e4: pd.DataFrame,
    e3_missing: list[str],
    e4_missing: list[str],
    e3_effects: pd.DataFrame,
    e3_best_mu_effects: pd.DataFrame,
    e4_effects: pd.DataFrame,
    e3_mu_summary: pd.DataFrame,
    main_effects: pd.DataFrame,
    e4_scatter: dict,
) -> None:
    mu_eta = float(e3_effects.loc[e3_effects["factor"] == "FedProx mu", "partial_eta2"].iloc[0])
    machine_eta = float(e3_effects.loc[e3_effects["factor"] == "Machine type", "partial_eta2"].iloc[0])
    snr_eta = float(e3_effects.loc[e3_effects["factor"] == "SNR", "partial_eta2"].iloc[0])
    best_mu = (
        e3_mu_summary.groupby("fedprox_mu")["auroc_mean"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .iloc[0]
    )
    default_mu_mean = float(
        e3_mu_summary[e3_mu_summary["fedprox_mu"] == 0.01].groupby("fedprox_mu")["auroc_mean"].mean().iloc[0]
    )
    best_mu_effects_rank = ", ".join(
        f"{r.factor}={float(r.partial_eta2):.3f}"
        for r in e3_best_mu_effects.sort_values("partial_eta2", ascending=False).itertuples()
    )
    top_e4 = e4_effects.iloc[0]
    condition_eta_e4 = max(
        float(e4_effects.loc[e4_effects["factor"] == "Machine type", "partial_eta2"].iloc[0]),
        float(e4_effects.loc[e4_effects["factor"] == "SNR", "partial_eta2"].iloc[0]),
    )
    algorithm_eta_e4 = float(e4_effects.loc[e4_effects["factor"] == "Algorithm", "partial_eta2"].iloc[0])

    e3_support = mu_eta < 0.06 and max(machine_eta, snr_eta) >= 3.0 * max(mu_eta, 1e-12)
    if e3_support:
        e3_decision = "PASS: FedProx mu is small and condition effects dominate by the preregistered rule."
    elif mu_eta < 0.14:
        e3_decision = "PARTIAL: FedProx mu is medium or condition dominance margin is below 3x; scope wording is required."
    else:
        e3_decision = "FAIL: FedProx mu is a large effect; do not claim optimizer robustness."

    rho = float(e4_scatter["spearman_rho"])
    e4_condition_top = top_e4["factor"] in {"Machine type", "SNR"}
    if rho >= 0.7 and e4_condition_top:
        e4_decision = "PASS: condition ranking is stable and a condition factor remains the top effect."
    elif rho >= 0.5:
        e4_decision = "PARTIAL: condition ranking is moderately stable; use autoencoder-family wording only."
    else:
        e4_decision = "FAIL: backbone sensitivity is too high for the current condition-dominance wording."

    report = OUT_DIR / "p1_e3_e4_reviewer_risk_report.md"
    with report.open("w", encoding="utf-8") as fh:
        fh.write("# P1 E3/E4 Reviewer-Risk Reduction Analysis\n\n")
        fh.write("Generated from completed remote E3/E4 logs after Claude reviewer-mode planning.\n\n")
        fh.write("## Completeness\n\n")
        fh.write(f"- E3 FedProx-mu rows joined: {len(e3)} / 400; missing metrics: {len(e3_missing)}\n")
        fh.write(f"- E4 LiteConvAE rows joined: {len(e4)} / 160; missing metrics: {len(e4_missing)}\n")
        fh.write("\n## E3 Decision\n\n")
        fh.write(f"- {e3_decision}\n")
        fh.write(f"- FedProx mu partial eta^2: {mu_eta:.4f} ({eta_magnitude(mu_eta)})\n")
        fh.write(f"- Machine partial eta^2: {machine_eta:.4f}; SNR partial eta^2: {snr_eta:.4f}\n")
        fh.write(
            f"- Best average mu: {best_mu['fedprox_mu']:g} with mean AUROC "
            f"{best_mu['auroc_mean']:.4f}; default mu=0.01 mean AUROC {default_mu_mean:.4f}\n"
        )
        fh.write(f"- At best mu, additive effect-size rank: {best_mu_effects_rank}\n")
        fh.write("\n## E4 Decision\n\n")
        fh.write(f"- {e4_decision}\n")
        fh.write(
            f"- Dense-vs-Lite per-condition Spearman rho: {rho:.4f} "
            f"(p={e4_scatter['spearman_p']:.4g})\n"
        )
        fh.write(
            f"- E4 strongest factor: {top_e4['factor']} "
            f"(partial eta^2={float(top_e4['partial_eta2']):.4f})\n"
        )
        fh.write(
            f"- E4 condition max eta^2: {condition_eta_e4:.4f}; "
            f"algorithm eta^2: {algorithm_eta_e4:.4f}\n"
        )
        fh.write("\n## Manuscript Action\n\n")
        if e3_support and rho >= 0.7:
            fh.write(
                "- Add scoped wording: condition dominance is preserved at the best FedProx mu "
                "and across a second compact autoencoder backbone in the evaluated MIMII slice.\n"
            )
            fh.write(
                "- Do not claim broad architecture or dataset generality; keep single-corpus "
                "external validity as an explicit limitation.\n"
            )
        else:
            fh.write(
                "- Do not claim FedProx-mu robustness. State instead that the sweep found FedProx "
                "to be mu-sensitive; use the best-mu subset to show the condition effect remains "
                "larger than algorithmic tuning in the tested slice.\n"
            )
        fh.write("\n## Files\n\n")
        fh.write("- `p1_e3_run_metrics.csv`\n")
        fh.write("- `p1_e3_effect_sizes.csv`\n")
        fh.write("- `p1_e3_best_mu_effect_sizes.csv`\n")
        fh.write("- `p1_e3_mu_summary.csv`\n")
        fh.write("- `p1_e4_run_metrics.csv`\n")
        fh.write("- `p1_e4_effect_sizes.csv`\n")
        fh.write("- `p1_e4_dense_vs_lite_condition_scatter.csv`\n")
        fh.write("- `figures/p1_e3_mu_sensitivity.png`\n")
        fh.write("- `figures/p1_e4_dense_vs_lite_condition_scatter.png`\n")
        fh.write("- `tex/p1_e3_mu_summary.tex`\n")
        fh.write("- `tex/p1_e3_e4_effect_sizes.tex`\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    e3, e3_missing = join_rows(load_grid(E3_GRID), load_log_metrics(E3_REMOTE), "E3")
    e4, e4_missing = join_rows(load_grid(E4_GRID), load_log_metrics(E4_REMOTE), "E4")
    if args.require_complete and (len(e3) != 400 or len(e4) != 160 or e3_missing or e4_missing):
        raise SystemExit(
            f"Incomplete E3/E4 logs: E3={len(e3)}/400 missing={len(e3_missing)}, "
            f"E4={len(e4)}/160 missing={len(e4_missing)}"
        )

    main_dense = normalize_main_subset(load_main_subset())
    if len(main_dense) != 160:
        raise SystemExit(f"Expected 160 matched main dense rows, found {len(main_dense)}")

    e3_effects = effect_sizes(
        e3,
        "auroc ~ C(machine_type) + C(db_level) + C(alpha) + C(fedprox_mu) + C(alpha):C(fedprox_mu) + C(machine_type):C(fedprox_mu)",
        {
            "C(machine_type)": "Machine type",
            "C(db_level)": "SNR",
            "C(alpha)": "Dirichlet alpha",
            "C(fedprox_mu)": "FedProx mu",
            "C(alpha):C(fedprox_mu)": "Alpha x mu",
            "C(machine_type):C(fedprox_mu)": "Machine x mu",
        },
    )
    best_mu_value = (
        grouped_summary(e3, ["fedprox_mu"])
        .sort_values("auroc_mean", ascending=False)
        .iloc[0]["fedprox_mu"]
    )
    e3_best_mu = e3[e3["fedprox_mu"] == best_mu_value].copy()
    e3_best_mu_effects = effect_sizes(
        e3_best_mu,
        "auroc ~ C(machine_type) + C(db_level) + C(alpha)",
        {
            "C(machine_type)": "Machine type",
            "C(db_level)": "SNR",
            "C(alpha)": "Dirichlet alpha",
        },
    )
    e4_effects = effect_sizes(
        e4,
        "auroc ~ C(machine_type) + C(db_level) + C(algorithm) + C(alpha)",
        {
            "C(machine_type)": "Machine type",
            "C(db_level)": "SNR",
            "C(algorithm)": "Algorithm",
            "C(alpha)": "Dirichlet alpha",
        },
    )
    main_effects = effect_sizes(
        main_dense,
        "auroc ~ C(machine_type) + C(db_level) + C(algorithm) + C(alpha)",
        {
            "C(machine_type)": "Machine type",
            "C(db_level)": "SNR",
            "C(algorithm)": "Algorithm",
            "C(alpha)": "Dirichlet alpha",
        },
    )

    e3_mu_summary = grouped_summary(e3, ["machine_type", "db_level", "fedprox_mu"])
    e3_by_mu = grouped_summary(e3, ["fedprox_mu"])
    e4_summary = grouped_summary(e4, ["model_family", "algorithm", "alpha", "machine_type", "db_level"])
    main_summary = grouped_summary(main_dense, ["model_family", "algorithm", "alpha", "machine_type", "db_level"])
    scatter_path = OUT_DIR / "figures" / "p1_e4_dense_vs_lite_condition_scatter.png"
    rho, p_value = plot_e4_scatter(main_dense, e4, scatter_path)
    e4_scatter = {"spearman_rho": rho, "spearman_p": p_value}

    keys = ["algorithm", "alpha", "machine_type", "db_level"]
    dense_cond = grouped_summary(main_dense, keys).rename(columns={"auroc_mean": "dense_auroc"})
    lite_cond = grouped_summary(e4, keys).rename(columns={"auroc_mean": "lite_auroc"})
    condition_scatter = dense_cond[keys + ["dense_auroc"]].merge(
        lite_cond[keys + ["lite_auroc"]], on=keys, how="inner"
    )

    plot_e3_mu(e3, OUT_DIR / "figures" / "p1_e3_mu_sensitivity.png")

    write_csv(OUT_DIR / "p1_e3_run_metrics.csv", e3)
    write_csv(OUT_DIR / "p1_e3_effect_sizes.csv", e3_effects)
    write_csv(OUT_DIR / "p1_e3_mu_summary.csv", e3_mu_summary)
    write_csv(OUT_DIR / "p1_e3_by_mu_summary.csv", e3_by_mu)
    write_csv(OUT_DIR / "p1_e3_best_mu_effect_sizes.csv", e3_best_mu_effects)
    write_csv(OUT_DIR / "p1_e4_run_metrics.csv", e4)
    write_csv(OUT_DIR / "p1_e4_effect_sizes.csv", e4_effects)
    write_csv(OUT_DIR / "p1_e4_matched_dense_effect_sizes.csv", main_effects)
    write_csv(OUT_DIR / "p1_e4_lite_summary.csv", e4_summary)
    write_csv(OUT_DIR / "p1_e4_matched_dense_summary.csv", main_summary)
    write_csv(OUT_DIR / "p1_e4_dense_vs_lite_condition_scatter.csv", condition_scatter)

    write_tex_tables(e3_mu_summary, e3_effects, e3_best_mu_effects, e4_effects, e4_scatter)
    write_report(
        e3,
        e4,
        e3_missing,
        e4_missing,
        e3_effects,
        e3_best_mu_effects,
        e4_effects,
        e3_mu_summary,
        main_effects,
        e4_scatter,
    )

    print(f"E3 rows: {len(e3)} missing={len(e3_missing)}")
    print(f"E4 rows: {len(e4)} missing={len(e4_missing)}")
    print(f"E3 effects written to {OUT_DIR / 'p1_e3_effect_sizes.csv'}")
    print(f"E4 effects written to {OUT_DIR / 'p1_e4_effect_sizes.csv'}")
    print(f"Report written to {OUT_DIR / 'p1_e3_e4_reviewer_risk_report.md'}")


if __name__ == "__main__":
    main()
