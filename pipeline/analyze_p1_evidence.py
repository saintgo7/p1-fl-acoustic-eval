"""Analyze P1 evidence-logged normalization-ablation reruns.

The input is the fetched bundle tree:
  analysis_outputs/p1_evidence_remote/{master,n3}/p1_evidence/<run>/*.csv

Outputs are paper-supporting diagnostics for valve difficulty, site-level
dispersion, and Clustered-FL assignment stability.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REMOTE = ROOT / "analysis_outputs" / "p1_evidence_remote"
DEFAULT_OUT = ROOT / "analysis_outputs" / "p1_evidence_analysis"
FIG_DIR = ROOT / "analysis_outputs" / "figures"

RUN_RE = re.compile(
    r"^p1_norm_(?P<normalization>central|federated_train|local_site)_"
    r"(?P<algorithm>clustered_fl|fedavg|personalized)_"
    r"(?P<alpha_slug>a0p05|a0p5|a100)_"
    r"(?P<machine_type>fan|slider|valve)_"
    r"(?P<db_slug>m6dB|0dB|6dB)_s(?P<seed>[0-9]+)$"
)

ALPHA_MAP = {"a0p05": 0.05, "a0p5": 0.5, "a100": 100.0}
DB_MAP = {"m6dB": "-6dB", "0dB": "0dB", "6dB": "6dB"}
MACHINE_ORDER = ["fan", "slider", "valve"]
SNR_ORDER = ["-6dB", "0dB", "6dB"]
NORM_ORDER = ["central", "federated_train", "local_site"]
DISPLAY_LABELS = {
    "machine_type": "Machine",
    "db_level": "SNR",
    "n_runs": "Runs",
    "mean_error_delta": "Delta",
    "mean_cohen_d": "Cohen d",
    "mean_error_ratio": "Ratio",
    "mean_run_auroc": "AUROC",
    "mean_site_dispersion": "Site std.",
    "mean_worst_site_auroc": "Worst site",
    "mean_ari": "ARI",
    "mean_min_ari": "Min ARI",
    "mean_final_ari": "Final ARI",
}


def _run_meta(run_id: str) -> dict[str, object]:
    match = RUN_RE.match(run_id)
    if not match:
        raise ValueError(f"Unexpected P1 run_id format: {run_id}")
    meta = match.groupdict()
    meta["alpha"] = ALPHA_MAP[meta["alpha_slug"]]
    meta["db_level"] = DB_MAP[meta["db_slug"]]
    meta["seed"] = int(meta["seed"])
    return meta


def _read_table(remote_root: Path, table_name: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for host_dir in sorted(remote_root.glob("*")):
        host = host_dir.name
        evidence_root = host_dir / "p1_evidence"
        for path in sorted(evidence_root.glob(f"*/{table_name}.csv")):
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            frame["host"] = host
            frame["bundle_dir"] = path.parent.name
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    meta = pd.DataFrame([_run_meta(str(run_id)) for run_id in out["run_id"]])
    meta.index = out.index
    for col in ["normalization", "algorithm", "alpha", "machine_type", "db_level", "seed"]:
        out[col] = meta[col]
    return out


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_tex_table(path: Path, frame: pd.DataFrame, columns: list[str], caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = frame[columns].copy()
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    table = table.rename(columns={col: DISPLAY_LABELS.get(col, col.replace("_", " ")) for col in table.columns})
    tex = table.to_latex(index=False, escape=True, caption=caption, label=label)
    tex = tex.replace("\\begin{table}\n", "\\begin{table}[t]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n", 1)
    path.write_text(tex, encoding="utf-8")


def _bootstrap_ci(values: pd.Series) -> float:
    vals = values.dropna().to_numpy(dtype=float)
    if len(vals) < 2:
        return 0.0
    return 1.96 * vals.std(ddof=1) / np.sqrt(len(vals))


def _run_site_summary(site: pd.DataFrame) -> pd.DataFrame:
    grouped = site.groupby(
        ["run_id", "normalization", "algorithm", "alpha", "machine_type", "db_level", "seed"],
        dropna=False,
    )
    return grouped.agg(
        n_sites=("site_id", "nunique"),
        mean_site_auroc=("auroc", "mean"),
        std_site_auroc=("auroc", "std"),
        min_site_auroc=("auroc", "min"),
        max_site_auroc=("auroc", "max"),
        mean_site_auprc=("auprc", "mean"),
        mean_site_pauc=("partial_auroc_fpr_0_1", "mean"),
    ).reset_index()


def _site_dispersion_summary(run_site: pd.DataFrame) -> pd.DataFrame:
    grouped = run_site.groupby(["machine_type", "db_level"], dropna=False)
    out = grouped.agg(
        n_runs=("run_id", "nunique"),
        mean_run_auroc=("mean_site_auroc", "mean"),
        mean_site_dispersion=("std_site_auroc", "mean"),
        ci95_site_dispersion=("std_site_auroc", _bootstrap_ci),
        mean_worst_site_auroc=("min_site_auroc", "mean"),
        mean_best_site_auroc=("max_site_auroc", "mean"),
    ).reset_index()
    out["db_level"] = pd.Categorical(out["db_level"], SNR_ORDER, ordered=True)
    out["machine_type"] = pd.Categorical(out["machine_type"], MACHINE_ORDER, ordered=True)
    return out.sort_values(["machine_type", "db_level"]).reset_index(drop=True)


def _reconstruction_run_summary(recon: pd.DataFrame) -> pd.DataFrame:
    grouped = recon.groupby(
        ["run_id", "normalization", "algorithm", "alpha", "machine_type", "db_level", "seed", "label"],
        dropna=False,
    )["reconstruction_error"]
    by_label = grouped.agg(["count", "mean", "median", "std"]).reset_index()
    wide = by_label.pivot_table(
        index=["run_id", "normalization", "algorithm", "alpha", "machine_type", "db_level", "seed"],
        columns="label",
        values=["count", "mean", "median", "std"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{'anomaly' if label == 1 else 'normal'}" for metric, label in wide.columns]
    wide = wide.reset_index()
    wide["mean_error_delta"] = wide["mean_anomaly"] - wide["mean_normal"]
    wide["median_error_delta"] = wide["median_anomaly"] - wide["median_normal"]
    pooled = np.sqrt((wide["std_anomaly"].pow(2) + wide["std_normal"].pow(2)) / 2.0)
    wide["cohen_d_error_separation"] = wide["mean_error_delta"] / pooled.replace(0, np.nan)
    wide["mean_error_ratio"] = wide["mean_anomaly"] / wide["mean_normal"].replace(0, np.nan)
    return wide


def _reconstruction_machine_summary(recon_run: pd.DataFrame) -> pd.DataFrame:
    grouped = recon_run.groupby(["machine_type", "db_level"], dropna=False)
    out = grouped.agg(
        n_runs=("run_id", "nunique"),
        mean_normal_error=("mean_normal", "mean"),
        mean_anomaly_error=("mean_anomaly", "mean"),
        mean_error_delta=("mean_error_delta", "mean"),
        ci95_error_delta=("mean_error_delta", _bootstrap_ci),
        mean_cohen_d=("cohen_d_error_separation", "mean"),
        mean_error_ratio=("mean_error_ratio", "mean"),
    ).reset_index()
    out["db_level"] = pd.Categorical(out["db_level"], SNR_ORDER, ordered=True)
    out["machine_type"] = pd.Categorical(out["machine_type"], MACHINE_ORDER, ordered=True)
    return out.sort_values(["machine_type", "db_level"]).reset_index(drop=True)


def _cluster_summary(stability: pd.DataFrame, assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if stability.empty:
        return pd.DataFrame(), pd.DataFrame()
    ari = stability[
        (stability["site_id"].astype(str) == "all")
        & (stability["stability_metric"].astype(str) == "ari")
    ].copy()
    run_ari = ari.groupby(
        ["run_id", "normalization", "alpha", "machine_type", "db_level", "seed"],
        dropna=False,
    ).agg(
        n_round_comparisons=("stability_value", "count"),
        mean_round_to_round_ari=("stability_value", "mean"),
        min_round_to_round_ari=("stability_value", "min"),
        final_round_to_round_ari=("stability_value", "last"),
    ).reset_index()
    grouped = run_ari.groupby(["machine_type", "db_level"], dropna=False)
    machine = grouped.agg(
        n_runs=("run_id", "nunique"),
        mean_ari=("mean_round_to_round_ari", "mean"),
        ci95_ari=("mean_round_to_round_ari", _bootstrap_ci),
        mean_min_ari=("min_round_to_round_ari", "mean"),
        mean_final_ari=("final_round_to_round_ari", "mean"),
    ).reset_index()

    if not assignments.empty:
        final_round = assignments.groupby("run_id")["round"].transform("max")
        final = assignments[assignments["round"] == final_round].copy()
        cluster_balance = final.groupby(
            ["run_id", "normalization", "alpha", "machine_type", "db_level", "seed"],
            dropna=False,
        ).agg(
            n_sites=("site_id", "nunique"),
            n_clusters=("cluster_id", "nunique"),
            largest_cluster_sites=("cluster_id", lambda s: s.value_counts().max()),
        ).reset_index()
        cluster_balance["largest_cluster_fraction"] = (
            cluster_balance["largest_cluster_sites"] / cluster_balance["n_sites"].replace(0, np.nan)
        )
        run_ari = run_ari.merge(cluster_balance, how="left")

    for frame in (machine, run_ari):
        if not frame.empty:
            frame["db_level"] = pd.Categorical(frame["db_level"], SNR_ORDER, ordered=True)
            frame["machine_type"] = pd.Categorical(frame["machine_type"], MACHINE_ORDER, ordered=True)
    return (
        machine.sort_values(["machine_type", "db_level"]).reset_index(drop=True),
        run_ari.sort_values(["machine_type", "db_level", "normalization", "alpha", "seed"]).reset_index(drop=True),
    )


def _plot_site_dispersion(run_site: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = [run_site.loc[run_site["machine_type"] == m, "std_site_auroc"].dropna() for m in MACHINE_ORDER]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.boxplot(data, tick_labels=MACHINE_ORDER, showfliers=False)
    ax.set_ylabel("Within-run site AUROC std.")
    ax.set_xlabel("Machine type")
    ax.set_title("P1 evidence rerun: site-level AUROC dispersion")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "p1_evidence_site_auroc_dispersion_by_machine.pdf")
    fig.savefig(fig_dir / "p1_evidence_site_auroc_dispersion_by_machine.png", dpi=200)
    plt.close(fig)


def _plot_reconstruction(recon_run: pd.DataFrame, fig_dir: Path) -> None:
    data = [recon_run.loc[recon_run["machine_type"] == m, "cohen_d_error_separation"].dropna() for m in MACHINE_ORDER]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.boxplot(data, tick_labels=MACHINE_ORDER, showfliers=False)
    ax.set_ylabel("Error separation (Cohen's d)")
    ax.set_xlabel("Machine type")
    ax.set_title("P1 evidence rerun: normal/anomaly reconstruction-error separation")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "p1_evidence_reconstruction_separation_by_machine.pdf")
    fig.savefig(fig_dir / "p1_evidence_reconstruction_separation_by_machine.png", dpi=200)
    plt.close(fig)


def _plot_cluster_stability(run_ari: pd.DataFrame, fig_dir: Path) -> None:
    if run_ari.empty:
        return
    data = [run_ari.loc[run_ari["machine_type"] == m, "mean_round_to_round_ari"].dropna() for m in MACHINE_ORDER]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.boxplot(data, tick_labels=MACHINE_ORDER, showfliers=False)
    ax.set_ylabel("Round-to-round cluster ARI")
    ax.set_xlabel("Machine type")
    ax.set_title("P1 evidence rerun: Clustered-FL assignment stability")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "p1_evidence_cluster_stability_by_machine.pdf")
    fig.savefig(fig_dir / "p1_evidence_cluster_stability_by_machine.png", dpi=200)
    plt.close(fig)


def _write_markdown(
    path: Path,
    site_summary: pd.DataFrame,
    recon_summary: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    counts: dict[str, int],
) -> None:
    valve_recon = recon_summary[recon_summary["machine_type"].astype(str) == "valve"]
    other_recon = recon_summary[recon_summary["machine_type"].astype(str) != "valve"]
    valve_disp = site_summary[site_summary["machine_type"].astype(str) == "valve"]
    other_disp = site_summary[site_summary["machine_type"].astype(str) != "valve"]
    lines = [
        "# P1 Evidence-Logged Normalization Ablation Analysis",
        "",
        "Generated from the completed 1,215-run evidence rerun on `master` and `n3`.",
        "",
        "## Coverage",
        "",
        f"- Evidence bundles: {counts['bundles']}",
        f"- `reconstruction_errors.csv`: {counts['reconstruction_tables']} tables, {counts['reconstruction_rows']:,} rows",
        f"- `site_auroc.csv`: {counts['site_tables']} tables, {counts['site_rows']:,} rows",
        f"- `cluster_assignments.csv`: {counts['cluster_assignment_tables']} tables, {counts['cluster_assignment_rows']:,} rows",
        f"- `cluster_stability.csv`: {counts['cluster_stability_tables']} tables, {counts['cluster_stability_rows']:,} rows",
        "",
        "## Main Diagnostic Readout",
        "",
        (
            f"- Valve mean reconstruction-error separation: {valve_recon['mean_cohen_d'].mean():.3f} "
            f"vs. non-valve {other_recon['mean_cohen_d'].mean():.3f}."
        ),
        (
            f"- Valve mean within-run site-AUROC dispersion: {valve_disp['mean_site_dispersion'].mean():.3f} "
            f"vs. non-valve {other_disp['mean_site_dispersion'].mean():.3f}."
        ),
    ]
    if not cluster_summary.empty:
        valve_cluster = cluster_summary[cluster_summary["machine_type"].astype(str) == "valve"]
        other_cluster = cluster_summary[cluster_summary["machine_type"].astype(str) != "valve"]
        lines.append(
            f"- Valve mean cluster ARI: {valve_cluster['mean_ari'].mean():.3f} "
            f"vs. non-valve {other_cluster['mean_ari'].mean():.3f}."
        )
    lines += [
        "",
        "## Manuscript Use",
        "",
        "- Use the reconstruction-error separation table to replace the current limitation that valve difficulty lacks error-distribution evidence.",
        "- Use the site-dispersion table to quantify whether hard conditions are aggregate-only or site-instability effects.",
        "- Use the cluster-stability table conservatively: it supports discussion of the tested state-dict k-means Clustered-FL variant, not all clustered FL methods.",
        "",
        "## Output Files",
        "",
        "- `p1_evidence_site_auroc_run_summary.csv`",
        "- `p1_evidence_site_dispersion_by_machine_snr.csv`",
        "- `p1_evidence_reconstruction_run_summary.csv`",
        "- `p1_evidence_reconstruction_by_machine_snr.csv`",
        "- `p1_evidence_cluster_stability_by_machine_snr.csv`",
        "- `analysis_outputs/figures/p1_evidence_*`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-root", type=Path, default=DEFAULT_REMOTE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    args = parser.parse_args()

    site = _read_table(args.remote_root, "site_auroc")
    recon = _read_table(args.remote_root, "reconstruction_errors")
    assignments = _read_table(args.remote_root, "cluster_assignments")
    stability = _read_table(args.remote_root, "cluster_stability")

    if site.empty or recon.empty:
        raise SystemExit("Missing required site_auroc or reconstruction_errors evidence tables.")

    args.out.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    run_site = _run_site_summary(site)
    site_summary = _site_dispersion_summary(run_site)
    recon_run = _reconstruction_run_summary(recon)
    recon_summary = _reconstruction_machine_summary(recon_run)
    cluster_machine, cluster_run = _cluster_summary(stability, assignments)

    _write_csv(args.out / "p1_evidence_site_auroc_all.csv", site)
    _write_csv(args.out / "p1_evidence_site_auroc_run_summary.csv", run_site)
    _write_csv(args.out / "p1_evidence_site_dispersion_by_machine_snr.csv", site_summary)
    _write_csv(args.out / "p1_evidence_reconstruction_run_summary.csv", recon_run)
    _write_csv(args.out / "p1_evidence_reconstruction_by_machine_snr.csv", recon_summary)
    if not cluster_machine.empty:
        _write_csv(args.out / "p1_evidence_cluster_stability_by_machine_snr.csv", cluster_machine)
        _write_csv(args.out / "p1_evidence_cluster_stability_run_summary.csv", cluster_run)

    _write_tex_table(
        args.out / "p1_evidence_reconstruction_by_machine_snr.tex",
        recon_summary,
        ["machine_type", "db_level", "n_runs", "mean_error_delta", "mean_cohen_d", "mean_error_ratio"],
        "P1 evidence rerun reconstruction-error separation by machine type and SNR.",
        "tab:p1-evidence-reconstruction",
    )
    _write_tex_table(
        args.out / "p1_evidence_site_dispersion_by_machine_snr.tex",
        site_summary,
        ["machine_type", "db_level", "n_runs", "mean_run_auroc", "mean_site_dispersion", "mean_worst_site_auroc"],
        "P1 evidence rerun site-level AUROC dispersion by machine type and SNR.",
        "tab:p1-evidence-site-dispersion",
    )
    if not cluster_machine.empty:
        _write_tex_table(
            args.out / "p1_evidence_cluster_stability_by_machine_snr.tex",
            cluster_machine,
            ["machine_type", "db_level", "n_runs", "mean_ari", "mean_min_ari", "mean_final_ari"],
            "P1 evidence rerun Clustered-FL round-to-round assignment stability.",
            "tab:p1-evidence-cluster-stability",
        )

    _plot_site_dispersion(run_site, args.fig_dir)
    _plot_reconstruction(recon_run, args.fig_dir)
    _plot_cluster_stability(cluster_run, args.fig_dir)

    counts = {
        "bundles": int(site["run_id"].nunique()),
        "site_tables": int(site["run_id"].nunique()),
        "site_rows": int(len(site)),
        "reconstruction_tables": int(recon["run_id"].nunique()),
        "reconstruction_rows": int(len(recon)),
        "cluster_assignment_tables": int(assignments["run_id"].nunique()) if not assignments.empty else 0,
        "cluster_assignment_rows": int(len(assignments)) if not assignments.empty else 0,
        "cluster_stability_tables": int(stability["run_id"].nunique()) if not stability.empty else 0,
        "cluster_stability_rows": int(len(stability)) if not stability.empty else 0,
    }
    _write_markdown(args.out / "P1_EVIDENCE_ANALYSIS_2026-06-07.md", site_summary, recon_summary, cluster_machine, counts)

    print(f"site rows: {len(site):,}")
    print(f"reconstruction rows: {len(recon):,}")
    print(f"cluster assignment rows: {len(assignments):,}")
    print(f"cluster stability rows: {len(stability):,}")
    print(f"outputs: {args.out}")


if __name__ == "__main__":
    main()
