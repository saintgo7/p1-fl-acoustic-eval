"""Generate statistical summaries for the P1 main 5,760-run sweep.

The manuscript reports a balanced Cartesian sweep over algorithm, alpha,
machine type, SNR, and seed. This script keeps the inferential claims
matched to that design: algorithm comparisons are paired on
alpha/machine/SNR/seed units, while factor ranges are descriptive summaries
of the completed aggregate.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols


ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "analysis_outputs" / "p1_results_aggregate.csv"
OUT_DIR = ROOT / "analysis_outputs" / "p1_main_stats"

ALGORITHM_LABELS = {
    "personalized": "Personalized",
    "clustered_fl": "Clustered FL",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
}

FACTOR_LABELS = {
    "algorithm": "Algorithm",
    "machine_type": "Machine type",
    "db_level": "SNR",
    "alpha": "Dirichlet alpha",
}


def mean_ci95(values: pd.Series) -> tuple[float, float]:
    vals = values.dropna().astype(float)
    if len(vals) < 2:
        return float(vals.mean()), 0.0
    ci = stats.t.ppf(0.975, len(vals) - 1) * vals.std(ddof=1) / math.sqrt(len(vals))
    return float(vals.mean()), float(ci)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    out = [math.nan] * len(p_values)
    running = 0.0
    n = len(p_values)
    for rank, idx in enumerate(order):
        adjusted = min(1.0, (n - rank) * p_values[idx])
        running = max(running, adjusted)
        out[idx] = running
    return out


def fmt_float(value: float, digits: int = 4) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def fmt_p(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if value < 1e-4:
        return r"$<10^{-4}$"
    return f"{value:.4f}"


def load_main() -> pd.DataFrame:
    df = pd.read_csv(IN_PATH)
    expected = 4 * 6 * 4 * 3 * 20
    if len(df) != expected:
        raise SystemExit(f"Expected {expected} rows, found {len(df)} in {IN_PATH}")
    required = {"algorithm", "alpha", "machine_type", "db_level", "seed", "auroc"}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")
    return df


def algorithm_pairwise(df: pd.DataFrame) -> list[dict]:
    unit_cols = ["alpha", "machine_type", "db_level", "seed"]
    wide = df.pivot_table(index=unit_cols, columns="algorithm", values="auroc", aggfunc="first")
    if wide.isna().any().any():
        raise SystemExit("Algorithm comparison is not fully paired.")

    baseline = "personalized"
    rows: list[dict] = []
    raw_p: list[float] = []
    for other in ["clustered_fl", "fedavg", "fedprox"]:
        delta = wide[baseline] - wide[other]
        mean, ci = mean_ci95(delta)
        stat, p_value = stats.wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
        dz = mean / float(delta.std(ddof=1)) if float(delta.std(ddof=1)) else math.nan
        rows.append(
            {
                "comparison": f"{ALGORITHM_LABELS[other]}",
                "n_pairs": len(delta),
                "mean_delta": mean,
                "ci95": ci,
                "median_delta": float(delta.median()),
                "cohen_dz": dz,
                "wilcoxon_stat": float(stat),
                "p_value": float(p_value),
            }
        )
        raw_p.append(float(p_value))

    adjusted = holm_adjust(raw_p)
    for row, p_holm in zip(rows, adjusted):
        row["p_holm"] = p_holm
    return rows


def factor_ranges(df: pd.DataFrame) -> list[dict]:
    pretty = {
        "personalized": "Personalized",
        "clustered_fl": "Clustered FL",
        "fedavg": "FedAvg",
        "fedprox": "FedProx",
        "slider": "slider",
        "valve": "valve",
        "fan": "fan",
        "pump": "pump",
        "6dB": r"6~dB",
        "0dB": r"0~dB",
        "-6dB": r"$-6$~dB",
        "a0p05": r"\texttt{a0p05}",
        "a0p1": r"\texttt{a0p1}",
        "a0p5": r"\texttt{a0p5}",
        "a1": r"\texttt{a1}",
        "a10": r"\texttt{a10}",
        "a100": r"\texttt{a100}",
    }
    rows: list[dict] = []
    for factor in ["algorithm", "machine_type", "db_level", "alpha"]:
        grouped = (
            df.groupby(factor)["auroc"]
            .agg(["count", "mean", "std"])
            .sort_values("mean", ascending=False)
            .reset_index()
        )
        high = grouped.iloc[0]
        low = grouped.iloc[-1]
        rows.append(
            {
                "factor": FACTOR_LABELS[factor],
                "best_condition": pretty.get(str(high[factor]), str(high[factor])),
                "best_mean": float(high["mean"]),
                "worst_condition": pretty.get(str(low[factor]), str(low[factor])),
                "worst_mean": float(low["mean"]),
                "range": float(high["mean"] - low["mean"]),
                "n_per_condition": int(grouped["count"].iloc[0]),
            }
        )
    return rows


def factor_effect_sizes(df: pd.DataFrame) -> list[dict]:
    """Fit an additive fixed-effects model and report partial eta-squared.

    This is a deliberately conservative quantification of "dominant observed
    axes": it does not claim causal variance decomposition, but it converts the
    balanced sweep into a reproducible effect-size table.
    """
    model = ols(
        "auroc ~ C(machine_type) + C(db_level) + C(algorithm) + C(alpha)",
        data=df,
    ).fit()
    table = sm.stats.anova_lm(model, typ=2)
    residual_ss = float(table.loc["Residual", "sum_sq"])
    label_map = {
        "C(machine_type)": "Machine type",
        "C(db_level)": "SNR",
        "C(algorithm)": "Algorithm",
        "C(alpha)": "Dirichlet alpha",
    }
    rows: list[dict] = []
    for term, label in label_map.items():
        ss = float(table.loc[term, "sum_sq"])
        df_term = int(table.loc[term, "df"])
        f_stat = float(table.loc[term, "F"])
        p_value = float(table.loc[term, "PR(>F)"])
        partial_eta2 = ss / (ss + residual_ss)
        rows.append(
            {
                "factor": label,
                "df": df_term,
                "sum_sq": ss,
                "f_stat": f_stat,
                "p_value": p_value,
                "partial_eta2": partial_eta2,
            }
        )
    rows.sort(key=lambda r: r["partial_eta2"], reverse=True)
    return rows


def logit_effect_sizes(df: pd.DataFrame) -> tuple[list[dict], dict]:
    """Effect sizes on a logit-transformed AUROC scale plus seed ICC.

    AUROC is bounded, so this acts as a sensitivity analysis for the raw-AUROC
    additive ANOVA. The fixed-effect model follows Claude/reviewer guidance:
    machine type, SNR, and algorithm are allowed to interact, with alpha added
    as a separate controlled non-IID factor. A one-way random-intercept model
    estimates how much residual logit-AUROC variation is attributable to seed.
    """
    work = df.copy()
    eps = 1e-6
    clipped = work["auroc"].clip(eps, 1.0 - eps)
    work["auroc_logit"] = np.log(clipped / (1.0 - clipped))
    model = ols(
        "auroc_logit ~ C(machine_type) * C(db_level) * C(algorithm) + C(alpha)",
        data=work,
    ).fit()
    table = sm.stats.anova_lm(model, typ=2)
    residual_ss = float(table.loc["Residual", "sum_sq"])
    label_map = {
        "C(machine_type)": "Machine type",
        "C(db_level)": "SNR",
        "C(algorithm)": "Algorithm",
        "C(alpha)": "Dirichlet alpha",
        "C(machine_type):C(db_level)": "Machine x SNR",
        "C(machine_type):C(algorithm)": "Machine x Algorithm",
        "C(db_level):C(algorithm)": "SNR x Algorithm",
        "C(machine_type):C(db_level):C(algorithm)": "Machine x SNR x Algorithm",
    }
    rows: list[dict] = []
    for term, label in label_map.items():
        ss = float(table.loc[term, "sum_sq"])
        partial_eta2 = ss / (ss + residual_ss)
        rows.append(
            {
                "factor": label,
                "df": int(table.loc[term, "df"]),
                "sum_sq": ss,
                "f_stat": float(table.loc[term, "F"]),
                "p_value": float(table.loc[term, "PR(>F)"]),
                "partial_eta2": partial_eta2,
            }
        )
    rows.sort(key=lambda r: r["partial_eta2"], reverse=True)

    icc = {"seed_group_var": float("nan"), "residual_var": float("nan"), "seed_icc": float("nan")}
    try:
        mixed = sm.MixedLM.from_formula(
            "auroc_logit ~ C(machine_type) * C(db_level) * C(algorithm) + C(alpha)",
            groups="seed",
            data=work,
        ).fit(reml=True, method="lbfgs", maxiter=200, disp=False)
        group_var = float(mixed.cov_re.iloc[0, 0])
        resid_var = float(mixed.scale)
        icc = {
            "seed_group_var": group_var,
            "residual_var": resid_var,
            "seed_icc": group_var / (group_var + resid_var) if group_var + resid_var > 0 else float("nan"),
        }
    except Exception as exc:
        icc["mixedlm_error"] = str(exc)
    return rows, icc


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_algorithm_tex(path: Path, rows: list[dict]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Matched paired algorithm comparisons in the main 5,760-run sweep. Deltas are Personalized AUROC minus the comparator AUROC on matched Dirichlet-alpha, machine-type, SNR, and seed units; $p_{\mathrm{Holm}}$ reports Holm-adjusted Wilcoxon signed-rank tests.}",
        r"\label{tab:p1-algorithm-paired-tests}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Comparator & $n$ & Mean $\Delta$ & 95\% CI & Median $\Delta$ & $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['comparison']} & {row['n_pairs']} & "
            f"{fmt_float(row['mean_delta'])} & $\\pm${fmt_float(row['ci95'])} & "
            f"{fmt_float(row['median_delta'])} & {fmt_p(row['p_holm'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_range_tex(path: Path, rows: list[dict]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Observed mean-AUROC ranges by sweep factor. These are descriptive separations in the completed balanced aggregate, not causal variance-decomposition estimates.}",
        r"\label{tab:p1-factor-ranges}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{@{}llrlrr@{}}",
        r"\toprule",
        r"Factor & Best & Mean & Worst & Mean & Range \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['factor']} & {row['best_condition']} & {fmt_float(row['best_mean'])} & "
            f"{row['worst_condition']} & {fmt_float(row['worst_mean'])} & "
            f"{fmt_float(row['range'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_effect_size_tex(path: Path, rows: list[dict]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Additive fixed-effects ANOVA effect sizes for the main 5,760-run sweep. Partial $\eta^2$ is used only to quantify observed AUROC separation in the balanced aggregate; it is not interpreted as a causal variance decomposition.}",
        r"\label{tab:p1-factor-effect-sizes}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Factor & df & $F$ & $p$ & Partial $\eta^2$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['factor']} & {row['df']} & {row['f_stat']:.1f} & "
            f"{fmt_p(row['p_value'])} & {row['partial_eta2']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_logit_effect_size_tex(path: Path, rows: list[dict], icc: dict) -> None:
    top = rows[:6]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Logit-AUROC sensitivity analysis for the main 5,760-run sweep. The fixed-effect model includes machine type, SNR, algorithm, their interactions, and Dirichlet alpha; partial $\eta^2$ again quantifies observed separation rather than causal dominance.}",
        r"\label{tab:p1-logit-effect-sizes}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Term & df & $F$ & $p$ & Partial $\eta^2$ \\",
        r"\midrule",
    ]
    for row in top:
        lines.append(
            f"{row['factor']} & {row['df']} & {row['f_stat']:.1f} & "
            f"{fmt_p(row['p_value'])} & {row['partial_eta2']:.3f} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            rf"Seed ICC & -- & -- & -- & {icc.get('seed_icc', float('nan')):.3f} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(
    path: Path,
    pair_rows: list[dict],
    range_rows: list[dict],
    effect_rows: list[dict],
    logit_rows: list[dict],
    seed_icc: dict,
) -> None:
    lines = [
        "# P1 Main Sweep Statistical Hardening",
        "",
        f"Input: `{IN_PATH.relative_to(ROOT)}`",
        "",
        "## Matched Algorithm Tests",
        "",
        "| comparison | n pairs | mean delta | 95% CI | median delta | Holm p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in pair_rows:
        lines.append(
            f"| {row['comparison']} | {row['n_pairs']} | {fmt_float(row['mean_delta'])} | "
            f"+/- {fmt_float(row['ci95'])} | {fmt_float(row['median_delta'])} | "
            f"{fmt_p(row['p_holm'])} |"
        )
    lines.extend(["", "## Factor Ranges", "", "| factor | best | worst | range |", "|---|---|---|---:|"])
    for row in range_rows:
        lines.append(
            f"| {row['factor']} | {row['best_condition']} ({fmt_float(row['best_mean'])}) | "
            f"{row['worst_condition']} ({fmt_float(row['worst_mean'])}) | {fmt_float(row['range'])} |"
        )
    lines.extend(
        [
            "",
            "## Additive Fixed-Effects Effect Sizes",
            "",
            "| factor | df | F | p | partial eta squared |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in effect_rows:
        lines.append(
            f"| {row['factor']} | {row['df']} | {row['f_stat']:.1f} | "
            f"{fmt_p(row['p_value'])} | {row['partial_eta2']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Logit-AUROC Sensitivity Model",
            "",
            "Formula: `logit(auroc) ~ machine * SNR * algorithm + alpha`; seed random-intercept ICC is estimated separately.",
            "",
            "| term | df | F | p | partial eta squared |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in logit_rows:
        lines.append(
            f"| {row['factor']} | {row['df']} | {row['f_stat']:.1f} | "
            f"{fmt_p(row['p_value'])} | {row['partial_eta2']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Seed Random-Intercept Sensitivity",
            "",
            f"- Seed group variance: {seed_icc.get('seed_group_var', float('nan')):.6f}",
            f"- Residual variance: {seed_icc.get('residual_var', float('nan')):.6f}",
            f"- Seed ICC: {seed_icc.get('seed_icc', float('nan')):.6f}",
        ]
    )
    if "mixedlm_error" in seed_icc:
        lines.append(f"- MixedLM warning: {seed_icc['mixedlm_error']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_main()
    pair_rows = algorithm_pairwise(df)
    range_rows = factor_ranges(df)
    effect_rows = factor_effect_sizes(df)
    logit_rows, seed_icc = logit_effect_sizes(df)
    write_csv(OUT_DIR / "p1_main_algorithm_paired_tests.csv", pair_rows)
    write_csv(OUT_DIR / "p1_main_factor_ranges.csv", range_rows)
    write_csv(OUT_DIR / "p1_main_factor_effect_sizes.csv", effect_rows)
    write_csv(OUT_DIR / "p1_main_logit_effect_sizes.csv", logit_rows)
    write_csv(OUT_DIR / "p1_main_seed_icc.csv", [seed_icc])
    write_algorithm_tex(OUT_DIR / "p1_main_algorithm_paired_tests.tex", pair_rows)
    write_range_tex(OUT_DIR / "p1_main_factor_ranges.tex", range_rows)
    write_effect_size_tex(OUT_DIR / "p1_main_factor_effect_sizes.tex", effect_rows)
    write_logit_effect_size_tex(OUT_DIR / "p1_main_logit_effect_sizes.tex", logit_rows, seed_icc)
    write_report(
        OUT_DIR / "P1_MAIN_STATS_2026-06-07.md",
        pair_rows,
        range_rows,
        effect_rows,
        logit_rows,
        seed_icc,
    )
    print(f"Wrote {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
