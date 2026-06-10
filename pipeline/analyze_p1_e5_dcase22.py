#!/usr/bin/env python3
# P1 E5(DCASE 2022 ToyCar/ToyTrain) 480-run 그리드의 효과크기·paired 분석 스크립트
"""Analyze the E5 second-dataset grid.

Input : a rows CSV (name, machine_type, section, algorithm, alpha, seed, auroc)
        produced by joining worker-log auroc lines with done config JSONs.
Output: factor effect sizes (partial eta-squared, additive OLS, mirrors
        analyze_p1_main_stats.factor_effect_sizes with section replacing SNR),
        factor ranges, per-condition means, matched paired algorithm deltas,
        and a Markdown report comparing against the MIMII main-sweep values.

Usage:
  python3 analyze_p1_e5_dcase22.py --rows analysis_outputs/p1_e5_dcase22/p1_e5_rows.csv \
      [--require-complete 480]
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis_outputs" / "p1_e5_dcase22"

# MIMII main-sweep reference values (analysis_outputs/p1_main_stats, 2026-06-07)
MIMII_ETA2 = {"Machine type": 0.829, "SNR/Section": 0.655, "Algorithm": 0.031, "Dirichlet alpha": 0.014}

FACTORS = {
    "C(machine_type)": "Machine type",
    "C(section)": "SNR/Section",
    "C(algorithm)": "Algorithm",
    "C(alpha)": "Dirichlet alpha",
}


def factor_effect_sizes(df: pd.DataFrame) -> pd.DataFrame:
    model = ols("auroc ~ C(machine_type) + C(section) + C(algorithm) + C(alpha)", data=df).fit()
    table = sm.stats.anova_lm(model, typ=2)
    residual_ss = float(table.loc["Residual", "sum_sq"])
    rows = []
    for term, label in FACTORS.items():
        ss = float(table.loc[term, "sum_sq"])
        rows.append({
            "factor": label,
            "df": int(table.loc[term, "df"]),
            "sum_sq": ss,
            "F": float(table.loc[term, "F"]),
            "p_value": float(table.loc[term, "PR(>F)"]),
            "partial_eta2": ss / (ss + residual_ss),
            "mimii_partial_eta2": MIMII_ETA2[label],
        })
    return pd.DataFrame(rows).sort_values("partial_eta2", ascending=False)


def factor_ranges(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in (("machine_type", "Machine type"), ("section", "SNR/Section"),
                       ("algorithm", "Algorithm"), ("alpha", "Dirichlet alpha")):
        means = df.groupby(col)["auroc"].mean()
        rows.append({
            "factor": label,
            "levels": len(means),
            "min_level": str(means.idxmin()), "min_mean": means.min(),
            "max_level": str(means.idxmax()), "max_mean": means.max(),
            "range": means.max() - means.min(),
        })
    return pd.DataFrame(rows).sort_values("range", ascending=False)


def paired_algorithm_tests(df: pd.DataFrame) -> pd.DataFrame:
    unit_cols = ["machine_type", "section", "alpha", "seed"]
    wide = df.pivot_table(index=unit_cols, columns="algorithm", values="auroc", aggfunc="first")
    wide = wide.dropna()
    algos = sorted(wide.columns)
    best = wide.mean().idxmax()
    rows = []
    for other in algos:
        if other == best:
            continue
        delta = wide[best] - wide[other]
        w_stat, w_p = stats.wilcoxon(delta)
        rows.append({
            "comparison": f"{best} - {other}",
            "n_matched_units": len(delta),
            "mean_delta": delta.mean(),
            "wilcoxon_p": w_p,
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze the P1 E5 DCASE-2022 grid.")
    parser.add_argument("--rows", default=str(OUT_DIR / "p1_e5_rows.csv"))
    parser.add_argument("--require-complete", type=int, default=0,
                        help="Fail unless the rows CSV has exactly this many runs.")
    args = parser.parse_args()

    df = pd.read_csv(args.rows)
    df["alpha"] = df["alpha"].astype(float)
    df["section"] = df["section"].astype(str).str.zfill(2)
    n = len(df)
    if args.require_complete and n != args.require_complete:
        raise SystemExit(f"rows={n} != required {args.require_complete}; refusing partial analysis")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eff = factor_effect_sizes(df)
    rng = factor_ranges(df)
    paired = paired_algorithm_tests(df)
    cond = df.groupby(["machine_type", "section"])["auroc"].agg(["mean", "std", "count"]).reset_index()

    eff.to_csv(OUT_DIR / "p1_e5_factor_effect_sizes.csv", index=False)
    rng.to_csv(OUT_DIR / "p1_e5_factor_ranges.csv", index=False)
    paired.to_csv(OUT_DIR / "p1_e5_algorithm_paired_tests.csv", index=False)
    cond.to_csv(OUT_DIR / "p1_e5_condition_means.csv", index=False)

    lines = [
        "# P1 E5 (DCASE 2022 ToyCar/ToyTrain) Analysis",
        "",
        f"Runs analyzed: {n} (complete grid = 480). Overall mean AUROC {df['auroc'].mean():.4f}, std {df['auroc'].std():.4f}.",
        "",
        "## Partial eta-squared (additive OLS, typ-2 ANOVA) vs MIMII main sweep",
        "",
        "| Factor | E5 partial eta2 | MIMII partial eta2 | E5 p |",
        "|---|---:|---:|---:|",
    ]
    for _, r in eff.iterrows():
        lines.append(f"| {r['factor']} | {r['partial_eta2']:.3f} | {r['mimii_partial_eta2']:.3f} | {r['p_value']:.2e} |")
    lines += ["", "## Factor ranges (mean AUROC max-min per factor)", "",
              "| Factor | range | min level | max level |", "|---|---:|---|---|"]
    for _, r in rng.iterrows():
        lines.append(f"| {r['factor']} | {r['range']:.4f} | {r['min_level']} ({r['min_mean']:.4f}) | {r['max_level']} ({r['max_mean']:.4f}) |")
    lines += ["", "## Matched paired algorithm tests (best vs others)", "",
              "| Comparison | n units | mean delta | Wilcoxon p |", "|---|---:|---:|---:|"]
    for _, r in paired.iterrows():
        lines.append(f"| {r['comparison']} | {r['n_matched_units']} | {r['mean_delta']:.4f} | {r['wilcoxon_p']:.2e} |")
    (OUT_DIR / "P1_E5_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
