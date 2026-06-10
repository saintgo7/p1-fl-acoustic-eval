#!/usr/bin/env python3
# P1 B1+B5+B4: 상호작용 효과크기, 시드-클러스터 부트스트랩 CI, forest figure 생성
"""Interaction effect sizes + bootstrap CIs + forest figure for P1.

B1: two-way model adding algorithm x condition interactions to the additive
    model, on the MIMII 5,760-run aggregate and the E5 480-run grid.
B5: seed-cluster bootstrap (resample seeds with replacement) 95% CIs for the
    additive-model partial eta-squared values.
B4: forest-style figure comparing MIMII vs E5 partial eta-squared with CIs.

Outputs under analysis_outputs/p1_interactions_ci/.
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs" / "p1_interactions_ci"

ADDITIVE = "auroc ~ C(machine_type) + C(cond) + C(algorithm) + C(alpha)"
INTERACT = (
    "auroc ~ C(machine_type) + C(cond) + C(algorithm) + C(alpha)"
    " + C(algorithm):C(machine_type) + C(algorithm):C(cond) + C(algorithm):C(alpha)"
)

LABELS = {
    "C(machine_type)": "Machine type",
    "C(cond)": "Condition (SNR/section)",
    "C(algorithm)": "Algorithm",
    "C(alpha)": "Dirichlet alpha",
    "C(algorithm):C(machine_type)": "Algorithm x Machine",
    "C(algorithm):C(cond)": "Algorithm x Condition",
    "C(algorithm):C(alpha)": "Algorithm x Alpha",
}


def load_mimii() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "analysis_outputs" / "p1_results_aggregate.csv")
    return df.rename(columns={"db_level": "cond"})


def load_e5() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "analysis_outputs" / "p1_e5_dcase22" / "p1_e5_rows.csv")
    df["section"] = df["section"].astype(str).str.zfill(2)
    return df.rename(columns={"section": "cond"})


def eta2_table(df: pd.DataFrame, formula: str) -> pd.DataFrame:
    model = ols(formula, data=df).fit()
    table = sm.stats.anova_lm(model, typ=2)
    residual_ss = float(table.loc["Residual", "sum_sq"])
    rows = []
    for term in table.index:
        if term == "Residual":
            continue
        ss = float(table.loc[term, "sum_sq"])
        rows.append({
            "term": LABELS.get(term, term),
            "df": int(table.loc[term, "df"]),
            "partial_eta2": ss / (ss + residual_ss),
            "p_value": float(table.loc[term, "PR(>F)"]),
        })
    return pd.DataFrame(rows)


def bootstrap_eta2(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """시드 단위 클러스터 부트스트랩 — additive 모델 partial eta2의 95% CI."""
    rng = np.random.default_rng(seed)
    seeds = sorted(df["seed"].unique())
    groups = {s: g for s, g in df.groupby("seed")}
    samples: dict[str, list[float]] = {}
    for _ in range(n_boot):
        pick = rng.choice(seeds, size=len(seeds), replace=True)
        boot = pd.concat([groups[s] for s in pick], ignore_index=True)
        tab = eta2_table(boot, ADDITIVE)
        for _, r in tab.iterrows():
            samples.setdefault(r["term"], []).append(r["partial_eta2"])
    point = eta2_table(df, ADDITIVE).set_index("term")["partial_eta2"]
    rows = []
    for term, vals in samples.items():
        v = np.asarray(vals)
        rows.append({
            "term": term,
            "eta2_point": float(point[term]),
            "ci_lo": float(np.percentile(v, 2.5)),
            "ci_hi": float(np.percentile(v, 97.5)),
        })
    return pd.DataFrame(rows)


def forest_figure(mimii_ci: pd.DataFrame, e5_ci: pd.DataFrame, path: pathlib.Path) -> None:
    order = ["Machine type", "Condition (SNR/section)", "Algorithm", "Dirichlet alpha"]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    y = np.arange(len(order))[::-1]
    for off, (ci, label, color) in enumerate((
        (mimii_ci, "MIMII (5,760 runs)", "#1f5fa8"),
        (e5_ci, "DCASE 2022 E5 (480 runs)", "#c2452f"),
    )):
        ci = ci.set_index("term").loc[order]
        yy = y + (0.16 if off == 0 else -0.16)
        ax.errorbar(ci["eta2_point"], yy,
                    xerr=[ci["eta2_point"] - ci["ci_lo"], ci["ci_hi"] - ci["eta2_point"]],
                    fmt="o", capsize=3, label=label, color=color, markersize=5)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("Partial $\\eta^2$ (additive model, seed-cluster bootstrap 95% CI)")
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, df in (("mimii", load_mimii()), ("e5", load_e5())):
        add = eta2_table(df, ADDITIVE)
        inter = eta2_table(df, INTERACT)
        ci = bootstrap_eta2(df)
        add.to_csv(OUT / f"p1_{name}_additive_eta2.csv", index=False)
        inter.to_csv(OUT / f"p1_{name}_interaction_eta2.csv", index=False)
        ci.to_csv(OUT / f"p1_{name}_eta2_bootstrap_ci.csv", index=False)
        results[name] = (add, inter, ci)
        print(f"== {name} interaction model ==")
        print(inter.to_string(index=False))
        print(f"== {name} bootstrap CI ==")
        print(ci.to_string(index=False))
    forest_figure(results["mimii"][2], results["e5"][2],
                  ROOT / "analysis_outputs" / "figures" / "p1_eta2_forest.pdf")
    print("figure written: analysis_outputs/figures/p1_eta2_forest.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
