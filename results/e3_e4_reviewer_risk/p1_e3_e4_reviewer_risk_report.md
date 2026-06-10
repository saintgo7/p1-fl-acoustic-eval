# P1 E3/E4 Reviewer-Risk Reduction Analysis

Generated from completed remote E3/E4 logs after Claude reviewer-mode planning.

## Completeness

- E3 FedProx-mu rows joined: 400 / 400; missing metrics: 0
- E4 LiteConvAE rows joined: 160 / 160; missing metrics: 0

## E3 Decision

- FAIL: FedProx mu is a large effect; do not claim optimizer robustness.
- FedProx mu partial eta^2: 0.3757 (large)
- Machine partial eta^2: 0.9197; SNR partial eta^2: 0.3825
- Best average mu: 0.001 with mean AUROC 0.7417; default mu=0.01 mean AUROC 0.7076
- At best mu, additive effect-size rank: Machine type=0.923, SNR=0.594, Dirichlet alpha=0.314

## E4 Decision

- PASS: condition ranking is stable and a condition factor remains the top effect.
- Dense-vs-Lite per-condition Spearman rho: 0.9857 (p=9.176e-25)
- E4 strongest factor: Machine type (partial eta^2=0.9011)
- E4 condition max eta^2: 0.9011; algorithm eta^2: 0.0707

## Manuscript Action

- Do not claim FedProx-mu robustness. State instead that the sweep found FedProx to be mu-sensitive; use the best-mu subset to show the condition effect remains larger than algorithmic tuning in the tested slice.

## Files

- `p1_e3_run_metrics.csv`
- `p1_e3_effect_sizes.csv`
- `p1_e3_best_mu_effect_sizes.csv`
- `p1_e3_mu_summary.csv`
- `p1_e4_run_metrics.csv`
- `p1_e4_effect_sizes.csv`
- `p1_e4_dense_vs_lite_condition_scatter.csv`
- `figures/p1_e3_mu_sensitivity.png`
- `figures/p1_e4_dense_vs_lite_condition_scatter.png`
- `tex/p1_e3_mu_summary.tex`
- `tex/p1_e3_e4_effect_sizes.tex`
