# P1 E5 (DCASE 2022 ToyCar/ToyTrain) Analysis

Runs analyzed: 480 (complete grid = 480). Overall mean AUROC 0.6710, std 0.1252.

## Partial eta-squared (additive OLS, typ-2 ANOVA) vs MIMII main sweep

| Factor | E5 partial eta2 | MIMII partial eta2 | E5 p |
|---|---:|---:|---:|
| SNR/Section | 0.553 | 0.655 | 2.43e-83 |
| Machine type | 0.324 | 0.829 | 4.63e-42 |
| Algorithm | 0.016 | 0.031 | 5.73e-02 |
| Dirichlet alpha | 0.010 | 0.014 | 2.72e-02 |

## Factor ranges (mean AUROC max-min per factor)

| Factor | range | min level | max level |
|---|---:|---|---|
| SNR/Section | 0.1982 | 00 (0.5558) | 02 (0.7540) |
| Machine type | 0.1046 | ToyTrain (0.6187) | ToyCar (0.7233) |
| Algorithm | 0.0236 | fedprox (0.6546) | personalized (0.6781) |
| Dirichlet alpha | 0.0154 | 100.0 (0.6633) | 0.05 (0.6787) |

## Matched paired algorithm tests (best vs others)

| Comparison | n units | mean delta | Wilcoxon p |
|---|---:|---:|---:|
| personalized - clustered_fl | 120 | 0.0028 | 2.05e-01 |
| personalized - fedavg | 120 | 0.0021 | 6.19e-01 |
| personalized - fedprox | 120 | 0.0236 | 2.58e-10 |
