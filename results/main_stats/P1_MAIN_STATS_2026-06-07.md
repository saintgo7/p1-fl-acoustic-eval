# P1 Main Sweep Statistical Hardening

Input: `analysis_outputs/p1_results_aggregate.csv`

## Matched Algorithm Tests

| comparison | n pairs | mean delta | 95% CI | median delta | Holm p |
|---|---:|---:|---:|---:|---:|
| Clustered FL | 1440 | 0.0079 | +/- 0.0011 | 0.0056 | $<10^{-4}$ |
| FedAvg | 1440 | 0.0082 | +/- 0.0007 | 0.0050 | $<10^{-4}$ |
| FedProx | 1440 | 0.0260 | +/- 0.0018 | 0.0256 | $<10^{-4}$ |

## Factor Ranges

| factor | best | worst | range |
|---|---|---|---:|
| Algorithm | Personalized (0.7463) | FedProx (0.7203) | 0.0260 |
| Machine type | slider (0.8665) | valve (0.5488) | 0.3176 |
| SNR | 6~dB (0.8229) | $-6$~dB (0.6444) | 0.1785 |
| Dirichlet alpha | \texttt{a0p5} (0.7428) | \texttt{a0p1} (0.7266) | 0.0162 |

## Additive Fixed-Effects Effect Sizes

| factor | df | F | p | partial eta squared |
|---|---:|---:|---:|---:|
| Machine type | 3 | 9290.5 | $<10^{-4}$ | 0.829 |
| SNR | 2 | 5466.6 | $<10^{-4}$ | 0.655 |
| Algorithm | 3 | 62.1 | $<10^{-4}$ | 0.031 |
| Dirichlet alpha | 5 | 16.4 | $<10^{-4}$ | 0.014 |

## Logit-AUROC Sensitivity Model

Formula: `logit(auroc) ~ machine * SNR * algorithm + alpha`; seed random-intercept ICC is estimated separately.

| term | df | F | p | partial eta squared |
|---|---:|---:|---:|---:|
| Machine type | 3 | 14245.2 | $<10^{-4}$ | 0.882 |
| SNR | 2 | 11633.1 | $<10^{-4}$ | 0.803 |
| Machine x SNR | 6 | 1354.6 | $<10^{-4}$ | 0.587 |
| Algorithm | 3 | 114.5 | $<10^{-4}$ | 0.057 |
| Dirichlet alpha | 5 | 62.4 | $<10^{-4}$ | 0.052 |
| Machine x Algorithm | 9 | 17.6 | $<10^{-4}$ | 0.027 |
| Machine x SNR x Algorithm | 18 | 2.4 | 0.0007 | 0.008 |
| SNR x Algorithm | 6 | 1.3 | 0.2371 | 0.001 |

## Seed Random-Intercept Sensitivity

- Seed group variance: 0.004946
- Residual variance: 0.052490
- Seed ICC: 0.086108
