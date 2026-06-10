# P1 E2 Baseline Anchor Analysis

- Completed anchor rows with AUROC: 360
- Completed done files missing log AUROC: 0

## Anchor Overall Summary

| anchor | alpha | n | mean AUROC | 95% CI | min | max |
|---|---|---:|---:|---:|---:|---:|
| Centralized pooled | pooled | 120 | 0.7618 | +/- 0.0247 | 0.5134 | 0.9658 |
| Local only | a0p05 | 120 | 0.7325 | +/- 0.0244 | 0.4016 | 0.9578 |
| Local only | a100 | 120 | 0.7128 | +/- 0.0264 | 0.4454 | 0.9196 |

## Local-Only Paired Delta Summary

| alpha | comparator | n | mean delta | 95% CI | min | max |
|---|---|---:|---:|---:|---:|---:|
| a0p05 | Clustered FL | 120 | 0.0030 | +/- 0.0097 | -0.1362 | 0.1885 |
| a0p05 | FedAvg | 120 | 0.0104 | +/- 0.0109 | -0.1476 | 0.1702 |
| a0p05 | FedProx | 120 | 0.0176 | +/- 0.0093 | -0.1913 | 0.1546 |
| a0p05 | Personalized FL | 120 | -0.0029 | +/- 0.0102 | -0.2066 | 0.1639 |
| a100 | Clustered FL | 120 | -0.0325 | +/- 0.0060 | -0.1108 | 0.1316 |
| a100 | FedAvg | 120 | -0.0324 | +/- 0.0066 | -0.1283 | 0.1304 |
| a100 | FedProx | 120 | -0.0103 | +/- 0.0033 | -0.0516 | 0.0333 |
| a100 | Personalized FL | 120 | -0.0380 | +/- 0.0060 | -0.1331 | 0.0899 |
