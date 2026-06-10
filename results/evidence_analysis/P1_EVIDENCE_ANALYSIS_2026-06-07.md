# P1 Evidence-Logged Normalization Ablation Analysis

Generated from the completed 1,215-run evidence rerun on `master` and `n3`.

## Coverage

- Evidence bundles: 1215
- `reconstruction_errors.csv`: 1215 tables, 2,036,097 rows
- `site_auroc.csv`: 1215 tables, 11,610 rows
- `cluster_assignments.csv`: 405 tables, 121,500 rows
- `cluster_stability.csv`: 405 tables, 129,195 rows

## Main Diagnostic Readout

- Valve mean reconstruction-error separation: 0.318 vs. non-valve 1.328.
- Valve mean within-run site-AUROC dispersion: 0.105 vs. non-valve 0.094.
- Valve mean cluster ARI: 0.824 vs. non-valve 0.832.

## Manuscript Use

- Use the reconstruction-error separation table to replace the current limitation that valve difficulty lacks error-distribution evidence.
- Use the site-dispersion table to quantify whether hard conditions are aggregate-only or site-instability effects.
- Use the cluster-stability table conservatively: it supports discussion of the tested state-dict k-means Clustered-FL variant, not all clustered FL methods.

## Output Files

- `p1_evidence_site_auroc_run_summary.csv`
- `p1_evidence_site_dispersion_by_machine_snr.csv`
- `p1_evidence_reconstruction_run_summary.csv`
- `p1_evidence_reconstruction_by_machine_snr.csv`
- `p1_evidence_cluster_stability_by_machine_snr.csv`
- `analysis_outputs/figures/p1_evidence_*`
