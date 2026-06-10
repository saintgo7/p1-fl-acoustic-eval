"""P1 evidence logging schema for anomaly reconstruction and site stability."""

from __future__ import annotations

from typing import Mapping


SCHEMA_VERSION = "p1-evidence-logging-v1"

RECONSTRUCTION_ERROR_COLUMNS = (
    "schema_version",
    "run_id",
    "paper_id",
    "experiment_id",
    "seed",
    "round",
    "site_id",
    "machine_id",
    "split",
    "sample_id",
    "sample_path_hash",
    "label",
    "anomaly_type",
    "normalization",
    "model_family",
    "reconstruction_error",
    "threshold",
    "predicted_anomaly",
)

SITE_AUROC_COLUMNS = (
    "schema_version",
    "run_id",
    "paper_id",
    "experiment_id",
    "seed",
    "round",
    "site_id",
    "machine_id",
    "normalization",
    "model_family",
    "n_normal",
    "n_anomaly",
    "auroc",
    "auprc",
    "partial_auroc_fpr_0_1",
    "f1_at_threshold",
    "threshold",
)

CLUSTER_ASSIGNMENT_COLUMNS = (
    "schema_version",
    "run_id",
    "paper_id",
    "experiment_id",
    "seed",
    "round",
    "site_id",
    "machine_id",
    "cluster_id",
    "cluster_method",
    "feature_space",
    "n_samples",
    "normalization",
)

CLUSTER_STABILITY_COLUMNS = (
    "schema_version",
    "run_id",
    "paper_id",
    "experiment_id",
    "seed",
    "round",
    "site_id",
    "cluster_method",
    "feature_space",
    "stability_metric",
    "stability_value",
    "baseline_round",
    "comparison_round",
)

TABLE_COLUMNS = {
    "reconstruction_errors": RECONSTRUCTION_ERROR_COLUMNS,
    "site_auroc": SITE_AUROC_COLUMNS,
    "cluster_assignments": CLUSTER_ASSIGNMENT_COLUMNS,
    "cluster_stability": CLUSTER_STABILITY_COLUMNS,
}


def csv_header(table_name: str) -> list[str]:
    """Return the canonical CSV header for a P1 evidence table."""
    try:
        return list(TABLE_COLUMNS[table_name])
    except KeyError as exc:
        raise KeyError(f"Unknown P1 evidence table: {table_name}") from exc


def schema_manifest() -> dict[str, object]:
    """Return a serializable manifest for documentation and run metadata."""
    return {
        "schema_version": SCHEMA_VERSION,
        "tables": {table: list(columns) for table, columns in TABLE_COLUMNS.items()},
        "join_keys": ["run_id", "experiment_id", "seed", "round", "site_id", "machine_id"],
    }


def missing_required_columns(table_name: str, row: Mapping[str, object]) -> list[str]:
    """Return required schema fields absent from a prospective row."""
    return [column for column in csv_header(table_name) if column not in row]


def validate_row(table_name: str, row: Mapping[str, object]) -> None:
    """Raise ValueError when a row is missing required P1 evidence fields."""
    missing = missing_required_columns(table_name, row)
    if missing:
        raise ValueError(f"{table_name} row missing required fields: {', '.join(missing)}")
