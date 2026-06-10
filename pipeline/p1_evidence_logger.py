"""Helpers for writing P1 evidence-logging CSV bundles."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping

import p1_logging_schema as schema


def stable_path_hash(value: str) -> str:
    """Return a short stable hash for a sample path or synthetic sample id."""
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def evidence_root_for_run(
    run_name: str,
    output_dir: str | None = None,
    env_var: str = "P1_EVIDENCE_OUTPUT_DIR",
) -> Path:
    """Resolve the root directory for one run's evidence bundle."""
    base = output_dir or os.environ.get(env_var) or "analysis_outputs/p1_evidence"
    return Path(base).expanduser() / run_name


def _normalize_row(table_name: str, row: Mapping[str, object]) -> dict[str, object]:
    header = schema.csv_header(table_name)
    schema.validate_row(table_name, row)
    return {key: row.get(key, "") for key in header}


def write_table(path: Path, table_name: str, rows: Iterable[Mapping[str, object]]) -> int:
    """Write a single evidence table to CSV and return the row count."""
    materialized = [_normalize_row(table_name, row) for row in rows]
    if not materialized:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=schema.csv_header(table_name))
        writer.writeheader()
        writer.writerows(materialized)
    return len(materialized)


def write_bundle(
    root: Path,
    tables: Mapping[str, Iterable[Mapping[str, object]]],
    manifest: Mapping[str, object] | None = None,
) -> dict[str, int]:
    """Write the evidence bundle and an index manifest."""
    root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for table_name, rows in tables.items():
        counts[table_name] = write_table(root / f"{table_name}.csv", table_name, rows)

    bundle_manifest = {
        "schema_version": schema.SCHEMA_VERSION,
        "tables": counts,
    }
    if manifest:
        bundle_manifest.update(manifest)
    (root / "manifest.json").write_text(
        json.dumps(bundle_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return counts
