import importlib.util
import unittest
from pathlib import Path


def _load_schema():
    module_path = Path(__file__).with_name("p1_logging_schema.py")
    spec = importlib.util.spec_from_file_location("p1_logging_schema", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class P1LoggingSchemaTest(unittest.TestCase):
    def test_manifest_exposes_four_evidence_tables(self):
        schema = _load_schema()
        manifest = schema.schema_manifest()

        self.assertEqual(manifest["schema_version"], "p1-evidence-logging-v1")
        self.assertEqual(
            set(manifest["tables"]),
            {"reconstruction_errors", "site_auroc", "cluster_assignments", "cluster_stability"},
        )
        self.assertIn("run_id", manifest["join_keys"])

    def test_reconstruction_error_header_has_claim_linkage_fields(self):
        schema = _load_schema()
        header = schema.csv_header("reconstruction_errors")

        for column in (
            "run_id",
            "site_id",
            "machine_id",
            "label",
            "normalization",
            "reconstruction_error",
            "threshold",
        ):
            self.assertIn(column, header)

    def test_validate_row_reports_missing_required_fields(self):
        schema = _load_schema()
        row = {column: "" for column in schema.csv_header("site_auroc")}
        row.pop("auroc")

        with self.assertRaises(ValueError) as ctx:
            schema.validate_row("site_auroc", row)

        self.assertIn("auroc", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
