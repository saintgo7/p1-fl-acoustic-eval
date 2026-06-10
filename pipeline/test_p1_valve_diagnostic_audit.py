import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import p1_valve_diagnostic_audit as audit


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF")


class P1ValveDiagnosticAuditTest(unittest.TestCase):
    def test_collect_file_index_counts_machine_id_files_without_feature_extraction(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for i in range(3):
                _touch(root / "-6dB" / "valve" / "id_00" / "normal" / f"n{i}.wav")
            for i in range(2):
                _touch(root / "-6dB" / "valve" / "id_00" / "abnormal" / f"a{i}.wav")
            _touch(root / "-6dB" / "valve" / "id_02" / "normal" / "n0.wav")

            labels, meta = audit.collect_file_index(str(root), "valve", "-6dB")

        self.assertEqual(len(labels), 6)
        self.assertEqual(sum(labels), 2)
        self.assertEqual({m["machine_id"] for m in meta}, {"id_00", "id_02"})

    def test_partition_audit_rows_report_site_and_summary_counts(self):
        labels = [0, 0, 0, 0, 1, 1]
        meta = [
            {"machine_id": "id_00"},
            {"machine_id": "id_00"},
            {"machine_id": "id_02"},
            {"machine_id": "id_02"},
            {"machine_id": "id_00"},
            {"machine_id": "id_02"},
        ]

        site_rows, summary = audit.partition_audit_rows(
            labels=labels,
            meta=meta,
            machine_type="valve",
            db_level="-6dB",
            alpha=100.0,
            seed=0,
            num_sites=2,
        )

        self.assertEqual(len(site_rows), 2)
        self.assertEqual(summary["total_anomaly"], 2)
        self.assertEqual(
            sum(int(row["anomaly_test_count"]) for row in site_rows),
            summary["assigned_anomaly"],
        )


if __name__ == "__main__":
    unittest.main()
