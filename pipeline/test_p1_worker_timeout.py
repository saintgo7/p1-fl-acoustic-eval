import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import p1_worker


class P1WorkerTimeoutTest(unittest.TestCase):
    def test_claim_skips_manifest_json(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            backlog = tmp / "backlog"
            running = tmp / "running"
            done = tmp / "done"
            backlog.mkdir()
            running.mkdir()
            done.mkdir()
            (backlog / "_manifest.json").write_text(
                json.dumps({"total_jobs": 1}), encoding="utf-8"
            )
            (backlog / "._job.json").write_text(
                json.dumps({"macos": "metadata"}), encoding="utf-8"
            )
            (backlog / "job.json").write_text(
                json.dumps({"name": "real_job"}), encoding="utf-8"
            )

            old_backlog, old_running, old_done = p1_worker.BACKLOG, p1_worker.RUNNING, p1_worker.DONE
            p1_worker.BACKLOG = str(backlog)
            p1_worker.RUNNING = str(running)
            p1_worker.DONE = str(done)
            try:
                claimed = p1_worker.claim()
            finally:
                p1_worker.BACKLOG = old_backlog
                p1_worker.RUNNING = old_running
                p1_worker.DONE = old_done

        self.assertIsNotNone(claimed)
        self.assertEqual(Path(str(claimed)).name, f"{p1_worker.WORKER}_job.json")

    def test_run_job_times_out_subprocess(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            fake_train = tmp / "fl_train.py"
            fake_train.write_text(
                textwrap.dedent(
                    """
                    import time
                    time.sleep(5)
                    """
                ),
                encoding="utf-8",
            )
            cfg = tmp / "job.json"
            cfg.write_text(json.dumps({"name": "timeout_case"}), encoding="utf-8")

            old_script_dir = p1_worker.SCRIPT_DIR
            p1_worker.SCRIPT_DIR = str(tmp)
            try:
                ok, detail = p1_worker._run_job(str(cfg), timeout_sec=1)
            finally:
                p1_worker.SCRIPT_DIR = old_script_dir

        self.assertFalse(ok)
        self.assertIn("timeout", detail)


if __name__ == "__main__":
    unittest.main()
