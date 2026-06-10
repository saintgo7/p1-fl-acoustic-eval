import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.expanduser(os.environ.get("ABADA_BASE", "~/abada-night"))
BACKLOG = os.path.join(BASE, "backlog")
RUNNING = os.path.join(BASE, "running")
DONE = os.path.join(BASE, "done")
WORKER = os.environ.get("WORKER_ID", uuid.uuid4().hex[:6])
DAEMON = os.environ.get("DAEMON", "0") == "1"
VIS = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
JOB_TIMEOUT_SEC = int(os.environ.get("JOB_TIMEOUT_SEC", "3600"))


def claim():
    """Atomically claim one JSON config from the backlog."""
    for path in sorted(glob.glob(os.path.join(BACKLOG, "*.json"))):
        base = os.path.basename(path)
        if base.startswith("_") or base.startswith("."):
            continue
        target = os.path.join(RUNNING, f"{WORKER}_{base}")
        try:
            os.rename(path, target)
            return target
        except (FileNotFoundError, OSError):
            continue
    return None


def _run_job(path: str, timeout_sec: int = JOB_TIMEOUT_SEC) -> tuple[bool, str]:
    """Run one config in a subprocess so a hung training job cannot trap the worker."""
    env = os.environ.copy()
    env.setdefault("WANDB_SILENT", "true")
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "fl_train.py"), "--config", path]
    try:
        proc = subprocess.run(
            cmd,
            cwd=SCRIPT_DIR,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        return False, f"timeout after {timeout_sec}s\n{out[-2000:]}"

    out = proc.stdout or ""
    if proc.returncode != 0:
        return False, f"exit={proc.returncode}\n{out[-4000:]}"

    match = re.search(r'"auroc"\s*:\s*([0-9.eE+-]+)', out)
    return True, match.group(1) if match else "ok"


def finish_claim(path: str, dst: str) -> bool:
    if not os.path.exists(path):
        print(f"[{WORKER}/gpu{VIS}] claim file missing; skip move {os.path.basename(path)}", flush=True)
        return False
    shutil.move(path, dst)
    return True


def main():
    for d in (RUNNING, DONE):
        os.makedirs(d, exist_ok=True)
    ran = 0
    while True:
        target = claim()
        if target is None:
            if DAEMON:
                time.sleep(15)
                continue
            break

        base = os.path.basename(target)
        print(f"[{WORKER}/gpu{VIS}] start {base}", flush=True)
        try:
            ok, detail = _run_job(target)
            if ok:
                print(f"[{WORKER}/gpu{VIS}] done {base} auroc={detail}", flush=True)
                if finish_claim(target, os.path.join(DONE, base)):
                    ran += 1
            else:
                print(f"[{WORKER}/gpu{VIS}] FAILED {base}: {detail}", flush=True)
                finish_claim(target, os.path.join(DONE, "FAILED_" + base))
        except Exception as exc:
            print(f"[{WORKER}/gpu{VIS}] FAILED {base}: {exc}", flush=True)
            traceback.print_exc()
            finish_claim(target, os.path.join(DONE, "FAILED_" + base))

    print(f"[{WORKER}/gpu{VIS}] exit - ran {ran}", flush=True)


if __name__ == "__main__":
    main()
