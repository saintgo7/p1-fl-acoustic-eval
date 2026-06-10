#!/usr/bin/env bash
# master에서 E5 run 결과(auroc)와 config 파라미터를 join해 rows CSV로 가져오는 스크립트
set -euo pipefail
OUT="${1:-analysis_outputs/p1_e5_dcase22/p1_e5_rows.csv}"
mkdir -p "$(dirname "$OUT")"
ssh -o BatchMode=yes master 'python3 - <<EOF
import json, re, glob, csv, sys, os
rows = {}
for log in glob.glob(os.path.expanduser("~/abada-night/p1/logs/p1_g*.log")):
    for line in open(log, errors="ignore"):
        m = re.search(r"done (g\d+_(p1e5_\S+?)\.json) auroc=([0-9.eE+-]+)", line)
        if m:
            rows[m.group(2)] = float(m.group(3))
w = csv.writer(sys.stdout)
w.writerow(["name","machine_type","section","algorithm","alpha","seed","auroc"])
for f in sorted(glob.glob(os.path.expanduser("~/abada-night/done/g*_p1e5_*.json"))):
    cfg = json.load(open(f))
    name = cfg["name"]
    if name in rows:
        w.writerow([name, cfg["machine_type"], cfg["section"], cfg["algorithm"],
                    cfg["alpha"], cfg["seed"], rows[name]])
EOF' > "$OUT"
echo "rows: $(($(wc -l < "$OUT") - 1)) -> $OUT"
