# non-IID(alpha)별 AUROC 저하 그래프 — W&B 수정런(n_frames=320, 21:04 KST 이후)만 필터해 alpha 축으로 알고리즘별 추세 시각화
import csv
import datetime
import os
import sys

import wandb

PROJECT = "e8both-wku/abada-night"
# 수정 config(n_frames=320 레시피) 재발사 시각 = 2026-06-03 21:04 KST = 12:04 UTC
CUTOFF = datetime.datetime(2026, 6, 3, 12, 4, tzinfo=datetime.timezone.utc)
OUT_DIR = os.path.expanduser("~/abada-night/reports")
ALGORITHMS = ["fedavg", "fedprox", "clustered_fl", "personalized"]

# 라벨 양국어 (§11.10). LANG=en|ko (인자 또는 환경변수)
LABELS = {
    "en": {"title": "Non-IID severity vs anomaly-detection AUROC (P1)",
           "x": "Dirichlet α  (smaller = more non-IID →)",
           "y": "Mean AUROC (over machine·dB·seed)", "suffix": "en"},
    "ko": {"title": "Non-IID 강도별 이상탐지 AUROC (P1)",
           "x": "Dirichlet α  (작을수록 non-IID 심함 →)",
           "y": "평균 AUROC (machine·dB·seed 통합)", "suffix": "ko"},
}


def _created_after_cutoff(run):
    """run.created_at(UTC ISO 'Z')이 레시피 재발사 시각 이후인지."""
    try:
        ts = datetime.datetime.strptime(run.created_at, "%Y-%m-%dT%H:%M:%SZ")
        return ts.replace(tzinfo=datetime.timezone.utc) >= CUTOFF
    except Exception:
        return False


def fetch_fixed_rows():
    """수정런만 → [{algorithm, alpha, machine, db, seed, auroc}] (§11: 버그런 제외)."""
    api = wandb.Api()
    rows = []
    for r in api.runs(PROJECT):
        if not r.name.startswith("p1_") or not _created_after_cutoff(r):
            continue
        auroc = r.summary.get("auroc")
        if auroc is None:
            continue
        rows.append({
            "algorithm": r.config.get("algorithm"),
            "alpha": float(r.config.get("alpha", -1)),
            "machine": r.config.get("machine_type"),
            "db": r.config.get("db_level"),
            "seed": r.config.get("seed"),
            "auroc": float(auroc),
        })
    return rows


def aggregate(rows):
    """algorithm × alpha 평균/표준편차/표본수 집계 (machine/db/seed 통합)."""
    import statistics
    agg = {}
    for r in rows:
        agg.setdefault((r["algorithm"], r["alpha"]), []).append(r["auroc"])
    out = {}
    for (alg, a), vals in agg.items():
        out[(alg, a)] = (statistics.mean(vals),
                         statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                         len(vals))
    return out


def save_csv(rows, path):
    """raw 데이터 CSV 저장 (§11 출처 보존)."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["algorithm", "alpha", "machine", "db", "seed", "auroc"])
        w.writeheader()
        w.writerows(rows)


def _set_korean_font():
    """한글 폰트 자동 탐지 (Noto CJK / Apple Gothic / Nanum). 없으면 False."""
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    candidates = ["Noto Sans CJK KR", "NanumGothic", "Apple SD Gothic Neo",
                  "Malgun Gothic", "Noto Sans KR"]
    avail = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in avail:
            plt.rcParams["font.family"] = c
            plt.rcParams["axes.unicode_minus"] = False
            return c
    return None


def plot(agg, out_dir, lang):
    """alpha(로그축) vs 평균 AUROC, 알고리즘별 라인. lang=en|ko. matplotlib 없으면 스킵."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 없음 — CSV만 저장. `pip install --user matplotlib` 후 재실행")
        return None
    L = LABELS[lang]
    if lang == "ko":
        font = _set_korean_font()
        if not font:
            print("⚠️ 한글 폰트 없음 → ko 그림은 두부박스. `Noto Sans CJK KR` 설치 필요(en은 정상)")
    alphas = sorted({a for (_, a) in agg})
    plt.figure(figsize=(7, 5))
    for alg in ALGORITHMS:
        xs = [a for a in alphas if (alg, a) in agg]
        ys = [agg[(alg, a)][0] for a in xs]
        es = [agg[(alg, a)][1] for a in xs]
        if xs:
            plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=alg)
    plt.xscale("log")
    plt.xlabel(L["x"]); plt.ylabel(L["y"]); plt.title(L["title"])
    plt.gca().invert_xaxis()  # 왼쪽이 더 non-IID(저하 추세 가시화)
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    path = os.path.join(out_dir, f"p1_alpha_auroc_{L['suffix']}.png")
    plt.savefig(path, dpi=150); plt.close()
    return path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = fetch_fixed_rows()
    print(f"수정런 데이터 포인트: {len(rows)}")
    if not rows:
        print("데이터 없음 (스윕 진행 중이면 나중에 재실행)")
        return
    save_csv(rows, os.path.join(OUT_DIR, "p1_alpha_auroc_raw.csv"))
    agg = aggregate(rows)
    print(f"\n{'algorithm':14s} {'alpha':>7s} {'meanAUROC':>10s} {'std':>6s} {'n':>4s}")
    for (alg, a) in sorted(agg, key=lambda k: (k[0], -k[1])):
        m, s, n = agg[(alg, a)]
        print(f"{alg:14s} {a:7.2f} {m:10.3f} {s:6.3f} {n:4d}")
    # 한/영 둘 다 생성 (§11.10)
    saved = [p for lang in ("en", "ko") if (p := plot(agg, OUT_DIR, lang))]
    print(f"\n[저장] {OUT_DIR}/p1_alpha_auroc_raw.csv")
    for p in saved:
        print(f"        {p}")


if __name__ == "__main__":
    main()
