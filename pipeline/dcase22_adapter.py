# DCASE 2022 Task 2 (ToyCar/ToyTrain) 데이터를 fl_train의 사이트별 payload로 변환하는 어댑터 (P1 E5)
"""E5 second-dataset adapter: DCASE 2022 Task 2 dev (ToyADMOS2-derived).

Mirrors mimii_adapter.make_mimii_site_data() so fl_train can consume a second
dataset family with zero changes to the training loop:

- run unit     : one (machine_type, section) pair, source domain only;
- train data   : section source-domain train normals, Dirichlet-partitioned
                 across simulated sites by attribute-derived pseudo machine id;
- test data    : section source-domain test normals/anomalies, assigned to
                 sites by matching pseudo machine id (disjoint);
- normalization: reuses mimii_adapter's leakage-safe mode implementations.

The target domain (10 train clips/section) is intentionally excluded from
training and from the default evaluation; it is a future generalization probe.
"""

import pathlib
import re

import numpy as np

import data_loader as dl
from mimii_adapter import (
    _assign_anomalies,
    _normalize_features_by_mode,
    _to_site_tensors,
)

MACHINE_TYPES = ("ToyCar", "ToyTrain")
SECTIONS = ("00", "01", "02")

_NAME_RE = re.compile(
    r"section_(?P<section>\d+)_(?P<domain>source|target)_"
    r"(?P<split>train|test)_(?P<label>normal|anomaly)_(?P<index>\d+)_?(?P<attr>.*)\.wav$"
)


def parse_filename(name: str) -> dict | None:
    """DCASE 2022 파일명 → 토큰 dict (불일치 시 None)."""
    m = _NAME_RE.search(name)
    if not m:
        return None
    d = m.groupdict()
    d["attr"] = d["attr"].strip("_")
    return d


def pseudo_machine_id(tokens: dict) -> str:
    """속성 문자열의 첫 key_value 쌍을 MIMII machine_id 대응물로 사용.

    예: 'car_A1_spd_28V_mic_1_noise_1' -> 'car_A1'. 속성이 없으면 섹션 기반 id.
    """
    attr = tokens.get("attr") or ""
    parts = attr.split("_")
    if len(parts) >= 2 and parts[0]:
        return f"{parts[0]}_{parts[1]}"
    return f"sec{tokens['section']}"


def _collect_files(machine_root: pathlib.Path, section: str):
    """(train_normals, test_normals, test_anomalies) 파일 목록 — source 도메인만."""
    train_n, test_n, test_a = [], [], []
    for sub in ("train", "test"):
        d = machine_root / sub
        if not d.exists():
            continue
        for p in sorted(d.glob("*.wav")):
            t = parse_filename(p.name)
            if not t or t["domain"] != "source":
                continue
            if int(t["section"]) != int(section):
                continue
            if t["split"] == "train" and t["label"] == "normal":
                train_n.append((p, t))
            elif t["split"] == "test" and t["label"] == "normal":
                test_n.append((p, t))
            elif t["split"] == "test" and t["label"] == "anomaly":
                test_a.append((p, t))
    return train_n, test_n, test_a


def load_dcase22_features(
    root: str,
    machine_type: str,
    section: str,
    sr: int = 16_000,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 512,
):
    """DCASE 2022 (machine, section, source) → (feats, labels, split_tags, meta).

    labels: 0=normal, 1=anomaly. split_tags: 'train'|'test' (normal만 train 존재).
    meta는 mimii_adapter가 요구하는 'machine_id' 키를 포함한다.
    """
    if machine_type not in MACHINE_TYPES:
        raise ValueError(f"machine_type must be one of {MACHINE_TYPES}, got {machine_type!r}")
    base = pathlib.Path(root).expanduser()
    machine_root = base / machine_type
    if not machine_root.exists():
        raise FileNotFoundError(f"경로를 찾을 수 없습니다: {machine_root}")

    train_n, test_n, test_a = _collect_files(machine_root, section)
    if not train_n:
        raise RuntimeError(f"train normal 파일이 없습니다: {machine_root} section {section}")

    feats, labels, tags, meta = [], [], [], []
    for group, label, tag in ((train_n, 0, "train"), (test_n, 0, "test"), (test_a, 1, "test")):
        for p, t in group:
            feats.append(dl._extract_features(str(p), sr, n_mels, n_fft, hop_length))
            labels.append(label)
            tags.append(tag)
            meta.append({
                "machine_type": machine_type,
                "machine_id": pseudo_machine_id(t),
                "section": t["section"],
                "domain": t["domain"],
                "db_level": f"section{t['section']}",
                "wav_path": str(p),
            })

    max_t = max(f.shape[-1] for f in feats)
    padded = [np.pad(f, ((0, 0), (0, max_t - f.shape[-1]))) for f in feats]
    return (
        np.stack(padded).astype(np.float32),
        np.array(labels, dtype=np.int8),
        np.array(tags),
        meta,
    )


def _assign_test_normals(meta, test_normal_idx, site_machine_ids, seed):
    """test normal을 machine_id 일치 사이트에 disjoint 배정 (anomaly 배정과 동일 규칙)."""
    rng = np.random.default_rng(seed + 13)
    site_tn = [[] for _ in site_machine_ids]
    for gi in test_normal_idx:
        mid = meta[gi]["machine_id"]
        cands = [s for s, mids in enumerate(site_machine_ids) if mid in mids]
        if cands:
            site_tn[rng.choice(cands)].append(gi)
    return [np.array(a, dtype=int) for a in site_tn]


def make_dcase22_site_data(config, num_sites, n_mels, n_frames, alpha, seed, device):
    """DCASE 2022 로드 → Dirichlet(train normal) 분할 + id 매칭 test 배정 → 사이트 payload."""
    root = config.get("data_root", "~/abada-night/data/dcase2022")
    machine_type = config["machine_type"]
    section = str(config.get("section", "00")).zfill(2)
    normalization_mode = config.get("normalization_mode", "central")

    feats, labels, tags, meta = load_dcase22_features(
        root, machine_type, section, n_mels=n_mels)

    train_normal_idx = np.where((labels == 0) & (tags == "train"))[0]
    test_normal_idx = np.where((labels == 0) & (tags == "test"))[0]
    anomaly_idx = np.where(labels == 1)[0]

    # Dirichlet 분할은 train normal에만 적용 (전용 test split이 있으므로 1/5 분할 불필요)
    tr_feats = feats[train_normal_idx]
    tr_labels = labels[train_normal_idx]
    tr_meta = [meta[int(i)] for i in train_normal_idx]
    site_local = dl.make_dirichlet_partition(tr_feats, tr_labels, tr_meta, num_sites, alpha, seed)
    site_train_idx = [train_normal_idx[np.asarray(loc, dtype=int)] for loc in site_local]

    site_mids = [set(meta[int(i)]["machine_id"] for i in idx) for idx in site_train_idx]
    site_test_idx = _assign_test_normals(meta, test_normal_idx, site_mids, seed)
    site_anom_idx = _assign_anomalies(meta, anomaly_idx, site_mids, seed)

    splits = list(zip(site_train_idx, site_test_idx))
    feats = _normalize_features_by_mode(
        feats,
        splits,
        mode=normalization_mode,
        site_anomaly_idx=site_anom_idx,
        fallback_idx=train_normal_idx,
    )

    payloads = []
    for s in range(num_sites):
        train_i, test_i = splits[s]
        anomaly_i = site_anom_idx[s]
        norm_tr, norm_te, anom_te = _to_site_tensors(
            feats, train_i, test_i, anomaly_i, device, n_frames
        )
        payloads.append(
            {
                "site_idx": s,
                "normal_train": norm_tr,
                "normal_test": norm_te,
                "anomaly_test": anom_te,
                "train_meta": [meta[int(i)] for i in np.asarray(train_i, dtype=int)],
                "test_meta": [meta[int(i)] for i in np.asarray(test_i, dtype=int)],
                "anomaly_meta": [meta[int(i)] for i in np.asarray(anomaly_i, dtype=int)],
                "site_machine_ids": sorted(site_mids[s]),
                "site_machine_id_joined": ";".join(sorted(site_mids[s])),
            }
        )
    return payloads
