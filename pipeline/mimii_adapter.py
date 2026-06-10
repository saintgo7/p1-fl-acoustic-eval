# 실제 MIMII 데이터를 fl_train의 사이트별 (normal_train, normal_test, anomaly_test) 텐서로 변환하는 어댑터
import numpy as np
import torch

import data_loader as dl

NORMALIZATION_MODES = ("central", "federated_train", "local_site", "none")


def _fix(arr, n_frames):
    """시간축 길이를 n_frames로 패딩/크롭 → (N,1,n_mels,T)."""
    x = arr[:, :, :n_frames]
    if x.shape[-1] < n_frames:
        x = np.pad(x, ((0, 0), (0, 0), (0, n_frames - x.shape[-1])), mode="constant")
    return x[:, None, :, :].astype(np.float32)


def _split_site(normal_idx, seed):
    """사이트 normal을 train/test 인덱스로 분할 (정규화 적합 전 호출)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(normal_idx))
    n_test = max(1, len(normal_idx) // 5) if len(normal_idx) else 0
    return normal_idx[perm[n_test:]], normal_idx[perm[:n_test]]  # (train, test)


def _to_site_tensors(feats, train_i, test_i, anomaly_idx, device, n_frames):
    """미리 분할된 train/test/anomaly 인덱스 → 텐서 3종 (split·정규화는 호출부에서 완료)."""
    empty = np.empty((0,) + feats.shape[1:], np.float32)
    norm_tr = feats[train_i] if len(train_i) else empty
    norm_te = feats[test_i] if len(test_i) else empty
    anom_te = feats[anomaly_idx] if len(anomaly_idx) else empty
    return (
        torch.from_numpy(_fix(norm_tr, n_frames)).to(device),
        torch.from_numpy(_fix(norm_te, n_frames)).to(device),
        torch.from_numpy(_fix(anom_te, n_frames)).to(device),
    )


def _assign_anomalies(meta, anomaly_global_idx, site_machine_ids, seed):
    """각 anomaly를 machine_id가 일치하는 사이트 중 하나에 disjoint 배정 (중복 방지, 사이트 매칭)."""
    rng = np.random.default_rng(seed + 7)
    site_anom = [[] for _ in site_machine_ids]
    for gi in anomaly_global_idx:
        mid = meta[gi]["machine_id"]
        cands = [s for s, mids in enumerate(site_machine_ids) if mid in mids]
        if cands:
            site_anom[rng.choice(cands)].append(gi)
    return [np.array(a, dtype=int) for a in site_anom]


def _concat_indices(parts):
    """Return one int array from possibly empty index arrays."""
    non_empty = [np.asarray(p, dtype=int) for p in parts if len(p)]
    return np.concatenate(non_empty) if non_empty else np.array([], dtype=int)


def _fit_stats(feats, idx):
    """Fit per-mel mean/std on selected examples over sample and time axes."""
    idx = np.asarray(idx, dtype=int)
    if len(idx) == 0:
        return None
    src = feats[idx].astype(np.float64, copy=False)
    mu = src.mean(axis=(0, 2), keepdims=True)
    sd = src.std(axis=(0, 2), keepdims=True) + 1e-6
    return mu, sd


def _fit_federated_train_stats(feats, splits, fallback_idx=None):
    """Aggregate site-local train sufficient statistics without raw-feature pooling."""
    n_mels = feats.shape[1]
    total = 0
    sums = np.zeros(n_mels, dtype=np.float64)
    sumsqs = np.zeros(n_mels, dtype=np.float64)
    for train_idx, _ in splits:
        idx = np.asarray(train_idx, dtype=int)
        if len(idx) == 0:
            continue
        src = feats[idx].astype(np.float64, copy=False)
        total += src.shape[0] * src.shape[2]
        sums += src.sum(axis=(0, 2))
        sumsqs += (src * src).sum(axis=(0, 2))
    if total == 0 and fallback_idx is not None:
        return _fit_stats(feats, fallback_idx)
    if total == 0:
        return None
    mu = sums / total
    var = np.maximum(sumsqs / total - mu * mu, 0.0)
    sd = np.sqrt(var) + 1e-6
    return mu.reshape(1, n_mels, 1), sd.reshape(1, n_mels, 1)


def _normalize_with_stats(feats, idx, stats, out):
    """Apply stats to selected rows of out."""
    idx = np.asarray(idx, dtype=int)
    if len(idx) == 0 or stats is None:
        return
    mu, sd = stats
    out[idx] = ((feats[idx] - mu) / sd).astype(np.float32)


def _normalize_features_by_mode(
    feats,
    splits,
    mode="central",
    site_anomaly_idx=None,
    fallback_idx=None,
):
    """Normalize MIMII features according to the requested ablation mode.

    Modes:
      central         : current simulation baseline; fit one train-normal stat centrally.
      federated_train : same statistic via site-local sums/sumsq aggregation.
      local_site      : fit/apply each site's own train-normal statistic.
      none            : no normalization, for optional sensitivity checks.
    """
    if mode not in NORMALIZATION_MODES:
        raise ValueError(f"normalization_mode must be one of {NORMALIZATION_MODES}, got {mode!r}")
    feats = feats.astype(np.float32, copy=False)
    if mode == "none":
        return feats.copy()

    train_idx_all = _concat_indices([tr for tr, _ in splits])
    fallback_idx = np.asarray(fallback_idx if fallback_idx is not None else train_idx_all, dtype=int)

    if mode == "central":
        stats = _fit_stats(feats, train_idx_all)
        if stats is None:
            stats = _fit_stats(feats, fallback_idx)
        out = feats.copy()
        if stats is not None:
            out = ((feats - stats[0]) / stats[1]).astype(np.float32)
        return out

    global_stats = _fit_federated_train_stats(feats, splits, fallback_idx=fallback_idx)
    if mode == "federated_train":
        out = feats.copy()
        if global_stats is not None:
            out = ((feats - global_stats[0]) / global_stats[1]).astype(np.float32)
        return out

    out = feats.copy()
    site_anomaly_idx = site_anomaly_idx or [np.array([], dtype=int) for _ in splits]
    for site_idx, (train_idx, test_idx) in enumerate(splits):
        local_stats = _fit_stats(feats, train_idx) or global_stats
        apply_idx = _concat_indices([
            train_idx,
            test_idx,
            site_anomaly_idx[site_idx] if site_idx < len(site_anomaly_idx) else np.array([], dtype=int),
        ])
        _normalize_with_stats(feats, apply_idx, local_stats, out)
    return out.astype(np.float32)


def make_mimii_site_data(config, num_sites, n_mels, n_frames, alpha, seed, device):
    """MIMII 로드 → Dirichlet(normal) 분할 + machine_id 매칭 anomaly 분할 → 사이트별 payload."""
    import os
    root = os.path.expanduser(config.get("data_root", "~/abada-night/data/mimii"))
    db_level = config.get("db_level", "6dB")
    normalization_mode = config.get("normalization_mode", "central")
    feats, labels, meta = dl.load_audio_features(
        root, config["machine_type"], db_level=db_level, n_mels=n_mels, include_abnormal=True)

    site_normal_idx = dl.make_dirichlet_partition(feats, labels, meta, num_sites, alpha, seed)
    # 먼저 사이트별 train/test 분할 (정규화 누수 방지 — test/anomaly는 통계에서 제외)
    splits = [_split_site(idx, seed) for idx in site_normal_idx]
    site_mids = [set(meta[i]["machine_id"] for i in idx) for idx in site_normal_idx]
    site_anom_idx = _assign_anomalies(meta, np.where(labels == 1)[0], site_mids, seed)
    feats = _normalize_features_by_mode(
        feats,
        splits,
        mode=normalization_mode,
        site_anomaly_idx=site_anom_idx,
        fallback_idx=np.where(labels == 0)[0],
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
