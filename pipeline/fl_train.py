# P1 비IID 연합학습 이상탐지 훈련 코어 — 합성곱 오토인코더 + 4개 FL 집계 알고리즘
"""
Federated anomaly detection training for P1.

Algorithms:
  fedavg      — McMahan et al. 2017 (cite: Communication-Efficient Learning of
                Deep Networks from Decentralized Data, AISTATS 2017)
  fedprox     — Li et al. 2020 (cite: Federated Optimization in Heterogeneous
                Networks, MLSys 2020)
  clustered_fl — Unsupervised cluster-then-aggregate (cite: verify — inspired
                 by IFCA Ghosh et al. 2020; Fan et al. TPDS 2024 ClusterFLADS)
  personalized — FedAvg global + per-site local fine-tuning head
                 (cite: verify — pFedAvg / Fallah et al. 2020 Per-FedAvg)
  centralized_pooled — non-federated pooled-training reference anchor
  local_only         — non-federated site-isolated training reference anchor
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import p1_evidence_logger as evidence_logger


def roc_auc_score(labels, scores) -> float:
    """numpy만으로 AUROC 계산 (sklearn↔numpy2 ABI 충돌 회피). 랭크 기반 Mann-Whitney U."""
    y = np.asarray(labels).astype(np.int64)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # 동점 보정: 평균 랭크
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    avg = {i: (cum[i] - counts[i] + 1 + cum[i]) / 2.0 for i in range(len(counts))}
    ranks = np.array([avg[i] for i in inv])
    sum_pos = ranks[y == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _binary_pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute PR-AUC with a rank-sorted step curve."""
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, 1)
    recall = np.concatenate([[0.0], recall.astype(np.float64)])
    precision = np.concatenate([[1.0], precision.astype(np.float64)])
    return float(np.trapz(precision, recall))


def _partial_roc_auc(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.1) -> float:
    """Compute ROC-AUC truncated at max_fpr with linear interpolation."""
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1).astype(np.float64)
    fp = np.cumsum(y_sorted == 0).astype(np.float64)
    tpr = np.concatenate([[0.0], tp / n_pos])
    fpr = np.concatenate([[0.0], fp / n_neg])
    if max_fpr <= 0:
        return 0.0
    if fpr[-1] < max_fpr:
        return float(np.trapz(tpr, fpr) / max_fpr)
    idx = np.searchsorted(fpr, max_fpr, side="right")
    fpr_cut = np.concatenate([fpr[:idx], [max_fpr]])
    tpr_cut = np.concatenate([
        tpr[:idx],
        [np.interp(max_fpr, fpr[max(0, idx - 1):idx + 1], tpr[max(0, idx - 1):idx + 1])],
    ])
    return float(np.trapz(tpr_cut, fpr_cut) / max_fpr)


def _f1_from_scores(normal_scores: np.ndarray, anomaly_scores: np.ndarray, threshold: float) -> float:
    labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))]).astype(int)
    preds = np.concatenate([normal_scores > threshold, anomaly_scores > threshold]).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def _adjusted_rand_index(labels_a: list[int], labels_b: list[int]) -> float:
    """Adjusted Rand Index without sklearn."""
    if len(labels_a) != len(labels_b):
        raise ValueError("cluster label vectors must have equal length")
    n = len(labels_a)
    if n < 2:
        return 1.0
    from collections import Counter

    def comb2(x: int) -> float:
        return x * (x - 1) / 2.0

    contingency: dict[tuple[int, int], int] = {}
    row = Counter(labels_a)
    col = Counter(labels_b)
    for a, b in zip(labels_a, labels_b):
        contingency[(a, b)] = contingency.get((a, b), 0) + 1
    sum_comb = sum(comb2(v) for v in contingency.values())
    sum_row = sum(comb2(v) for v in row.values())
    sum_col = sum(comb2(v) for v in col.values())
    total = comb2(n)
    if total == 0:
        return 1.0
    expected = (sum_row * sum_col) / total
    max_index = 0.5 * (sum_row + sum_col)
    denom = max_index - expected
    if denom == 0:
        return 1.0
    return float((sum_comb - expected) / denom)


def _stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 1. MODEL
# ---------------------------------------------------------------------------

class ConvAutoencoder(nn.Module):
    """Log-mel spectrogram용 소형 합성곱 오토인코더.

    Input shape: (B, 1, n_mels, n_frames)
    Default: n_mels=128, n_frames=64  → bottleneck 256-d flat vector.
    Keep tiny — runs hundreds of times across 20 GPUs.
    """

    def __init__(self, n_mels: int = 128, n_frames: int = 320, bottleneck: int = 1024) -> None:
        super().__init__()
        # Encoder: 128x64 → 32x16 (2 stride-2 convs)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),   # 64x32
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 32x16
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 8, kernel_size=1),                        # channel squeeze
            nn.ReLU(inplace=True),
        )
        # Flatten → bottleneck
        enc_h, enc_w = n_mels // 4, n_frames // 4
        enc_flat = 8 * enc_h * enc_w
        self.fc_enc = nn.Linear(enc_flat, bottleneck)
        self.fc_dec = nn.Linear(bottleneck, enc_flat)
        self._enc_shape = (8, enc_h, enc_w)

        # Decoder: mirrors encoder with transposed convs
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (reconstruction, bottleneck_code)."""
        h = self.encoder(x)
        flat = h.flatten(1)
        code = self.fc_enc(flat)
        dec_flat = self.fc_dec(F.relu(code))
        h2 = dec_flat.view(-1, *self._enc_shape)
        recon = self.decoder(h2)
        return recon, code

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction error (no grad)."""
        with torch.no_grad():
            recon, _ = self.forward(x)
            err = F.mse_loss(recon, x, reduction="none").mean(dim=(1, 2, 3))
        return err


class LiteConvAutoencoder(nn.Module):
    """Smaller convolutional autoencoder for reviewer-risk backbone sensitivity.

    This intentionally changes channel widths and bottleneck capacity while
    preserving the same input/output contract as ConvAutoencoder. It is not a
    new proposed method; it is a compact sanity check for whether the P1
    condition-vs-algorithm conclusion is tied to one capacity point.
    """

    def __init__(self, n_mels: int = 128, n_frames: int = 320, bottleneck: int = 256) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 4, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        enc_h, enc_w = n_mels // 4, n_frames // 4
        enc_flat = 4 * enc_h * enc_w
        self.fc_enc = nn.Linear(enc_flat, bottleneck)
        self.fc_dec = nn.Linear(bottleneck, enc_flat)
        self._enc_shape = (4, enc_h, enc_w)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        flat = h.flatten(1)
        code = self.fc_enc(flat)
        dec_flat = self.fc_dec(F.relu(code))
        h2 = dec_flat.view(-1, *self._enc_shape)
        recon = self.decoder(h2)
        return recon, code

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            recon, _ = self.forward(x)
            err = F.mse_loss(recon, x, reduction="none").mean(dim=(1, 2, 3))
        return err


def _make_model(
    model_family: str,
    n_mels: int,
    n_frames: int,
    bottleneck: int,
) -> nn.Module:
    family = str(model_family or "ConvAutoencoder").lower().replace("-", "_")
    if family in {"convautoencoder", "conv_autoencoder", "cae", "default"}:
        return ConvAutoencoder(n_mels=n_mels, n_frames=n_frames, bottleneck=bottleneck)
    if family in {"liteconvautoencoder", "lite_conv_autoencoder", "conv_autoencoder_lite", "lite"}:
        return LiteConvAutoencoder(n_mels=n_mels, n_frames=n_frames, bottleneck=bottleneck)
    raise ValueError(f"Unknown model_family: {model_family!r}")


# ---------------------------------------------------------------------------
# 2. FL AGGREGATION ALGORITHMS
# ---------------------------------------------------------------------------

def _weighted_average(state_dicts: list[dict], weights: list[float]) -> dict:
    """Immutably produce a new state_dict as a weighted average."""
    total = sum(weights)
    if total == 0:  # 전 사이트 0샘플 → 균등 평균 (0 나눗셈 방지)
        weights = [1.0] * len(state_dicts)
        total = float(len(state_dicts))
    result: dict = {}
    for key in state_dicts[0]:
        stacked = torch.stack(
            [sd[key].float() * (w / total) for sd, w in zip(state_dicts, weights)]
        )
        result[key] = stacked.sum(dim=0).to(state_dicts[0][key].dtype)
    return result


def fedavg(state_dicts: list[dict], sample_counts: list[int]) -> dict:
    """FedAvg: weighted average by site sample count.

    McMahan et al. 2017 — Communication-Efficient Learning of Deep Networks
    from Decentralized Data, AISTATS 2017.
    """
    return _weighted_average(state_dicts, [float(n) for n in sample_counts])


def fedprox(state_dicts: list[dict], sample_counts: list[int]) -> dict:
    """FedProx server-side aggregation is identical to FedAvg.

    The proximal term mu is applied during LOCAL training (see _local_train).
    Li et al. 2020 — Federated Optimization in Heterogeneous Networks, MLSys 2020.
    """
    return _weighted_average(state_dicts, [float(n) for n in sample_counts])


def clustered_fl(
    state_dicts: list[dict],
    sample_counts: list[int],
    n_clusters: int = 2,
    seed: int = 0,
) -> tuple[list[dict], list[int]]:
    """Cluster sites by model-parameter similarity, aggregate within clusters.

    Returns (cluster_models, cluster_ids) — cluster_ids[i]는 site i의 클러스터.
    호출부는 같은 cluster_ids로 사이트를 배정해야 함(이중 k-means 금지).
    Unsupervised — no labels required (cite: verify — inspired by IFCA
    Ghosh et al. 2020; Fan et al. TPDS 2024 ClusterFLADS).
    """
    # Flatten each state_dict to a 1-D feature vector for clustering
    vecs = []
    for sd in state_dicts:
        parts = [v.float().flatten() for v in sd.values()]
        vecs.append(torch.cat(parts))
    mat = torch.stack(vecs)  # (K, D)

    # k-means 단 1회 — seed 일관 (호출부와 동일 클러스터링 보장)
    cluster_ids = _torch_kmeans(mat, n_clusters, seed=seed)

    # Aggregate within each cluster
    cluster_models: list[dict] = []
    for cid in range(n_clusters):
        idxs = [i for i, c in enumerate(cluster_ids) if c == cid]
        if not idxs:          # empty cluster: fall back to global average
            idxs = list(range(len(state_dicts)))
        c_sds = [state_dicts[i] for i in idxs]
        c_ns  = [sample_counts[i] for i in idxs]
        cluster_models.append(_weighted_average(c_sds, [float(n) for n in c_ns]))
    return cluster_models, cluster_ids


def personalized(
    global_sd: dict,
    local_sd: dict,
    mix_alpha: float = 0.2,
) -> dict:
    """Blend global and local state_dicts for per-site personalization.

    Returns a new state_dict = (1-mix_alpha)*global + mix_alpha*local.
    cite: verify — pFedAvg / Fallah et al. 2020 Per-FedAvg (MAML-based);
    interpolation variant as in Yu et al. 2020.
    """
    result: dict = {}
    for key in global_sd:
        g = global_sd[key].float()
        loc = local_sd[key].float()
        blended = (1.0 - mix_alpha) * g + mix_alpha * loc
        result[key] = blended.to(global_sd[key].dtype)
    return result


# ---------------------------------------------------------------------------
# 3. UTILITY: TORCH K-MEANS (pure torch, immutable)
# ---------------------------------------------------------------------------

def _torch_kmeans(
    mat: torch.Tensor,
    k: int,
    n_iter: int = 20,
    seed: int = 0,
) -> list[int]:
    """Simple Lloyd's k-means on rows of mat. Returns cluster id per row."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    n = mat.shape[0]
    k = min(k, n)
    # Random init: pick k rows without replacement
    perm = torch.randperm(n, generator=rng)[:k]
    centroids = mat[perm].clone()

    ids = [0] * n
    for _ in range(n_iter):
        dists = torch.cdist(mat, centroids)       # (n, k)
        ids = dists.argmin(dim=1).tolist()
        new_centroids = []
        for c in range(k):
            members = [i for i, cid in enumerate(ids) if cid == c]
            if members:
                new_centroids.append(mat[members].mean(dim=0))
            else:
                new_centroids.append(centroids[c])
        centroids = torch.stack(new_centroids)
    return ids


# ---------------------------------------------------------------------------
# 4. LOCAL TRAINING
# ---------------------------------------------------------------------------

def _local_train(
    model: nn.Module,
    data: torch.Tensor,
    epochs: int,
    device: torch.device,
    lr: float = 1e-3,
    mu: float = 0.0,          # FedProx proximal term coefficient
    global_params: list | None = None,
) -> tuple[nn.Module, float, int]:
    """Train model on normal-only data; return (trained_model, avg_loss, n_samples).

    mu > 0 activates FedProx proximal regularisation:
      loss += (mu/2) * ||w - w_global||^2
    """
    model = model.to(device)
    # 극단 non-IID에서 샘플 0개 사이트: 학습 없이 모델 그대로 반환(n_samples=0 → fedavg 가중 0으로 제외)
    if data.shape[0] == 0:
        return model.to(device), 0.0, 0
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = torch.utils.data.TensorDataset(data.to(device))
    loader  = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # FedProx: global 파라미터를 device로 1회 이동+detach (CPU↔CUDA 불일치 방지)
    prox_params = ([g.detach().to(device) for g in global_params]
                   if mu > 0.0 and global_params is not None else None)

    total_loss = 0.0
    for _ in range(epochs):
        for (batch,) in loader:
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = F.mse_loss(recon, batch)
            if prox_params is not None:
                prox = sum(
                    ((p - g) ** 2).sum()
                    for p, g in zip(model.parameters(), prox_params)
                )
                loss = loss + (mu / 2.0) * prox
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)

    return model, total_loss / max(len(data), 1), len(data)


# ---------------------------------------------------------------------------
# 5. SYNTHETIC DATA (for --selftest and when real data is absent)
# ---------------------------------------------------------------------------

def _make_synthetic_site_data(
    n_normal: int,
    n_mels: int,
    n_frames: int,
    alpha: float,
    site_idx: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (normal_train, normal_test, anomaly_test) tensors.

    Anomalies have slightly higher energy in a frequency band controlled by
    site_idx — simulates non-IID distribution across sites.
    """
    rng = np.random.default_rng(seed + site_idx * 1000)
    # Non-IID: each site has a dominant frequency band
    band_lo = int((site_idx % 4) / 4 * n_mels)
    band_hi = band_lo + n_mels // 4

    def _make(n: int, anomaly: bool) -> np.ndarray:
        base = rng.standard_normal((n, 1, n_mels, n_frames)).astype(np.float32)
        if anomaly:
            # Inject energy scaled by 1/alpha (stronger anomaly for smaller alpha)
            scale = 1.0 + 1.0 / max(alpha, 0.01)
            base[:, :, band_lo:band_hi, :] += rng.uniform(
                scale * 0.5, scale, size=(n, 1, band_hi - band_lo, n_frames)
            ).astype(np.float32)
        return base

    normal_train = torch.from_numpy(_make(n_normal, anomaly=False)).to(device)
    normal_test  = torch.from_numpy(_make(n_normal // 4, anomaly=False)).to(device)
    anomaly_test = torch.from_numpy(_make(n_normal // 4, anomaly=True)).to(device)
    return normal_train, normal_test, anomaly_test


def _make_synthetic_site_payloads(
    n_normal: int,
    n_mels: int,
    n_frames: int,
    alpha: float,
    num_sites: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, object]]:
    """Return synthetic site payloads with metadata for evidence logging."""
    payloads: list[dict[str, object]] = []
    for site_idx in range(num_sites):
        normal_train, normal_test, anomaly_test = _make_synthetic_site_data(
            n_normal, n_mels, n_frames, alpha, site_idx, seed, device
        )
        payloads.append(
            {
                "site_idx": site_idx,
                "normal_train": normal_train,
                "normal_test": normal_test,
                "anomaly_test": anomaly_test,
                "train_meta": [
                    {
                        "machine_type": "synthetic",
                        "machine_id": f"site_{site_idx:02d}",
                        "db_level": "synthetic",
                        "wav_path": f"synthetic/site_{site_idx:02d}/normal/train_{i:05d}.npy",
                    }
                    for i in range(int(normal_train.shape[0]))
                ],
                "test_meta": [
                    {
                        "machine_type": "synthetic",
                        "machine_id": f"site_{site_idx:02d}",
                        "db_level": "synthetic",
                        "wav_path": f"synthetic/site_{site_idx:02d}/normal/test_{i:05d}.npy",
                    }
                    for i in range(int(normal_test.shape[0]))
                ],
                "anomaly_meta": [
                    {
                        "machine_type": "synthetic",
                        "machine_id": f"site_{site_idx:02d}",
                        "db_level": "synthetic",
                        "wav_path": f"synthetic/site_{site_idx:02d}/abnormal/test_{i:05d}.npy",
                    }
                    for i in range(int(anomaly_test.shape[0]))
                ],
                "site_machine_ids": [f"site_{site_idx:02d}"],
                "site_machine_id_joined": f"site_{site_idx:02d}",
            }
        )
    return payloads


# ---------------------------------------------------------------------------
# 6. AUROC + COMMUNICATION COST
# ---------------------------------------------------------------------------

def _compute_auroc(
    model: ConvAutoencoder,
    normal_test: torch.Tensor,
    anomaly_test: torch.Tensor,
    device: torch.device,
) -> float:
    """Compute AUROC from reconstruction errors on test sets."""
    model = model.to(device)
    model.eval()
    errs_n = model.reconstruction_error(normal_test.to(device)).cpu().numpy()
    errs_a = model.reconstruction_error(anomaly_test.to(device)).cpu().numpy()
    scores = np.concatenate([errs_n, errs_a])
    labels = np.concatenate([np.zeros(len(errs_n)), np.ones(len(errs_a))])
    if labels.std() == 0:
        return 0.5
    return float(roc_auc_score(labels, scores))


def _state_dict_mb(sd: dict) -> float:
    """Approximate size of a state_dict in MB (float32 baseline)."""
    total = sum(v.numel() for v in sd.values())
    return total * 4 / (1024 ** 2)


def _compute_f1(
    model: ConvAutoencoder,
    normal_test: torch.Tensor,
    anomaly_test: torch.Tensor,
    device: torch.device,
) -> float:
    """F1 at the 95th-percentile normal reconstruction error threshold."""
    model = model.to(device)
    model.eval()
    errs_n = model.reconstruction_error(normal_test.to(device)).cpu().numpy()
    errs_a = model.reconstruction_error(anomaly_test.to(device)).cpu().numpy()
    threshold = float(np.percentile(errs_n, 95))
    preds = np.concatenate(
        [errs_n > threshold, errs_a > threshold]
    ).astype(int)
    labels = np.concatenate([np.zeros(len(errs_n)), np.ones(len(errs_a))]).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# 7. MAIN FL TRAINING LOOP
# ---------------------------------------------------------------------------

def train_federated(config: dict[str, Any]) -> dict[str, Any]:
    """Run FL experiment; return metrics dict.

    Config keys (required):
      algorithm   : str  — fedavg | fedprox | clustered_fl | personalized |
                           centralized_pooled | local_only
      alpha       : float — Dirichlet concentration (non-IID severity)
      num_sites   : int
      rounds      : int
      local_epochs: int
      seed        : int
      machine_type: str  — fan | pump | slider | valve | synthetic

    Config keys (optional):
      n_mels       : int  (default 128)
      n_frames     : int  (default 64)
      bottleneck   : int  (default 256)
      model_family : str  (default ConvAutoencoder; optional LiteConvAutoencoder)
      lr           : float (default 1e-3)
      fedprox_mu   : float (default 0.01)
      mix_alpha    : float (default 0.2, personalized blend)
      n_clusters   : int  (default 2, clustered_fl)
      n_normal     : int  (default 200, samples per site for synthetic data)
      normalization_mode: str (central | federated_train | local_site | none)
      wandb_project: str  (default 'abada-night')
      use_wandb    : bool (default True if wandb available)
    """
    # -- Seed
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # -- Device (worker pins via CUDA_VISIBLE_DEVICES before launching)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # -- Hyperparameters
    algorithm    = config["algorithm"]
    alpha        = float(config["alpha"])
    num_sites    = int(config["num_sites"])
    rounds       = int(config["rounds"])
    local_epochs = int(config["local_epochs"])
    n_mels       = int(config.get("n_mels", 128))
    n_frames     = int(config.get("n_frames", 320))
    bottleneck   = int(config.get("bottleneck", 1024))
    model_family = str(config.get("model_family", "ConvAutoencoder"))
    lr           = float(config.get("lr", 1e-3))
    mu           = float(config.get("fedprox_mu", 0.01))
    mix_alpha    = float(config.get("mix_alpha", 0.2))
    n_clusters   = int(config.get("n_clusters", 2))
    n_normal     = int(config.get("n_normal", 200))
    normalization_mode = config.get("normalization_mode", "central")
    machine_type = config.get("machine_type", "synthetic")
    run_name     = config.get("name", f"p1_{algorithm}_a{alpha}_s{seed}")
    log_evidence = bool(config.get("log_evidence", False) or os.environ.get("P1_EVIDENCE_LOGGING") == "1")
    evidence_output_dir = config.get("evidence_output_dir")

    # -- W&B (optional)
    wb_run = _init_wandb(config, run_name)

    # -- Per-site data: real MIMII via adapter, else synthetic fallback
    if machine_type == "synthetic":
        site_payloads = _make_synthetic_site_payloads(
            n_normal, n_mels, n_frames, alpha, num_sites, seed, device
        )
    else:
        import mimii_adapter
        site_payloads = mimii_adapter.make_mimii_site_data(
            config, num_sites, n_mels, n_frames, alpha, seed, device
        )
    site_data = [
        (payload["normal_train"], payload["normal_test"], payload["anomaly_test"])
        for payload in site_payloads
    ]

    # -- Global model initialisation
    global_model = _make_model(model_family, n_mels, n_frames, bottleneck)
    global_sd = copy.deepcopy(global_model.state_dict())
    model_mb = _state_dict_mb(global_sd)

    # Personalised FL: keep per-site local models
    site_models: list[nn.Module | None] = [None] * num_sites

    comm_up_mb   = 0.0   # client→server upload MB
    comm_down_mb = 0.0   # server→client download MB
    round_losses: list[float] = []
    cluster_history: list[dict[str, object]] = []

    if algorithm in ("centralized_pooled", "local_only"):
        equivalent_epochs = rounds * local_epochs
        if algorithm == "centralized_pooled":
            pooled_train = torch.cat([payload["normal_train"] for payload in site_payloads], dim=0)
            trained, loss, _ = _local_train(
                global_model, pooled_train, equivalent_epochs, device, lr=lr
            )
            global_model = trained.to(device)
            round_losses.append(loss)
        else:
            for site_idx, payload in enumerate(site_payloads):
                local_model = _make_model(model_family, n_mels, n_frames, bottleneck)
                trained, loss, _ = _local_train(
                    local_model,
                    payload["normal_train"],  # type: ignore[index]
                    equivalent_epochs,
                    device,
                    lr=lr,
                )
                site_models[site_idx] = trained.to(device)
                round_losses.append(loss)

        per_site_auroc: list[float] = []
        site_f1: list[float] = []
        for site_idx, payload in enumerate(site_payloads):
            normal_test = payload["normal_test"]  # type: ignore[index]
            anomaly_test = payload["anomaly_test"]  # type: ignore[index]
            if normal_test.shape[0] == 0 or anomaly_test.shape[0] == 0:
                continue
            eval_model = (
                site_models[site_idx].to(device)  # type: ignore[union-attr]
                if algorithm == "local_only" and site_models[site_idx] is not None
                else global_model.to(device)
            )
            auroc = _compute_auroc(eval_model, normal_test, anomaly_test, device)
            f1 = _compute_f1(eval_model, normal_test, anomaly_test, device)
            per_site_auroc.append(float(auroc))
            site_f1.append(float(f1))

        metrics = {
            "auroc": float(np.mean(per_site_auroc)) if per_site_auroc else float("nan"),
            "f1": float(np.mean(site_f1)) if site_f1 else float("nan"),
            "final_round": rounds,
            "equivalent_epochs": equivalent_epochs,
            "communication_cost_mb": 0.0,
            "per_site_auroc": per_site_auroc,
            "algorithm": algorithm,
            "alpha": alpha,
            "num_sites": num_sites,
            "normalization_mode": normalization_mode,
            "model_family": model_family,
            "baseline_training_mode": algorithm,
        }
        if wb_run is not None:
            wb_run.log(metrics)
            wb_run.finish()
        return metrics

    for rnd in range(1, rounds + 1):
        # -- Local training
        site_sds: list[dict]  = []
        site_ns:  list[int]   = []
        round_loss = 0.0

        global_params = list(global_model.parameters()) if algorithm == "fedprox" else None

        for site_idx in range(num_sites):
            normal_train, _, _ = site_data[site_idx]

            # Choose starting weights
            if algorithm == "personalized" and site_models[site_idx] is not None:
                start_sd = site_models[site_idx].state_dict()  # type: ignore[union-attr]
            else:
                start_sd = copy.deepcopy(global_sd)

            site_model = _make_model(model_family, n_mels, n_frames, bottleneck)
            site_model.load_state_dict(start_sd)
            trained, loss, n_samples = _local_train(
                site_model, normal_train, local_epochs, device,
                lr=lr, mu=(mu if algorithm == "fedprox" else 0.0),
                global_params=global_params,
            )
            site_sds.append(trained.state_dict())
            site_ns.append(n_samples)
            round_loss += loss
            comm_up_mb += model_mb

        avg_round_loss = round_loss / num_sites

        # -- Server aggregation (immutable: returns new state_dicts)
        if algorithm in ("fedavg", "fedprox"):
            global_sd = fedavg(site_sds, site_ns) if algorithm == "fedavg" \
                        else fedprox(site_sds, site_ns)
            global_model.load_state_dict(global_sd)
            comm_down_mb += model_mb * num_sites

        elif algorithm == "clustered_fl":
            # k-means 단 1회 — 집계와 배정이 동일 클러스터링 사용 (seed 일관)
            cluster_sds, cluster_ids = clustered_fl(
                site_sds, site_ns, n_clusters=n_clusters, seed=seed)
            for site_idx, cid in enumerate(cluster_ids):
                site_models[site_idx] = _make_model(model_family, n_mels, n_frames, bottleneck)
                site_models[site_idx].load_state_dict(cluster_sds[cid])  # type: ignore[union-attr]
            # Global = mean of cluster models (for logging / AUROC reference)
            global_sd = _weighted_average(cluster_sds, [1.0] * len(cluster_sds))
            global_model.load_state_dict(global_sd)
            comm_down_mb += model_mb * num_sites
            cluster_history.append(
                {
                    "round": rnd,
                    "cluster_ids": list(cluster_ids),
                    "site_ns": list(site_ns),
                }
            )

        elif algorithm == "personalized":
            # Global update via FedAvg; then blend per-site
            new_global_sd = fedavg(site_sds, site_ns)
            global_model.load_state_dict(new_global_sd)
            global_sd = new_global_sd
            for site_idx in range(num_sites):
                pers_sd = personalized(global_sd, site_sds[site_idx], mix_alpha)
                m = _make_model(model_family, n_mels, n_frames, bottleneck)
                m.load_state_dict(pers_sd)
                site_models[site_idx] = m
            comm_down_mb += model_mb * num_sites
        else:
            raise ValueError(f"Unknown algorithm: {algorithm!r}")

        round_losses.append(avg_round_loss)
        if wb_run is not None:
            wb_run.log({"round": rnd, "train_loss": avg_round_loss})

    # -- Final evaluation (빈 test 사이트는 제외 — nan 방지)
    per_site_auroc: list[float] = []
    site_site_rows: list[dict[str, object]] = []
    reconstruction_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []

    for site_idx in range(num_sites):
        payload = site_payloads[site_idx]
        _, normal_test, anomaly_test = site_data[site_idx]
        if normal_test.shape[0] == 0 or anomaly_test.shape[0] == 0:
            continue

        eval_model = _resolve_eval_model(
            algorithm, site_idx, site_models, global_model,
            n_mels, n_frames, bottleneck, model_family, global_sd, device,
        )
        normal_scores = eval_model.reconstruction_error(normal_test.to(device)).cpu().numpy()
        anomaly_scores = eval_model.reconstruction_error(anomaly_test.to(device)).cpu().numpy()
        scores = np.concatenate([normal_scores, anomaly_scores])
        labels = np.concatenate([
            np.zeros(len(normal_scores), dtype=int),
            np.ones(len(anomaly_scores), dtype=int),
        ])
        threshold = float(np.percentile(normal_scores, 95)) if len(normal_scores) else float("nan")
        auroc = float(roc_auc_score(labels, scores))
        auprc = float(_binary_pr_auc(labels, scores))
        pauc = float(_partial_roc_auc(labels, scores, max_fpr=0.1))
        f1_score = float(_f1_from_scores(normal_scores, anomaly_scores, threshold))
        per_site_auroc.append(auroc)

        site_machine_id = payload.get("site_machine_id_joined", "")
        site_site_rows.append(
            {
                "schema_version": "p1-evidence-logging-v1",
                "run_id": run_name,
                "paper_id": "P1",
                "experiment_id": run_name,
                "seed": seed,
                "round": rounds,
                "site_id": site_idx,
                "machine_id": site_machine_id,
                "normalization": normalization_mode,
                "model_family": model_family,
                "n_normal": int(len(normal_scores)),
                "n_anomaly": int(len(anomaly_scores)),
                "auroc": auroc,
                "auprc": auprc,
                "partial_auroc_fpr_0_1": pauc,
                "f1_at_threshold": f1_score,
                "threshold": threshold,
            }
        )

        for sample_idx, err in enumerate(normal_scores):
            meta = payload["test_meta"][sample_idx]
            sample_path = str(meta.get("wav_path", f"{run_name}/site{site_idx}/normal/{sample_idx}"))
            reconstruction_rows.append(
                {
                    "schema_version": "p1-evidence-logging-v1",
                    "run_id": run_name,
                    "paper_id": "P1",
                    "experiment_id": run_name,
                    "seed": seed,
                    "round": rounds,
                    "site_id": site_idx,
                    "machine_id": meta.get("machine_id", site_machine_id),
                    "split": "normal_test",
                    "sample_id": sample_path,
                    "sample_path_hash": evidence_logger.stable_path_hash(sample_path),
                    "label": 0,
                    "anomaly_type": "normal",
                    "normalization": normalization_mode,
                    "model_family": model_family,
                    "reconstruction_error": float(err),
                    "threshold": threshold,
                    "predicted_anomaly": int(err > threshold),
                }
            )

        for sample_idx, err in enumerate(anomaly_scores):
            meta = payload["anomaly_meta"][sample_idx]
            sample_path = str(meta.get("wav_path", f"{run_name}/site{site_idx}/anomaly/{sample_idx}"))
            reconstruction_rows.append(
                {
                    "schema_version": "p1-evidence-logging-v1",
                    "run_id": run_name,
                    "paper_id": "P1",
                    "experiment_id": run_name,
                    "seed": seed,
                    "round": rounds,
                    "site_id": site_idx,
                    "machine_id": meta.get("machine_id", site_machine_id),
                    "split": "anomaly_test",
                    "sample_id": sample_path,
                    "sample_path_hash": evidence_logger.stable_path_hash(sample_path),
                    "label": 1,
                    "anomaly_type": str(meta.get("anomaly_type", "unknown")),
                    "normalization": normalization_mode,
                    "model_family": model_family,
                    "reconstruction_error": float(err),
                    "threshold": threshold,
                    "predicted_anomaly": int(err > threshold),
                }
            )

    if algorithm == "clustered_fl" and cluster_history:
        prev_ids: list[int] | None = None
        prev_round: int | None = None
        for record in cluster_history:
            rnd = int(record["round"])
            cluster_ids = [int(x) for x in record["cluster_ids"]]
            site_ns = [int(x) for x in record["site_ns"]]
            for site_idx, cid in enumerate(cluster_ids):
                payload = site_payloads[site_idx]
                cluster_rows.append(
                    {
                        "schema_version": "p1-evidence-logging-v1",
                        "run_id": run_name,
                        "paper_id": "P1",
                        "experiment_id": run_name,
                        "seed": seed,
                        "round": rnd,
                        "site_id": site_idx,
                        "machine_id": payload.get("site_machine_id_joined", ""),
                        "cluster_id": cid,
                        "cluster_method": f"state_dict_kmeans_k{n_clusters}",
                        "feature_space": "state_dict",
                        "n_samples": site_ns[site_idx] if site_idx < len(site_ns) else 0,
                        "normalization": normalization_mode,
                    }
                )
            if prev_ids is not None and prev_round is not None:
                ari = _adjusted_rand_index(prev_ids, cluster_ids)
                stability_rows.append(
                    {
                        "schema_version": "p1-evidence-logging-v1",
                        "run_id": run_name,
                        "paper_id": "P1",
                        "experiment_id": run_name,
                        "seed": seed,
                        "round": rnd,
                        "site_id": "all",
                        "cluster_method": f"state_dict_kmeans_k{n_clusters}",
                        "feature_space": "state_dict",
                        "stability_metric": "ari",
                        "stability_value": ari,
                        "baseline_round": prev_round,
                        "comparison_round": rnd,
                    }
                )
                for site_idx, (prev_id, cur_id) in enumerate(zip(prev_ids, cluster_ids)):
                    stability_rows.append(
                        {
                            "schema_version": "p1-evidence-logging-v1",
                            "run_id": run_name,
                            "paper_id": "P1",
                            "experiment_id": run_name,
                            "seed": seed,
                            "round": rnd,
                            "site_id": site_idx,
                            "cluster_method": f"state_dict_kmeans_k{n_clusters}",
                            "feature_space": "state_dict",
                            "stability_metric": "same_cluster_with_prev_round",
                            "stability_value": 1.0 if prev_id == cur_id else 0.0,
                            "baseline_round": prev_round,
                            "comparison_round": rnd,
                        }
                    )
            prev_ids = cluster_ids
            prev_round = rnd

    mean_auroc = float(np.mean(per_site_auroc)) if per_site_auroc else float("nan")
    f1_score = site_site_rows[0]["f1_at_threshold"] if site_site_rows else float("nan")

    metrics = {
        "auroc":                mean_auroc,
        "f1":                   f1_score,
        "final_round":          rounds,
        "communication_cost_mb": round(comm_up_mb + comm_down_mb, 3),
        "per_site_auroc":       per_site_auroc,
        "algorithm":            algorithm,
        "alpha":                alpha,
        "num_sites":            num_sites,
        "normalization_mode":    normalization_mode,
        "model_family":          model_family,
    }

    if log_evidence:
        evidence_root = evidence_logger.evidence_root_for_run(
            run_name, output_dir=evidence_output_dir
        )
        evidence_logger.write_bundle(
            evidence_root,
            {
                "reconstruction_errors": reconstruction_rows,
                "site_auroc": site_site_rows,
                "cluster_assignments": cluster_rows,
                "cluster_stability": stability_rows,
            },
            manifest={
                "run_name": run_name,
                "algorithm": algorithm,
                "machine_type": machine_type,
                "seed": seed,
                "rounds": rounds,
                "num_sites": num_sites,
                "normalization_mode": normalization_mode,
                "model_family": model_family,
            },
        )

    if wb_run is not None:
        wb_run.log(metrics)
        wb_run.finish()

    return metrics


def _resolve_eval_model(
    algorithm: str,
    site_idx: int,
    site_models: list,
    global_model: ConvAutoencoder,
    n_mels: int,
    n_frames: int,
    bottleneck: int,
    model_family: str,
    global_sd: dict,
    device: torch.device,
) -> nn.Module:
    """Return the model to use for evaluation at a given site."""
    if algorithm in ("clustered_fl", "personalized") and site_models[site_idx] is not None:
        return site_models[site_idx].to(device)  # type: ignore[union-attr, return-value]
    return global_model.to(device)


# ---------------------------------------------------------------------------
# 8. W&B HELPER
# ---------------------------------------------------------------------------

def _init_wandb(config: dict, run_name: str):
    """Initialise W&B run; return run object or None if unavailable."""
    use_wandb = config.get("use_wandb", True)
    if not use_wandb:
        return None
    try:
        import wandb  # noqa: PLC0415
        tags = [
            config.get("algorithm", "unknown"),
            f"alpha{config.get('alpha', '?')}",
            f"norm:{config.get('normalization_mode', 'central')}",
            f"model:{config.get('model_family', 'ConvAutoencoder')}",
        ]
        tags.extend(config.get("wandb_tags", []))
        return wandb.init(
            project=config.get("wandb_project", "abada-night"),
            name=run_name,
            config=config,
            group=config.get("wandb_group"),
            reinit=True,
            tags=sorted(set(tags)),
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 9. CONFIG-FILE ENTRY POINT (called by worker.py)
# ---------------------------------------------------------------------------

def run_from_config(path: str) -> dict[str, Any]:
    """Load a JSON config file and run train_federated.

    Designed for atomic-claim dispatch from ~/abada-night/worker.py.
    """
    with open(path, encoding="utf-8") as fh:
        config = json.load(fh)
    return train_federated(config)


# ---------------------------------------------------------------------------
# 10. SELF-TEST
# ---------------------------------------------------------------------------

def _selftest() -> None:
    """Run all 4 algorithms on synthetic data, CPU, no W&B. Prints AUROC."""
    print("=== P1 fl_train.py self-test ===")
    print(f"torch {torch.__version__} | device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    base_config: dict[str, Any] = {
        "num_sites":   4,
        "rounds":      2,
        "local_epochs": 1,
        "seed":        42,
        "machine_type": "synthetic",
        "n_mels":      32,     # tiny for speed
        "n_frames":    16,
        "bottleneck":  32,
        "n_normal":    40,
        "lr":          1e-3,
        "alpha":       0.5,
        "use_wandb":   False,
    }

    algorithms = ["fedavg", "fedprox", "clustered_fl", "personalized"]
    all_passed = True
    for alg in algorithms:
        cfg = {**base_config, "algorithm": alg, "name": f"selftest_{alg}"}
        t0 = time.time()
        metrics = train_federated(cfg)
        elapsed = time.time() - t0
        auroc = metrics["auroc"]
        per_site = [f"{v:.4f}" for v in metrics["per_site_auroc"]]
        ok = isinstance(auroc, float) and 0.0 <= auroc <= 1.0
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_passed = False
        print(
            f"  [{status}] {alg:<16} "
            f"AUROC={auroc:.4f}  F1={metrics['f1']:.4f}  "
            f"comm={metrics['communication_cost_mb']:.2f}MB  "
            f"time={elapsed:.1f}s  per_site={per_site}"
        )

    print()
    print("All algorithms PASSED." if all_passed else "SOME ALGORITHMS FAILED.")
    sys.exit(0 if all_passed else 1)


# ---------------------------------------------------------------------------
# 11. CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P1 FL training core")
    p.add_argument("--selftest", action="store_true",
                   help="Run end-to-end self-test with synthetic data and exit.")
    p.add_argument("--config", metavar="PATH",
                   help="Path to JSON config file (calls run_from_config).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.selftest:
        _selftest()
    elif args.config:
        result = run_from_config(args.config)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: fl_train.py --selftest  |  fl_train.py --config path/to/config.json",
              file=sys.stderr)
        sys.exit(1)
