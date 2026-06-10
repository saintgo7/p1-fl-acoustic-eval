# MIMII 데이터셋을 연합학습 이상탐지 실험(P1)을 위해 로드·분할하는 모듈

"""
MIMII Dataset Structure (verified via Zenodo 3384388 + MIMII-hitachi/mimii_baseline)
=====================================================================================
Source  : https://zenodo.org/records/3384388
Paper   : Purohit et al., "MIMII Dataset", arXiv 1909.09347
Baseline: https://github.com/MIMII-hitachi/mimii_baseline

Download: one .zip per (machine_type, dB_level) combination, e.g.
  fan_6dB.zip, fan_0dB.zip, fan_-6dB.zip
  pump_6dB.zip, pump_0dB.zip, pump_-6dB.zip
  slider_6dB.zip, slider_0dB.zip, slider_-6dB.zip
  valve_6dB.zip, valve_0dB.zip, valve_-6dB.zip
  (12 zip files total for the public v1.0 release)

After extraction the expected directory layout is:

  <root>/
  └── <db_level>/          # one of: "6dB", "0dB", "-6dB"
      └── <machine_type>/  # one of: "fan", "pump", "slider", "valve"
          ├── id_00/
          │   ├── normal/      # normal WAV files (8-ch, 16 kHz, 16-bit)
          │   └── abnormal/    # anomalous WAV files
          ├── id_02/
          │   ├── normal/
          │   └── abnormal/
          ├── id_04/
          │   ├── normal/
          │   └── abnormal/
          └── id_06/
              ├── normal/
              └── abnormal/

  WAV files are named like 00000000.wav, 00000001.wav, …
  Recording: 8 microphone channels mixed to 1 ch at 16 kHz, 16-bit PCM.
  SNR variants: -6 dB (hardest), 0 dB, 6 dB (easiest).
  Machine IDs: id_00, id_02, id_04, id_06 in public release (v1.0).
  Normal sounds: ~5000–10000 s per machine-id; anomalous: ~1000 s.
  Task: train autoencoder on NORMAL only; detect anomaly via recon-error (AUROC).

  NOTE: The original zip files use the layout above.  Some community re-packs
  omit the top-level <db_level>/ directory — set `db_level=None` in that case.
"""

from __future__ import annotations

import os
import pathlib
import warnings
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Audio feature extraction
# ---------------------------------------------------------------------------
_HAVE_LIBROSA = False
_HAVE_TORCHAUDIO = False

try:
    import librosa  # type: ignore
    _HAVE_LIBROSA = True
except ImportError:
    pass

if not _HAVE_LIBROSA:
    try:
        import torchaudio  # type: ignore
        import torch  # type: ignore
        _HAVE_TORCHAUDIO = True
    except ImportError:
        pass


def _log_mel_librosa(wav_path: str, sr: int, n_mels: int,
                     n_fft: int, hop_length: int) -> np.ndarray:
    """librosa 백엔드로 log-mel 스펙트로그램 추출 → (n_mels, T) float32."""
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def _log_mel_torchaudio(wav_path: str, sr: int, n_mels: int,
                         n_fft: int, hop_length: int) -> np.ndarray:
    """torchaudio 백엔드로 log-mel 스펙트로그램 추출 → (n_mels, T) float32."""
    waveform, file_sr = torchaudio.load(wav_path)
    if file_sr != sr:
        resampler = torchaudio.transforms.Resample(file_sr, sr)
        waveform = resampler(waveform)
    mono = waveform.mean(dim=0, keepdim=True)  # (1, T)
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel = transform(mono).squeeze(0)  # (n_mels, T)
    log_mel = (mel + 1e-9).log2()
    return log_mel.numpy().astype(np.float32)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """표준 mel 필터뱅크 (n_mels, n_fft//2+1). stdlib+numpy만."""
    def hz2mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel2hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    m_pts = np.linspace(hz2mel(0), hz2mel(sr / 2), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel2hz(m_pts) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        for k in range(l, c):
            if c > l: fb[i - 1, k] = (k - l) / (c - l)
        for k in range(c, r):
            if r > c: fb[i - 1, k] = (r - k) / (r - c)
    return fb


def _log_mel_wave_stdlib(wav_path: str, sr: int, n_mels: int,
                         n_fft: int, hop_length: int) -> np.ndarray:
    """RIFF 직접 파싱 + numpy FFT로 log-mel 추출 (WAVE_FORMAT_EXTENSIBLE 65534 지원)."""
    return _log_mel_from_raw(_read_wav_pcm16(wav_path), sr, n_mels, n_fft, hop_length)


def _read_wav_pcm16(wav_path):
    """RIFF 청크를 직접 파싱해 (channels, data_bytes) 반환. format 태그 무시(PCM/EXTENSIBLE 모두)."""
    import struct
    with open(wav_path, "rb") as f:
        data = f.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"WAV 아님: {wav_path}")
    ch, i = 1, 12
    raw = b""
    while i < len(data) - 8:
        cid = data[i:i + 4]
        sz = struct.unpack("<I", data[i + 4:i + 8])[0]
        body = data[i + 8:i + 8 + sz]
        if cid == b"fmt ":
            ch = struct.unpack("<H", body[2:4])[0]
        elif cid == b"data":
            raw = body
            break
        i += 8 + sz + (sz & 1)
    return ch, raw


def _log_mel_from_raw(ch_raw, sr, n_mels, n_fft, hop_length):
    """(channels, int16 raw bytes) → log-mel (n_mels, T)."""
    ch, raw = ch_raw
    y = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        n = (len(y) // ch) * ch
        y = y[:n].reshape(-1, ch).mean(axis=1)
    win = np.hanning(n_fft).astype(np.float32)
    starts = range(0, max(1, len(y) - n_fft), hop_length)
    frames = [y[i:i + n_fft] * win for i in starts] or [
        np.pad(y, (0, max(0, n_fft - len(y))))[:n_fft] * win]
    spec = np.abs(np.fft.rfft(np.stack(frames), n=n_fft, axis=1)) ** 2
    mel = spec @ _mel_filterbank(sr, n_fft, n_mels).T
    return np.log(mel + 1e-9).T.astype(np.float32)


def _extract_features(wav_path: str, sr: int, n_mels: int,
                       n_fft: int, hop_length: int) -> np.ndarray:
    """log-mel 추출: librosa → torchaudio → stdlib wave 폴백."""
    if _HAVE_LIBROSA:
        return _log_mel_librosa(wav_path, sr, n_mels, n_fft, hop_length)
    if _HAVE_TORCHAUDIO:
        try:
            return _log_mel_torchaudio(wav_path, sr, n_mels, n_fft, hop_length)
        except Exception:
            pass  # torchcodec 미설치 등 → stdlib 폴백
    return _log_mel_wave_stdlib(wav_path, sr, n_mels, n_fft, hop_length)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

MACHINE_TYPES = ("fan", "pump", "slider", "valve")
DB_LEVELS = ("6dB", "0dB", "-6dB")
MACHINE_IDS = ("id_00", "id_02", "id_04", "id_06")


def load_audio_features(
    root: str,
    machine_type: str,
    db_level: Optional[str] = "6dB",
    machine_ids: Optional[List[str]] = None,
    sr: int = 16_000,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 512,
    include_abnormal: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """
    MIMII 디렉터리를 순회하여 log-mel 특징, 레이블, 메타데이터를 반환합니다.

    Parameters
    ----------
    root          : 데이터셋 루트 경로 (db_level 디렉터리의 부모).
    machine_type  : "fan" | "pump" | "slider" | "valve"
    db_level      : "6dB" | "0dB" | "-6dB" | None(루트 바로 아래 machine_type)
    machine_ids   : 로드할 machine-id 목록. None이면 존재하는 모든 id.
    sr            : 목표 샘플링 레이트 (Hz).
    n_mels        : mel 필터 수.
    n_fft         : FFT 윈도우 크기.
    hop_length    : 홉 길이.
    include_abnormal : False이면 normal 파일만 로드.

    Returns
    -------
    features : np.ndarray, shape (N, n_mels, T), float32
    labels   : np.ndarray, shape (N,), int8  — 0=normal, 1=abnormal
    meta     : List[dict] — machine_type, machine_id, db_level, wav_path per sample
    """
    if machine_type not in MACHINE_TYPES:
        raise ValueError(f"machine_type must be one of {MACHINE_TYPES}")

    base = pathlib.Path(root)
    machine_root = base / db_level / machine_type if db_level else base / machine_type

    if not machine_root.exists():
        raise FileNotFoundError(f"경로를 찾을 수 없습니다: {machine_root}")

    resolved_ids = machine_ids or [
        d.name for d in sorted(machine_root.iterdir())
        if d.is_dir() and d.name.startswith("id_")
    ]

    all_features: list[np.ndarray] = []
    all_labels: list[int] = []
    all_meta: list[dict] = []

    splits = ["normal"] + (["abnormal"] if include_abnormal else [])

    for mid in resolved_ids:
        for split in splits:
            split_dir = machine_root / mid / split
            if not split_dir.exists():
                warnings.warn(f"존재하지 않는 경로 건너뜀: {split_dir}")
                continue
            label = 0 if split == "normal" else 1
            for wav_path in sorted(split_dir.glob("*.wav")):
                feat = _extract_features(str(wav_path), sr, n_mels, n_fft, hop_length)
                all_features.append(feat)
                all_labels.append(label)
                all_meta.append({
                    "machine_type": machine_type,
                    "machine_id": mid,
                    "db_level": db_level,
                    "wav_path": str(wav_path),
                })

    if not all_features:
        raise RuntimeError(f"WAV 파일을 찾을 수 없습니다: {machine_root}")

    # Pad/crop to uniform time dimension
    max_t = max(f.shape[-1] for f in all_features)
    padded = [
        np.pad(f, ((0, 0), (0, max_t - f.shape[-1]))) for f in all_features
    ]
    features = np.stack(padded, axis=0).astype(np.float32)   # (N, n_mels, T)
    labels = np.array(all_labels, dtype=np.int8)

    return features, labels, all_meta


# ---------------------------------------------------------------------------
# Dirichlet partitioner  (CORE non-IID mechanism for P1)
# ---------------------------------------------------------------------------

def make_dirichlet_partition(
    features: np.ndarray,
    labels: np.ndarray,
    meta: List[dict],
    num_sites: int,
    alpha: float,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    NORMAL 훈련 데이터를 Dirichlet(alpha) 분포로 연합학습 클라이언트에 분할합니다.

    Federated non-IID 메커니즘 (P1 핵심):
      - 각 클라이언트의 데이터 구성은 Dirichlet(alpha) 분포에서 샘플링됩니다.
      - alpha → 0   : 극단적 non-IID (한 클라이언트가 특정 machine-id 독점)
      - alpha → ∞   : IID에 근접 (모든 클라이언트가 동일 분포)
      - 분할 기준 변수: machine_id (산업 현장별 장비 종류 차이를 모사)

    Parameters
    ----------
    features  : (N, n_mels, T) float32 — load_audio_features()의 출력
    labels    : (N,) int8 — 0=normal, 1=abnormal
    meta      : List[dict] — machine_id 포함 메타데이터
    num_sites : 연합학습 클라이언트(사이트) 수
    alpha     : Dirichlet 농도 파라미터 (권장: 0.1=고 skew, 1.0=중간, 10.0=low skew)
    seed      : 재현성을 위한 RNG 시드

    Returns
    -------
    site_indices : List[np.ndarray]  — 사이트별 샘플 인덱스 배열 (길이 num_sites)

    Algorithm
    ---------
    1. NORMAL 샘플(label==0)만 선택.
    2. 각 machine_id 클래스별로 샘플 인덱스를 모읍니다.
    3. 각 클래스에 대해 Dirichlet(alpha * ones(num_sites))로 비율을 샘플링합니다.
    4. 비율에 따라 해당 클래스의 인덱스를 num_sites 버킷으로 분배합니다.
    5. 각 사이트의 인덱스를 셔플 후 반환합니다.
    """
    rng = np.random.default_rng(seed)

    # 1. NORMAL 샘플 인덱스만 추출 (입력 배열 불변)
    normal_mask = labels == 0
    normal_indices = np.where(normal_mask)[0]

    # 2. machine_id별 인덱스 그룹화
    mid_to_indices: dict[str, list[int]] = {}
    for idx in normal_indices:
        mid = meta[idx]["machine_id"]
        mid_to_indices.setdefault(mid, [])
        mid_to_indices[mid].append(idx)

    machine_ids_present = sorted(mid_to_indices.keys())
    num_classes = len(machine_ids_present)

    # 사이트별 인덱스 버킷 (새 리스트 — 입력 불변)
    site_buckets: list[list[int]] = [[] for _ in range(num_sites)]

    # 3–4. 각 machine_id 클래스를 Dirichlet 비율로 분배
    for mid in machine_ids_present:
        class_indices = np.array(mid_to_indices[mid], dtype=np.int64)
        rng.shuffle(class_indices)                                    # in-place but on new copy

        proportions = rng.dirichlet(alpha=np.full(num_sites, alpha))
        proportions = proportions / proportions.sum()                 # 수치 안정화

        split_points = (np.cumsum(proportions[:-1]) * len(class_indices)).astype(int)
        splits = np.split(class_indices, split_points)

        for site_idx, chunk in enumerate(splits):
            site_buckets[site_idx].extend(chunk.tolist())

    # 5. 각 사이트 인덱스를 셔플 후 np.ndarray로 반환 (입력 배열 불변)
    return [
        rng.permutation(np.array(bucket, dtype=np.int64))
        for bucket in site_buckets
    ]


# ---------------------------------------------------------------------------
# Self-test  (runs on SYNTHETIC data — no real download needed)
# ---------------------------------------------------------------------------

def _synthetic_meta_and_labels(
    n_per_mid: int = 60,
    machine_ids: tuple[str, ...] = ("id_00", "id_02", "id_04", "id_06"),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """합성 데이터로 features/labels/meta를 생성합니다 (테스트 전용)."""
    rng = np.random.default_rng(seed)
    features_list: list[np.ndarray] = []
    labels_list: list[int] = []
    meta_list: list[dict] = []

    for mid in machine_ids:
        for _ in range(n_per_mid):
            feat = rng.standard_normal((128, 32)).astype(np.float32)
            features_list.append(feat)
            labels_list.append(0)
            meta_list.append({
                "machine_type": "fan",
                "machine_id": mid,
                "db_level": "0dB",
                "wav_path": f"synthetic/{mid}/normal/fake.wav",
            })

    features = np.stack(features_list, axis=0)
    labels = np.array(labels_list, dtype=np.int8)
    return features, labels, meta_list


def _print_site_distribution(
    site_indices: List[np.ndarray],
    meta: List[dict],
    alpha: float,
) -> None:
    """사이트별 machine_id 분포를 출력합니다."""
    print(f"\n{'='*60}")
    print(f"  alpha={alpha:.1f}  (낮을수록 non-IID가 강함)")
    print(f"{'='*60}")
    all_mids = sorted({m["machine_id"] for m in meta})
    header = f"{'Site':<8}" + "".join(f"{mid:>10}" for mid in all_mids) + f"{'Total':>8}"
    print(header)
    print("-" * len(header))

    for i, indices in enumerate(site_indices):
        counts = {mid: 0 for mid in all_mids}
        for idx in indices:
            counts[meta[idx]["machine_id"]] += 1
        row = f"site-{i:<3}" + "".join(f"{counts[mid]:>10}" for mid in all_mids)
        row += f"{len(indices):>8}"
        print(row)


if __name__ == "__main__":
    print("MIMII 연합학습 데이터 로더 자체 테스트 (합성 데이터)")
    print("실제 오디오 다운로드 없이 Dirichlet 분할기를 검증합니다.\n")

    features, labels, meta = _synthetic_meta_and_labels(n_per_mid=60)
    print(f"합성 데이터: {len(features)}개 샘플, "
          f"machine_id={sorted({m['machine_id'] for m in meta})}")

    for alpha in (0.1, 1.0):
        site_indices = make_dirichlet_partition(
            features=features,
            labels=labels,
            meta=meta,
            num_sites=4,
            alpha=alpha,
            seed=42,
        )
        _print_site_distribution(site_indices, meta, alpha)

    # Sanity checks
    total_normal = int((labels == 0).sum())
    all_assigned = np.concatenate(site_indices).tolist()
    assert len(all_assigned) == total_normal, "일부 샘플이 누락됨"
    assert len(set(all_assigned)) == total_normal, "인덱스 중복 발생"
    print("\n[OK] 총 샘플 수 일치 + 인덱스 중복 없음")
    print("[OK] Dirichlet 분할기 자체 테스트 통과")
