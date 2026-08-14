"""
paralinguistic_extractor.py — prosodic/paralinguistic feature extraction.
Extracts 15 features (F0, energy, ZCR, spectral centroid, MFCC 2-5 statistics)
complementary to wav2vec2 representations; only the first 10 minutes are used.
"""
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from audio_extractor import build_audio_index, _load_audio_full, MAX_SEGMENTS, SEGMENT_SECONDS

PARA_DIM = 15
MAX_PARA_SAMPLES = MAX_SEGMENTS * SEGMENT_SECONDS * 16000  # 10 minutes

FEATURE_NAMES = [
    "f0_mean", "f0_std", "voiced_ratio",          # F0 / voicing (3)
    "rms_mean", "rms_std",                         # energy (2)
    "zcr_mean",                                    # speech rate proxy (1)
    "spec_centroid_mean",                          # spectral brightness (1)
    "mfcc2_mean", "mfcc3_mean", "mfcc4_mean", "mfcc5_mean",   # spectral shape (4)
    "mfcc2_std",  "mfcc3_std",  "mfcc4_std",  "mfcc5_std",    # spectral variation (4)
]
assert len(FEATURE_NAMES) == PARA_DIM


def _extract_para_one(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    One audio array → 15-dim paralinguistic feature vector (zeros on error).
    """
    import librosa

    # truncate to 10 minutes (same limit as wav2vec2)
    if len(audio) > MAX_PARA_SAMPLES:
        audio = audio[:MAX_PARA_SAMPLES]

    hop = 1024
    n_fft = 2048

    try:
        # ── F0 (yin) ──────────────────────────────────────────────────────────
        # yin returns F0 for all frames; voiced frames estimated via an energy threshold
        f0 = librosa.yin(audio, fmin=60.0, fmax=400.0,
                         sr=sr, frame_length=n_fft, hop_length=hop)
        rms_for_gate = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop)[0]

        # voiced: frames whose RMS is at least 3% of the max
        voiced_mask = rms_for_gate > 0.03 * (rms_for_gate.max() + 1e-8)
        voiced_ratio = float(voiced_mask.mean())
        voiced_f0 = f0[voiced_mask]
        if len(voiced_f0) > 0:
            f0_mean = float(voiced_f0.mean())
            f0_std  = float(voiced_f0.std())
        else:
            f0_mean = 0.0
            f0_std  = 0.0

        # ── Energy (RMS) ──────────────────────────────────────────────────────
        rms = rms_for_gate  # (frames,)
        rms_mean = float(rms.mean())
        rms_std  = float(rms.std())

        # ── ZCR ───────────────────────────────────────────────────────────────
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=n_fft, hop_length=hop)[0]
        zcr_mean = float(zcr.mean())

        # ── Spectral centroid ─────────────────────────────────────────────────
        sc = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop)[0]
        sc_mean = float(sc.mean())

        # ── MFCC 2-5 (mean + std) ─────────────────────────────────────────────
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=6,
                                      n_fft=n_fft, hop_length=hop)  # (6, T)
        mfcc_mean = mfccs[1:5].mean(axis=1)   # indices 1-4 → MFCC 2-5, shape (4,)
        mfcc_std  = mfccs[1:5].std(axis=1)    # (4,)

        feat = np.array([
            f0_mean, f0_std, voiced_ratio,
            rms_mean, rms_std,
            zcr_mean,
            sc_mean,
            *mfcc_mean.tolist(),
            *mfcc_std.tolist(),
        ], dtype=np.float32)

    except Exception:
        feat = np.zeros(PARA_DIM, dtype=np.float32)

    return feat


def _extract_worker(args):
    """Worker function for parallel extraction."""
    i, key, audio_path_str = args
    if audio_path_str is None:
        return i, None, False
    try:
        audio = _load_audio_full(Path(audio_path_str))
        if len(audio) < 400:
            return i, None, False
        feat = _extract_para_one(audio)
        return i, feat, True
    except Exception:
        return i, None, False


def compute_paralinguistic_features(
    samples: List[dict],
    audio_index: Dict[Tuple[str, str], Path],
    cache_path: Optional[Path] = None,
    n_workers: int = 24,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract paralinguistic features for all samples (multiprocessing).
    Returns (feats (N, PARA_DIM), has_audio (N,)); samples without audio get zeros.
    """
    import multiprocessing as mp

    cache_keys = [(s["corpus"], s["patient_id"]) for s in samples]

    if cache_path and cache_path.exists():
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        if data.get("cache_keys") == cache_keys:
            print(f"  [Para cache] loaded {cache_path.name} ({len(samples)} samples)")
            return data["feats"].numpy(), data["has_audio"].numpy().astype(bool)

    n = len(samples)
    print(f"  [Paralinguistic] extracting features ({n} samples, {PARA_DIM} dims, workers={n_workers}) …")

    worker_args = []
    for i, s in enumerate(samples):
        key = (s["corpus"], s["patient_id"])
        audio_path = audio_index.get(key)
        worker_args.append((i, key, str(audio_path) if audio_path else None))

    feats     = np.zeros((n, PARA_DIM), dtype=np.float32)
    has_audio = np.zeros(n, dtype=bool)
    n_found = n_missing = 0

    with mp.Pool(processes=n_workers) as pool:
        for idx, (i, feat, found) in enumerate(pool.imap_unordered(_extract_worker, worker_args, chunksize=4)):
            if found and feat is not None:
                feats[i]     = feat
                has_audio[i] = True
                n_found += 1
            else:
                n_missing += 1
            if (idx + 1) % 200 == 0:
                print(f"    {idx+1}/{n}  found={n_found}  missing={n_missing}")

    print(f"  [Paralinguistic] done: {n_found} ok, {n_missing} missing")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "cache_keys": cache_keys,
            "feats":      torch.from_numpy(feats),
            "has_audio":  torch.from_numpy(has_audio.astype(np.uint8)),
        }, cache_path)
        print(f"  [Para cache] saved: {cache_path.name}")

    return feats, has_audio


def normalize_para_features(feats: np.ndarray,
                              mean: Optional[np.ndarray] = None,
                              std:  Optional[np.ndarray] = None,
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalization; compute mean/std from the input when None."""
    if mean is None:
        mean = feats.mean(axis=0)
        std  = feats.std(axis=0)
    normed = (feats - mean) / (std + 1e-8)
    return normed.astype(np.float32), mean, std
