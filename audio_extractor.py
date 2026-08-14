"""
audio_extractor.py — audio-modality feature extraction with frozen wav2vec2
(model set by config.WAV2VEC_MODEL_NAME).
Each recording is split into 30-s segments; per-segment embeddings are kept
(and optionally mean-pooled). A corpus-scoped (corpus_tag, patient_id) index
is used to avoid patient-ID collisions across corpora.
"""
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import AUDIO_BASE_DIR, WAV2VEC_MODEL_NAME, AUDIO_DIM

SEGMENT_SECONDS  = 30        # segment length (seconds)
SEGMENT_SAMPLES  = SEGMENT_SECONDS * 16000  # samples per segment at 16 kHz
MIN_SEGMENT_SAMPLES = 400   # ignore segments shorter than this
MAX_SEGMENTS     = 20       # max segments per sample

# corpus_tag must match sample["corpus"] from data_loader
_ADRESSO_TAG = "ADReSSo"
_TBI_TAG     = "Coelho"      # data_loader: corpus="Coelho"
_RHD_TAG     = "RHD-English" # data_loader: corpus="RHD-English"
# Aphasia: corpus = metadata corpus column value (e.g. "NEURAL", "Wright", ...)


def build_audio_index(audio_base_dir: Optional[Path] = None
                      ) -> Dict[Tuple[str, str], Path]:
    """
    Build a corpus-scoped mapping {(corpus_tag, patient_id): audio_path}.
    corpus_tag matches the sample["corpus"] field from data_loader.
    """
    base = audio_base_dir or AUDIO_BASE_DIR
    index: Dict[Tuple[str, str], Path] = {}

    # ADReSSo — corpus_tag="ADReSSo"
    for split in ("train", "test"):
        d = base / "ADReSSo" / split
        if d.exists():
            for f in d.glob("*.wav"):
                index[(_ADRESSO_TAG, f.stem)] = f

    # Aphasia — corpus_tag = subdirectory name (e.g. "NEURAL", "Wright")
    aphasia_dir = base / "Aphasia"
    if aphasia_dir.exists():
        for corpus_dir in aphasia_dir.iterdir():
            if corpus_dir.is_dir():
                ctag = corpus_dir.name  # "NEURAL", "Wright", ...
                for sub_dir in corpus_dir.iterdir():
                    if sub_dir.is_dir():
                        for f in sub_dir.glob("*.mp3"):
                            index[(ctag, f.stem)] = f

    # TBI — corpus_tag="Coelho", audio_stem + "_participant" = patient_id
    tbi_base = base / "TBI_audio_mp3" / "Coelho"
    for sub in ("TB", "N"):
        d = tbi_base / sub
        if d.exists():
            for f in d.glob("*.mp3"):
                patient_id = f.stem + "_participant"
                index[(_TBI_TAG, patient_id)] = f

    # RHD — corpus_tag="RHD-English"
    rhd_base = base / "RHD_audio_mp3" / "English"
    for sub in ("Control", "RHD"):
        d = rhd_base / sub
        if d.exists():
            for f in d.glob("*.mp3"):
                index[(_RHD_TAG, f.stem)] = f

    # MCI (Delaware) — corpus_tag="Delaware"
    mci_base = base / "MCI" / "Delaware"
    for sub in ("Control", "MCI"):
        d = mci_base / sub
        if d.exists():
            for f in d.glob("*.mp3"):
                index[("Delaware", f.stem)] = f

    return index


def _load_audio_full(path: Path, target_sr: int = 16000) -> np.ndarray:
    """Load a full audio file as 16 kHz mono float32 (no length limit)."""
    suffix = path.suffix.lower()

    if suffix == ".wav":
        import soundfile as sf
        data, sr = sf.read(str(path))
        if data.ndim > 1:
            data = data.mean(axis=1).astype(np.float32)
        else:
            data = data.astype(np.float32)
        if sr != target_sr:
            from scipy.signal import resample_poly
            import math
            g = math.gcd(target_sr, sr)
            data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)
    elif suffix == ".mp3":
        from pydub import AudioSegment
        audio = AudioSegment.from_mp3(str(path))
        audio = audio.set_frame_rate(target_sr).set_channels(1)
        data = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    else:
        raise ValueError(f"Unsupported audio format: {suffix}")

    return data.astype(np.float32)


def _segment_audio(audio: np.ndarray,
                   seg_samples: int = SEGMENT_SAMPLES,
                   min_samples: int = MIN_SEGMENT_SAMPLES,
                   max_segments: int = MAX_SEGMENTS,
                   ) -> List[np.ndarray]:
    """
    Split an audio array into seg_samples-sized segments.
    A trailing segment shorter than min_samples and anything past max_segments are dropped.
    """
    segments = []
    total = len(audio)
    start = 0
    while start < total and len(segments) < max_segments:
        end = start + seg_samples
        seg = audio[start:end]
        if len(seg) >= min_samples:
            segments.append(seg)
        start = end
    return segments if segments else [audio]   # whole audio shorter than min_samples


def _extract_one(audio: np.ndarray,
                 processor,
                 w2v_model,
                 device: str,
                 use_gpu: bool) -> np.ndarray:
    """
    One audio segment → time-averaged wav2vec2 last_hidden_state, shape (AUDIO_DIM,).
    """
    inputs = processor(
        audio, sampling_rate=16000,
        return_tensors="pt", padding=True,
    )
    input_values = inputs.input_values
    if use_gpu:
        input_values = input_values.to(device)
    with torch.no_grad():
        out = w2v_model(input_values)
        feat = out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
    return feat


def compute_wav2vec2_features(
    samples: List[dict],
    audio_index: Dict[Tuple[str, str], Path],
    device: str = "cuda",
    cache_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract per-patient wav2vec2 features (30-s segments, mean-pooled).
    Returns (feats (N, AUDIO_DIM), has_audio (N,)); samples without audio get zeros.
    """
    cache_keys = [(s["corpus"], s["patient_id"]) for s in samples]

    if cache_path and cache_path.exists():
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        stored_keys = data.get("cache_keys", [])
        if stored_keys == cache_keys:
            print(f"  [Audio cache] loaded {cache_path.name} ({len(samples)} samples)")
            return data["feats"].numpy(), data["has_audio"].numpy().astype(bool)
        # order mismatch → try dict-based reordering
        stored_set = set(map(tuple, stored_keys)) if stored_keys else set()
        need_set   = set(map(tuple, cache_keys))
        if need_set.issubset(stored_set):
            idx_map = {tuple(k): i for i, k in enumerate(stored_keys)}
            idxs = [idx_map[tuple(k)] for k in cache_keys]
            feats_t    = data["feats"][idxs]
            has_t      = data["has_audio"][idxs]
            print(f"  [Audio cache] reorder-loaded {cache_path.name} ({len(samples)} samples)")
            return feats_t.numpy(), has_t.numpy().astype(bool)

    n = len(samples)
    print(f"  [wav2vec2] extracting audio features ({n} samples, segment={SEGMENT_SECONDS}s) …")
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    use_gpu = device != "cpu" and torch.cuda.is_available()
    processor = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC_MODEL_NAME)
    w2v_model = Wav2Vec2Model.from_pretrained(WAV2VEC_MODEL_NAME)
    w2v_model.eval()
    if use_gpu:
        w2v_model = w2v_model.to(device)

    feats     = np.zeros((n, AUDIO_DIM), dtype=np.float32)
    has_audio = np.zeros(n, dtype=bool)
    n_found = n_missing = 0
    total_segs = 0

    for i, s in enumerate(samples):
        if i > 0 and i % 100 == 0:
            print(f"    {i}/{n}  found={n_found}  missing={n_missing}  "
                  f"avg_segs={total_segs/max(n_found,1):.1f}")

        key = (s["corpus"], s["patient_id"])
        audio_path = audio_index.get(key)
        if audio_path is None:
            n_missing += 1
            continue

        try:
            audio = _load_audio_full(audio_path)
            if len(audio) < MIN_SEGMENT_SAMPLES:
                n_missing += 1
                continue

            segments = _segment_audio(audio)
            seg_feats = []
            for seg in segments:
                f = _extract_one(seg, processor, w2v_model, device, use_gpu)
                seg_feats.append(f)

            feats[i]     = np.mean(seg_feats, axis=0)   # segment-level mean pooling
            has_audio[i] = True
            n_found      += 1
            total_segs   += len(segments)

        except Exception as e:
            print(f"    [Warning] {s['patient_id']} ({s['corpus']}): {e}")
            n_missing += 1

    avg_segs = total_segs / max(n_found, 1)
    print(f"  [wav2vec2] done: {n_found} ok, {n_missing} missing/failed, "
          f"avg {avg_segs:.1f} segments/sample")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "cache_keys": cache_keys,
            "feats":      torch.from_numpy(feats),
            "has_audio":  torch.from_numpy(has_audio),
        }, cache_path)
        print(f"  [Audio cache] saved: {cache_path.name}")

    return feats, has_audio


def compute_wav2vec2_segment_features(
    samples: List[dict],
    audio_index: Dict[Tuple[str, str], Path],
    device: str = "cuda",
    cache_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract per-segment wav2vec2 features (for attention pooling over segments).
    Returns (seg_feats (N, MAX_SEGMENTS, AUDIO_DIM) zero-padded, n_segs (N,), has_audio (N,)).
    """
    cache_keys = [(s["corpus"], s["patient_id"]) for s in samples]

    if cache_path and cache_path.exists():
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        stored_keys = data.get("cache_keys", [])
        if stored_keys == cache_keys:
            print(f"  [Seg cache] loaded {cache_path.name} ({len(samples)} samples)")
            return (data["seg_feats"].numpy(),
                    data["n_segs"].numpy(),
                    data["has_audio"].numpy().astype(bool))
        # order mismatch → try dict-based reordering
        stored_set = set(map(tuple, stored_keys)) if stored_keys else set()
        need_set   = set(map(tuple, cache_keys))
        if need_set.issubset(stored_set):
            idx_map  = {tuple(k): i for i, k in enumerate(stored_keys)}
            idxs     = [idx_map[tuple(k)] for k in cache_keys]
            sf_t     = data["seg_feats"][idxs]
            ns_t     = data["n_segs"][idxs]
            ha_t     = data["has_audio"][idxs]
            print(f"  [Seg cache] reorder-loaded {cache_path.name} ({len(samples)} samples)")
            return sf_t.numpy(), ns_t.numpy(), ha_t.numpy().astype(bool)

    n = len(samples)
    print(f"  [wav2vec2-seg] extracting segment features ({n} samples) …")
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    use_gpu = device != "cpu" and torch.cuda.is_available()
    processor = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC_MODEL_NAME)
    w2v_model = Wav2Vec2Model.from_pretrained(WAV2VEC_MODEL_NAME)
    w2v_model.eval()
    if use_gpu:
        w2v_model = w2v_model.to(device)

    from config import AUDIO_DIM as _AUDIO_DIM
    seg_feats = np.zeros((n, MAX_SEGMENTS, _AUDIO_DIM), dtype=np.float32)
    n_segs    = np.zeros(n, dtype=np.int32)
    has_audio = np.zeros(n, dtype=bool)
    n_found = n_missing = 0

    for i, s in enumerate(samples):
        if i > 0 and i % 100 == 0:
            print(f"    {i}/{n}  found={n_found}  missing={n_missing}")

        key = (s["corpus"], s["patient_id"])
        audio_path = audio_index.get(key)
        if audio_path is None:
            n_missing += 1
            continue

        try:
            audio    = _load_audio_full(audio_path)
            if len(audio) < MIN_SEGMENT_SAMPLES:
                n_missing += 1
                continue

            segments = _segment_audio(audio)
            for j, seg in enumerate(segments):
                f = _extract_one(seg, processor, w2v_model, device, use_gpu)
                seg_feats[i, j] = f

            n_segs[i]    = len(segments)
            has_audio[i] = True
            n_found      += 1
        except Exception as e:
            print(f"    [Warning] {s['patient_id']} ({s['corpus']}): {e}")
            n_missing += 1

    print(f"  [wav2vec2-seg] done: {n_found} ok, {n_missing} missing, "
          f"avg {n_segs[has_audio].mean():.1f} segments/sample")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "cache_keys": cache_keys,
            "seg_feats":  torch.from_numpy(seg_feats),
            "n_segs":     torch.from_numpy(n_segs),
            "has_audio":  torch.from_numpy(has_audio.astype(np.uint8)),
        }, cache_path)
        print(f"  [Seg cache] saved: {cache_path.name}")

    return seg_feats, n_segs, has_audio
