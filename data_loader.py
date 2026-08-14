"""
data_loader.py — Data loading and preprocessing.
Loads transcript samples from AphasiaBank, ADReSSo, TBI (Coelho),
RHD-English, and MCI (Delaware) corpora into PyTorch NeuroBrainDataset objects.
"""
import re
from collections import defaultdict
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import Dataset

from config import (
    APHASIA_TEXT_DIR, AD_TEST_DIR, AD_TRAIN_DIR,
    METADATA_CSV, AD_LABELS_CSV,
    AD_TRAIN_META_CSV, AD_TEST_META_CSV,
    TBI_CTRL_DIR, TBI_TBI_DIR,
    RHD_CTRL_DIR, RHD_RHD_DIR,
    MCI_CTRL_DIR, MCI_MCI_DIR,
    TBI_TBI_DEMO_XLSX, TBI_CTRL_DEMO_XLSX,
    RHD_CTRL_DEMO_XLSX, RHD_RHD_DEMO_XLSX,
    MCI_DEMO_XLSX,
    LABEL_MAP, CONTROL_WAB_TYPES, VAL_RATIO, SEED,
    BERT_MODEL_NAME, BERT_DIM, AUDIO_DIM, WAV2VEC_MODEL_NAME,
)

def _wav2vec_cache_name(suffix: str = "") -> str:
    tag = "large" if "large" in WAV2VEC_MODEL_NAME else "base"
    return f"wav2vec2_{tag}_cache{suffix}.pt"
from feature_extractor import extract_features, normalize_features


# ─── Internal utilities ──────────────────────────────────────────────────────

def _read_text(path: Path) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=enc, errors="ignore").strip()
        except Exception:
            continue
    return ""


def _is_control(wab_type: str) -> bool:
    if not isinstance(wab_type, str):
        return False
    return wab_type.strip().lower() in {w.lower() for w in CONTROL_WAB_TYPES}


# ─── Corpus loaders ──────────────────────────────────────────────────────────

def _load_aphasia_samples(text_dir: Path, meta_df: pd.DataFrame):
    samples = []
    dedup = meta_df.drop_duplicates("participant_id", keep="first")
    pid_to_meta = dedup.set_index("participant_id").to_dict("index")

    for split in ("train", "test"):
        split_dir = text_dir / split
        if not split_dir.exists():
            continue
        for txt_file in split_dir.glob("*.txt"):
            pid = txt_file.stem
            if re.match(r"^\d", pid):
                continue
            meta = pid_to_meta.get(pid)
            if meta is None:
                continue
            text = _read_text(txt_file)
            if len(text.split()) < 5:
                continue

            wab_type = meta.get("wab_type", "")
            wab_aq   = meta.get("wab_aq", np.nan)
            age      = meta.get("age", np.nan)
            sex_str  = str(meta.get("sex", "")).lower()

            label = LABEL_MAP["Control"] if _is_control(wab_type) else LABEL_MAP["Aphasia"]
            samples.append({
                "patient_id": pid,
                "text":       text,
                "label":      label,
                "wab_aq":     float(wab_aq) if pd.notna(wab_aq) else -1.0,
                "age":        float(age) if pd.notna(age) else 60.0,
                "sex":        1.0 if sex_str == "female" else 0.0,
                "corpus":     meta.get("corpus", "AphasiaBank"),
            })
    return samples


def _load_ad_samples(test_dir: Path, labels_csv: Path):
    if not labels_csv.exists():
        return []
    label_df = pd.read_csv(labels_csv)
    label_df["stem"] = label_df["file_name"].apply(lambda x: Path(str(x).strip()).stem)
    stem_to_dx = dict(zip(label_df["stem"], label_df["diagnosis"]))

    samples = []
    for txt_file in sorted(test_dir.glob("*.txt")):
        dx = stem_to_dx.get(txt_file.stem)
        if dx is None:
            continue
        text = _read_text(txt_file)
        if len(text.split()) < 5:
            continue
        label = LABEL_MAP["AD"] if "AD" in dx.upper() else LABEL_MAP["Control"]
        samples.append({
            "patient_id": txt_file.stem,
            "text":       text,
            "label":      label,
            "wab_aq":     -1.0,
            "age":        72.0,
            "sex":        0.5,
            "corpus":     "ADReSSo",
        })
    return samples


def _load_tbi_demo() -> dict:
    """Load TBI (Coelho) demographics. Returns {stem: {age, sex}}."""
    lookup = {}
    for xlsx_path, prefix, sep in [
        (TBI_TBI_DEMO_XLSX,  "TB", "tbi"),
        (TBI_CTRL_DEMO_XLSX, "N",  "ctrl"),
    ]:
        if not xlsx_path.exists():
            continue
        df = pd.read_excel(xlsx_path)
        for _, row in df.iterrows():
            id_str = str(row.get("ID", "")).strip()
            if not id_str or id_str == "nan":
                continue
            if sep == "tbi":
                num_str = re.sub(r"^TB0*", "", id_str, flags=re.IGNORECASE)
                stem = f"tb{int(num_str):02d}_participant"
            else:
                num_str = re.sub(r"^N0*", "", id_str, flags=re.IGNORECASE)
                stem = f"n{int(num_str):02d}_participant"
            age_val = row.get("Age", np.nan)
            age = float(age_val) if pd.notna(age_val) else 60.0
            sex_str = str(row.get("Sex", "")).strip().upper()
            sex = 1.0 if sex_str == "F" else 0.0
            lookup[stem] = {"age": age, "sex": sex}
    return lookup


def _load_rhd_demo() -> dict:
    """Load RHD-English demographics. Returns {stem: {age, sex}}."""
    lookup = {}
    for xlsx_path in [RHD_CTRL_DEMO_XLSX, RHD_RHD_DEMO_XLSX]:
        if not xlsx_path.exists():
            continue
        df = pd.read_excel(xlsx_path, sheet_name=1)
        for _, row in df.iterrows():
            pid = str(row.get("Participant ID", "")).strip()
            if not pid or pid == "nan":
                continue
            age_val = row.get("Age at Testing", np.nan)
            age = float(age_val) if pd.notna(age_val) else 60.0
            gender_str = str(row.get("Gender", "")).strip().upper()
            sex = 1.0 if gender_str == "F" else 0.0
            lookup[pid] = {"age": age, "sex": sex}
    return lookup


def _load_mci_demo() -> dict:
    """Load Delaware MCI demographics.
    File stems are '{record_id:02d}-{visit}'; returns {stem: {age, sex}}.
    Multiple visits per participant: use first-visit age/sex for all visits.
    SEX encoding: 1=male→0.0, 2=female→1.0 (standard US clinical survey).
    """
    if not MCI_DEMO_XLSX.exists():
        return {}
    df = pd.read_excel(MCI_DEMO_XLSX)
    # Build per-record_id lookup from first visit
    id_to_demo: dict = {}
    for _, row in df.iterrows():
        rid = row.get("RECORD ID")
        if pd.isna(rid):
            continue
        record_id = int(rid)
        if record_id in id_to_demo:
            continue
        age_val = row.get("AGE AT TESTING", np.nan)
        age = float(age_val) if pd.notna(age_val) else 60.0
        sex_val = row.get("SEX", np.nan)
        sex = 1.0 if (pd.notna(sex_val) and int(sex_val) == 2) else 0.0
        id_to_demo[record_id] = {"age": age, "sex": sex}

    # Build stem-level lookup for both MCI/ and Control/ directories
    lookup: dict = {}
    for src_dir in [MCI_MCI_DIR, MCI_CTRL_DIR]:
        if not src_dir.exists():
            continue
        for txt_file in src_dir.glob("*.txt"):
            stem = txt_file.stem
            try:
                record_id = int(stem.split("-")[0])
                if record_id in id_to_demo:
                    lookup[stem] = id_to_demo[record_id]
            except (ValueError, IndexError):
                pass
    return lookup


def _load_txt_dir_samples(ctrl_dir: Path, disease_dir: Path,
                           disease_label: str, corpus: str,
                           demo_lookup: Optional[dict] = None) -> list:
    """Load .txt samples from a Control/Disease directory pair.

    demo_lookup: {stem: {'age': float, 'sex': float}} — defaults used if absent.
    """
    samples = []
    for label_name, src_dir in [("Control", ctrl_dir), (disease_label, disease_dir)]:
        if not src_dir.exists():
            continue
        for txt_file in sorted(src_dir.glob("*.txt")):
            text = _read_text(txt_file)
            if len(text.split()) < 5:
                continue
            stem = txt_file.stem
            demo = demo_lookup.get(stem) if demo_lookup else None
            age = demo["age"] if demo else 60.0
            sex = demo["sex"] if demo else 0.5
            samples.append({
                "patient_id": stem,
                "text":       text,
                "label":      LABEL_MAP[label_name],
                "wab_aq":     -1.0,
                "age":        age,
                "sex":        sex,
                "corpus":     corpus,
            })
    return samples


# ─── Precomputed BERT features ───────────────────────────────────────────────

def _compute_bert_features(texts: list[str], device: str = "cuda",
                            cache_path: Optional[Path] = None) -> np.ndarray:
    """
    Batch-compute frozen DistilBERT [CLS] embeddings.
    Loads from cache_path when available; otherwise computes and saves.
    """
    if cache_path and cache_path.exists():
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        cached_texts = data["texts"]
        if cached_texts == texts:
            print(f"  [BERT cache] loaded {cache_path.name} ({len(texts)} samples)")
            return data["feats"].numpy()

    print(f"  [BERT] computing DistilBERT features ({len(texts)} samples) …")
    from transformers import DistilBertTokenizer, DistilBertModel

    tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL_NAME)
    model     = DistilBertModel.from_pretrained(BERT_MODEL_NAME)
    model.eval()
    use_gpu = device != "cpu" and torch.cuda.is_available()
    if use_gpu:
        model = model.to(device)

    feats = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            enc = tokenizer(
                batch_texts, return_tensors="pt",
                truncation=True, max_length=512,
                padding=True,
            )
            if use_gpu:
                enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            cls_emb = out.last_hidden_state[:, 0, :].cpu()
            feats.append(cls_emb)
            if (i // batch_size + 1) % 10 == 0:
                print(f"    {i + len(batch_texts)}/{len(texts)}")

    feat_arr = torch.cat(feats, dim=0)   # (N, 768)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"texts": texts, "feats": feat_arr}, cache_path)
        print(f"  [BERT cache] saved: {cache_path.name}")

    return feat_arr.numpy()


# ─── Tabular features ────────────────────────────────────────────────────────

def _build_tabular(sample: dict, wab_mean: float, wab_std: float) -> np.ndarray:
    age_norm    = (sample["age"] - 60.0) / 15.0
    sex         = sample["sex"]
    has_wab     = 1.0 if sample["wab_aq"] >= 0 else 0.0
    wab_aq_norm = ((sample["wab_aq"] - wab_mean) / (wab_std + 1e-8)
                   if sample["wab_aq"] >= 0 else 0.0)
    return np.array([age_norm, sex, has_wab, wab_aq_norm], dtype=np.float32)


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────

class NeuroBrainDataset(Dataset):
    def __init__(self, samples: list,
                 feat_mean: Optional[np.ndarray] = None,
                 feat_std:  Optional[np.ndarray] = None,
                 wab_mean: float = 70.0,
                 wab_std:  float = 20.0,
                 bert_feats:  Optional[np.ndarray] = None,
                 audio_feats: Optional[np.ndarray] = None,
                 has_audio:   Optional[np.ndarray] = None,
                 para_feats:  Optional[np.ndarray] = None,
                 audio_seg_feats: Optional[np.ndarray] = None,
                 n_segs:      Optional[np.ndarray] = None):
        self.samples  = samples
        self.wab_mean = wab_mean
        self.wab_std  = wab_std

        raw_feats = np.stack([extract_features(s["text"]) for s in samples])
        if feat_mean is None:
            raw_feats, self.feat_mean, self.feat_std = normalize_features(raw_feats)
        else:
            self.feat_mean = feat_mean
            self.feat_std  = feat_std
            raw_feats = (raw_feats - feat_mean) / (feat_std + 1e-8)
        self.text_feats = raw_feats.astype(np.float32)

        self.tab_feats = np.stack([
            _build_tabular(s, wab_mean, wab_std) for s in samples
        ]).astype(np.float32)

        self.bert_feats      = bert_feats.astype(np.float32) if bert_feats  is not None else None
        self.audio_feats     = audio_feats.astype(np.float32) if audio_feats is not None else None
        self.has_audio       = has_audio.astype(bool) if has_audio is not None else None
        self.para_feats      = para_feats.astype(np.float32) if para_feats is not None else None
        self.audio_seg_feats = audio_seg_feats.astype(np.float32) if audio_seg_feats is not None else None
        self.n_segs          = n_segs.astype(np.int32) if n_segs is not None else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        item = {
            "text_feat":  torch.tensor(self.text_feats[idx]),
            "tab_feat":   torch.tensor(self.tab_feats[idx]),
            "label":      torch.tensor(s["label"],  dtype=torch.long),
            "wab_aq":     torch.tensor(s["wab_aq"], dtype=torch.float32),
            "patient_id": s["patient_id"],
            "corpus":     s["corpus"],
        }
        if self.bert_feats is not None:
            item["bert_feat"] = torch.tensor(self.bert_feats[idx])
        if self.audio_feats is not None:
            item["audio_feat"] = torch.tensor(self.audio_feats[idx])
            item["has_audio"]  = torch.tensor(
                bool(self.has_audio[idx]) if self.has_audio is not None else False
            )
        if self.para_feats is not None:
            item["para_feat"] = torch.tensor(self.para_feats[idx])
        if self.audio_seg_feats is not None:
            item["audio_seg_feat"] = torch.tensor(self.audio_seg_feats[idx])
            item["n_segs"] = torch.tensor(int(self.n_segs[idx]), dtype=torch.long)
            if "has_audio" not in item:
                item["has_audio"] = torch.tensor(int(self.n_segs[idx]) > 0)
        return item


# ─── Public interface ────────────────────────────────────────────────────────

def load_dataset(verbose: bool = True,
                 use_bert: bool = False,
                 bert_device: str = "cuda",
                 use_audio: bool = False,
                 use_para: bool = False,
                 use_seg_audio: bool = False):
    """
    Load the full dataset and return (train_dataset, val_dataset, test_dataset).

    use_bert / use_audio precompute and inject DistilBERT / wav2vec2 features.
    """
    meta_df = pd.read_csv(METADATA_CSV)

    tbi_demo = _load_tbi_demo()
    rhd_demo = _load_rhd_demo()
    mci_demo = _load_mci_demo()

    aphasia_samples = _load_aphasia_samples(APHASIA_TEXT_DIR, meta_df)
    ad_samples      = (
        _load_ad_samples(AD_TRAIN_DIR, AD_TRAIN_META_CSV) +
        _load_ad_samples(AD_TEST_DIR,  AD_TEST_META_CSV)
    )
    tbi_samples     = _load_txt_dir_samples(TBI_CTRL_DIR, TBI_TBI_DIR, "TBI", "Coelho",
                                             demo_lookup=tbi_demo)
    rhd_samples     = _load_txt_dir_samples(RHD_CTRL_DIR, RHD_RHD_DIR, "RHD", "RHD-English",
                                             demo_lookup=rhd_demo)
    mci_samples     = _load_txt_dir_samples(MCI_CTRL_DIR, MCI_MCI_DIR, "MCI", "Delaware",
                                             demo_lookup=mci_demo)

    all_samples = aphasia_samples + ad_samples + tbi_samples + rhd_samples + mci_samples

    if verbose:
        from collections import Counter
        lbl_counts = Counter(s["label"] for s in all_samples)
        print(f"[Data] total samples: {len(all_samples)}")
        for lbl in sorted(lbl_counts):
            from config import IDX_TO_LABEL
            print(f"       {IDX_TO_LABEL[lbl]:10s}: {lbl_counts[lbl]}")

    if len(all_samples) == 0:
        raise RuntimeError("Data loading failed — check dataset paths.")

    valid_wab = [s["wab_aq"] for s in all_samples if s["wab_aq"] >= 0]
    wab_mean  = float(np.mean(valid_wab)) if valid_wab else 70.0
    wab_std   = float(np.std(valid_wab))  if valid_wab else 20.0

    # ── Stratified per-class 3-way split (70 / 15 / 15) ──────────────────────
    by_label = defaultdict(list)
    for s in all_samples:
        by_label[s["label"]].append(s)

    rng = np.random.default_rng(SEED)
    train_raw, val_raw, test_raw = [], [], []
    for label_id in sorted(by_label.keys()):
        samps = by_label[label_id]
        idx   = rng.permutation(len(samps))
        n_test = max(1, round(len(samps) * VAL_RATIO))
        n_val  = max(1, round(len(samps) * VAL_RATIO))
        test_raw.extend([samps[i] for i in idx[:n_test]])
        val_raw.extend( [samps[i] for i in idx[n_test:n_test + n_val]])
        train_raw.extend([samps[i] for i in idx[n_test + n_val:]])

    # ── Precompute BERT features ──────────────────────────────────────────────
    bert_train = bert_val = bert_test = None
    if use_bert:
        from config import RESULTS_DIR
        cache_path = RESULTS_DIR / "bert_features_cache.pt"

        all_texts = [s["text"] for s in all_samples]
        all_feats = _compute_bert_features(all_texts, device=bert_device,
                                            cache_path=cache_path)
        text_to_idx = {id(s): i for i, s in enumerate(all_samples)}

        def _get_bert(split_samples):
            return np.stack([all_feats[text_to_idx[id(s)]] for s in split_samples])

        bert_train = _get_bert(train_raw)
        bert_val   = _get_bert(val_raw)
        bert_test  = _get_bert(test_raw)

    # ── Precompute audio features ─────────────────────────────────────────────
    audio_train = audio_val = audio_test = None
    has_audio_train = has_audio_val = has_audio_test = None
    if use_audio:
        from config import RESULTS_DIR
        from audio_extractor import build_audio_index, compute_wav2vec2_features
        cache_path = RESULTS_DIR / _wav2vec_cache_name()

        audio_index = build_audio_index()
        all_afeats, all_has = compute_wav2vec2_features(
            all_samples, audio_index, device=bert_device, cache_path=cache_path,
        )
        sample_to_idx = {id(s): i for i, s in enumerate(all_samples)}

        def _get_audio(split_samples):
            idxs = [sample_to_idx[id(s)] for s in split_samples]
            return all_afeats[idxs], all_has[idxs]

        audio_train, has_audio_train = _get_audio(train_raw)
        audio_val,   has_audio_val   = _get_audio(val_raw)
        audio_test,  has_audio_test  = _get_audio(test_raw)

    # ── Precompute paralinguistic features ────────────────────────────────────
    para_train = para_val = para_test = None
    if use_para:
        from config import RESULTS_DIR
        from audio_extractor import build_audio_index
        from paralinguistic_extractor import compute_paralinguistic_features, normalize_para_features
        cache_path = RESULTS_DIR / "para_cache.pt"

        audio_index = build_audio_index()
        all_pfeats, _phas = compute_paralinguistic_features(
            all_samples, audio_index, cache_path=cache_path,
        )
        s2i = {id(s): i for i, s in enumerate(all_samples)}

        def _get_para(split_samples):
            idxs = [s2i[id(s)] for s in split_samples]
            return all_pfeats[idxs]

        para_train_raw = _get_para(train_raw)
        para_val_raw   = _get_para(val_raw)
        para_test_raw  = _get_para(test_raw)
        para_train, _pmean, _pstd = normalize_para_features(para_train_raw)
        para_val,   _, _          = normalize_para_features(para_val_raw, _pmean, _pstd)
        para_test,  _, _          = normalize_para_features(para_test_raw, _pmean, _pstd)

    # ── Segment-level audio features (for attention pooling) ─────────────────
    seg_train = seg_val = seg_test = None
    nsegs_train = nsegs_val = nsegs_test = None
    has_seg_train = has_seg_val = has_seg_test = None
    if use_seg_audio:
        from config import RESULTS_DIR
        from audio_extractor import build_audio_index, compute_wav2vec2_segment_features
        cache_path = RESULTS_DIR / _wav2vec_cache_name("_seg")

        audio_index = build_audio_index()
        all_sfeats, all_nsegs, all_sha = compute_wav2vec2_segment_features(
            all_samples, audio_index, device=bert_device, cache_path=cache_path,
        )
        s2i_seg = {id(s): i for i, s in enumerate(all_samples)}

        def _get_seg(split_samples):
            idxs = [s2i_seg[id(s)] for s in split_samples]
            return all_sfeats[idxs], all_nsegs[idxs], all_sha[idxs]

        seg_train, nsegs_train, has_seg_train = _get_seg(train_raw)
        seg_val,   nsegs_val,   has_seg_val   = _get_seg(val_raw)
        seg_test,  nsegs_test,  has_seg_test  = _get_seg(test_raw)
        # derive has_audio from segment availability if not already set
        if has_audio_train is None:
            has_audio_train, has_audio_val, has_audio_test = has_seg_train, has_seg_val, has_seg_test

    train_ds = NeuroBrainDataset(train_raw, wab_mean=wab_mean, wab_std=wab_std,
                                  bert_feats=bert_train,
                                  audio_feats=audio_train, has_audio=has_audio_train,
                                  para_feats=para_train,
                                  audio_seg_feats=seg_train, n_segs=nsegs_train)
    val_ds   = NeuroBrainDataset(val_raw,
                                  feat_mean=train_ds.feat_mean,
                                  feat_std=train_ds.feat_std,
                                  wab_mean=wab_mean, wab_std=wab_std,
                                  bert_feats=bert_val,
                                  audio_feats=audio_val, has_audio=has_audio_val,
                                  para_feats=para_val,
                                  audio_seg_feats=seg_val, n_segs=nsegs_val)
    test_ds  = NeuroBrainDataset(test_raw,
                                  feat_mean=train_ds.feat_mean,
                                  feat_std=train_ds.feat_std,
                                  wab_mean=wab_mean, wab_std=wab_std,
                                  bert_feats=bert_test,
                                  audio_feats=audio_test, has_audio=has_audio_test,
                                  para_feats=para_test,
                                  audio_seg_feats=seg_test, n_segs=nsegs_test)

    if verbose:
        from collections import Counter
        from config import IDX_TO_LABEL
        _lbl_name = {k: v[:4] for k, v in IDX_TO_LABEL.items()}
        for split_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
            cnt   = Counter(s["label"] for s in ds.samples)
            parts = "  ".join(f"{_lbl_name[l]}={cnt[l]}" for l in sorted(cnt))
            print(f"[Split] {split_name:5s}={len(ds):4d}  ({parts})")

    return train_ds, val_ds, test_ds


def load_loco_dataset(exclude_corpus: str,
                      verbose: bool = True,
                      use_bert: bool = False,
                      bert_device: str = "cuda",
                      use_audio: bool = False,
                      use_para: bool = False,
                      use_seg_audio: bool = False):
    """
    Leave-One-Corpus-Out: hold out exclude_corpus as the test set;
    remaining samples get a stratified per-class 85/15 train/val split.
    Supported corpus tags: "ADReSSo", "Coelho", "Delaware", "RHD-English".
    """
    meta_df = pd.read_csv(METADATA_CSV)

    tbi_demo = _load_tbi_demo()
    rhd_demo = _load_rhd_demo()
    mci_demo = _load_mci_demo()

    aphasia_samples = _load_aphasia_samples(APHASIA_TEXT_DIR, meta_df)
    ad_samples      = (
        _load_ad_samples(AD_TRAIN_DIR, AD_TRAIN_META_CSV) +
        _load_ad_samples(AD_TEST_DIR,  AD_TEST_META_CSV)
    )
    tbi_samples  = _load_txt_dir_samples(TBI_CTRL_DIR, TBI_TBI_DIR, "TBI", "Coelho",
                                          demo_lookup=tbi_demo)
    rhd_samples  = _load_txt_dir_samples(RHD_CTRL_DIR, RHD_RHD_DIR, "RHD", "RHD-English",
                                          demo_lookup=rhd_demo)
    mci_samples  = _load_txt_dir_samples(MCI_CTRL_DIR, MCI_MCI_DIR, "MCI", "Delaware",
                                          demo_lookup=mci_demo)

    all_samples = aphasia_samples + ad_samples + tbi_samples + rhd_samples + mci_samples

    test_raw  = [s for s in all_samples if s["corpus"] == exclude_corpus]
    remaining = [s for s in all_samples if s["corpus"] != exclude_corpus]

    if not test_raw:
        raise ValueError(f"No samples found for corpus='{exclude_corpus}'. "
                         f"Available: {sorted(set(s['corpus'] for s in all_samples))}")

    valid_wab = [s["wab_aq"] for s in all_samples if s["wab_aq"] >= 0]
    wab_mean  = float(np.mean(valid_wab)) if valid_wab else 70.0
    wab_std   = float(np.std(valid_wab))  if valid_wab else 20.0

    # Stratified per-class 85/15 train/val split of remaining samples
    by_label = defaultdict(list)
    for s in remaining:
        by_label[s["label"]].append(s)

    rng = np.random.default_rng(SEED)
    train_raw, val_raw = [], []
    for label_id in sorted(by_label.keys()):
        samps = by_label[label_id]
        idx   = rng.permutation(len(samps))
        n_val = max(1, round(len(samps) * VAL_RATIO))
        val_raw.extend([samps[i] for i in idx[:n_val]])
        train_raw.extend([samps[i] for i in idx[n_val:]])

    if verbose:
        from collections import Counter
        from config import IDX_TO_LABEL
        print(f"[LOCO] exclude_corpus={exclude_corpus!r} | "
              f"test={len(test_raw)} train={len(train_raw)} val={len(val_raw)}")
        disease_cnt = Counter(s["label"] for s in test_raw)
        print(f"[LOCO] test classes: {dict(disease_cnt)}")

    # BERT features (computed over all samples so the cache is reusable)
    bert_train = bert_val = bert_test = None
    if use_bert:
        from config import RESULTS_DIR
        cache_path = RESULTS_DIR / "bert_features_cache.pt"

        all_texts = [s["text"] for s in all_samples]
        all_feats = _compute_bert_features(all_texts, device=bert_device,
                                            cache_path=cache_path)
        text_to_idx = {id(s): i for i, s in enumerate(all_samples)}

        def _get_bert(split_samples):
            return np.stack([all_feats[text_to_idx[id(s)]] for s in split_samples])

        bert_train = _get_bert(train_raw)
        bert_val   = _get_bert(val_raw)
        bert_test  = _get_bert(test_raw)

    # Audio features
    audio_train = audio_val = audio_test = None
    has_audio_train = has_audio_val = has_audio_test = None
    if use_audio:
        from config import RESULTS_DIR
        from audio_extractor import build_audio_index, compute_wav2vec2_features
        cache_path = RESULTS_DIR / _wav2vec_cache_name()

        audio_index = build_audio_index()
        all_afeats, all_has = compute_wav2vec2_features(
            all_samples, audio_index, device=bert_device, cache_path=cache_path,
        )
        sample_to_idx = {id(s): i for i, s in enumerate(all_samples)}

        def _get_audio_split(split_samples):
            idxs = [sample_to_idx[id(s)] for s in split_samples]
            return all_afeats[idxs], all_has[idxs]

        audio_train, has_audio_train = _get_audio_split(train_raw)
        audio_val,   has_audio_val   = _get_audio_split(val_raw)
        audio_test,  has_audio_test  = _get_audio_split(test_raw)

    # ── Paralinguistic features ───────────────────────────────────────────────
    para_train = para_val = para_test = None
    if use_para:
        from config import RESULTS_DIR
        from audio_extractor import build_audio_index
        from paralinguistic_extractor import compute_paralinguistic_features, normalize_para_features
        cache_path  = RESULTS_DIR / "para_cache.pt"
        audio_index = build_audio_index()
        all_pfeats, _phas = compute_paralinguistic_features(
            all_samples, audio_index, cache_path=cache_path,
        )
        s2ip = {id(s): i for i, s in enumerate(all_samples)}

        def _get_para_loco(split_samples):
            return all_pfeats[[s2ip[id(s)] for s in split_samples]]

        para_train_raw = _get_para_loco(train_raw)
        para_val_raw   = _get_para_loco(val_raw)
        para_test_raw  = _get_para_loco(test_raw)
        para_train, _pmean, _pstd = normalize_para_features(para_train_raw)
        para_val,   _, _ = normalize_para_features(para_val_raw,  _pmean, _pstd)
        para_test,  _, _ = normalize_para_features(para_test_raw, _pmean, _pstd)

    # ── Segment-level audio features ──────────────────────────────────────────
    seg_train = seg_val = seg_test = None
    nsegs_train = nsegs_val = nsegs_test = None
    has_seg_train = has_seg_val = has_seg_test = None
    if use_seg_audio:
        from config import RESULTS_DIR
        from audio_extractor import build_audio_index, compute_wav2vec2_segment_features
        cache_path  = RESULTS_DIR / _wav2vec_cache_name("_seg")
        audio_index = build_audio_index()
        all_sfeats, all_nsegs, all_sha = compute_wav2vec2_segment_features(
            all_samples, audio_index, device=bert_device, cache_path=cache_path,
        )
        s2is = {id(s): i for i, s in enumerate(all_samples)}

        def _get_seg_loco(split_samples):
            idxs = [s2is[id(s)] for s in split_samples]
            return all_sfeats[idxs], all_nsegs[idxs], all_sha[idxs]

        seg_train, nsegs_train, has_seg_train = _get_seg_loco(train_raw)
        seg_val,   nsegs_val,   has_seg_val   = _get_seg_loco(val_raw)
        seg_test,  nsegs_test,  has_seg_test  = _get_seg_loco(test_raw)
        if has_audio_train is None:
            has_audio_train, has_audio_val, has_audio_test = has_seg_train, has_seg_val, has_seg_test

    train_ds = NeuroBrainDataset(train_raw, wab_mean=wab_mean, wab_std=wab_std,
                                  bert_feats=bert_train,
                                  audio_feats=audio_train, has_audio=has_audio_train,
                                  para_feats=para_train,
                                  audio_seg_feats=seg_train, n_segs=nsegs_train)
    val_ds   = NeuroBrainDataset(val_raw,
                                  feat_mean=train_ds.feat_mean, feat_std=train_ds.feat_std,
                                  wab_mean=wab_mean, wab_std=wab_std,
                                  bert_feats=bert_val,
                                  audio_feats=audio_val, has_audio=has_audio_val,
                                  para_feats=para_val,
                                  audio_seg_feats=seg_val, n_segs=nsegs_val)
    test_ds  = NeuroBrainDataset(test_raw,
                                  feat_mean=train_ds.feat_mean, feat_std=train_ds.feat_std,
                                  wab_mean=wab_mean, wab_std=wab_std,
                                  bert_feats=bert_test,
                                  audio_feats=audio_test, has_audio=has_audio_test,
                                  para_feats=para_test,
                                  audio_seg_feats=seg_test, n_segs=nsegs_test)
    return train_ds, val_ds, test_ds


def make_longitudinal_pairs(dataset: NeuroBrainDataset, n_pairs: int = 50):
    from collections import defaultdict
    by_corpus_label = defaultdict(list)
    for i, s in enumerate(dataset.samples):
        key = (s["corpus"], s["label"])
        by_corpus_label[key].append(i)

    pairs = []
    rng   = np.random.default_rng(SEED)
    for key, idxs in by_corpus_label.items():
        if len(idxs) < 2:
            continue
        chosen = rng.choice(idxs, size=min(n_pairs, len(idxs)//2 * 2), replace=False)
        for j in range(0, len(chosen)-1, 2):
            pairs.append((chosen[j], chosen[j+1]))
        if len(pairs) >= n_pairs:
            break
    return pairs[:n_pairs]
