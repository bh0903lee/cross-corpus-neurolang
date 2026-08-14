#!/usr/bin/env python3
"""Pre-compute frozen RoBERTa-large [CLS] features for all transcripts.

Writes results/feats_roberta-large.pt with keys {"texts", "feats", "model_name"},
which run_rl_all_experiments.py loads at training time.
"""
import sys
from pathlib import Path

import numpy as np
import torch

_POC = Path(__file__).resolve().parent
if str(_POC) not in sys.path:
    sys.path.insert(0, str(_POC))

from config import RESULTS_DIR
from data_loader import load_dataset

MODEL_NAME = "roberta-large"
MAX_LEN = 512
BATCH = 16


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    train_ds, val_ds, test_ds = load_dataset(
        verbose=False, use_bert=False, use_audio=False)
    texts = []
    seen = set()
    for ds in (train_ds, val_ds, test_ds):
        for s in ds.samples:
            if s["text"] not in seen:
                seen.add(s["text"])
                texts.append(s["text"])

    feats = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i + BATCH]
            enc = tokenizer(batch, truncation=True, max_length=MAX_LEN,
                            padding=True, return_tensors="pt").to(device)
            out = model(**enc)
            feats.append(out.last_hidden_state[:, 0, :].cpu())
            if (i // BATCH) % 20 == 0:
                print(f"  {i}/{len(texts)}")
    feat_arr = torch.cat(feats, dim=0)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"feats_{MODEL_NAME}.pt"
    torch.save({"texts": texts, "feats": feat_arr, "model_name": MODEL_NAME},
               out_path)
    print(f"Saved {feat_arr.shape} -> {out_path}")


if __name__ == "__main__":
    main()
