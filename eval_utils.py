"""
eval_utils.py — shared model-evaluation utilities.

Includes prior-correction calibration: search the optimal α on the val set,
then apply it to the test set (α=0: no correction, α=1: full Bayes correction).
"""
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
)
from config import N_CLASSES, IDX_TO_LABEL, BATCH_SIZE


def evaluate_model(model, train_ds, val_ds, test_ds, device,
                   encoder_type: str = "bert",
                   use_audio: bool = False,
                   use_prior_correction: bool = True,
                   batch_size: int = None) -> dict:
    """
    Evaluate on val/test sets, with prior-correction calibration.
    Returns a dict of accuracy / macro-F1 / weighted-F1 per split, per-class
    metrics, predictions, labels, softmax probabilities, and prior_alpha.
    """
    bs       = batch_size or BATCH_SIZE
    feat_key = "bert_feat" if encoder_type == "bert" else "text_feat"

    def _loader(ds):
        return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0,
                          pin_memory=(device.type == "cuda"))

    def _get_audio(batch):
        if "audio_feat" not in batch:
            return None, None
        return (batch["audio_feat"].to(device, non_blocking=True),
                batch["has_audio"].to(device,  non_blocking=True))

    def _get_extras(batch):
        sf = (batch["audio_seg_feat"].to(device, non_blocking=True)
              if "audio_seg_feat" in batch else None)
        ns = (batch["n_segs"].to(device, non_blocking=True)
              if "n_segs" in batch else None)
        pf = (batch["para_feat"].to(device, non_blocking=True)
              if "para_feat" in batch else None)
        return sf, ns, pf

    # Prior-correction setup
    log_prior_adj = None
    if use_prior_correction:
        train_counts = np.bincount(
            [s["label"] for s in train_ds.samples], minlength=N_CLASSES
        ).astype(float)
        p_natural    = train_counts / train_counts.sum()
        log_prior_adj = torch.tensor(
            [math.log(p_natural[c] * N_CLASSES + 1e-9) for c in range(N_CLASSES)],
            dtype=torch.float32,
        )

    # Collect raw logits + labels
    raw_logits_map = {}
    raw_labels_map = {}
    raw_probs_map  = {}

    model.eval()
    with torch.no_grad():
        for split_name, ds in [("val", val_ds), ("test", test_ds)]:
            logits_list, labels_list = [], []
            for batch in _loader(ds):
                tf  = batch[feat_key].to(device, non_blocking=True)
                tab = batch["tab_feat"].to(device, non_blocking=True)
                af, ha = _get_audio(batch)
                sf, ns, pf = _get_extras(batch)
                out = model(tf, tab, af, ha, sf, ns, pf)
                logits_list.append(out["cls_logits"].cpu())
                labels_list.extend(batch["label"].tolist())
            raw_logits = torch.cat(logits_list, dim=0)
            raw_logits_map[split_name] = raw_logits
            raw_labels_map[split_name] = labels_list
            raw_probs_map[split_name]  = torch.softmax(raw_logits, dim=-1).numpy()

    # Find best alpha on val set
    best_alpha = 0.0
    if use_prior_correction:
        best_score = -1.0
        for alpha in [round(a * 0.1, 1) for a in range(-5, 16)]:
            adj   = raw_logits_map["val"] + alpha * log_prior_adj.unsqueeze(0)
            preds = adj.argmax(-1).tolist()
            acc   = accuracy_score(raw_labels_map["val"], preds)
            mf1   = f1_score(raw_labels_map["val"], preds, average="macro", zero_division=0)
            score = 0.5 * acc + 0.5 * mf1
            if score > best_score:
                best_score = score
                best_alpha = alpha

    results = {"prior_alpha": best_alpha}

    for split_name in ("val", "test"):
        logits = raw_logits_map[split_name]
        if use_prior_correction:
            logits = logits + best_alpha * log_prior_adj.unsqueeze(0)

        labels = raw_labels_map[split_name]
        preds  = logits.argmax(-1).tolist()

        acc = accuracy_score(labels, preds)
        mf1 = f1_score(labels, preds, average="macro",    zero_division=0)
        wf1 = f1_score(labels, preds, average="weighted", zero_division=0)

        present = sorted(set(labels))
        prec, rec, f1s, supp = precision_recall_fscore_support(
            labels, preds, labels=present, zero_division=0
        )
        per_class = {
            IDX_TO_LABEL[present[i]]: {
                "precision": float(prec[i]), "recall": float(rec[i]),
                "f1": float(f1s[i]), "support": int(supp[i]),
            }
            for i in range(len(present))
        }

        results[f"{split_name}_accuracy"]    = float(acc)
        results[f"{split_name}_macro_f1"]    = float(mf1)
        results[f"{split_name}_weighted_f1"] = float(wf1)
        results[f"{split_name}_per_class"]   = per_class
        results[f"{split_name}_preds"]       = preds
        results[f"{split_name}_labels"]      = labels
        results[f"{split_name}_probs"]       = raw_probs_map[split_name].tolist()

    return results


def get_raw_predictions(model, ds, device, encoder_type: str = "bert",
                        use_audio: bool = False, batch_size: int = None):
    """
    Return raw predictions for a dataset:
    logits (N, C), probs (N, C), labels list, patient_ids list, corpora list.
    """
    bs       = batch_size or BATCH_SIZE
    feat_key = "bert_feat" if encoder_type == "bert" else "text_feat"

    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0,
                        pin_memory=(device.type == "cuda"))

    all_logits, all_labels = [], []
    all_pids, all_corpora  = [], []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            tf  = batch[feat_key].to(device, non_blocking=True)
            tab = batch["tab_feat"].to(device, non_blocking=True)
            af  = batch.get("audio_feat", None)
            ha  = batch.get("has_audio",  None)
            if af is not None:
                af = af.to(device, non_blocking=True)
                ha = ha.to(device, non_blocking=True)
            out = model(tf, tab, af, ha)
            all_logits.append(out["cls_logits"].cpu())
            all_labels.extend(batch["label"].tolist())
            all_pids.extend(batch["patient_id"])
            all_corpora.extend(batch["corpus"])

    logits = torch.cat(all_logits, dim=0)
    probs  = torch.softmax(logits, dim=-1).numpy()
    return logits.numpy(), probs, all_labels, all_pids, all_corpora
