"""
error_analysis.py — Misclassification pattern analysis.

Analyses:
  1. Full misclassified-sample list (patient_id, corpus, true/pred label, confidence)
  2. Per-corpus error rates (quantifying domain heterogeneity)
  3. MCI↔Control confusion: linguistic feature comparison (ttr, filler_ratio)
  4. Per-class confidence calibration — 10 bins
  5. Confusion matrix

Usage:
  from error_analysis import run_error_analysis
  results = run_error_analysis(model, train_ds, val_ds, test_ds, device)
"""
import math
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from config import N_CLASSES, IDX_TO_LABEL, LABEL_MAP
from eval_utils import get_raw_predictions


def run_error_analysis(model, train_ds, val_ds, test_ds, device,
                       encoder_type: str = "bert",
                       use_audio: bool = False,
                       output_dir: Path = None) -> dict:
    """
    Misclassification analysis over the full test set.

    Returns dict:
      misclassifications    : list[dict] — details of misclassified samples
      corpus_error_rates    : {corpus: {"n_total","n_wrong","error_rate","accuracy"}}
      mci_control_analysis  : linguistic feature comparison dict
      calibration           : list[{"bin","accuracy","count","mean_confidence"}]
      confusion_matrix      : NxN list (true × pred)
      overall_accuracy      : float
      overall_macro_f1      : float
      class_metrics         : per-class accuracy, precision, recall, f1
    """
    # --- prior-correction alpha (from val set) --------------------------------
    from eval_utils import evaluate_model
    eval_result = evaluate_model(
        model, train_ds, val_ds, test_ds, device,
        encoder_type=encoder_type, use_audio=use_audio,
        use_prior_correction=True,
    )
    best_alpha   = eval_result["prior_alpha"]
    test_preds   = eval_result["test_preds"]
    test_labels  = eval_result["test_labels"]
    test_probs   = np.array(eval_result["test_probs"])  # (N, C)

    # adjusted confidence = max prob after alpha shift (approx)
    confidence   = test_probs.max(axis=1).tolist()

    # --- Collect sample metadata ----------------------------------------------
    samples = test_ds.samples
    text_feats = test_ds.text_feats   # (N, 12) normalized ling features
    # raw features (un-normalize to get interpretable values)
    feat_mean = test_ds.feat_mean
    feat_std  = test_ds.feat_std
    raw_feats = text_feats * feat_std + feat_mean   # (N, 12)

    FEATURE_NAMES = [
        "n_words_log", "n_sent_log", "avg_sent_len", "ttr", "mattr",
        "long_word_ratio", "pronoun_ratio", "filler_ratio",
        "repetition_rate", "content_density", "pause_density", "avg_word_len",
    ]

    # --- Misclassification list -----------------------------------------------
    misclassifications = []
    for i, (true_l, pred_l) in enumerate(zip(test_labels, test_preds)):
        if true_l == pred_l:
            continue
        s = samples[i]
        misclassifications.append({
            "idx":          i,
            "patient_id":   s["patient_id"],
            "corpus":       s["corpus"],
            "true_label":   IDX_TO_LABEL[true_l],
            "pred_label":   IDX_TO_LABEL[pred_l],
            "confidence":   float(confidence[i]),
            "features":     {FEATURE_NAMES[j]: float(raw_feats[i, j])
                             for j in range(len(FEATURE_NAMES))},
        })

    # --- Per-corpus error rates ------------------------------------------------
    corpus_stats = {}
    for i, s in enumerate(samples):
        corp = s["corpus"]
        if corp not in corpus_stats:
            corpus_stats[corp] = {"n_total": 0, "n_wrong": 0}
        corpus_stats[corp]["n_total"] += 1
        if test_preds[i] != test_labels[i]:
            corpus_stats[corp]["n_wrong"] += 1

    corpus_error_rates = {}
    for corp, st in corpus_stats.items():
        n = st["n_total"]
        w = st["n_wrong"]
        corpus_error_rates[corp] = {
            "n_total":    n,
            "n_wrong":    w,
            "error_rate": round(w / n, 4) if n > 0 else 0.0,
            "accuracy":   round((n - w) / n, 4) if n > 0 else 0.0,
        }

    # --- MCI↔Control confusion: linguistic feature comparison ---------------
    # Groups: MCI correct, MCI predicted as Control, Control correct, Control predicted as MCI
    feat_idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    mci_lbl  = LABEL_MAP["MCI"]
    ctrl_lbl = LABEL_MAP["Control"]

    def _group_features(cond_fn):
        idxs = [i for i in range(len(samples)) if cond_fn(i)]
        if not idxs:
            return {}
        f = raw_feats[idxs]
        return {name: {"mean": float(f[:, j].mean()), "std": float(f[:, j].std())}
                for j, name in enumerate(FEATURE_NAMES)}

    mci_control_analysis = {
        "mci_correct":       _group_features(
            lambda i: test_labels[i] == mci_lbl and test_preds[i] == mci_lbl),
        "mci_as_control":    _group_features(
            lambda i: test_labels[i] == mci_lbl and test_preds[i] == ctrl_lbl),
        "control_correct":   _group_features(
            lambda i: test_labels[i] == ctrl_lbl and test_preds[i] == ctrl_lbl),
        "control_as_mci":    _group_features(
            lambda i: test_labels[i] == ctrl_lbl and test_preds[i] == mci_lbl),
        "n_mci_total":       sum(1 for l in test_labels if l == mci_lbl),
        "n_mci_as_control":  sum(1 for tl, pl in zip(test_labels, test_preds)
                                 if tl == mci_lbl and pl == ctrl_lbl),
    }

    # --- Calibration curve -----------------------------------------------------
    n_bins = 10
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    calibration = []
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask   = [(lo <= confidence[i] < hi) for i in range(len(confidence))]
        if not any(mask):
            continue
        idxs = [i for i, m in enumerate(mask) if m]
        acc  = np.mean([test_preds[i] == test_labels[i] for i in idxs])
        calibration.append({
            "bin":              f"{lo:.1f}-{hi:.1f}",
            "mean_confidence":  float(np.mean([confidence[i] for i in idxs])),
            "accuracy":         float(acc),
            "count":            len(idxs),
        })

    # --- Confusion matrix --------------------------------------------------------
    present_labels = sorted(set(test_labels))
    cm = confusion_matrix(test_labels, test_preds, labels=list(range(N_CLASSES))).tolist()

    # --- Per-class metrics -------------------------------------------------------
    from sklearn.metrics import precision_recall_fscore_support
    prec, rec, f1s, supp = precision_recall_fscore_support(
        test_labels, test_preds, labels=list(range(N_CLASSES)), zero_division=0
    )
    class_metrics = {
        IDX_TO_LABEL[i]: {
            "precision": float(prec[i]), "recall": float(rec[i]),
            "f1": float(f1s[i]), "support": int(supp[i]),
            "accuracy": float(np.mean([
                test_preds[j] == test_labels[j]
                for j in range(len(test_labels)) if test_labels[j] == i
            ])) if supp[i] > 0 else 0.0,
        }
        for i in range(N_CLASSES)
    }

    results = {
        "overall_accuracy":   float(accuracy_score(test_labels, test_preds)),
        "overall_macro_f1":   float(f1_score(test_labels, test_preds,
                                             average="macro", zero_division=0)),
        "prior_alpha":        best_alpha,
        "n_misclassified":    len(misclassifications),
        "n_total":            len(test_labels),
        "misclassifications": misclassifications,
        "corpus_error_rates": corpus_error_rates,
        "mci_control_analysis": mci_control_analysis,
        "calibration":        calibration,
        "confusion_matrix":   cm,
        "confusion_labels":   [IDX_TO_LABEL[i] for i in range(N_CLASSES)],
        "class_metrics":      class_metrics,
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "error_analysis.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[Error Analysis] saved: {out_path}")

    return results
