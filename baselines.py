"""
baselines.py — Baseline comparison models.

Layer 1 — Trivial:
  Random, MostFrequent, ClassPrior

Layer 2 — Traditional ML (same data, same split):
  SVM-RBF (ling+tab),  SVM-RBF (BERT+tab)
  RandomForest (ling+tab),  LogisticRegression (ling+tab)

Layer 3 — Neural baselines:
  DistilBERT Fine-tune (6-way, end-to-end)
  No-Q-Former concat (proposed architecture without the Q-Former; same as ablation A1)

Entry point: run_all_baselines(train_ds, val_ds, test_ds, device, seeds) → dict
"""
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from config import N_CLASSES, IDX_TO_LABEL, LABEL_MAP, BERT_MODEL_NAME
from statistical_analysis import aggregate_seeds

logger = logging.getLogger(__name__)


# ─── Feature extraction ───────────────────────────────────────────────────────

def _extract_features(ds, feature_type: str = "ling+tab"):
    """
    Extract (X, y) arrays for sklearn from a NeuroBrainDataset.

    feature_type:
      "ling+tab"  : 12-dim ling + 4-dim tab = 16-dim
      "bert+tab"  : 768-dim BERT + 4-dim tab = 772-dim (requires bert_feats)
      "ling"      : 12-dim ling features only
      "tab"       : 4-dim tabular features only
    """
    text  = ds.text_feats   # (N, 12)
    tab   = ds.tab_feats    # (N, 4)
    y     = np.array([s["label"] for s in ds.samples])

    if feature_type == "ling+tab":
        X = np.concatenate([text, tab], axis=1)
    elif feature_type == "bert+tab":
        if ds.bert_feats is None:
            raise ValueError("bert+tab requested but bert_feats is missing")
        X = np.concatenate([ds.bert_feats, tab], axis=1)
    elif feature_type == "ling":
        X = text
    elif feature_type == "tab":
        X = tab
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    return X.astype(np.float32), y


# ─── Evaluation utilities ─────────────────────────────────────────────────────

def _sklearn_metrics(y_true, y_pred) -> dict:
    prec, rec, f1s, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(N_CLASSES)), zero_division=0
    )
    return {
        "test_accuracy":    float(accuracy_score(y_true, y_pred)),
        "test_macro_f1":    float(f1_score(y_true, y_pred, average="macro",    zero_division=0)),
        "test_weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "test_per_class": {
            IDX_TO_LABEL[i]: {
                "precision": float(prec[i]), "recall": float(rec[i]),
                "f1": float(f1s[i]), "support": int(supp[i]),
            }
            for i in range(N_CLASSES)
        },
        "test_preds":  y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred),
        "test_labels": y_true.tolist() if hasattr(y_true, "tolist") else list(y_true),
    }


# ─── Layer 1: Trivial baselines ───────────────────────────────────────────────

def run_trivial_baselines(train_ds, test_ds) -> dict:
    """Random, MostFrequent, ClassPrior — analytic computation (no training)."""
    train_labels = np.array([s["label"] for s in train_ds.samples])
    test_labels  = np.array([s["label"] for s in test_ds.samples])

    counts = np.bincount(train_labels, minlength=N_CLASSES)
    priors = counts / counts.sum()
    n_test = len(test_labels)

    # Random uniform
    rng = np.random.default_rng(42)
    random_preds = rng.integers(0, N_CLASSES, size=n_test)

    # Most-frequent class
    most_freq_cls  = int(np.argmax(counts))
    mf_preds = np.full(n_test, most_freq_cls)

    # Class-prior sampling
    prior_preds = rng.choice(N_CLASSES, size=n_test, p=priors)

    def _wrap(m):
        return {"seed_results": [m], "aggregated": aggregate_seeds([m]), **m}

    return {
        "Random":       _wrap(_sklearn_metrics(test_labels, random_preds)),
        "MostFrequent": _wrap(_sklearn_metrics(test_labels, mf_preds)),
        "ClassPrior":   _wrap(_sklearn_metrics(test_labels, prior_preds)),
    }


# ─── Layer 2: Traditional ML baselines ────────────────────────────────────────

def _run_single_sklearn(clf_factory, X_train, y_train, X_test, y_test) -> dict:
    clf = clf_factory()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return _sklearn_metrics(y_test, preds)


def run_sklearn_baselines(train_ds, test_ds, seeds: list = None) -> dict:
    """
    SVM, RF, LR — based on ling+tab features.
    If seeds are given, RF/LR are run with multiple seeds and aggregated as mean±std.
    SVM is deterministic, so it runs once.
    """
    if seeds is None:
        seeds = [42]

    X_tr, y_tr = _extract_features(train_ds, "ling+tab")
    X_te, y_te = _extract_features(test_ds,  "ling+tab")

    results = {}

    # SVM-RBF (deterministic)
    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",   SVC(kernel="rbf", C=10, class_weight="balanced",
                      probability=False, random_state=42)),
    ])
    def _wrap_single(m):
        return {"seed_results": [m], "aggregated": aggregate_seeds([m]), **m}

    results["SVM-RBF (ling+tab)"] = _wrap_single(_run_single_sklearn(
        lambda: svm_pipe, X_tr, y_tr, X_te, y_te
    ))

    # SVM-RBF with BERT features (if available)
    if train_ds.bert_feats is not None:
        X_tr_b, _ = _extract_features(train_ds, "bert+tab")
        X_te_b, _ = _extract_features(test_ds,  "bert+tab")
        svm_bert  = Pipeline([
            ("scaler", StandardScaler()),
            ("svm",   SVC(kernel="rbf", C=10, class_weight="balanced",
                          probability=False, random_state=42)),
        ])
        results["SVM-RBF (BERT+tab)"] = _wrap_single(_run_single_sklearn(
            lambda: svm_bert, X_tr_b, y_tr, X_te_b, y_te
        ))

    # RF and LR: multi-seed
    for name, factory in [
        ("RandomForest (ling+tab)", lambda s: Pipeline([
            ("rf", RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                max_depth=None, random_state=s, n_jobs=-1)),
        ])),
        ("LogisticRegression (ling+tab)", lambda s: Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=1.0, class_weight="balanced",
                max_iter=2000, random_state=s, solver="lbfgs")),
        ])),
    ]:
        seed_results = []
        for s in seeds:
            clf   = factory(s)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)
            seed_results.append(_sklearn_metrics(y_te, preds))

        results[name] = {
            "seed_results": seed_results,
            "aggregated":   aggregate_seeds(seed_results),
        }
        if len(seed_results) == 1:
            results[name].update(seed_results[0])

    return results


# ─── Layer 3: DistilBERT Fine-tune ────────────────────────────────────────────

class _BertFinetuneDataset(Dataset):
    def __init__(self, samples: list, tokenizer, max_length: int = 256):
        self.samples    = samples
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        enc = self.tokenizer(
            s["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(s["label"], dtype=torch.long),
        }


class BertFinetuneClassifier(nn.Module):
    """DistilBERT + linear classification head (end-to-end fine-tune)."""
    def __init__(self, n_classes: int = N_CLASSES, dropout: float = 0.2):
        super().__init__()
        from transformers import DistilBertModel
        self.bert       = DistilBertModel.from_pretrained(BERT_MODEL_NAME)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, n_classes)

    def forward(self, input_ids, attention_mask):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_h  = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_h))


def run_bert_finetune(train_ds, val_ds, test_ds, device,
                      seeds: list = None,
                      n_epochs: int = 20,
                      lr: float = 2e-5,
                      batch_size: int = 16,
                      output_dir: Path = None) -> dict:
    """
    DistilBERT end-to-end fine-tune (6-way classification).
    CrossEntropyLoss with class weights.
    """
    if seeds is None:
        seeds = [42]

    from transformers import DistilBertTokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL_NAME)

    # Class weights (from train set)
    train_labels = np.array([s["label"] for s in train_ds.samples])
    counts       = np.bincount(train_labels, minlength=N_CLASSES).astype(float)
    cls_weights  = torch.tensor(
        1.0 / np.maximum(counts, 1) / (1.0 / np.maximum(counts, 1)).sum() * N_CLASSES,
        dtype=torch.float32,
    ).to(device)

    seed_results = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        model    = BertFinetuneClassifier().to(device)
        train_ft = _BertFinetuneDataset(train_ds.samples, tokenizer)
        val_ft   = _BertFinetuneDataset(val_ds.samples,   tokenizer)
        test_ft  = _BertFinetuneDataset(test_ds.samples,  tokenizer)

        tr_loader  = DataLoader(train_ft, batch_size=batch_size, shuffle=True,  num_workers=0)
        val_loader = DataLoader(val_ft,   batch_size=batch_size, shuffle=False, num_workers=0)
        te_loader  = DataLoader(test_ft,  batch_size=batch_size, shuffle=False, num_workers=0)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss(weight=cls_weights)

        use_amp  = (device.type == "cuda")
        scaler   = torch.amp.GradScaler("cuda", enabled=use_amp)
        best_val = -1.0
        best_state = None

        for epoch in range(n_epochs):
            model.train()
            for batch in tr_loader:
                ids  = batch["input_ids"].to(device,      non_blocking=True)
                mask = batch["attention_mask"].to(device, non_blocking=True)
                lbl  = batch["label"].to(device,          non_blocking=True)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(ids, mask)
                    loss   = criterion(logits, lbl)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            # Val accuracy
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for batch in val_loader:
                    ids  = batch["input_ids"].to(device)
                    mask = batch["attention_mask"].to(device)
                    lbl  = batch["label"].to(device)
                    logits = model(ids, mask)
                    correct += (logits.argmax(-1) == lbl).sum().item()
                    total   += lbl.size(0)
            val_acc = correct / max(total, 1)
            if val_acc > best_val:
                best_val   = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info("BertFinetune seed=%d epoch=%d/%d val_acc=%.4f",
                        seed, epoch + 1, n_epochs, val_acc)

        # Load best and evaluate on test
        model.load_state_dict(best_state)
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in te_loader:
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                logits = model(ids, mask)
                all_preds.extend(logits.argmax(-1).tolist())
                all_labels.extend(batch["label"].tolist())

        metrics = _sklearn_metrics(np.array(all_labels), np.array(all_preds))
        metrics["best_val_acc"] = best_val
        seed_results.append(metrics)

    return {
        "seed_results": seed_results,
        "aggregated":   aggregate_seeds(seed_results),
    }


# ─── Main entry point ─────────────────────────────────────────────────────────

def run_all_baselines(train_ds, val_ds, test_ds, device,
                      seeds: list = None,
                      n_epochs_bert: int = 20,
                      output_dir: Path = None) -> dict:
    """
    Run all baselines and return a results dict.
    Saves JSON if output_dir is given.
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 1234]

    results = {}

    logger.info("=== Layer 1: Trivial baselines ===")
    results["trivial"] = run_trivial_baselines(train_ds, test_ds)

    logger.info("=== Layer 2: Traditional ML baselines ===")
    results["sklearn"] = run_sklearn_baselines(train_ds, test_ds, seeds=seeds)

    logger.info("=== Layer 3: DistilBERT Fine-tune ===")
    results["bert_finetune"] = run_bert_finetune(
        train_ds, val_ds, test_ds, device,
        seeds=seeds, n_epochs=n_epochs_bert,
        output_dir=output_dir,
    )

    if output_dir:
        out = Path(output_dir) / "baseline_results.json"
        _save_json(results, out)
        logger.info("Baselines saved: %s", out)

    return results


def _save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
