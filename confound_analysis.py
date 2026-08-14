#!/usr/bin/env python3
"""
confound_analysis.py — corpus-confound analyses for the proposed model.

Steps:
  e2      : per-corpus Control recall (re-aggregated from saved 5-seed predictions)
  e2prime : LOCO Control-F1 re-aggregation + held-out disease prediction distribution
  e3prime : tabular permutation test (proposed checkpoints, tab_feat shuffled at inference)
  e3w     : WAB-features-fixed evaluation (has_wab / wab_aq_norm set to population default)
  e1      : corpus-identity linear probe (128-d shared repr vs raw 27-d baseline)

Usage:
  python confound_analysis.py --step e2
  python confound_analysis.py --step e2prime
  python confound_analysis.py --step e3prime --device cuda:0
  python confound_analysis.py --step e1      --device cuda:0

Outputs: results/confound_analysis/
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_POC = Path(__file__).resolve().parent
if str(_POC) not in sys.path:
    sys.path.insert(0, str(_POC))

from config import IDX_TO_LABEL, LABEL_MAP, N_CLASSES

OUT = _POC / "results" / "confound_analysis"
OUT.mkdir(parents=True, exist_ok=True)
RL_DIR = _POC / "results" / "rl_experiments"
SEEDS = [42, 123, 456, 789, 1234]
TOP_CORPORA = {"ADReSSo", "Coelho", "RHD-English", "Delaware"}
LOCO_EXPS = ["LOCO_ADReSSo", "LOCO_Coelho", "LOCO_RHD", "LOCO_Delaware"]


def top_corpus(tag: str) -> str:
    """Map AphasiaBank sub-corpus names to 'AphasiaBank'; keep other corpora as-is."""
    return tag if tag in TOP_CORPORA else "AphasiaBank"


def _mean_std(xs):
    return {"mean": float(np.mean(xs)), "std": float(np.std(xs)), "values": [float(x) for x in xs]}


# ─────────────────────────────────────────────────────────────────────────────
# Shared: data loading (same arguments as evaluation to preserve split order)
# ─────────────────────────────────────────────────────────────────────────────
def load_full_data(device_str="cpu", light=False):
    from data_loader import load_dataset
    from run_rl_all_experiments import _inject_roberta_large
    if light:
        # For sample-order (corpus tag) checks — load without features
        train_ds, val_ds, test_ds = load_dataset(
            verbose=False, use_bert=False, use_audio=False,
            use_para=False, use_seg_audio=False, bert_device="cpu")
    else:
        train_ds, val_ds, test_ds = load_dataset(
            verbose=False, use_bert=False, use_audio=True,
            use_para=True, use_seg_audio=True, bert_device=device_str)
        _inject_roberta_large(train_ds, val_ds, test_ds)
    return train_ds, val_ds, test_ds


def _check_alignment(ds, stored_labels, name):
    ds_labels = [int(s["label"]) for s in ds.samples]
    if ds_labels != [int(x) for x in stored_labels]:
        raise RuntimeError(f"[{name}] stored predictions and dataset label order mismatch — needs review")


# ─────────────────────────────────────────────────────────────────────────────
# E2 — per-corpus Control recall (proposed, re-aggregated from saved predictions)
# ─────────────────────────────────────────────────────────────────────────────
def run_e2():
    result = json.load(open(RL_DIR / "proposed" / "result.json"))
    seed_results = {sr["seed"]: sr for sr in result["seed_results"]}

    _, _, test_ds = load_full_data(light=True)
    _check_alignment(test_ds, seed_results[SEEDS[0]]["test_labels"], "proposed/test")
    corpora_raw = [s["corpus"] for s in test_ds.samples]
    corpora = [top_corpus(c) for c in corpora_raw]
    pids = [s["patient_id"] for s in test_ds.samples]
    ctrl = LABEL_MAP["Control"]

    # CSV: all samples (patient_id, corpus, true, pred, seed)
    csv_path = OUT / "predictions_test_5seed.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "corpus_raw", "corpus", "true", "pred", "seed"])
        for seed in SEEDS:
            sr = seed_results[seed]
            for i, (t, p) in enumerate(zip(sr["test_labels"], sr["test_preds"])):
                w.writerow([pids[i], corpora_raw[i], corpora[i],
                            IDX_TO_LABEL[int(t)], IDX_TO_LABEL[int(p)], seed])

    corpus_names = sorted(set(corpora))
    per_corpus = {}
    for cname in corpus_names:
        idxs = [i for i, c in enumerate(corpora)
                if c == cname and seed_results[SEEDS[0]]["test_labels"][i] == ctrl]
        recalls, err_counter = [], Counter()
        for seed in SEEDS:
            sr = seed_results[seed]
            preds = [int(sr["test_preds"][i]) for i in idxs]
            recalls.append(float(np.mean([p == ctrl for p in preds])) if idxs else float("nan"))
            err_counter.update(IDX_TO_LABEL[p] for p in preds if p != ctrl)
        per_corpus[cname] = {
            "n_control_test": len(idxs),
            "control_recall": _mean_std(recalls),
            "misclassified_to_5seed_total": dict(err_counter),
        }

    # Overall Control recall (sanity check: must match the per-class Control-F1 results)
    all_idxs = [i for i, t in enumerate(seed_results[SEEDS[0]]["test_labels"]) if t == ctrl]
    overall = [float(np.mean([int(seed_results[s]["test_preds"][i]) == ctrl for i in all_idxs]))
               for s in SEEDS]

    out = {
        "description": "E2: per-corpus Control recall on fixed test split, proposed 5-seed",
        "n_test": len(corpora), "n_control_test": len(all_idxs),
        "overall_control_recall": _mean_std(overall),
        "per_corpus": per_corpus,
    }
    json.dump(out, open(OUT / "e2_control_recall.json", "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"[saved] {OUT/'e2_control_recall.json'}, {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# E2′ — LOCO Control-F1 (re-aggregated from saved predictions)
# ─────────────────────────────────────────────────────────────────────────────
def run_e2prime():
    out = {"description": "E2': LOCO held-out corpus Control P/R/F1 + disease pred distribution"}
    for exp in LOCO_EXPS:
        result = json.load(open(RL_DIR / exp / "result.json"))
        disease = result["loco_disease"]
        d_idx = LABEL_MAP[disease]
        ctrl = LABEL_MAP["Control"]
        prec, rec, f1 = [], [], []
        dis_dist, ctrl_err = Counter(), Counter()
        n_ctrl = n_dis = None
        for sr in result["seed_results"]:
            pc = sr["test_per_class"]["Control"]
            prec.append(pc["precision"]); rec.append(pc["recall"]); f1.append(pc["f1"])
            n_ctrl = pc["support"]
            dis_idxs = [i for i, t in enumerate(sr["test_labels"]) if t == d_idx]
            n_dis = len(dis_idxs)
            dis_dist.update(IDX_TO_LABEL[int(sr["test_preds"][i])] for i in dis_idxs)
            ctrl_err.update(IDX_TO_LABEL[int(sr["test_preds"][i])]
                            for i, t in enumerate(sr["test_labels"])
                            if t == ctrl and sr["test_preds"][i] != ctrl)
        n_seeds = len(result["seed_results"])
        out[exp] = {
            "held_out_disease": disease,
            "n_control": n_ctrl, "n_disease": n_dis,
            "control_precision": _mean_std(prec),
            "control_recall": _mean_std(rec),
            "control_f1": _mean_std(f1),
            "disease_pred_distribution_frac": {
                k: round(v / (n_dis * n_seeds), 4) for k, v in dis_dist.most_common()},
            "control_misclassified_to_frac": {
                k: round(v / (n_ctrl * n_seeds), 4) for k, v in ctrl_err.most_common()},
        }
    json.dump(out, open(OUT / "e2prime_loco_control_f1.json", "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"[saved] {OUT/'e2prime_loco_control_f1.json'}")


# ─────────────────────────────────────────────────────────────────────────────
# Shared: forward pass with proposed checkpoints (E3′, E1)
# ─────────────────────────────────────────────────────────────────────────────
def _load_model(seed, device):
    from run_rl_all_experiments import EXPERIMENTS, _build_model
    cfg = EXPERIMENTS["proposed"]
    model = _build_model(cfg).to(device)
    ckpt = RL_DIR / "proposed" / f"seed_{seed}" / "best_model.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    return model


def _forward_ds(model, ds, device, tab_override=None, want="cls_logits", batch_size=64):
    """Forward pass over the full dataset. tab_override: (N, tab_dim) numpy array replacing tab_feat."""
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    outs, labels = [], []
    pos = 0
    with torch.no_grad():
        for b in loader:
            bs = b["bert_feat"].size(0)
            tf = b["bert_feat"].to(device)
            if tab_override is not None:
                tab = torch.tensor(tab_override[pos:pos + bs], dtype=torch.float32).to(device)
            else:
                tab = b["tab_feat"].to(device)
            pos += bs
            af = b.get("audio_feat"); ha = b.get("has_audio")
            af = af.to(device) if af is not None else None
            ha = ha.to(device) if ha is not None else None
            sf = b.get("audio_seg_feat"); ns = b.get("n_segs"); pf = b.get("para_feat")
            sf = sf.to(device) if sf is not None else None
            ns = ns.to(device) if ns is not None else None
            pf = pf.to(device) if pf is not None else None
            o = model(tf, tab, af, ha, sf, ns, pf)
            outs.append(o[want].cpu())
            labels.extend(b["label"].tolist())
    return torch.cat(outs, 0), np.array(labels)


# ─────────────────────────────────────────────────────────────────────────────
# E3′ — tabular permutation test
# ─────────────────────────────────────────────────────────────────────────────
def run_e3prime(device_str, n_perm=100):
    from sklearn.metrics import accuracy_score, f1_score
    device = torch.device(device_str)
    result = json.load(open(RL_DIR / "proposed" / "result.json"))
    seed_results = {sr["seed"]: sr for sr in result["seed_results"]}

    _, _, test_ds = load_full_data(device_str)
    _check_alignment(test_ds, seed_results[SEEDS[0]]["test_labels"], "proposed/test")
    tab_all = np.stack([test_ds.tab_feats[i] for i in range(len(test_ds))]).astype(np.float32)

    # Prior-correction constant (same as eval_utils: based on train counts)
    train_ds, _, _ = load_full_data(light=True)
    import math
    counts = np.bincount([s["label"] for s in train_ds.samples], minlength=N_CLASSES).astype(float)
    p_nat = counts / counts.sum()
    log_prior_adj = torch.tensor([math.log(p_nat[c] * N_CLASSES + 1e-9) for c in range(N_CLASSES)])

    out = {"description": f"E3': test-time tabular permutation ({n_perm} reps), proposed 5-seed",
           "n_perm": n_perm, "per_seed": {}}
    drops_f1, drops_acc = [], []
    for seed in SEEDS:
        model = _load_model(seed, device)
        alpha = seed_results[seed]["prior_alpha"]
        labels_stored = seed_results[seed]["test_labels"]
        base_f1 = seed_results[seed]["test_macro_f1"]
        base_acc = seed_results[seed]["test_accuracy"]

        # Sanity check: reproduce without shuffling
        logits, labels = _forward_ds(model, test_ds, device)
        preds = (logits + alpha * log_prior_adj.unsqueeze(0)).argmax(-1).tolist()
        rep_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        if abs(rep_f1 - base_f1) > 1e-6:
            print(f"  [warn] seed={seed} reproduced mF1 {rep_f1:.4f} vs stored {base_f1:.4f}")

        rng = np.random.default_rng(seed)
        f1s, accs = [], []
        for _ in range(n_perm):
            perm = rng.permutation(len(test_ds))
            logits_p, _ = _forward_ds(model, test_ds, device, tab_override=tab_all[perm])
            preds_p = (logits_p + alpha * log_prior_adj.unsqueeze(0)).argmax(-1).tolist()
            f1s.append(f1_score(labels, preds_p, average="macro", zero_division=0))
            accs.append(accuracy_score(labels, preds_p))
        out["per_seed"][seed] = {
            "baseline_macro_f1": base_f1, "baseline_accuracy": base_acc,
            "reproduced_macro_f1": float(rep_f1),
            "perm_macro_f1": _mean_std(f1s), "perm_accuracy": _mean_std(accs),
            "drop_macro_f1": _mean_std([base_f1 - x for x in f1s]),
            "drop_accuracy": _mean_std([base_acc - x for x in accs]),
        }
        drops_f1.append(base_f1 - float(np.mean(f1s)))
        drops_acc.append(base_acc - float(np.mean(accs)))
        print(f"  [seed={seed}] perm mF1 {np.mean(f1s):.4f} (base {base_f1:.4f}, "
              f"drop {drops_f1[-1]*100:.2f}pp)")
    out["drop_macro_f1_5seed"] = _mean_std(drops_f1)
    out["drop_accuracy_5seed"] = _mean_std(drops_acc)
    json.dump(out, open(OUT / "e3prime_tab_permutation.json", "w"), indent=2)
    print(f"[saved] {OUT/'e3prime_tab_permutation.json'}")


# ─────────────────────────────────────────────────────────────────────────────
# E3″ — WAB availability flag + score fixed to population default (eval-only)
# ─────────────────────────────────────────────────────────────────────────────
def run_e3w(device_str):
    """Fix has_wab=0 and wab_aq_norm=0 in tab_feat = [age_norm, sex, has_wab, wab_aq_norm].
    Directly quantifies the extent to which WAB availability acts as an AphasiaBank indicator."""
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    import math
    device = torch.device(device_str)
    result = json.load(open(RL_DIR / "proposed" / "result.json"))
    seed_results = {sr["seed"]: sr for sr in result["seed_results"]}

    _, _, test_ds = load_full_data(device_str)
    _check_alignment(test_ds, seed_results[SEEDS[0]]["test_labels"], "proposed/test")
    tab_fixed = np.stack([test_ds.tab_feats[i] for i in range(len(test_ds))]).astype(np.float32)
    tab_fixed[:, 2] = 0.0   # has_wab
    tab_fixed[:, 3] = 0.0   # wab_aq_norm

    train_ds, _, _ = load_full_data(light=True)
    counts = np.bincount([s["label"] for s in train_ds.samples], minlength=N_CLASSES).astype(float)
    p_nat = counts / counts.sum()
    log_prior_adj = torch.tensor([math.log(p_nat[c] * N_CLASSES + 1e-9) for c in range(N_CLASSES)])

    out = {"description": "E3'': WAB availability flag + score fixed to population default "
                          "(has_wab=0, wab_aq_norm=0) at inference, proposed 5-seed",
           "per_seed": {}}
    drops_f1, drops_acc = [], []
    cls_recalls = {c: [] for c in range(N_CLASSES)}
    for seed in SEEDS:
        model = _load_model(seed, device)
        alpha = seed_results[seed]["prior_alpha"]
        base_f1 = seed_results[seed]["test_macro_f1"]
        base_acc = seed_results[seed]["test_accuracy"]
        logits, labels = _forward_ds(model, test_ds, device, tab_override=tab_fixed)
        preds = (logits + alpha * log_prior_adj.unsqueeze(0)).argmax(-1).numpy()
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        acc = accuracy_score(labels, preds)
        rec = recall_score(labels, preds, average=None,
                           labels=list(range(N_CLASSES)), zero_division=0)
        base_pc = seed_results[seed]["test_per_class"]
        per_class = {}
        for c in range(N_CLASSES):
            name = IDX_TO_LABEL[c]
            cls_recalls[c].append(float(rec[c]))
            per_class[name] = {"recall_fixed": float(rec[c]),
                               "recall_base": base_pc.get(name, {}).get("recall")}
        out["per_seed"][seed] = {
            "baseline_macro_f1": base_f1, "fixed_macro_f1": float(f1),
            "baseline_accuracy": base_acc, "fixed_accuracy": float(acc),
            "per_class_recall": per_class,
        }
        drops_f1.append(base_f1 - f1); drops_acc.append(base_acc - acc)
        print(f"  [seed={seed}] fixed mF1 {f1:.4f} (base {base_f1:.4f}, drop {drops_f1[-1]*100:.2f}pp)")
    out["drop_macro_f1_5seed"] = _mean_std(drops_f1)
    out["drop_accuracy_5seed"] = _mean_std(drops_acc)
    out["per_class_recall_fixed_5seed"] = {
        IDX_TO_LABEL[c]: _mean_std(v) for c, v in cls_recalls.items()}
    json.dump(out, open(OUT / "e3w_wab_default.json", "w"), indent=2)
    print(f"[saved] {OUT/'e3w_wab_default.json'}")


# ─────────────────────────────────────────────────────────────────────────────
# E1 — corpus-identity linear probe
# ─────────────────────────────────────────────────────────────────────────────
def _probe(X_tr, y_tr, X_te, y_te):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(X_tr), y_tr)
    pred = clf.predict(sc.transform(X_te))
    classes = sorted(set(list(y_tr) + list(y_te)))
    return {
        "accuracy": float(accuracy_score(y_te, pred)),
        "macro_f1": float(f1_score(y_te, pred, average="macro", zero_division=0)),
        "chance_majority": float(np.max(np.bincount(
            [classes.index(y) for y in y_te], minlength=len(classes))) / len(y_te)),
        "confusion_matrix": confusion_matrix(y_te, pred, labels=classes).tolist(),
        "classes": classes,
    }


def run_e1(device_str):
    device = torch.device(device_str)
    train_ds, val_ds, test_ds = load_full_data(device_str)

    def corpus_arr(ds):
        return np.array([top_corpus(s["corpus"]) for s in ds.samples])

    def label_arr(ds):
        return np.array([int(s["label"]) for s in ds.samples])

    c_tr = np.concatenate([corpus_arr(train_ds), corpus_arr(val_ds)])
    c_te = corpus_arr(test_ds)
    l_tr = np.concatenate([label_arr(train_ds), label_arr(val_ds)])
    l_te = label_arr(test_ds)
    ctrl = LABEL_MAP["Control"]

    # Baseline: raw 27-d input (12 ling + 15 para) — seed-independent
    def feats27(ds):
        return np.concatenate([ds.text_feats, ds.para_feats], axis=1)
    X27_tr = np.concatenate([feats27(train_ds), feats27(val_ds)])
    X27_te = feats27(test_ds)
    raw_all = _probe(X27_tr, c_tr, X27_te, c_te)
    m_tr, m_te = l_tr == ctrl, l_te == ctrl
    raw_ctrl = _probe(X27_tr[m_tr], c_tr[m_tr], X27_te[m_te], c_te[m_te])
    print(f"[raw27] all acc={raw_all['accuracy']:.4f}  control-only acc={raw_ctrl['accuracy']:.4f}")

    per_seed = {}
    accA, f1A, accB, f1B = [], [], [], []
    for seed in SEEDS:
        model = _load_model(seed, device)
        E_tr, _ = _forward_ds(model, train_ds, device, want="shared_repr")
        E_va, _ = _forward_ds(model, val_ds, device, want="shared_repr")
        E_te, _ = _forward_ds(model, test_ds, device, want="shared_repr")
        X_tr = np.concatenate([E_tr.numpy(), E_va.numpy()])
        X_te = E_te.numpy()
        pa = _probe(X_tr, c_tr, X_te, c_te)                          # Probe A: all samples
        pb = _probe(X_tr[m_tr], c_tr[m_tr], X_te[m_te], c_te[m_te])  # Probe B: Control-only
        per_seed[seed] = {"probeA_all": pa, "probeB_control_only": pb}
        accA.append(pa["accuracy"]); f1A.append(pa["macro_f1"])
        accB.append(pb["accuracy"]); f1B.append(pb["macro_f1"])
        print(f"  [seed={seed}] A acc={pa['accuracy']:.4f} mF1={pa['macro_f1']:.4f} | "
              f"B(ctrl) acc={pb['accuracy']:.4f} mF1={pb['macro_f1']:.4f}")

    out = {
        "description": "E1: 5-way corpus-identity linear probe on 128-d shared repr "
                       "(fit train+val, eval test) vs raw 27-d input baseline",
        "n_train_fit": int(len(c_tr)), "n_test": int(len(c_te)),
        "n_control_fit": int(m_tr.sum()), "n_control_test": int(m_te.sum()),
        "probeA_all_accuracy": _mean_std(accA), "probeA_all_macro_f1": _mean_std(f1A),
        "probeB_control_accuracy": _mean_std(accB), "probeB_control_macro_f1": _mean_std(f1B),
        "baseline_raw27_all": raw_all, "baseline_raw27_control_only": raw_ctrl,
        "per_seed": per_seed,
    }
    json.dump(out, open(OUT / "e1_corpus_probe.json", "w"), indent=2)
    print(f"[saved] {OUT/'e1_corpus_probe.json'}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True,
                    choices=["e2", "e2prime", "e3prime", "e3w", "e1"])
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-perm", type=int, default=100)
    args = ap.parse_args()
    if args.step == "e2":
        run_e2()
    elif args.step == "e2prime":
        run_e2prime()
    elif args.step == "e3prime":
        run_e3prime(args.device, args.n_perm)
    elif args.step == "e3w":
        run_e3w(args.device)
    elif args.step == "e1":
        run_e1(args.device)
