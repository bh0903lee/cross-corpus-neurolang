#!/usr/bin/env python3
"""
correlation_analysis.py — Inter-disorder correlation analysis.

(A) Embedding-space inter-disorder proximity  → 4 subfigures (a)-(d)
    (a) PCA scatter, (b) t-SNE scatter,
    (c) centroid cosine-similarity heatmap, (d) dendrogram
(B) Feature-level disorder profile             → 3 subfigures (a)-(c)
    (a) 27-feature z-score profile heatmap,
    (b) profile correlation heatmap, (c) dendrogram

Additional analyses:
    - Top discriminative features by ANOVA F-statistic
    - Shared / distinguishing feature contributions per disorder pair
    - Linguistic-only vs paralinguistic-only profile correlations (modality split)

Each panel is saved as a separate PNG.

Outputs: results/correlation_analysis/
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, f_oneway

_POC = Path(__file__).resolve().parent
if str(_POC) not in sys.path:
    sys.path.insert(0, str(_POC))

from config import IDX_TO_LABEL, N_CLASSES, N_LING_FEATURES
from data_loader import load_dataset
from paralinguistic_extractor import FEATURE_NAMES as PARA_NAMES
from run_rl_all_experiments import EXPERIMENTS, _build_model, _inject_roberta_large

OUT = _POC / "results" / "correlation_analysis"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 123, 456, 789, 1234]
SCATTER_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_ORDER = ["Control", "MCI", "AD", "Aphasia", "TBI", "RHD"]
COLORS = {
    "Control": "#4C72B0", "MCI": "#55A868", "AD": "#C44E52",
    "Aphasia": "#8172B3", "TBI": "#CCB974", "RHD": "#64B5CD",
}
# Linguistic feature names
# (features #0, #1, #5 are log/log/ratio-transformed values, not raw names)
LING_NAMES = [
    "n_words_log", "n_sent_log", "avg_sent_len", "ttr", "mattr", "long_word_ratio",
    "pronoun_ratio", "filler_ratio", "repetition_rate", "content_density",
    "pause_density", "avg_word_len",
]
FEAT_NAMES = LING_NAMES + list(PARA_NAMES)            # 12 + 15 = 27

# ── Global font sizes ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 16, "axes.titlesize": 19, "axes.labelsize": 17,
    "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 13,
})
DENDRO_CUT = 0.7   # color_threshold = DENDRO_CUT * max cophenetic distance
ABOVE_COL = "#9a9a9a"


def _cidx(cname):
    return [k for k, v in IDX_TO_LABEL.items() if v == cname][0]


# ─────────────────────────────────────────────────────────────────────────────
# Data / embeddings
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    train_ds, val_ds, test_ds = load_dataset(
        verbose=False, use_bert=False, use_audio=True,
        use_para=True, use_seg_audio=True, bert_device=str(DEVICE),
    )
    _inject_roberta_large(train_ds, val_ds, test_ds)
    return train_ds, val_ds, test_ds


def extract_embeddings(test_ds, seed):
    from torch.utils.data import DataLoader
    cfg = EXPERIMENTS["proposed"]
    model = _build_model(cfg).to(DEVICE)
    ckpt = _POC / "results" / "rl_experiments" / "proposed" / f"seed_{seed}" / "best_model.pt"
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.eval()
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    embs, labels = [], []
    with torch.no_grad():
        for b in loader:
            tf, tab = b["bert_feat"].to(DEVICE), b["tab_feat"].to(DEVICE)
            af = b.get("audio_feat"); ha = b.get("has_audio")
            af = af.to(DEVICE) if af is not None else None
            ha = ha.to(DEVICE) if ha is not None else None
            sf = b.get("audio_seg_feat"); ns = b.get("n_segs"); pf = b.get("para_feat")
            sf = sf.to(DEVICE) if sf is not None else None
            ns = ns.to(DEVICE) if ns is not None else None
            pf = pf.to(DEVICE) if pf is not None else None
            out = model(tf, tab, af, ha, sf, ns, pf)
            embs.append(out["shared_repr"].cpu().numpy())
            labels.extend(b["label"].tolist())
    return np.concatenate(embs, 0), np.array(labels)


def cosine_centroid_matrix(emb, labels):
    cents = np.stack([emb[labels == _cidx(c)].mean(0) for c in CLASS_ORDER])
    norm = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9)
    return norm @ norm.T


# ─────────────────────────────────────────────────────────────────────────────
# (A) Visualization — individual panels
# ─────────────────────────────────────────────────────────────────────────────
def _scatter_panel(red, labels, title, fname, legend=True,
                   xlabel="dim 1", ylabel="dim 2"):
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for cname in CLASS_ORDER:
        pts = red[labels == _cidx(cname)]
        if len(pts) == 0:
            continue
        ax.scatter(pts[:, 0], pts[:, 1], s=34, color=COLORS[cname],
                   alpha=0.22, edgecolors="none", zorder=1)
        c = pts.mean(0)
        ax.scatter(c[0], c[1], s=420, color=COLORS[cname], edgecolors="black",
                   linewidths=2.0, marker="o", zorder=3, label=f"{cname} (n={len(pts)})")
        ax.annotate(cname, (c[0], c[1]), xytext=(0, 16), textcoords="offset points",
                    fontsize=15, fontweight="bold", ha="center", va="bottom", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=COLORS[cname],
                              lw=1.6, alpha=0.92))
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(alpha=0.15)
    if legend:
        ax.legend(loc="best", fontsize=12, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _heatmap_panel(M, fname, vmin=-1, vmax=1, cmap="RdYlBu_r", annot_fs=15):
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(CLASS_ORDER))); ax.set_yticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_ORDER, rotation=40, ha="right", fontsize=20)
    ax.set_yticklabels(CLASS_ORDER, fontsize=20)
    for i in range(len(CLASS_ORDER)):
        for j in range(len(CLASS_ORDER)):
            val = M[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=annot_fs,
                    color="white" if abs(val) > 0.6 else "black",
                    fontweight="bold" if i != j and abs(val) > 0.55 else "normal")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=16)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / fname, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _dendro_panel(M, fname, ylabel="cophenetic distance (1 - cosine)"):
    dist = 1.0 - M
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2
    Z = linkage(squareform(dist, checks=False), method="average")
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    dendrogram(Z, labels=CLASS_ORDER, ax=ax, leaf_font_size=20,
               color_threshold=DENDRO_CUT * Z[:, 2].max(),
               above_threshold_color=ABOVE_COL)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=16)
    ax.axhline(DENDRO_CUT * Z[:, 2].max(), ls="--", lw=1.2, color="gray", alpha=0.7)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / fname, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return Z


# ─────────────────────────────────────────────────────────────────────────────
# (B) Feature profiles
# ─────────────────────────────────────────────────────────────────────────────
def collect_features(datasets):
    X, y = [], []
    for ds in datasets:
        X.append(np.concatenate([ds.text_feats, ds.para_feats], axis=1))
        y.extend([s["label"] for s in ds.samples])
    return np.concatenate(X, 0), np.array(y)


def class_profiles(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xz = (X - mu) / sd
    prof = np.stack([Xz[y == _cidx(c)].mean(0) for c in CLASS_ORDER])
    counts = [int((y == _cidx(c)).sum()) for c in CLASS_ORDER]
    return prof, counts


def profile_corr(prof):
    C = np.eye(len(CLASS_ORDER))
    for i in range(len(CLASS_ORDER)):
        for j in range(len(CLASS_ORDER)):
            C[i, j] = pearsonr(prof[i], prof[j])[0]
    return C


def fig_profile_heatmap(prof, counts):
    fig, ax = plt.subplots(figsize=(15.5, 4.2))
    norm = TwoSlopeNorm(vmin=min(-2, prof.min()), vcenter=0, vmax=max(2, prof.max()))
    im = ax.imshow(prof, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_yticks(range(len(CLASS_ORDER)))
    ax.set_yticklabels([f"{c} (n={n})" for c, n in zip(CLASS_ORDER, counts)], fontsize=16)
    ax.set_xticks(range(len(FEAT_NAMES)))
    disp_names = [n.replace("spec_centroid_mean", "spec_cent_mean") for n in FEAT_NAMES]
    ax.set_xticklabels(disp_names, rotation=90, fontsize=13)
    ax.axvline(N_LING_FEATURES - 0.5, color="black", lw=2.0)
    ax.text(N_LING_FEATURES / 2 - 0.5, -0.66, "linguistic (12)", ha="center",
            fontsize=15, style="italic")
    ax.text(N_LING_FEATURES + len(PARA_NAMES) / 2 - 0.5, -0.66, "paralinguistic (15)",
            ha="center", fontsize=15, style="italic")
    cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.015)
    cb.set_label("z-score (mean per disorder)", fontsize=14)
    cb.ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "figB_a_profile.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Additional analyses
# ─────────────────────────────────────────────────────────────────────────────
def anova_ranking(X, y, k=10):
    groups_idx = [np.where(y == _cidx(c))[0] for c in CLASS_ORDER]
    F = []
    for f in range(X.shape[1]):
        gs = [X[g, f] for g in groups_idx]
        F.append(f_oneway(*gs).statistic)
    F = np.array(F)
    order = np.argsort(-F)
    return [(FEAT_NAMES[m], round(float(F[m]), 1)) for m in order[:k]]


def pair_signature(prof, a, b, k=5):
    """Shared / distinguishing feature contributions for disorder pair (a, b)."""
    i, j = CLASS_ORDER.index(a), CLASS_ORDER.index(b)
    pi = prof[i] - prof[i].mean()
    pj = prof[j] - prof[j].mean()
    contrib = pi * pj                       # >0: same direction (shared), <0: opposite (distinguishing)
    shared = [(FEAT_NAMES[m], round(float(prof[i][m]), 2), round(float(prof[j][m]), 2))
              for m in np.argsort(-contrib)[:k] if contrib[m] > 0]
    distinct = [(FEAT_NAMES[m], round(float(prof[i][m]), 2), round(float(prof[j][m]), 2))
                for m in np.argsort(contrib)[:k] if contrib[m] < 0]
    return shared, distinct


def modality_split_corr(prof):
    """Linguistic-only vs paralinguistic-only profile correlations (key pairs)."""
    ling = profile_corr(prof[:, :N_LING_FEATURES])
    para = profile_corr(prof[:, N_LING_FEATURES:])
    pairs = [("Control", "MCI"), ("Control", "Aphasia"), ("MCI", "AD"), ("TBI", "RHD")]
    out = {}
    for a, b in pairs:
        i, j = CLASS_ORDER.index(a), CLASS_ORDER.index(b)
        out[f"{a}-{b}"] = {"linguistic": round(float(ling[i, j]), 3),
                           "paralinguistic": round(float(para[i, j]), 3)}
    return out, ling, para


def top_discriminative(prof, k=5):
    out = {}
    for i, c in enumerate(CLASS_ORDER):
        order = np.argsort(-np.abs(prof[i]))[:k]
        out[c] = [(FEAT_NAMES[m], round(float(prof[i][m]), 2)) for m in order]
    return out


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"[device] {DEVICE}")
    train_ds, val_ds, test_ds = load_data()
    print(f"[data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    # Remove outdated combined figures
    for old in ["figA1_embedding_scatter.png", "figA2_embedding_dendrogram.png",
                "figB1_feature_profile_heatmap.png", "figB2_profile_correlation_dendrogram.png"]:
        (OUT / old).unlink(missing_ok=True)

    # ── (A) ─────────────────────────────────────────────────────────────────
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    sims, scat_emb, scat_lab = [], None, None
    for seed in SEEDS:
        emb, lab = extract_embeddings(test_ds, seed)
        sims.append(cosine_centroid_matrix(emb, lab))
        if seed == SCATTER_SEED:
            scat_emb, scat_lab = emb, lab
    sim_mean = np.mean(sims, 0)

    pca = PCA(n_components=2, random_state=0).fit(scat_emb)
    emb_pca = pca.transform(scat_emb); evr = pca.explained_variance_ratio_
    perp = max(5, min(30, (len(scat_emb) - 1) // 3))
    emb_tsne = TSNE(n_components=2, perplexity=perp, init="pca",
                    learning_rate="auto", random_state=0).fit_transform(scat_emb)

    _scatter_panel(emb_pca, scat_lab, "PCA", "figA_a_pca.png", legend=False,
                   xlabel=f"PC1 ({evr[0]*100:.1f}%)",
                   ylabel=f"PC2 ({evr[1]*100:.1f}%)")
    _scatter_panel(emb_tsne, scat_lab, f"t-SNE (perplexity={perp})",
                   "figA_b_tsne.png", legend=False,
                   xlabel="t-SNE 1", ylabel="t-SNE 2")
    _heatmap_panel(sim_mean, "figA_c_cosine.png")
    Za = _dendro_panel(sim_mean, "figA_d_dendrogram.png",
                       ylabel="cophenetic distance (1 - cosine)")
    print("[A] saved a,b,c,d")

    # ── (B) ─────────────────────────────────────────────────────────────────
    X, y = collect_features([train_ds, val_ds, test_ds])
    prof, counts = class_profiles(X, y)
    C = profile_corr(prof)
    fig_profile_heatmap(prof, counts)
    _heatmap_panel(C, "figB_b_corr.png")
    Zb = _dendro_panel(C, "figB_c_dendrogram.png",
                       ylabel="cophenetic distance (1 - r)")
    print("[B] saved a,b,c")

    # ── Additional analyses ───────────────────────────────────────────────────
    anova = anova_ranking(X, y, k=10)
    disc = top_discriminative(prof)
    split, ling_C, para_C = modality_split_corr(prof)
    sigs = {f"{a}-{b}": pair_signature(prof, a, b)
            for a, b in [("Control", "MCI"), ("Control", "Aphasia"),
                         ("MCI", "AD"), ("TBI", "RHD")]}

    summary = {
        "class_order": CLASS_ORDER,
        "class_counts_all": dict(zip(CLASS_ORDER, counts)),
        "A_embedding_cosine_sim_mean_5seed": sim_mean.round(4).tolist(),
        "B_profile_correlation": C.round(4).tolist(),
        "B_profile_correlation_linguistic_only": ling_C.round(4).tolist(),
        "B_profile_correlation_paralinguistic_only": para_C.round(4).tolist(),
        "anova_top_features": anova,
        "top_discriminative_per_disorder": disc,
        "modality_split_keypairs": split,
        "pair_signatures": sigs,
        "feature_names": FEAT_NAMES,
    }
    with open(OUT / "correlation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Console output
    def _pm(M, name):
        print(f"\n=== {name} ===")
        print("        " + "  ".join(f"{c[:4]:>6}" for c in CLASS_ORDER))
        for i, c in enumerate(CLASS_ORDER):
            print(f"{c[:7]:<7} " + "  ".join(f"{M[i,j]:6.2f}" for j in range(len(CLASS_ORDER))))
    _pm(sim_mean, "(A) embedding cosine sim (5-seed mean)")
    _pm(C, "(B) profile correlation (all 27 feats)")
    _pm(ling_C, "(B) profile correlation — linguistic only")
    _pm(para_C, "(B) profile correlation — paralinguistic only")
    print("\n=== ANOVA top features (F across 6 classes) ===")
    print("  " + ", ".join(f"{n}({v})" for n, v in anova))
    print("\n=== modality split (key pairs) ===")
    for k, v in split.items():
        print(f"  {k:<16} ling r={v['linguistic']:+.2f}  para r={v['paralinguistic']:+.2f}")
    print("\n=== pair signatures ===")
    for k, (sh, di) in sigs.items():
        print(f"  [{k}] shared: " + ", ".join(f"{n}" for n, _, _ in sh[:4]))
        print(f"  [{k}] distinct: " + ", ".join(f"{n}" for n, _, _ in di[:4]))
    print("\n=== top discriminative per disorder ===")
    for c, feats in disc.items():
        print(f"  {c:<8}: " + ", ".join(f"{n}({v:+.1f})" for n, v in feats))
    print(f"\n[done] {OUT}")


if __name__ == "__main__":
    main()
