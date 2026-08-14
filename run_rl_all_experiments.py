#!/usr/bin/env python3
"""
run_rl_all_experiments.py — parallel launcher for all RoBERTa-large experiments.

Usage:
  # Run all phases (Phase 1→2→3→4 in order; experiments within a phase run on GPUs in parallel):
  python run_rl_all_experiments.py

  # Run a specific phase only:
  python run_rl_all_experiments.py --phase 1
  python run_rl_all_experiments.py --phase 2
  python run_rl_all_experiments.py --phase 3

  # Single experiment (worker mode):
  python run_rl_all_experiments.py --worker --exp proposed --device cuda:0
  python run_rl_all_experiments.py --worker --exp A1_no_qformer --device cuda:1
  python run_rl_all_experiments.py --worker --exp LOCO_ADReSSo --device cuda:0
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_POC_DIR = Path(__file__).resolve().parent
if str(_POC_DIR) not in sys.path:
    sys.path.insert(0, str(_POC_DIR))

logger = logging.getLogger(__name__)

# ─── Output directory ─────────────────────────────────────────────────────────
OUTPUT_DIR = _POC_DIR / "results" / "rl_experiments"
SEEDS      = [42, 123, 456, 789, 1234]
N_EPOCHS   = 150

# ─── Hyperparameter settings ──────────────────────────────────────────────────
# Base config: ablations A1–A5 (config.py defaults, Trainer default HP)
BASE_HP = dict(
    lr=2e-4,
    label_smoothing=None,
    sampler_power=None,
    focal_alpha_power=None,
    warmup_epochs=None,
)

# Full proposed-config HP (proposed model + audio experiments)
FULL_HP = dict(
    lr=1e-4,
    label_smoothing=0.10,
    sampler_power=0.5,
    focal_alpha_power=1.0,
    warmup_epochs=15,
)

# ─── Experiment registry ──────────────────────────────────────────────────────
#
# Per-experiment config keys:
#   phase       : execution phase (1-4)
#   gpu         : GPU index (0-7)
#   use_audio   : use wav2vec2 mean-pooled audio
#   no_qformer  : DirectFusion (Q-Former removed)
#   n_query_tokens: None → model default (config.N_QUERY_TOKENS=16)
#   use_focal   : use Focal Loss
#   use_sampler : use WeightedRandomSampler
#   use_prior   : Prior-correction calibration
#   use_missing : M1 learned missing-audio token
#   use_para    : M2 paralinguistic features
#   use_attn    : M3 attention-weighted audio pooling
#   use_seg     : segment-level audio (data loading for M3)
#   lambda_reg  : auxiliary regression loss weight override
#   lambda_sim  : auxiliary similarity loss weight override
#   hp          : "base" or "full"
#   loco_corpus : corpus excluded for LOCO (set only for LOCO experiments)

EXPERIMENTS = {
    # ── Phase 1: proposed model + core ablations ───────────────────────────
    "proposed": dict(
        phase=1, gpu=0,
        use_audio=True, no_qformer=False, n_query_tokens=8,
        use_focal=True, use_sampler=True, use_prior=True,
        use_missing=True, use_para=True, use_attn=True, use_seg=True,
        lambda_reg=0.0, lambda_sim=0.0, hp="full",
        description="Full proposed (RoBERTa-large + M1+M2+M3 + Nq=8)",
    ),
    # ── Tabular-ablation experiment ────────────────────────────────────────
    "A_notab": dict(
        phase=5, gpu=0,
        use_audio=True, no_qformer=False, n_query_tokens=8,
        use_focal=True, use_sampler=True, use_prior=True,
        use_missing=True, use_para=True, use_attn=True, use_seg=True,
        use_tabular=False,
        lambda_reg=0.0, lambda_sim=0.0, hp="full",
        description="w/o Tabular modality (proposed config, TabularEncoder removed)",
    ),
    "A1_no_qformer": dict(
        phase=1, gpu=1,
        use_audio=True, no_qformer=True, n_query_tokens=None,
        use_focal=True, use_sampler=True, use_prior=True,
        use_missing=False, use_para=False, use_attn=False, use_seg=False,
        lambda_reg=None, lambda_sim=None, hp="base",
        description="w/o Q-Former (DirectFusion), base config",
    ),
    "A2_no_focal": dict(
        phase=1, gpu=2,
        use_audio=True, no_qformer=False, n_query_tokens=None,
        use_focal=False, use_sampler=True, use_prior=True,
        use_missing=False, use_para=False, use_attn=False, use_seg=False,
        lambda_reg=None, lambda_sim=None, hp="base",
        description="w/o Focal Loss → CrossEntropy, base config",
    ),
    "A3_no_sampler": dict(
        phase=1, gpu=3,
        use_audio=True, no_qformer=False, n_query_tokens=None,
        use_focal=True, use_sampler=False, use_prior=True,
        use_missing=False, use_para=False, use_attn=False, use_seg=False,
        lambda_reg=None, lambda_sim=None, hp="base",
        description="w/o WeightedRandomSampler, base config",
    ),
    "A5_classif_only": dict(
        phase=1, gpu=4,
        use_audio=True, no_qformer=False, n_query_tokens=None,
        use_focal=True, use_sampler=True, use_prior=True,
        use_missing=False, use_para=False, use_attn=False, use_seg=False,
        lambda_reg=0.0, lambda_sim=0.0, hp="base",
        description="w/o Multitask (classif-only, λ=0), base config",
    ),
    "A1_star": dict(
        phase=1, gpu=5,
        use_audio=True, no_qformer=True, n_query_tokens=8,
        use_focal=True, use_sampler=True, use_prior=True,
        use_missing=True, use_para=True, use_attn=True, use_seg=True,
        lambda_reg=0.0, lambda_sim=0.0, hp="full",
        description="w/o Q-Former (DirectFusion), full proposed config (A1*)",
    ),
    "A_text": dict(
        phase=1, gpu=6,
        use_audio=False, no_qformer=False, n_query_tokens=8,
        use_focal=True, use_sampler=True, use_prior=True,
        use_missing=False, use_para=False, use_attn=False, use_seg=False,
        lambda_reg=0.0, lambda_sim=0.0, hp="full",
        description="Text+Tabular only (no audio), full proposed config",
    ),
    "A_Nq4": dict(
        phase=1, gpu=7,
        use_audio=True, no_qformer=False, n_query_tokens=4,
        use_focal=True, use_sampler=True, use_prior=True,
        use_missing=True, use_para=True, use_attn=True, use_seg=True,
        lambda_reg=0.0, lambda_sim=0.0, hp="full",
        description="Q-Former Nq=4, full proposed config",
    ),
    # ── Phase 2: audio-improvement experiments + A_Nq16 ────────────────────
    "base_audio": dict(
        phase=2, gpu=0,
        use_audio=True, no_qformer=False, n_query_tokens=8,
        use_focal=True, use_sampler=True, use_prior=True,
        use_missing=False, use_para=False, use_attn=False, use_seg=False,
        lambda_reg=0.0, lambda_sim=0.0, hp="full",
        description="Base audio (mean-pooled, no M1/M2/M3), full proposed config",
    ),
    "M1_only": dict(
        phase=2, gpu=1,
        use_audio=True, no_qformer=False, n_query_tokens=8,
        use_focal=True, use_sampler=True, use_prior=True,
        use_missing=True, use_para=False, use_attn=False, use_seg=False,
        lambda_reg=0.0, lambda_sim=0.0, hp="full",
        description="M1 only (learned missing-audio token)",
    ),
    "M2_only": dict(
        phase=2, gpu=2,
        use_audio=True, no_qformer=False, n_query_tokens=8,
        use_focal=True, use_sampler=True, use_prior=True,
        use_missing=False, use_para=True, use_attn=False, use_seg=False,
        lambda_reg=0.0, lambda_sim=0.0, hp="full",
        description="M2 only (paralinguistic features)",
    ),
    "M3_only": dict(
        phase=2, gpu=3,
        use_audio=True, no_qformer=False, n_query_tokens=8,
        use_focal=True, use_sampler=True, use_prior=True,
        use_missing=False, use_para=False, use_attn=True, use_seg=True,
        lambda_reg=0.0, lambda_sim=0.0, hp="full",
        description="M3 only (attention-weighted segment pooling)",
    ),
    "M1M2": dict(
        phase=2, gpu=4,
        use_audio=True, no_qformer=False, n_query_tokens=8,
        use_focal=True, use_sampler=True, use_prior=True,
        use_missing=True, use_para=True, use_attn=False, use_seg=False,
        lambda_reg=0.0, lambda_sim=0.0, hp="full",
        description="M1 + M2 combined",
    ),
    "M1M3": dict(
        phase=2, gpu=5,
        use_audio=True, no_qformer=False, n_query_tokens=8,
        use_focal=True, use_sampler=True, use_prior=True,
        use_missing=True, use_para=False, use_attn=True, use_seg=True,
        lambda_reg=0.0, lambda_sim=0.0, hp="full",
        description="M1 + M3 combined",
    ),
    "M2M3": dict(
        phase=2, gpu=6,
        use_audio=True, no_qformer=False, n_query_tokens=8,
        use_focal=True, use_sampler=True, use_prior=True,
        use_missing=False, use_para=True, use_attn=True, use_seg=True,
        lambda_reg=0.0, lambda_sim=0.0, hp="full",
        description="M2 + M3 combined",
    ),
    "A_Nq16": dict(
        phase=2, gpu=7,
        use_audio=True, no_qformer=False, n_query_tokens=16,
        use_focal=True, use_sampler=True, use_prior=True,
        use_missing=True, use_para=True, use_attn=True, use_seg=True,
        lambda_reg=0.0, lambda_sim=0.0, hp="full",
        description="Q-Former Nq=16, full proposed config",
    ),
    # ── Phase 3: LOCO ────────────────────────────────────────────────────────
    "LOCO_ADReSSo": dict(
        phase=3, gpu=0,
        use_audio=True, no_qformer=False, n_query_tokens=8,
        use_focal=True, use_sampler=True, use_prior=True,
        use_missing=True, use_para=True, use_attn=True, use_seg=True,
        lambda_reg=0.0, lambda_sim=0.0, hp="full",
        loco_corpus="ADReSSo", loco_disease="AD",
        description="LOCO: exclude ADReSSo, zero-shot AD",
    ),
    "LOCO_Coelho": dict(
        phase=3, gpu=1,
        use_audio=True, no_qformer=False, n_query_tokens=8,
        use_focal=True, use_sampler=True, use_prior=True,
        use_missing=True, use_para=True, use_attn=True, use_seg=True,
        lambda_reg=0.0, lambda_sim=0.0, hp="full",
        loco_corpus="Coelho", loco_disease="TBI",
        description="LOCO: exclude Coelho, zero-shot TBI",
    ),
    "LOCO_RHD": dict(
        phase=3, gpu=2,
        use_audio=True, no_qformer=False, n_query_tokens=8,
        use_focal=True, use_sampler=True, use_prior=True,
        use_missing=True, use_para=True, use_attn=True, use_seg=True,
        lambda_reg=0.0, lambda_sim=0.0, hp="full",
        loco_corpus="RHD-English", loco_disease="RHD",
        description="LOCO: exclude RHD-English, zero-shot RHD",
    ),
    "LOCO_Delaware": dict(
        phase=3, gpu=3,
        use_audio=True, no_qformer=False, n_query_tokens=8,
        use_focal=True, use_sampler=True, use_prior=True,
        use_missing=True, use_para=True, use_attn=True, use_seg=True,
        lambda_reg=0.0, lambda_sim=0.0, hp="full",
        loco_corpus="Delaware", loco_disease="MCI",
        description="LOCO: exclude Delaware, zero-shot MCI",
    ),
    # ── Sub-corpus hold-out experiments: AphasiaBank (disease label overlap preserved) ──
    "E5_Richardson": dict(
        phase=5, gpu=0,
        use_audio=True, no_qformer=False, n_query_tokens=8,
        use_focal=True, use_sampler=True, use_prior=True,
        use_missing=True, use_para=True, use_attn=True, use_seg=True,
        lambda_reg=0.0, lambda_sim=0.0, hp="full",
        loco_corpus="Richardson", loco_disease="Aphasia",
        description="E5: hold out AphasiaBank/Richardson (Aphasia 16 / Control 63 patients)",
    ),
    "E5_UNH": dict(
        phase=5, gpu=1,
        use_audio=True, no_qformer=False, n_query_tokens=8,
        use_focal=True, use_sampler=True, use_prior=True,
        use_missing=True, use_para=True, use_attn=True, use_seg=True,
        lambda_reg=0.0, lambda_sim=0.0, hp="full",
        loco_corpus="UNH", loco_disease="Aphasia",
        description="E5: hold out AphasiaBank/UNH (Aphasia 30 / Control 20 patients)",
    ),
    "E5_UMD": dict(
        phase=5, gpu=2,
        use_audio=True, no_qformer=False, n_query_tokens=8,
        use_focal=True, use_sampler=True, use_prior=True,
        use_missing=True, use_para=True, use_attn=True, use_seg=True,
        lambda_reg=0.0, lambda_sim=0.0, hp="full",
        loco_corpus="UMD", loco_disease="Aphasia",
        description="E5: hold out AphasiaBank/UMD (Aphasia 15 / Control 26 patients)",
    ),
    # ── Phase 4: Eval-only ───────────────────────────────────────────────────
    "A4_no_prior_corr": dict(
        phase=4, gpu=0,
        use_audio=True, no_qformer=False, n_query_tokens=8,
        use_focal=True, use_sampler=True, use_prior=False,
        use_missing=True, use_para=True, use_attn=True, use_seg=True,
        lambda_reg=0.0, lambda_sim=0.0, hp="full",
        reuse_proposed_ckpt=True,
        description="w/o Prior-correction calibration (eval only, reuse proposed ckpt)",
    ),
}


# ─── ResidualBertTextEncoder ──────────────────────────────────────────────────

class ResidualBertTextEncoder(nn.Module):
    def __init__(self, input_dim: int = 768, dropout: float = 0.1):
        from config import HIDDEN_DIM
        super().__init__()
        D = HIDDEN_DIM
        self.main = nn.Sequential(
            nn.Linear(input_dim, D * 2), nn.LayerNorm(D * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D * 2, D), nn.LayerNorm(D), nn.GELU(),
        )
        self.skip     = nn.Linear(input_dim, D, bias=False)
        self.out_norm = nn.LayerNorm(D)

    def forward(self, x):
        return self.out_norm(self.main(x) + self.skip(x))


# ─── Utilities ────────────────────────────────────────────────────────────────

def _setup_logging(output_dir: Path, exp_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = output_dir / f"{exp_name}_{ts}.log"
    fmt     = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt,
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(logfile, encoding="utf-8")],
        force=True,
    )
    logger.info("Log: %s", logfile)


def _save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def _inject_roberta_large(train_ds, val_ds, test_ds):
    from config import RESULTS_DIR
    cache = RESULTS_DIR / "feats_roberta-large.pt"
    if not cache.exists():
        raise FileNotFoundError(f"RoBERTa-large cache not found: {cache}")
    data  = torch.load(cache, map_location="cpu", weights_only=True)
    feats = data["feats"].numpy()
    texts = data["texts"]
    t2i   = {t: i for i, t in enumerate(texts)}

    for ds in [d for d in (train_ds, val_ds, test_ds) if d is not None]:
        missing = [s["text"] for s in ds.samples if s["text"] not in t2i]
        if missing:
            raise KeyError(f"Texts not in RoBERTa cache: {missing[:3]}")
        ds.bert_feats = np.stack(
            [feats[t2i[s["text"]]] for s in ds.samples]
        ).astype(np.float32)
    logger.info("[feat] RoBERTa-large features injected (1024-dim) from %s", cache.name)


def _build_model(cfg: dict):
    from models.model import MultimodalFoundationModel
    n_query_tokens = cfg.get("n_query_tokens", None)
    model = MultimodalFoundationModel(
        encoder_type="bert",
        use_audio=cfg["use_audio"],
        no_qformer=cfg["no_qformer"],
        n_query_tokens=n_query_tokens,
        use_missing_token=cfg["use_missing"],
        use_para=cfg["use_para"],
        use_attn_audio=cfg["use_attn"],
        use_tabular=cfg.get("use_tabular", True),
    )
    model.text_enc = ResidualBertTextEncoder(input_dim=1024, dropout=0.1)
    return model


def _get_trainer_hp(cfg: dict) -> dict:
    hp_key = cfg.get("hp", "base")
    hp     = FULL_HP if hp_key == "full" else BASE_HP
    return {k: v for k, v in hp.items() if v is not None}


def _train_one_seed(model, train_ds, val_ds, test_ds, device, seed,
                    n_epochs, seed_dir, cfg, proposed_ckpt=None):
    from trainer import Trainer
    from eval_utils import evaluate_model

    seed_dir.mkdir(parents=True, exist_ok=True)

    reuse_ckpt = cfg.get("reuse_proposed_ckpt", False)
    if reuse_ckpt and proposed_ckpt and Path(proposed_ckpt).exists():
        model.load_state_dict(
            torch.load(proposed_ckpt, map_location=device, weights_only=True)
        )
        model.to(device).eval()
        logger.info("[seed=%d] Reusing checkpoint: %s", seed, proposed_ckpt)
    else:
        hp = _get_trainer_hp(cfg)
        trainer = Trainer(
            model=model, train_ds=train_ds, val_ds=val_ds,
            device=device, n_epochs=n_epochs,
            encoder_type="bert",
            seed=seed,
            use_focal_loss=cfg["use_focal"],
            use_weighted_sampler=cfg["use_sampler"],
            lambda_reg=cfg.get("lambda_reg"),
            lambda_sim=cfg.get("lambda_sim"),
            output_dir=seed_dir,
            ckpt_name="best_model.pt",
            **hp,
        )
        if hasattr(model, "cls_head") and hasattr(model.cls_head, "focal"):
            model.cls_head.focal.gamma = 2.0
        t0 = time.time()
        trainer.train()
        trainer.load_best()
        logger.info("[seed=%d] Trained in %.1fs", seed, time.time() - t0)

    metrics = evaluate_model(
        model=model, train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
        device=device, encoder_type="bert",
        use_audio=cfg["use_audio"],
        use_prior_correction=cfg["use_prior"],
    )
    metrics["seed"] = seed
    metrics["ckpt"] = str(seed_dir / "best_model.pt")
    return metrics


# ─── Worker: run a single experiment ──────────────────────────────────────────

def run_experiment_worker(exp_name: str, device_str: str, n_epochs: int = N_EPOCHS,
                          seeds: list = None):
    if exp_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {exp_name}")

    cfg       = EXPERIMENTS[exp_name]
    device    = torch.device(device_str)
    exp_dir   = OUTPUT_DIR / exp_name
    result_path = exp_dir / "result.json"

    _setup_logging(exp_dir, exp_name)
    logger.info("=" * 60)
    logger.info("Experiment: %s  |  GPU: %s", exp_name, device_str)
    logger.info("Description: %s", cfg["description"])
    logger.info("Output: %s", exp_dir)

    if result_path.exists():
        logger.info("Already done, loading existing result.")
        with open(result_path) as f:
            return json.load(f)

    # ── Data loading ─────────────────────────────────────────────────────────
    is_loco  = "loco_corpus" in cfg
    use_para = cfg["use_para"]
    use_seg  = cfg["use_seg"]

    if is_loco:
        from data_loader import load_loco_dataset
        corpus = cfg["loco_corpus"]
        logger.info("LOCO: loading data, exclude corpus=%s  (para=%s seg=%s)",
                    corpus, use_para, use_seg)
        train_ds, val_ds, test_ds = load_loco_dataset(
            exclude_corpus=corpus, verbose=True,
            use_bert=False, use_audio=True,
            use_para=use_para, use_seg_audio=use_seg,
            bert_device=device_str,
        )
    else:
        from data_loader import load_dataset
        logger.info("Loading data: audio=%s para=%s seg=%s",
                    cfg["use_audio"], use_para, use_seg)
        train_ds, val_ds, test_ds = load_dataset(
            verbose=True,
            use_bert=False,
            use_audio=cfg["use_audio"],
            use_para=use_para,
            use_seg_audio=use_seg,
            bert_device=device_str,
        )

    _inject_roberta_large(train_ds, val_ds, test_ds)
    logger.info("Data: train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds))

    # ── A4: eval-only (reuse proposed checkpoint) ────────────────────────────
    reuse_ckpt = cfg.get("reuse_proposed_ckpt", False)
    proposed_ckpts = {}
    if reuse_ckpt:
        proposed_dir = OUTPUT_DIR / "proposed"
        for seed in SEEDS:
            ckpt = proposed_dir / f"seed_{seed}" / "best_model.pt"
            if ckpt.exists():
                proposed_ckpts[seed] = str(ckpt)
            else:
                logger.warning("Proposed ckpt not found for seed=%d: %s", seed, ckpt)

    # ── Multi-seed training ──────────────────────────────────────────────────
    from statistical_analysis import aggregate_seeds

    seed_results = []
    ckpts        = {}

    for seed in (seeds or SEEDS):
        model = _build_model(cfg).to(device)
        seed_dir = exp_dir / f"seed_{seed}"
        proposed_ckpt = proposed_ckpts.get(seed) if reuse_ckpt else None

        metrics = _train_one_seed(
            model=model, train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
            device=device, seed=seed, n_epochs=n_epochs,
            seed_dir=seed_dir, cfg=cfg, proposed_ckpt=proposed_ckpt,
        )
        seed_results.append(metrics)
        ckpts[seed] = metrics["ckpt"]
        logger.info("[seed=%d] mF1=%.4f acc=%.4f",
                    seed, metrics.get("test_macro_f1", 0),
                    metrics.get("test_accuracy", 0))

    agg = aggregate_seeds(seed_results)
    mf1 = agg.get("test_macro_f1", {})
    acc = agg.get("test_accuracy", {})
    logger.info("AGG: mF1=%.4f±%.4f  acc=%.4f±%.4f",
                mf1.get("mean", 0), mf1.get("std", 0),
                acc.get("mean", 0), acc.get("std", 0))

    result = {
        "name":         exp_name,
        "description":  cfg["description"],
        "config":       {k: v for k, v in cfg.items() if k not in ("description",)},
        "seed_results": seed_results,
        "aggregated":   agg,
        "ckpts":        {str(k): v for k, v in ckpts.items()},
    }
    if is_loco:
        result["loco_corpus"]  = cfg["loco_corpus"]
        result["loco_disease"] = cfg["loco_disease"]

    _save_json(result, result_path)
    logger.info("Result saved: %s", result_path)
    return result


# ─── Launcher: run each phase in parallel ─────────────────────────────────────

def _launch_phase(phase: int, n_epochs: int = N_EPOCHS):
    """Run all experiments of a phase as parallel subprocesses, one per GPU."""
    exps_in_phase = {name: cfg for name, cfg in EXPERIMENTS.items()
                     if cfg["phase"] == phase}
    if not exps_in_phase:
        logger.info("Phase %d: no experiments.", phase)
        return {}

    logger.info("=" * 70)
    logger.info("Phase %d: launching %d experiments in parallel",
                phase, len(exps_in_phase))

    procs = {}
    log_files = {}
    for exp_name, cfg in exps_in_phase.items():
        result_path = OUTPUT_DIR / exp_name / "result.json"
        if result_path.exists():
            logger.info("  [SKIP] %s: already done", exp_name)
            continue

        gpu     = cfg["gpu"]
        logfile = OUTPUT_DIR / exp_name / f"stdout_{datetime.now().strftime('%H%M%S')}.log"
        logfile.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--worker",
            "--exp", exp_name,
            "--device", f"cuda:{gpu}",
            "--n-epochs", str(n_epochs),
        ]
        logger.info("  Launching: %-20s  GPU cuda:%d  log=%s", exp_name, gpu, logfile)

        fh = open(logfile, "w")
        proc = subprocess.Popen(
            cmd, stdout=fh, stderr=subprocess.STDOUT,
            cwd=str(_POC_DIR),
        )
        procs[exp_name]     = proc
        log_files[exp_name] = (fh, logfile)

    # Wait for all subprocesses to finish
    failed = []
    for exp_name, proc in procs.items():
        rc = proc.wait()
        fh, logfile = log_files[exp_name]
        fh.close()
        if rc == 0:
            logger.info("  [OK]   %s (rc=%d)", exp_name, rc)
        else:
            logger.error("  [FAIL] %s (rc=%d) — see %s", exp_name, rc, logfile)
            failed.append(exp_name)

    if failed:
        logger.error("Phase %d: %d experiment(s) FAILED: %s", phase, len(failed), failed)
    else:
        logger.info("Phase %d: all experiments completed successfully.", phase)

    return failed


def _collect_results(phases: list) -> dict:
    """Collect result JSON files into a single dictionary."""
    all_results = {}
    for exp_name in EXPERIMENTS:
        if EXPERIMENTS[exp_name]["phase"] not in phases:
            continue
        rp = OUTPUT_DIR / exp_name / "result.json"
        if rp.exists():
            with open(rp) as f:
                all_results[exp_name] = json.load(f)
    return all_results


def _print_summary(all_results: dict):
    print("\n" + "=" * 75)
    print(f"{'Experiment':<22} {'MF1 mean':>10} {'±std':>7} {'Acc mean':>10} {'ΔMF1 vs proposed':>18}")
    print("-" * 75)

    proposed_mf1 = None
    if "proposed" in all_results:
        agg = all_results["proposed"].get("aggregated", {})
        proposed_mf1 = agg.get("test_macro_f1", {}).get("mean", 0)
        std = agg.get("test_macro_f1", {}).get("std", 0)
        acc = agg.get("test_accuracy",  {}).get("mean", 0)
        print(f"{'proposed':<22} {proposed_mf1:>10.4f} ±{std:.4f} {acc:>10.4f}  {'(ref)':>18}")

    for name, res in all_results.items():
        if name == "proposed":
            continue
        agg = res.get("aggregated", {})
        mf1 = agg.get("test_macro_f1", {}).get("mean", 0)
        std = agg.get("test_macro_f1", {}).get("std", 0)
        acc = agg.get("test_accuracy",  {}).get("mean", 0)
        delta = ""
        if proposed_mf1 is not None:
            d = (mf1 - proposed_mf1) * 100
            delta = f"{'+' if d >= 0 else ''}{d:.2f}pp"
        print(f"  {name:<20} {mf1:>10.4f} ±{std:.4f} {acc:>10.4f}  {delta:>18}")

    print("=" * 75)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parallel launcher for RoBERTa-large experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--worker", action="store_true",
                        help="Worker mode (run a single experiment)")
    parser.add_argument("--exp",    type=str, default=None,
                        help="[worker] experiment name")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="[worker] CUDA device")
    parser.add_argument("--n-epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="[worker] list of seeds to run (default: all SEEDS)")
    parser.add_argument("--phase", type=int, nargs="+",
                        default=[1, 2, 3, 4],
                        help="[launcher] phases to run (1-4). Default: all")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results/rl_experiments/)")
    args = parser.parse_args()

    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Worker mode ──────────────────────────────────────────────────────────
    if args.worker:
        if not args.exp:
            parser.error("--worker mode requires --exp.")
        run_experiment_worker(
            exp_name=args.exp,
            device_str=args.device,
            n_epochs=args.n_epochs,
            seeds=args.seeds,
        )
        return

    # ── Launcher mode ────────────────────────────────────────────────────────
    _setup_logging(OUTPUT_DIR, "launcher")
    logger.info("Starting RoBERTa-large experiment launcher")
    logger.info("Output directory: %s", OUTPUT_DIR)
    logger.info("Phases to run: %s", args.phase)
    logger.info("Each experiment: %d seeds × %d epochs", len(SEEDS), args.n_epochs)

    phases_to_run = sorted(set(args.phase))
    all_failed    = []

    for phase in phases_to_run:
        t0     = time.time()
        failed = _launch_phase(phase, args.n_epochs)
        elapsed = time.time() - t0
        logger.info("Phase %d done (%.1fs). Failed: %s", phase, elapsed, failed or "none")
        all_failed.extend(failed)

    # Collect and print results
    all_results = _collect_results(phases_to_run)
    _save_json(all_results, OUTPUT_DIR / "all_results.json")
    _print_summary(all_results)

    if all_failed:
        logger.error("Failed experiments: %s", all_failed)
        sys.exit(1)
    else:
        logger.info("All experiments completed. Results: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
