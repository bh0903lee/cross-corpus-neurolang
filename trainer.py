"""
trainer.py — multitask training loop (classification + regression + contrastive).

L_total = λ_cls·L_cls + λ_reg·L_reg + λ_sim·L_sim
"""
import logging
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import numpy as np

from config import (BATCH_SIZE, LR, N_EPOCHS, SEED, RESULTS_DIR, N_CLASSES,
                    WARMUP_EPOCHS, FOCAL_ALPHA_POWER, SAMPLER_POWER)
from models.model import MultimodalFoundationModel

logger = logging.getLogger(__name__)


def _seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """
    Multitask trainer. Ablation parameters:
    seed, use_focal_loss, use_weighted_sampler, lambda_reg, lambda_sim, ckpt_name.
    """
    def __init__(self, model: MultimodalFoundationModel,
                 train_ds, val_ds,
                 device: torch.device,
                 n_epochs: int = N_EPOCHS,
                 lr: float = LR,
                 batch_size: int = BATCH_SIZE,
                 num_workers: int = 0,
                 output_dir=None,
                 encoder_type: str = "ling",
                 focal_alpha_power: float = None,
                 sampler_power: float = None,
                 warmup_epochs: int = None,
                 label_smoothing: float = None,
                 seed: int = SEED,
                 use_focal_loss: bool = True,
                 use_weighted_sampler: bool = True,
                 lambda_reg: float = None,
                 lambda_sim: float = None,
                 ckpt_name: str = "best_model.pt"):
        _seed_everything(seed)
        self.model        = model.to(device)
        self.device       = device
        self.n_epochs     = n_epochs
        self.encoder_type = encoder_type
        self.output_dir   = output_dir or RESULTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lambda_reg  = lambda_reg
        self._lambda_sim  = lambda_sim

        _focal_alpha_power = focal_alpha_power if focal_alpha_power is not None else FOCAL_ALPHA_POWER
        _sampler_power     = sampler_power     if sampler_power     is not None else SAMPLER_POWER
        _warmup_epochs     = warmup_epochs     if warmup_epochs     is not None else WARMUP_EPOCHS

        # ── Class statistics ───────────────────────────────────────────────────
        labels_arr = np.array([s["label"] for s in train_ds.samples])
        counts     = np.bincount(labels_arr, minlength=N_CLASSES).astype(float)
        total      = counts.sum()
        logger.info("Train class counts: %s",
                    {i: int(counts[i]) for i in range(N_CLASSES)})

        # ── Loss function: Focal Loss or CrossEntropy ─────────────────────────
        if use_focal_loss:
            inv_freq           = total / (N_CLASSES * np.maximum(counts, 1))
            inv_freq           = np.clip(inv_freq ** _focal_alpha_power, 0.0, 5.0)
            focal_alpha_tensor = torch.from_numpy(inv_freq.astype(np.float32))
            model.cls_head.focal.alpha = focal_alpha_tensor
            if label_smoothing is not None:
                model.cls_head.focal.label_smoothing = label_smoothing
            logger.info("Loss: FocalLoss (alpha_power=%.2f, gamma=%.1f)",
                        _focal_alpha_power, model.cls_head.focal.gamma)
        else:
            model.cls_head.focal = nn.CrossEntropyLoss()
            logger.info("Loss: CrossEntropyLoss (A2 ablation)")

        # ── WeightedRandomSampler or uniform sampling ─────────────────────────
        use_pin = (device.type == "cuda")
        use_pw  = (num_workers > 0)
        if use_weighted_sampler:
            cls_w    = 1.0 / np.maximum(counts, 1) ** _sampler_power
            sample_w = torch.from_numpy(cls_w[labels_arr]).float()
            sampler  = WeightedRandomSampler(sample_w,
                                             num_samples=len(sample_w),
                                             replacement=True)
            self.train_loader = DataLoader(
                train_ds, batch_size=batch_size, sampler=sampler,
                drop_last=True, num_workers=num_workers,
                pin_memory=use_pin, persistent_workers=use_pw,
            )
            logger.info("Sampler: WeightedRandomSampler (power=%.2f)", _sampler_power)
        else:
            self.train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True,
                drop_last=True, num_workers=num_workers,
                pin_memory=use_pin, persistent_workers=use_pw,
            )
            logger.info("Sampler: uniform shuffle (A3 ablation)")

        self.val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=use_pin, persistent_workers=use_pw,
        )

        # ── Optimizer ─────────────────────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        # ── LR scheduler: linear warmup → cosine decay ────────────────────────
        warmup = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0,
                          total_iters=_warmup_epochs)
        cosine = CosineAnnealingLR(self.optimizer,
                                   T_max=max(n_epochs - _warmup_epochs, 1),
                                   eta_min=lr * 0.05)
        self.scheduler = SequentialLR(self.optimizer,
                                      schedulers=[warmup, cosine],
                                      milestones=[_warmup_epochs])
        self._warmup_epochs = _warmup_epochs

        # ── AMP ───────────────────────────────────────────────────────────────
        self.use_amp = (device.type == "cuda")
        self.scaler  = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.best_val_acc = -1.0
        self.best_ckpt    = self.output_dir / ckpt_name
        self.history      = []

    def _get_audio(self, batch):
        if "audio_feat" not in batch:
            return None, None
        af = batch["audio_feat"].to(self.device, non_blocking=True)
        ha = batch["has_audio"].to(self.device,  non_blocking=True)
        return af, ha

    def _get_seg_audio(self, batch):
        if "audio_seg_feat" not in batch:
            return None, None
        sf = batch["audio_seg_feat"].to(self.device, non_blocking=True)
        ns = batch["n_segs"].to(self.device, non_blocking=True)
        return sf, ns

    def _get_para(self, batch):
        if "para_feat" not in batch:
            return None
        return batch["para_feat"].to(self.device, non_blocking=True)

    def _train_epoch(self) -> dict:
        self.model.train()
        total_loss = total_cls = total_reg = total_sim = 0.0
        n_batches  = 0

        for batch in self.train_loader:
            feat_key  = "bert_feat" if self.encoder_type == "bert" else "text_feat"
            text_feat = batch[feat_key].to(self.device, non_blocking=True)

            if self.encoder_type == "bert":
                text_feat = text_feat + torch.randn_like(text_feat) * 0.01

            tab_feat  = batch["tab_feat"].to(self.device,  non_blocking=True)
            labels    = batch["label"].to(self.device,     non_blocking=True)
            wab_tgt   = batch["wab_aq"].to(self.device,   non_blocking=True)
            audio_feat, has_audio = self._get_audio(batch)
            seg_feat, n_segs      = self._get_seg_audio(batch)
            para_feat             = self._get_para(batch)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(text_feat, tab_feat, audio_feat, has_audio,
                                     seg_feat, n_segs, para_feat)
                loss, loss_dict = self.model.compute_loss(
                    outputs, labels, wab_tgt,
                    lambda_reg=self._lambda_reg, lambda_sim=self._lambda_sim,
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            total_cls  += loss_dict["cls"]
            total_reg  += loss_dict["reg"]
            total_sim  += loss_dict["sim"]
            n_batches  += 1

        n = max(n_batches, 1)
        return {"loss": total_loss / n, "cls": total_cls / n,
                "reg": total_reg / n,  "sim": total_sim / n}

    @torch.no_grad()
    def _val_epoch(self) -> dict:
        self.model.eval()
        correct = total = 0
        val_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            feat_key  = "bert_feat" if self.encoder_type == "bert" else "text_feat"
            text_feat = batch[feat_key].to(self.device, non_blocking=True)
            tab_feat  = batch["tab_feat"].to(self.device,  non_blocking=True)
            labels    = batch["label"].to(self.device,     non_blocking=True)
            wab_tgt   = batch["wab_aq"].to(self.device,   non_blocking=True)
            audio_feat, has_audio = self._get_audio(batch)
            seg_feat, n_segs      = self._get_seg_audio(batch)
            para_feat             = self._get_para(batch)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(text_feat, tab_feat, audio_feat, has_audio,
                                     seg_feat, n_segs, para_feat)
                loss, _ = self.model.compute_loss(
                    outputs, labels, wab_tgt,
                    lambda_reg=self._lambda_reg, lambda_sim=self._lambda_sim,
                )

            val_loss += loss.item()
            preds     = outputs["cls_logits"].argmax(dim=-1)
            correct  += (preds == labels).sum().item()
            total    += labels.size(0)
            n_batches += 1

        return {"loss": val_loss / max(n_batches, 1), "acc": correct / max(total, 1)}

    def train(self) -> list:
        logger.info("=" * 60)
        logger.info("Multi-task Training | epochs=%d | device=%s | AMP=%s",
                    self.n_epochs, self.device, self.use_amp)
        focal_gamma = getattr(self.model.cls_head.focal, "gamma", None)
        if focal_gamma is not None:
            logger.info("Focal gamma=%.1f | LR warmup=%d epochs", focal_gamma, self._warmup_epochs)
        else:
            logger.info("CE loss | LR warmup=%d epochs", self._warmup_epochs)
        logger.info("=" * 60)

        for epoch in range(1, self.n_epochs + 1):
            t0 = time.time()
            tr = self._train_epoch()
            va = self._val_epoch()
            self.scheduler.step()
            elapsed = time.time() - t0

            current_lr = self.optimizer.param_groups[0]["lr"]
            record = {"epoch": epoch, "train": tr, "val": va, "lr": current_lr, "time": elapsed}
            self.history.append(record)

            if epoch % 10 == 0 or epoch in (1, self.n_epochs):
                logger.info(
                    "Epoch %3d/%d | lr=%.2e | train_loss=%.4f (cls=%.3f reg=%.3f sim=%.3f)"
                    " | val_loss=%.4f val_acc=%.4f | %.1fs",
                    epoch, self.n_epochs, current_lr,
                    tr["loss"], tr["cls"], tr["reg"], tr["sim"],
                    va["loss"], va["acc"], elapsed,
                )

            if va["acc"] > self.best_val_acc:
                self.best_val_acc = va["acc"]
                torch.save(self.model.state_dict(), self.best_ckpt)

        logger.info("Training done | best_val_acc=%.4f | saved: %s",
                    self.best_val_acc, self.best_ckpt)

        hist_path = self.output_dir / "training_history.json"
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

        return self.history

    def load_best(self):
        self.model.load_state_dict(
            torch.load(self.best_ckpt, map_location=self.device, weights_only=True)
        )
        logger.info("Best model loaded (val_acc=%.4f)", self.best_val_acc)
