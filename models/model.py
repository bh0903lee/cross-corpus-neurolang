"""
model.py — Full multimodal foundation model assembly.

Data flow:
  [text features]    → TextEncoder    ─┐
  [audio features]   → AudioEncoder  ──┼→ QFormer / DirectFusion → SharedLatentRepr → [5 task heads]
  [clinical metadata]→ TabularEncoder ─┘

no_qformer=True replaces fusion with DirectFusion (concat+proj) (ablation A1).
n_query_tokens override changes the number of QFormer query tokens (ablation A7).
"""
from typing import Optional
import torch
import torch.nn as nn
from .encoders import (TextEncoder, BertTextEncoder, TabularEncoder, AudioEncoder,
                        AudioEncoderAttn, ParalinguisticEncoder)
from .qformer  import QFormer
from .memory   import MemoryAugmentedModule
from .heads    import (ClassificationHead, RegressionHead,
                       SimilarityHead, ProgressionHead, GenerationHead,
                       multitask_loss)
from config import HIDDEN_DIM


class DirectFusion(nn.Module):
    """Simple concat + linear projection without Q-Former (ablation A1)."""
    def __init__(self, n_modalities: int):
        super().__init__()
        self.proj = nn.Linear(n_modalities * HIDDEN_DIM, HIDDEN_DIM)

    def forward(self, text_emb: torch.Tensor,
                tab_emb: Optional[torch.Tensor] = None,
                audio_emb: Optional[torch.Tensor] = None,
                para_emb: Optional[torch.Tensor] = None):
        parts = [text_emb]
        if audio_emb is not None:
            parts.append(audio_emb)
        if para_emb is not None:
            parts.append(para_emb)
        if tab_emb is not None:
            parts.append(tab_emb)
        shared = self.proj(torch.cat(parts, dim=-1))
        return shared, shared.unsqueeze(1)   # (B, D), (B, 1, D) — API compatible with QFormer


class MultimodalFoundationModel(nn.Module):
    """
    Full multimodal foundation model.

    encoder_type : 'ling' (12-dim features) or 'bert' (pre-computed CLS embeddings)
    use_audio    : True → include AudioEncoder
    no_qformer   : True → DirectFusion (ablation A1)
    n_query_tokens: None → config default; integer → override (ablation A7)
    """
    def __init__(self, encoder_type: str = "ling", use_audio: bool = False,
                 no_qformer: bool = False, n_query_tokens: int = None,
                 use_missing_token: bool = False,
                 use_para: bool = False,
                 use_attn_audio: bool = False,
                 use_tabular: bool = True):
        super().__init__()
        self.encoder_type      = encoder_type
        self.use_audio         = use_audio
        self.use_missing_token = use_missing_token
        self.use_para          = use_para
        self.use_attn_audio    = use_attn_audio
        self.use_tabular       = use_tabular

        self.text_enc = BertTextEncoder() if encoder_type == "bert" else TextEncoder()
        self.tab_enc  = TabularEncoder() if use_tabular else None
        if use_audio:
            if use_attn_audio:
                self.audio_enc = AudioEncoderAttn()
            else:
                self.audio_enc = AudioEncoder()
            if use_missing_token:
                self.missing_audio_token = nn.Parameter(torch.zeros(1, HIDDEN_DIM))
        if use_para:
            from paralinguistic_extractor import PARA_DIM
            self.para_enc = ParalinguisticEncoder(para_dim=PARA_DIM)

        n_modality_types = 4 if use_para else 3
        n_mod = (3 if use_audio else 2) + (1 if use_para else 0) - (0 if use_tabular else 1)
        self.qformer = (DirectFusion(n_mod) if no_qformer
                        else QFormer(n_query_tokens=n_query_tokens,
                                     n_modality_types=n_modality_types))

        self.cls_head  = ClassificationHead()
        self.reg_head  = RegressionHead()
        self.sim_head  = SimilarityHead()
        self.prog_head = ProgressionHead()
        self.gen_head  = GenerationHead()
        self.memory    = MemoryAugmentedModule()

    def forward(self, text_feat: torch.Tensor,
                tab_feat: torch.Tensor,
                audio_feat: Optional[torch.Tensor] = None,
                has_audio: Optional[torch.Tensor] = None,
                audio_seg_feat: Optional[torch.Tensor] = None,
                n_segs: Optional[torch.Tensor] = None,
                para_feat: Optional[torch.Tensor] = None) -> dict:
        text_emb = self.text_enc(text_feat)
        tab_emb  = self.tab_enc(tab_feat) if self.use_tabular else None

        audio_emb = None
        if self.use_audio:
            if self.use_attn_audio and audio_seg_feat is not None:
                audio_emb = self.audio_enc(audio_seg_feat, n_segs)
            elif audio_feat is not None:
                audio_emb = self.audio_enc(audio_feat)

            if audio_emb is not None and has_audio is not None:
                gate = has_audio.float().unsqueeze(1).to(audio_emb.device)
                if self.use_missing_token:
                    missing = self.missing_audio_token.expand(audio_emb.size(0), -1)
                    audio_emb = gate * audio_emb + (1.0 - gate) * missing
                else:
                    audio_emb = audio_emb * gate

        para_emb = None
        if self.use_para and para_feat is not None:
            para_emb = self.para_enc(para_feat)
            # gate paralinguistic by has_audio (same availability)
            if has_audio is not None:
                gate_p = has_audio.float().unsqueeze(1).to(para_emb.device)
                para_emb = para_emb * gate_p

        shared_repr, query_out = self.qformer(text_emb, tab_emb, audio_emb, para_emb)

        return {
            "cls_logits":  self.cls_head(shared_repr),
            "reg_pred":    self.reg_head(shared_repr),
            "sim_emb":     self.sim_head(shared_repr),
            "shared_repr": shared_repr,
            "query_out":   query_out,
        }

    def compute_loss(self, outputs: dict, labels: torch.Tensor,
                     wab_targets: torch.Tensor,
                     lambda_reg: float = None, lambda_sim: float = None):
        return multitask_loss(
            outputs["cls_logits"], labels,
            outputs["reg_pred"],   wab_targets,
            outputs["sim_emb"],    labels,
            self.cls_head, self.reg_head, self.sim_head,
            lambda_reg=lambda_reg, lambda_sim=lambda_sim,
        )

    def generate_explanation(self, pred_label: int,
                             feature_dict: dict, wab_pred: float) -> str:
        return self.gen_head.generate(pred_label, feature_dict, wab_pred)

    def predict(self, text_feat: torch.Tensor,
                tab_feat: torch.Tensor,
                audio_feat: Optional[torch.Tensor] = None,
                has_audio: Optional[torch.Tensor] = None,
                patient_id: str = None,
                write_memory: bool = True) -> dict:
        self.eval()
        with torch.no_grad():
            out = self.forward(text_feat, tab_feat, audio_feat, has_audio)
        pred_label = out["cls_logits"].argmax(dim=-1).item()
        wab_pred   = out["reg_pred"].item()
        if patient_id and write_memory:
            self.memory.write(patient_id, out["shared_repr"].squeeze(0))
        return {
            "pred_label":  pred_label,
            "wab_pred":    wab_pred,
            "sim_emb":     out["sim_emb"],
            "shared_repr": out["shared_repr"],
        }

    def get_similar_patients(self, patient_id: str,
                             device: torch.device, top_k: int = 3):
        traj = self.memory.get_trajectory(patient_id, device)
        if traj is None:
            return []
        return self.memory.longitudinal_search(traj, device, top_k)
