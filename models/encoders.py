"""
encoders.py — Input modality encoders.

TextEncoder    : 12-dim linguistic features → TEXT_ENC_DIM embedding
TabularEncoder : 4-dim clinical metadata → TAB_ENC_DIM embedding

Pretrained text/audio encoders stay frozen; the lightweight encoders here
project their pre-computed features (plus tabular metadata) into the shared
space, with small rank-8 LoRA-style adapters added to each encoder layer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import N_LING_FEATURES, N_TAB_FEATURES, TEXT_ENC_DIM, TAB_ENC_DIM, HIDDEN_DIM, BERT_DIM, AUDIO_DIM


class LoRAAdapter(nn.Module):
    """
    Low-Rank Adaptation (LoRA): low-rank adapter added to a linear layer.
    W' = W + B @ A   (rank r << d_in, d_out)
    """
    def __init__(self, d_in: int, d_out: int, rank: int = 8):
        super().__init__()
        self.A = nn.Linear(d_in,  rank,  bias=False)
        self.B = nn.Linear(rank,  d_out, bias=False)
        nn.init.kaiming_uniform_(self.A.weight)
        nn.init.zeros_(self.B.weight)   # B initialized to 0 → no LoRA contribution at start

    def forward(self, x):
        return self.B(self.A(x))


class TextEncoder(nn.Module):
    """
    Linguistic feature vector (12-dim) → fixed-length embedding (TEXT_ENC_DIM).
    MLP over linguistic features plus a LoRA adapter.
    """
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(N_LING_FEATURES, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, TEXT_ENC_DIM),
            nn.LayerNorm(TEXT_ENC_DIM),
            nn.GELU(),
        )
        self.lora = LoRAAdapter(N_LING_FEATURES, TEXT_ENC_DIM, rank=4)
        self.proj = nn.Linear(TEXT_ENC_DIM, HIDDEN_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N_LING_FEATURES)
        base_out = self.base(x)           # (B, TEXT_ENC_DIM)
        lora_out = self.lora(x)           # (B, TEXT_ENC_DIM)  ← lightweight adapter
        fused    = base_out + lora_out    # residual-style sum
        return self.proj(fused)           # (B, HIDDEN_DIM)


class BertTextEncoder(nn.Module):
    """
    Pre-computed DistilBERT [CLS] embedding (BERT_DIM=768) → HIDDEN_DIM.
    BERT itself stays frozen; embeddings are pre-computed in data_loader.
    Only the projection + normalization layers here are trained.
    """
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(BERT_DIM, HIDDEN_DIM * 2),
            nn.LayerNorm(HIDDEN_DIM * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, BERT_DIM=768)
        return self.proj(x)   # (B, HIDDEN_DIM)


class ResidualBertTextEncoder(nn.Module):
    """
    Residual projection: frozen CLS embedding (input_dim) → HIDDEN_DIM.
    main path + skip connection + output LayerNorm.
    """
    def __init__(self, input_dim: int = BERT_DIM, dropout: float = 0.1):
        super().__init__()
        D = HIDDEN_DIM
        self.main = nn.Sequential(
            nn.Linear(input_dim, D * 2),
            nn.LayerNorm(D * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D * 2, D),
            nn.LayerNorm(D),
            nn.GELU(),
        )
        self.skip     = nn.Linear(input_dim, D, bias=False)
        self.out_norm = nn.LayerNorm(D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_norm(self.main(x) + self.skip(x))


class AudioEncoder(nn.Module):
    """
    Pre-computed wav2vec2 output (AUDIO_DIM) → HIDDEN_DIM.
    wav2vec2 itself stays frozen; features are pre-computed in audio_extractor.
    Only the projection + normalization layers here are trained.
    """
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(AUDIO_DIM, HIDDEN_DIM * 2),
            nn.LayerNorm(HIDDEN_DIM * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, AUDIO_DIM)
        return self.proj(x)   # (B, HIDDEN_DIM)


class TabularEncoder(nn.Module):
    """
    Clinical metadata (4-dim: age, sex, has_wab, wab_aq_norm) → TAB_ENC_DIM embedding.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_TAB_FEATURES, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(16, TAB_ENC_DIM),
            nn.LayerNorm(TAB_ENC_DIM),
            nn.GELU(),
        )
        self.proj = nn.Linear(TAB_ENC_DIM, HIDDEN_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N_TAB_FEATURES)
        return self.proj(self.net(x))     # (B, HIDDEN_DIM)


class ParalinguisticEncoder(nn.Module):
    """
    Paralinguistic features (PARA_DIM=15) → HIDDEN_DIM.
    F0, energy, ZCR, spectral centroid, MFCC 2-5 mean/std.
    """
    def __init__(self, para_dim: int = 15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(para_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, PARA_DIM)
        return self.net(x)   # (B, HIDDEN_DIM)


class AudioEncoderAttn(nn.Module):
    """
    Per-segment wav2vec2 features (B, S, AUDIO_DIM) → attention-pooled (B, HIDDEN_DIM).
    Learnable attention weights over segments instead of mean-pooling.
    n_segs: number of valid (non-padded) segments per sample; if None, all segments are used.
    """
    def __init__(self):
        super().__init__()
        self.seg_proj = nn.Linear(AUDIO_DIM, HIDDEN_DIM)
        self.attn_q   = nn.Linear(HIDDEN_DIM, 1)
        self.out_norm = nn.LayerNorm(HIDDEN_DIM)

    def forward(self, x: torch.Tensor,
                n_segs: torch.Tensor = None) -> torch.Tensor:
        # x: (B, S, AUDIO_DIM)
        h = self.seg_proj(x)          # (B, S, HIDDEN_DIM)
        scores = self.attn_q(h).squeeze(-1)   # (B, S)

        if n_segs is not None:
            S = x.size(1)
            mask = torch.arange(S, device=x.device).unsqueeze(0) >= n_segs.unsqueeze(1)
            scores = scores.masked_fill(mask, -1e4)  # -1e9 overflows float16 under AMP

        weights = F.softmax(scores, dim=-1)             # (B, S)
        pooled  = (weights.unsqueeze(-1) * h).sum(dim=1)   # (B, HIDDEN_DIM)
        return self.out_norm(pooled)
