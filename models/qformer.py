"""
qformer.py — Q-Former-based modality fusion module.

Q-Former structure:
  - N learnable query tokens
  - Each transformer block: Cross-Attention → Self-Attention → FeedForward
  - Input modalities (text, tabular, ...) serve as Key/Value; queries as Query
  - Output: (B, N_QUERY_TOKENS, HIDDEN_DIM) → fixed-length representation

Adapts the BLIP-2 (Salesforce 2023) architecture to clinical multimodal data.
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import HIDDEN_DIM, N_QUERY_TOKENS, N_ATTN_HEADS, N_QFORMER_LAYERS


class CrossAttention(nn.Module):
    """
    Queries (Q) cross-attend to modality embeddings (K, V).
    Query size stays fixed (N_QUERY_TOKENS) regardless of context size.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale    = self.d_head ** -0.5

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # query:   (B, Nq, D)
        # context: (B, Nk, D)  ← modality embeddings (Key, Value)
        B, Nq, D = query.shape
        Nk       = context.shape[1]

        Q = self.q_proj(query).view(B, Nq, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(context).view(B, Nk, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(context).view(B, Nk, self.n_heads, self.d_head).transpose(1, 2)

        attn  = (Q @ K.transpose(-2, -1)) * self.scale   # (B, H, Nq, Nk)
        attn  = F.softmax(attn, dim=-1)
        out   = (attn @ V).transpose(1, 2).contiguous().view(B, Nq, D)
        return self.out_proj(out)


class QFormerBlock(nn.Module):
    """
    Single Q-Former block: Cross-Attention → Self-Attention → FeedForward
    """
    def __init__(self, d_model: int, n_heads: int, ff_dim: int):
        super().__init__()
        # Cross-Attention (queries ← modality context)
        self.cross_attn = CrossAttention(d_model, n_heads)
        self.norm1      = nn.LayerNorm(d_model)

        # Self-Attention (information exchange among queries)
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2      = nn.LayerNorm(d_model)

        # FeedForward
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(ff_dim, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(0.1)

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Cross-Attention: queries select key information from modality embeddings
        queries = queries + self.drop(self.cross_attn(self.norm1(queries), context))
        # Self-Attention: interaction among queries
        sa_out, _ = self.self_attn(self.norm2(queries), self.norm2(queries), self.norm2(queries))
        queries   = queries + self.drop(sa_out)
        # FeedForward
        queries   = queries + self.drop(self.ff(self.norm3(queries)))
        return queries


class QFormer(nn.Module):
    """
    Full Q-Former module.
    Inputs:
      text_emb  (B, HIDDEN_DIM)          : text encoder output
      tab_emb   (B, HIDDEN_DIM)          : tabular encoder output
      audio_emb (B, HIDDEN_DIM) optional : audio encoder output (2-modality if absent)
    Outputs:
      shared_repr (B, HIDDEN_DIM): shared latent representation
      query_out   (B, Nq, D)     : query sequence usable by task heads

    Modality tokens [TEXT]=0, [TAB]=1, [AUDIO]=2 are added before cross-attention.
    """
    def __init__(self, n_query_tokens: int = None, n_modality_types: int = 3):
        super().__init__()
        D  = HIDDEN_DIM
        Nq = n_query_tokens if n_query_tokens is not None else N_QUERY_TOKENS
        ff = D * 2

        self.query_tokens = nn.Parameter(torch.randn(1, Nq, D) * 0.02)

        # Modality token embeddings: [TEXT]=0, [TAB]=1, [AUDIO]=2, [PARA]=3 (optional)
        self.modality_embed = nn.Embedding(n_modality_types, D)

        self.blocks = nn.ModuleList([
            QFormerBlock(D, N_ATTN_HEADS, ff)
            for _ in range(N_QFORMER_LAYERS)
        ])
        self.norm = nn.LayerNorm(D)
        self.pool_proj = nn.Linear(Nq * D, D)

    def forward(self, text_emb: torch.Tensor,
                tab_emb: Optional[torch.Tensor] = None,
                audio_emb: Optional[torch.Tensor] = None,
                para_emb: Optional[torch.Tensor] = None):
        B = text_emb.size(0)
        dev = text_emb.device

        t_tok  = self.modality_embed(torch.zeros(B, 1, dtype=torch.long, device=dev))   # TEXT=0

        text_ctx = text_emb.unsqueeze(1) + t_tok   # (B, 1, D)

        parts = [text_ctx]
        if audio_emb is not None:
            a_tok     = self.modality_embed(
                torch.full((B, 1), 2, dtype=torch.long, device=dev)  # AUDIO=2
            )
            parts.append(audio_emb.unsqueeze(1) + a_tok)
        if para_emb is not None:
            p_tok     = self.modality_embed(
                torch.full((B, 1), 3, dtype=torch.long, device=dev)  # PARA=3
            )
            parts.append(para_emb.unsqueeze(1) + p_tok)
        if tab_emb is not None:
            tb_tok  = self.modality_embed(torch.ones(B, 1, dtype=torch.long, device=dev))  # TAB=1
            parts.append(tab_emb.unsqueeze(1) + tb_tok)
        context = torch.cat(parts, dim=1)  # (B, 1-4, D)

        queries = self.query_tokens.expand(B, -1, -1).clone()
        for block in self.blocks:
            queries = block(queries, context)
        queries = self.norm(queries)

        shared_repr = self.pool_proj(queries.reshape(B, -1))
        return shared_repr, queries
