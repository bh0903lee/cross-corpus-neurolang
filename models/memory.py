"""
memory.py — Memory-Augmented Neural Network for per-patient longitudinal
trajectory tracking via a key-value external memory.

Structure:
  - Key-Value external memory: {patient_id → [emb_t1, emb_t2, ...]}
  - New visits append to the patient's slot, accumulating the record
  - Trajectory embedding: LSTM compresses the temporal sequence
  - Longitudinal similarity retrieval
  - Disease progression prediction (Δrepresentation = r(t2) - r(t1))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from config import HIDDEN_DIM, MEMORY_SIZE, MEMORY_DIM


class PatientMemoryStore:
    """
    Python dict-based per-patient representation store.
    Could be replaced by a key-value or vector DB in deployment.
    """
    def __init__(self):
        self._store: Dict[str, List[torch.Tensor]] = {}

    def write(self, patient_id: str, embedding: torch.Tensor):
        """Store representation vector (value) under patient ID (key)."""
        emb = embedding.detach().cpu()
        if patient_id not in self._store:
            self._store[patient_id] = []
        self._store[patient_id].append(emb)

    def read(self, patient_id: str) -> Optional[torch.Tensor]:
        """Return the patient's time-ordered representation sequence. (T, D)"""
        if patient_id not in self._store:
            return None
        seq = self._store[patient_id]
        return torch.stack(seq, dim=0)  # (T, D)

    def all_patients(self) -> List[str]:
        return list(self._store.keys())

    def clear(self):
        self._store.clear()


class TrajectoryEncoder(nn.Module):
    """
    Time-ordered representation sequence (T, D) → trajectory embedding (D).
    LSTM captures long-range dependencies.
    """
    def __init__(self, d_model: int = HIDDEN_DIM):
        super().__init__()
        self.lstm = nn.LSTM(d_model, d_model // 2,
                            num_layers=1, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq: (1, T, D) — single-patient time series
        returns: (1, D) — trajectory embedding
        """
        _, (h, _) = self.lstm(seq)     # h: (2, 1, D//2)
        traj = h.transpose(0,1).contiguous().view(1, -1)   # (1, D)
        return self.norm(self.proj(traj))


class MemoryAugmentedModule(nn.Module):
    """
    Full memory-augmented network interface.

    Usage:
      1. write(patient_id, embedding)   — store representation at each visit
      2. get_trajectory(patient_id)     — query trajectory embedding
      3. compute_delta(emb_t1, emb_t2)  — Δrepresentation for progression risk
      4. longitudinal_search(query_emb) — similar trajectory retrieval (Top-K)
    """
    def __init__(self):
        super().__init__()
        self.store   = PatientMemoryStore()
        self.traj_enc = TrajectoryEncoder(HIDDEN_DIM)

        # Progression risk head:
        # input Δrepresentation (D) → risk score (1)
        self.prog_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),   # risk in 0-1
        )

    def write(self, patient_id: str, embedding: torch.Tensor):
        self.store.write(patient_id, embedding)

    def get_trajectory(self, patient_id: str,
                       device: torch.device = torch.device("cpu")
                       ) -> Optional[torch.Tensor]:
        """Return the patient's trajectory embedding, or None if no visits."""
        seq = self.store.read(patient_id)
        if seq is None or len(seq) < 1:
            return None
        seq = seq.unsqueeze(0).to(device)  # (1, T, D)
        return self.traj_enc(seq)           # (1, D)

    def compute_delta(self, emb_t1: torch.Tensor,
                      emb_t2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Δrepresentation = r(t2) - r(t1), fed to the progression risk head.
        returns: (delta, progression_risk_score)
        """
        delta = emb_t2 - emb_t1            # (B, D)
        risk  = self.prog_head(delta)       # (B, 1)
        return delta, risk

    def longitudinal_search(self, query_emb: torch.Tensor,
                            device: torch.device = torch.device("cpu"),
                            top_k: int = 3
                            ) -> List[Tuple[str, float]]:
        """
        Cosine similarity against all stored patients' trajectory embeddings.
        Returns Top-K most similar patients.
        """
        results = []
        q_norm = F.normalize(query_emb, dim=-1)   # (1, D)

        for pid in self.store.all_patients():
            traj = self.get_trajectory(pid, device)
            if traj is None:
                continue
            t_norm = F.normalize(traj, dim=-1)
            sim    = (q_norm @ t_norm.T).item()
            results.append((pid, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
