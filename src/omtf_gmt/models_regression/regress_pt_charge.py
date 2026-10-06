"""
GMT edge-transformer model — Branch B, Phase B2.

Architecture
------------
One round of message passing on the full valid-stub graph (all cross-stub pairs):

  1. Node encoder:   MLP(F → H)            per-stub embedding
  2. Edge encoder:   MLP(2H → 1) → σ       pairwise compatibility score in [0,1]
  3. Node aggregation:
       ctx_i = Σ_j  score(i,j) · emb_j     score-weighted sum of neighbour embeddings
               ─────────────────────────
               Σ_j  score(i,j)             (normalised, avoids scale dependence on n_stubs)
  4. Node updater:   TransformerConv       concat(emb_i, ctx_i) → updated embedding
  5. Node head:      Linear(H → 1)         node logit (signal/noise classification)
  6. Global pool:    masked mean of updated embeddings → global context (H,)
  7. Candidate head: MLP(H → K)            candidate logits
  8. pT head:        MLP(H → K)            positive pT prediction (softplus)
  9. Charge head:    MLP(H → K)            charge prediction (tanh)

Edge policy (section 13 of study)
----------------------------------
All valid stub pairs are included in the first software version — no hardware
sparsification yet.  The model learns which cross-layer pairs are informative.
Self-edges (i==i) and edges to padding nodes are masked out before aggregation.

Memory
------
The dominant tensor is edge_in: (N, Nmax, Nmax, 2H).
At batch=2048, Nmax=24, H=64 with AMP (fp16): ~150 MB.  Fine on A100.
Reduce --batch-size if running on smaller GPUs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import TransformerConv
from omtf_gmt.features import N_FEATURES

K_MAX = 3


def _mlp(dims: list[int], dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class GMTRegressPtCharge(nn.Module):
    def __init__(self, hidden: int = 64, K: int = K_MAX, dropout: float = 0.0):
        super().__init__()
        self.K = K
        H = hidden

        self.pt_head      = _mlp([H, H, K], dropout)
        self.chg_head     = _mlp([H, H, K], dropout)

        self.name         = "regress_pt_charge"
    
    def forward(
        self,
        global_ctx:  torch.Tensor,   # (N, H)
        **_,
    ) -> dict[str, torch.Tensor]:

        pt_pred     = F.softplus(self.pt_head(global_ctx))        # (N, K)
        chg_pred    = torch.tanh(self.chg_head(global_ctx))       # (N, K)

        return {
            "pt_pred":          pt_pred,
            "charge_pred":      chg_pred,
        } 


def build_regress_pt_charge(
    hidden: int = 64, K: int = K_MAX, dropout: float = 0.0
) -> GMTRegressPtCharge:
    return GMTRegressPtCharge(hidden=hidden, K=K, dropout=dropout)
