"""
GMT edge-compatibility model — Branch B, Phase B2.

Architecture
------------
One round of message passing on the full valid-stub graph (all cross-stub pairs):

  1. Node encoder:   MLP(F → H)            per-stub embedding
  2. Edge encoder:   MLP(2H → 1) → σ       pairwise compatibility score in [0,1]
  3. Node aggregation:
       ctx_i = Σ_j  score(i,j) · emb_j     score-weighted sum of neighbour embeddings
               ─────────────────────────
               Σ_j  score(i,j)             (normalised, avoids scale dependence on n_stubs)
  4. Node updater:   MLP(2H → H)           concat(emb_i, ctx_i) → updated embedding
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


class GMTEdgeCompatEdges(nn.Module):
    def __init__(self, hidden: int = 64, K: int = K_MAX, dropout: float = 0.0):
        super().__init__()
        self.K = K
        H = hidden

        self.node_encoder = _mlp([N_FEATURES, H, H], dropout)
        self.edge_encoder = _mlp([2 * N_FEATURES, H, 1],      dropout)   # → scalar edge score
        self.node_updater = _mlp([H + N_FEATURES, H, H],       dropout)   # concat(emb, ctx) → updated
        self.node_head    = nn.Linear(H, 1)
        self.cand_head    = _mlp([H, H, K], dropout)
        self.pt_head      = _mlp([H, H, K], dropout)
        self.chg_head     = _mlp([H, H, K], dropout)
        self.name         = "edge_compat_edges"

    def forward(
        self,
        stubs:      torch.Tensor,   # (N, Nmax, F)
        valid_mask: torch.Tensor,   # (N, Nmax) bool
        edge_node1: torch.Tensor,   # (N, Nmax, Nmax, F)
        edge_node2: torch.Tensor,   # (N, Nmax, Nmax, F)
        edge_mask:  torch.Tensor,   # (N, Nmax, Nmax) bool
        **_,
    ) -> dict[str, torch.Tensor]:
        N, Nmax, _ = stubs.shape
        device     = stubs.device

        # ---- 1. node encoding ----
        node_emb = self.node_encoder(stubs)                        # (N, Nmax, H)


        # ---- 2. edge encoding ----
        edge_score = torch.sigmoid(
            self.edge_encoder(torch.cat([edge_node1, edge_node2], dim=-1)).squeeze(-1)
        )                                                          # (N, Nmax, Nmax)
        vm = valid_mask                                            # (N, Nmax)

        # ---- 3. normalised score-weighted aggregation ----
        w = edge_score * edge_mask.float()                        # (N, Nmax, Nmax)
        denom = w.sum(dim=2, keepdim=True).clamp(min=1e-6)        # (N, Nmax, 1)
        w_n = (w / denom).unsqueeze(-1)                           # (N, Nmax, Nmax, 1)
        ctx = (w_n * edge_node2).sum(dim=2)                       # (N, Nmax, F)

        # ---- 4. node update ----
        node_upd = self.node_updater(
            torch.cat([node_emb, ctx], dim=-1)
        )                                                          # (N, Nmax, H)

        # ---- 5. masked global pool ----
        vmf   = valid_mask.unsqueeze(-1).float()                  # (N, Nmax, 1)
        denom_g = vmf.sum(dim=1).clamp(min=1.0)                   # (N, 1)
        global_ctx = (node_upd * vmf).sum(dim=1) / denom_g        # (N, H)

        # ---- 6-8. heads ----
        node_logit  = self.node_head(node_upd).squeeze(-1)        # (N, Nmax)
        cand_logits = self.cand_head(global_ctx)                  # (N, K)
        pt_pred     = F.softplus(self.pt_head(global_ctx))        # (N, K)
        chg_pred    = torch.tanh(self.chg_head(global_ctx))       # (N, K)

        return {
            "node_logit":       node_logit,
            "candidate_logits": cand_logits,
            "pt_pred":          pt_pred,
            "charge_pred":      chg_pred,
        }


def build_edge_compat_edges(
    hidden: int = 64, K: int = K_MAX, dropout: float = 0.0
) -> GMTEdgeCompatEdges:
    return GMTEdgeCompatEdges(hidden=hidden, K=K, dropout=dropout)
