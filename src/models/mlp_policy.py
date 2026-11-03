"""Model C: Pooled Feedforward MLP Policy Network for General RAG.

Consumes the 164-dim aggregated summary vector of query features and pooled passage
statistics. Tests whether fixed-size pooled features are sufficient without explicit
set attention across competing retrieved passages.
"""
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPGeneralRAGNet(nn.Module):
    """Three-layer feedforward network over pooled query and passage summary features."""

    def __init__(self, input_dim: int = 164, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.macro_head = nn.Linear(hidden_dim, 4)
        self.question_head = nn.Linear(hidden_dim + 74, 1)
        self.rewrite_head = nn.Linear(hidden_dim + 72, 1)
        # [prompt_cert_logit, evid_cert_logit, readiness_logit, est_regret]
        self.aux_head = nn.Linear(hidden_dim, 4)
        self.quadrant_head = nn.Linear(hidden_dim, 4)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        h = self.backbone(batch["flat_summary"])
        macro_scores = self.macro_head(h)

        # Score candidate questions and rewrites conditioned on pooled state h
        Q = batch["question_vecs"].shape[1]
        h_q = h.unsqueeze(1).expand(-1, Q, -1)
        q_scores = self.question_head(torch.cat([h_q, batch["question_vecs"]], dim=-1)).squeeze(-1)

        S = batch["rewrite_vecs"].shape[1]
        h_s = h.unsqueeze(1).expand(-1, S, -1)
        s_scores = self.rewrite_head(torch.cat([h_s, batch["rewrite_vecs"]], dim=-1)).squeeze(-1)

        aux = self.aux_head(h)
        quad_logits = self.quadrant_head(h)

        return {
            "macro_action_scores": macro_scores,
            "question_scores": q_scores,
            "rewrite_scores": s_scores,
            "prompt_cert_logit": aux[:, 0],
            "evid_cert_logit": aux[:, 1],
            "readiness_logit": aux[:, 2],
            "est_regret": F.softplus(aux[:, 3]),
            "quadrant_logits": quad_logits,
        }
