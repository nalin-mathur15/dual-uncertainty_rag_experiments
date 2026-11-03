"""Model D: Cross-Passage Set Transformer for General RAG.

Uses permutation-equivariant self-attention over the set of K retrieved passages and
cross-attention between the user query, competing passages, candidate clarification
questions, and candidate search rewrites.
Tests whether reasoning over inter-document contradictions and multi-interpretation
divergence improves action & question selection over pooled MLPs.
"""
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class SetAttentionBlock(nn.Module):
    """Permutation-equivariant Transformer block over a set of retrieved passages."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class SetTransformerGeneralRAGNet(nn.Module):
    """Model D: Set Transformer over Query, Retrieved Passage Set, Questions, and Rewrites."""

    def __init__(self, hidden_dim: int = 256, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.query_proj = nn.Sequential(nn.Linear(76, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.passage_proj = nn.Sequential(nn.Linear(80, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.question_proj = nn.Sequential(nn.Linear(74, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.rewrite_proj = nn.Sequential(nn.Linear(72, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())

        # Cross-passage self-attention (detects inter-document contradictions & interpretation splits)
        self.passage_sab = nn.ModuleList([
            SetAttentionBlock(hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.query_passage_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.question_passage_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.rewrite_passage_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.ctx_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.macro_head = nn.Linear(hidden_dim, 4)
        self.question_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.rewrite_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.aux_head = nn.Linear(hidden_dim, 4)
        self.quadrant_head = nn.Linear(hidden_dim, 4)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        q_emb = self.query_proj(batch["query_vec"]).unsqueeze(1)        # (B, 1, H)
        p_emb = self.passage_proj(batch["passage_vecs"])                # (B, K, H)
        p_mask = batch["passage_mask"]                                  # (B, K)

        for sab in self.passage_sab:
            p_emb = sab(p_emb, key_padding_mask=p_mask)

        pooled_p, _ = self.query_passage_attn(q_emb, p_emb, p_emb, key_padding_mask=p_mask)
        global_ctx = self.ctx_fusion(torch.cat([q_emb.squeeze(1), pooled_p.squeeze(1)], dim=-1))  # (B, H)

        cand_q = self.question_proj(batch["question_vecs"])             # (B, Q, H)
        q_attn, _ = self.question_passage_attn(cand_q, p_emb, p_emb, key_padding_mask=p_mask)
        q_ctx = cand_q + q_attn
        g_q = global_ctx.unsqueeze(1).expand(-1, cand_q.shape[1], -1)
        q_scores = self.question_head(torch.cat([q_ctx, g_q], dim=-1)).squeeze(-1)

        cand_s = self.rewrite_proj(batch["rewrite_vecs"])               # (B, S, H)
        s_attn, _ = self.rewrite_passage_attn(cand_s, p_emb, p_emb, key_padding_mask=p_mask)
        s_ctx = cand_s + s_attn
        g_s = global_ctx.unsqueeze(1).expand(-1, cand_s.shape[1], -1)
        s_scores = self.rewrite_head(torch.cat([s_ctx, g_s], dim=-1)).squeeze(-1)

        macro_scores = self.macro_head(global_ctx)
        aux = self.aux_head(global_ctx)
        quad_logits = self.quadrant_head(global_ctx)

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
