"""Model F: Dual-Uncertainty Set Transformer + Deep Ensemble + Calibrated VOI Controller.

This is the flagship architecture for General-Purpose Interactive RAG:
1. Cross-Passage Set Transformer (`SetAttentionBlock`) to detect inter-passage NLI contradictions
   and multi-interpretation facet splits across retrieved documents.
2. Explicit Decoupled Uncertainty Branches:
   - Prompt Intent Uncertainty Branch: Estimates posterior distribution \theta over latent query facets
     and calibrated Prompt Certainty P(Prompt Clear).
   - Database Evidence Uncertainty Branch: Attends over passage entailment, contradiction, and hop
     coverage to estimate calibrated Evidence Sufficiency P(Evidence Verified).
3. Deep Ensemble of K=4 VOI Decision Heads to quantify epistemic uncertainty on out-of-distribution domains.
4. Post-hoc Temperature Scaling parameter for calibrated readiness/selective abstention.
"""
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data_pipeline.schema import DEFAULT_EMBED_DIM
from .set_transformer import SetAttentionBlock


class DualUncertaintySetTransformerRAGNet(nn.Module):
    """Model F: Decoupled Prompt/Evidence Uncertainty Set Transformer + Calibrated Ensemble VOI."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        ensemble_size: int = 4,
        dropout: float = 0.1,
        separate_evidence_branch: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ensemble_size = ensemble_size
        self.separate_evidence_branch = separate_evidence_branch

        self.query_proj = nn.Sequential(nn.Linear(76, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.passage_proj = nn.Sequential(nn.Linear(80, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.question_proj = nn.Sequential(nn.Linear(74, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.rewrite_proj = nn.Sequential(nn.Linear(72, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())

        # Cross-passage self-attention layers
        self.passage_sab = nn.ModuleList([
            SetAttentionBlock(hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.query_passage_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.question_passage_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.rewrite_passage_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # Branch 1: Prompt / Query Intent Uncertainty Encoder
        self.prompt_branch = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.intent_mean_head = nn.Linear(hidden_dim, 8)
        self.intent_logstd_head = nn.Linear(hidden_dim, 8)
        self.prompt_cert_head = nn.Linear(hidden_dim, 1)

        # Branch 2: Database / Evidence Uncertainty Encoder (explicitly incorporates cross-passage conflict/dispersion)
        evid_in_dim = hidden_dim * 2 + 32 if separate_evidence_branch else hidden_dim * 2
        self.evidence_branch = nn.Sequential(
            nn.Linear(evid_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.evid_cert_head = nn.Linear(hidden_dim, 1)

        # Joint Dual-Uncertainty Context (2 * hidden_dim) -> 2x2 Quadrant & Regret Heads
        self.quadrant_head = nn.Linear(hidden_dim * 2, 4)
        self.readiness_head = nn.Linear(hidden_dim * 2, 2)  # [readiness_logit, est_regret_raw]

        # Deep Ensemble of K=4 VOI Heads for Macro Actions (4), Candidate Questions (Q), and Search Rewrites (S)
        self.ens_macro_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 4))
            for _ in range(ensemble_size)
        ])
        self.ens_question_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
            for _ in range(ensemble_size)
        ])
        self.ens_rewrite_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
            for _ in range(ensemble_size)
        ])

        # Learnable post-hoc calibration temperature
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        q_emb = self.query_proj(batch["query_vec"]).unsqueeze(1)        # (B, 1, H)
        p_emb = self.passage_proj(batch["passage_vecs"])                # (B, K, H)
        p_mask = batch["passage_mask"]                                  # (B, K)

        for sab in self.passage_sab:
            p_emb = sab(p_emb, key_padding_mask=p_mask)

        pooled_p, _ = self.query_passage_attn(q_emb, p_emb, p_emb, key_padding_mask=p_mask)
        q_flat = q_emb.squeeze(1)
        p_flat = pooled_p.squeeze(1)

        # 1. Prompt Uncertainty Branch
        prompt_h = self.prompt_branch(torch.cat([q_flat, p_flat], dim=-1))
        intent_mean = F.softmax(self.intent_mean_head(prompt_h), dim=-1)
        intent_std = F.softplus(self.intent_logstd_head(prompt_h)) + 0.01
        prompt_cert_logit = self.prompt_cert_head(prompt_h).squeeze(-1)

        # 2. Evidence Uncertainty Branch
        if self.separate_evidence_branch:
            # Extract passage mean(16) and passage std(16) from flat_summary[76:108]
            p_stats_summary = batch["flat_summary"][:, 76:108]
            evid_in = torch.cat([q_flat, p_flat, p_stats_summary], dim=-1)
        else:
            evid_in = torch.cat([q_flat, p_flat], dim=-1)

        evid_h = self.evidence_branch(evid_in)
        evid_cert_logit = self.evid_cert_head(evid_h).squeeze(-1)

        dual_ctx = torch.cat([prompt_h, evid_h], dim=-1)  # (B, 2H)
        quad_logits = self.quadrant_head(dual_ctx)
        ready_out = self.readiness_head(dual_ctx)
        raw_readiness_logit = ready_out[:, 0]
        est_regret = F.softplus(ready_out[:, 1])

        temp = torch.clamp(self.temperature, min=0.2, max=5.0)
        calibrated_readiness_logit = raw_readiness_logit / temp

        # 3. Candidate Question & Rewrite Cross-Attention
        cand_q = self.question_proj(batch["question_vecs"])             # (B, Q, H)
        q_attn, _ = self.question_passage_attn(cand_q, p_emb, p_emb, key_padding_mask=p_mask)
        q_ctx = cand_q + q_attn
        dual_q = dual_ctx.unsqueeze(1).expand(-1, cand_q.shape[1], -1)
        q_in = torch.cat([q_ctx, dual_q], dim=-1)                       # (B, Q, 3H)

        cand_s = self.rewrite_proj(batch["rewrite_vecs"])               # (B, S, H)
        s_attn, _ = self.rewrite_passage_attn(cand_s, p_emb, p_emb, key_padding_mask=p_mask)
        s_ctx = cand_s + s_attn
        dual_s = dual_ctx.unsqueeze(1).expand(-1, cand_s.shape[1], -1)
        s_in = torch.cat([s_ctx, dual_s], dim=-1)                       # (B, S, 3H)

        # 4. Deep Ensemble Predictions + Analytical Posterior EVSI Blending
        ens_macro, ens_q, ens_s = [], [], []
        for k in range(self.ensemble_size):
            ens_macro.append(self.ens_macro_heads[k](dual_ctx))
            ens_q.append(self.ens_question_heads[k](q_in).squeeze(-1))
            ens_s.append(self.ens_rewrite_heads[k](s_in).squeeze(-1))

        ens_macro_stack = torch.stack(ens_macro, dim=0)                 # (K, B, 4)
        ens_q_stack = torch.stack(ens_q, dim=0)                         # (K, B, Q)
        ens_s_stack = torch.stack(ens_s, dim=0)                         # (K, B, S)

        macro_scores = torch.mean(ens_macro_stack, dim=0)
        macro_epistemic_std = torch.std(ens_macro_stack, dim=0)
        q_scores = torch.mean(ens_q_stack, dim=0)
        s_scores = torch.mean(ens_s_stack, dim=0)

        return {
            "macro_action_scores": macro_scores,
            "macro_epistemic_std": macro_epistemic_std,
            "ens_macro_stack": ens_macro_stack,
            "ens_q_stack": ens_q_stack,
            "ens_s_stack": ens_s_stack,
            "question_scores": q_scores,
            "rewrite_scores": s_scores,
            "intent_mean": intent_mean,
            "intent_std": intent_std,
            "prompt_cert_logit": prompt_cert_logit,
            "evid_cert_logit": evid_cert_logit,
            "readiness_logit": calibrated_readiness_logit,
            "raw_readiness_logit": raw_readiness_logit,
            "est_regret": est_regret,
            "quadrant_logits": quad_logits,
        }
