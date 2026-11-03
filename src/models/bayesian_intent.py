"""Model E: Bayesian Latent Intent Model + Explicit VOI Controller for General RAG.

Models the posterior distribution over the user's hidden semantic intent facets
\theta = (w_1, ..., w_8) with mean and variance, plus calibrated probabilities
of prompt clarity and evidence sufficiency. Uses an explicit decision-theoretic
VOI calculation to select between ANSWER, ASK, SEARCH, and ABSTAIN.
"""
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data_pipeline.schema import DEFAULT_EMBED_DIM


class BayesianLatentIntentRAGNet(nn.Module):
    """Model E: Probabilistic posterior over latent query intents + analytical VOI rule."""

    def __init__(self, input_dim: int = 164, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        # Posterior mean and log-std over the 8 latent query intent facets
        self.intent_mean_head = nn.Linear(hidden_dim, 8)
        self.intent_logstd_head = nn.Linear(hidden_dim, 8)

        # Dual uncertainty logits: [prompt_cert_logit, evid_cert_logit, readiness_logit, est_regret]
        self.aux_head = nn.Linear(hidden_dim, 4)
        self.quadrant_head = nn.Linear(hidden_dim, 4)
        self.macro_head = nn.Linear(hidden_dim, 4)
        self.rewrite_head = nn.Linear(hidden_dim + 72, 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        h = self.encoder(batch["flat_summary"])
        intent_mean = F.softmax(self.intent_mean_head(h), dim=-1)
        intent_std = F.softplus(self.intent_logstd_head(h)) + 0.01

        aux = self.aux_head(h)
        prompt_cert_prob = torch.sigmoid(aux[:, 0])  # (B,)
        evid_cert_prob = torch.sigmoid(aux[:, 1])    # (B,)
        est_regret = F.softplus(aux[:, 3])           # (B,)

        # Explicit Decision-Theoretic VOI calculation over candidate questions (Q=8):
        # Extract each question's target facet vector (8-dim) and distractor/cost flags
        D_emb = DEFAULT_EMBED_DIM
        q_stats = batch["question_vecs"][:, :, D_emb:]  # (B, 8, 10)
        q_facets = q_stats[:, :, :8]                    # (B, 8, 8)
        q_is_distractor = q_stats[:, :, 8]              # (B, 8)
        q_cost = q_stats[:, :, 9] * 0.040               # (B, 8)

        # Also compute passage facet dispersion from passage_vecs
        p_facets = batch["passage_vecs"][:, :, D_emb + 8 : D_emb + 16]  # (B, 6, 8)
        p_disp = torch.std(p_facets, dim=1)                             # (B, 8)

        # Expected regret reduction if question q is asked:
        # Proportional to (1 - prompt_cert_prob) * evid_cert_prob * <q_facets, intent_mean + p_disp>
        facet_value = intent_mean + 1.5 * p_disp + 0.5 * intent_std     # (B, 8)
        targeted_gain = torch.einsum("bqf,bf->bq", q_facets, facet_value)
        ask_gate = ((1.0 - prompt_cert_prob) * evid_cert_prob * est_regret).unsqueeze(1)
        analytic_q_voi = ask_gate * targeted_gain * (1.0 - q_is_distractor) - q_cost

        S = batch["rewrite_vecs"].shape[1]
        h_s = h.unsqueeze(1).expand(-1, S, -1)
        s_scores = self.rewrite_head(torch.cat([h_s, batch["rewrite_vecs"]], dim=-1)).squeeze(-1)

        macro_scores = self.macro_head(h)
        quad_logits = self.quadrant_head(h)

        return {
            "macro_action_scores": macro_scores,
            "question_scores": analytic_q_voi,
            "rewrite_scores": s_scores,
            "intent_mean": intent_mean,
            "intent_std": intent_std,
            "prompt_cert_logit": aux[:, 0],
            "evid_cert_logit": aux[:, 1],
            "readiness_logit": aux[:, 2],
            "est_regret": est_regret,
            "quadrant_logits": quad_logits,
        }
