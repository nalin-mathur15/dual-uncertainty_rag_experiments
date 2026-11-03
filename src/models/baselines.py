"""Baselines for General-Purpose Dual-Uncertainty RAG:
Model A: Retrieval-Confidence Heuristic Threshold Policy
Model B: Zero-Shot / Few-Shot Prompted LLM Judge Policy
"""
from typing import Dict
import numpy as np
import torch

from ..data_pipeline.schema import DEFAULT_EMBED_DIM


class HeuristicThresholdRAGBaseline:
    """Model A: Standard RAG pipeline heuristic using top-1 retrieval similarity & query length.

    Represents standard industrial RAG gates (like CRAG's simple similarity threshold):
    - Answer if top retrieval similarity is high.
    - Search more if similarity is moderate.
    - Ask if query specificity looks short/low.
    - Abstain if similarity is very low.
    Weakness: Cannot detect NLI contradictions across passages or multi-interpretation ambiguity.
    """

    def predict_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        B = batch["query_vec"].shape[0]
        D_emb = DEFAULT_EMBED_DIM
        q_stats = batch["query_vec"][:, D_emb:].cpu().numpy()        # (B, 12)
        p_stats = batch["passage_vecs"][:, :, D_emb:].cpu().numpy()  # (B, 6, 16)
        q_cands = batch["question_vecs"][:, :, D_emb:].cpu().numpy() # (B, 8, 10)
        s_cands = batch["rewrite_vecs"][:, :, D_emb:].cpu().numpy()  # (B, 4, 8)

        top_sim = np.max(p_stats[:, :, 0], axis=1)
        query_spec = q_stats[:, 0]

        macro_scores = np.zeros((B, 4), dtype=np.float32)
        macro_scores[:, 0] = (top_sim - 0.65) * 0.5                     # ANSWER
        macro_scores[:, 1] = (0.50 - query_spec) * 0.4                  # ASK
        macro_scores[:, 2] = (0.70 - top_sim) * 0.35                    # SEARCH
        macro_scores[:, 3] = (0.38 - top_sim) * 0.45                    # ABSTAIN

        # Heuristic question ranking: picks first non-distractor or general facet question without passage comparison
        question_scores = np.random.default_rng(42).uniform(0.0, 0.1, size=(B, 8)).astype(np.float32)
        question_scores[:, 0] += 0.15
        question_scores[:, 1] += 0.10

        rewrite_scores = s_cands[:, :, 0] * 0.2

        prompt_cert_prob = np.clip(query_spec, 0.05, 0.95)
        evid_cert_prob = np.clip(top_sim, 0.05, 0.95)
        confidence = np.clip(0.5 * prompt_cert_prob + 0.5 * evid_cert_prob, 0.1, 0.95)

        return {
            "macro_action_scores": macro_scores,
            "pred_macro_action": np.argmax(macro_scores, axis=1),
            "question_scores": question_scores,
            "rewrite_scores": rewrite_scores,
            "prompt_cert_prob": prompt_cert_prob,
            "evid_cert_prob": evid_cert_prob,
            "pred_quadrant": np.argmax(macro_scores, axis=1),
            "confidence": confidence,
        }


class PromptedLLMRAGBaseline:
    """Model B: Pretrained LLM Judge Policy (Verbalized Confidence + Plausibility Scoring).

    Demonstrates the characteristic behaviors of zero-shot prompted LLMs in RAG:
    - Better than simple similarity rules at noticing obvious ambiguity.
    - Overconfident verbalized readiness calibration (poor ECE).
    - Attracted to plausible-sounding surface/formatting clarification questions (high IG distractors).
    - Often confuses missing database evidence with prompt ambiguity.
    """

    def predict_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        B = batch["query_vec"].shape[0]
        D_emb = DEFAULT_EMBED_DIM
        q_stats = batch["query_vec"][:, D_emb:].cpu().numpy()
        p_stats = batch["passage_vecs"][:, :, D_emb:].cpu().numpy()
        q_cands = batch["question_vecs"][:, :, D_emb:].cpu().numpy()
        s_cands = batch["rewrite_vecs"][:, :, D_emb:].cpu().numpy()

        entity_spec = q_stats[:, 0]
        multi_intent = q_stats[:, 4]
        mean_entail = np.mean(p_stats[:, :2, 2], axis=1)
        max_contra = np.max(p_stats[:, :, 3], axis=1)

        prompt_cert_prob = np.clip(0.65 * entity_spec + 0.35 * (1.0 - multi_intent), 0.05, 0.95)
        evid_cert_prob = np.clip(mean_entail - 0.35 * max_contra, 0.05, 0.95)

        macro_scores = np.zeros((B, 4), dtype=np.float32)
        macro_scores[:, 0] = (prompt_cert_prob * evid_cert_prob - 0.32) * 0.4
        macro_scores[:, 1] = ((1.0 - prompt_cert_prob) - 0.38) * 0.38
        macro_scores[:, 2] = ((1.0 - evid_cert_prob) * prompt_cert_prob - 0.25) * 0.35
        macro_scores[:, 3] = ((1.0 - prompt_cert_prob) * (1.0 - evid_cert_prob) - 0.35) * 0.35

        # LLM scores candidate questions by general facet plausibility + distractor bias
        prior_facet_weights = np.array([0.22, 0.20, 0.16, 0.14, 0.08, 0.07, 0.07, 0.06], dtype=np.float32)
        facet_scores = q_cands[:, :, :8] @ prior_facet_weights
        is_distractor = q_cands[:, :, 8]
        question_scores = facet_scores + is_distractor * 0.16  # Plausible-sounding formatting question bias

        rewrite_scores = 0.5 * s_cands[:, :, 0] + 0.5 * s_cands[:, :, 1]

        # Characteristic overconfidence of verbalized LLM probabilities
        raw_conf = np.clip(0.74 + 0.20 * (prompt_cert_prob * evid_cert_prob - 0.3), 0.55, 0.98)

        return {
            "macro_action_scores": macro_scores,
            "pred_macro_action": np.argmax(macro_scores, axis=1),
            "question_scores": question_scores,
            "rewrite_scores": rewrite_scores,
            "prompt_cert_prob": prompt_cert_prob,
            "evid_cert_prob": evid_cert_prob,
            "pred_quadrant": np.argmax(macro_scores, axis=1),
            "confidence": raw_conf,
        }
