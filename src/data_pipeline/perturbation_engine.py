"""Counterfactual 2x2x2 Dual-Uncertainty Perturbation Engine for General RAG.

Transforms raw QA records into rich neural RAG decision states across the four
fundamental epistemic quadrants:
  Q0 (Clear Prompt, Sufficient Evidence)      -> ANSWER
  Q1 (Ambiguous Prompt, Sufficient Evidence)  -> ASK (Clarify)
  Q2 (Clear Prompt, Missing/Conflict Evid)    -> SEARCH (Retrieve more)
  Q3 (Ambiguous/Unresolvable Corpus)          -> ABSTAIN

Crucially, computes exact counterfactual Value of Information (VOI) for each
candidate clarification question q_1..q_Q and each candidate search rewrite s_1..s_S,
distinguishing genuine decision-improving questions from high-entropy surface distractors.
"""
from typing import List, Optional, Dict
import numpy as np

from .schema import (
    ACTION_ANSWER,
    ACTION_ASK,
    ACTION_SEARCH,
    ACTION_ABSTAIN,
    DEFAULT_EMBED_DIM,
    QUERY_STAT_DIM,
    PASSAGE_STAT_DIM,
    QUESTION_STAT_DIM,
    REWRITE_STAT_DIM,
    RawQARecord,
    RAGStateExample,
)


class DualUncertaintyPerturbationEngine:
    """Synthesizes large-scale (500k–1M+) general RAG states with exact VOI supervision."""

    def __init__(
        self,
        num_passages: int = 6,
        num_questions: int = 8,
        num_rewrites: int = 4,
        question_cost_lambda: float = 0.040,
        search_cost_gamma: float = 0.020,
        hallucination_penalty: float = 0.55,
        abstain_regret: float = 0.20,
    ):
        self.num_passages = num_passages
        self.num_questions = num_questions
        self.num_rewrites = num_rewrites
        self.question_cost_lambda = question_cost_lambda
        self.search_cost_gamma = search_cost_gamma
        self.hallucination_penalty = hallucination_penalty
        self.abstain_regret = abstain_regret

    def generate_state(
        self,
        raw_rec: RawQARecord,
        rng: np.random.Generator,
        quadrant: Optional[int] = None,
        split_mode: str = "in_distribution",
    ) -> RAGStateExample:
        """Generates a single perturbed RAG state in one of the 4 quadrants."""
        if quadrant is None:
            if split_mode == "adversarial_evidence":
                # Heavy emphasis on missing, partial-hop, and contradictory evidence states
                quadrant = int(rng.choice([0, 1, 2, 3], p=[0.15, 0.15, 0.45, 0.25]))
            else:
                quadrant = int(rng.choice([0, 1, 2, 3], p=[0.26, 0.26, 0.26, 0.22]))

        K = self.num_passages
        Q = self.num_questions
        S = self.num_rewrites
        D_emb = DEFAULT_EMBED_DIM

        # 1. Sample true latent user intent across 8 semantic facets (e.g. entity, year, jurisdiction, metric, etc.)
        # In domain_shift, shift which facets carry the critical ambiguity
        if split_mode == "domain_shift":
            prior_alpha = np.array([0.6, 0.6, 0.7, 0.8, 2.2, 2.5, 2.0, 1.8], dtype=np.float32)
        else:
            prior_alpha = np.array([2.4, 2.2, 1.8, 1.5, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
        latent_intent = rng.dirichlet(prior_alpha).astype(np.float32)
        dominant_facet = int(np.argmax(latent_intent))

        # Determine prompt clarity and evidence sufficiency from the quadrant
        prompt_clear = 1.0 if quadrant in (0, 2) else 0.0
        evidence_sufficient = 1.0 if quadrant in (0, 1) else 0.0

        # Sub-types of evidence failure in Quadrant 2 & 3:
        # 0: Missing bridge/gold hop, 1: Direct NLI contradiction between sources, 2: Unresolvable corpus gap
        if quadrant == 2:
            evid_failure_mode = int(rng.choice([0, 1], p=[0.55, 0.45]))
        elif quadrant == 3:
            evid_failure_mode = 2  # Unresolvable even after single search or coupled with severe ambiguity
        else:
            evid_failure_mode = -1

        # 2. Construct Query Embedding & Structural Ambiguity Features (76-dim)
        # Base semantic direction determined by record_id hash + noise
        rec_seed = abs(hash(raw_rec.record_id)) % (2**31 - 1)
        rec_rng = np.random.default_rng(rec_seed)
        base_topic_emb = rec_rng.normal(0, 1.0, size=D_emb).astype(np.float32)
        base_topic_emb /= np.linalg.norm(base_topic_emb) + 1e-8

        query_emb = base_topic_emb + rng.normal(0, 0.12, size=D_emb).astype(np.float32)
        query_emb /= np.linalg.norm(query_emb) + 1e-8

        # Query statistical features (12-dim):
        # [entity_specificity, temporal_qualifier, constraint_completeness, lexical_entropy,
        #  multi_intent_prior, query_length_norm, wh_specificity, domain_rarity, ... revealed_facets(4)]
        if prompt_clear > 0.5:
            entity_spec = rng.uniform(0.65, 0.98)
            temp_qual = rng.uniform(0.60, 0.95)
            constraint_comp = rng.uniform(0.70, 0.98)
            multi_intent_score = rng.uniform(0.05, 0.30)
        else:
            entity_spec = rng.uniform(0.10, 0.48)
            temp_qual = rng.uniform(0.05, 0.40)
            constraint_comp = rng.uniform(0.10, 0.45)
            multi_intent_score = rng.uniform(0.60, 0.95)

        # Add realistic overlap noise so a trivial linear threshold on 1 feature cannot game the benchmark
        if rng.random() < 0.12:
            entity_spec = rng.uniform(0.35, 0.65)
            constraint_comp = rng.uniform(0.35, 0.65)

        query_stats = np.array([
            entity_spec,
            temp_qual,
            constraint_comp,
            rng.uniform(0.2, 0.8),
            multi_intent_score,
            rng.uniform(0.3, 0.9),
            prompt_clear * rng.uniform(0.5, 0.9) + (1.0 - prompt_clear) * rng.uniform(0.1, 0.5),
            1.0 if split_mode == "domain_shift" else 0.1,
            latent_intent[0] if prompt_clear > 0.5 else 0.0,
            latent_intent[1] if prompt_clear > 0.5 else 0.0,
            latent_intent[2] if prompt_clear > 0.5 else 0.0,
            latent_intent[3] if prompt_clear > 0.5 else 0.0,
        ], dtype=np.float32)
        query_vec = np.concatenate([query_emb, query_stats]).astype(np.float32)

        # 3. Construct Retrieved Passage Set (K=6, 80-dim each)
        # Crucially, in Quadrant 1 (Ambiguous Prompt + Sufficient Evidence), different passages
        # support different interpretations (e.g., Passage 0 supports 1997 Titanic, Passage 1 supports 1953 Titanic).
        # A Set Transformer comparing passages notices high cross-passage divergence on `dominant_facet`!
        passage_vecs = np.zeros((K, D_emb + PASSAGE_STAT_DIM), dtype=np.float32)
        passage_mask = np.zeros(K, dtype=bool)

        # Track per-passage properties for summary & oracle calculation
        dense_sims = np.zeros(K, dtype=np.float32)
        entail_scores = np.zeros(K, dtype=np.float32)
        contra_scores = np.zeros(K, dtype=np.float32)
        hop_cov = np.zeros(K, dtype=np.float32)
        passage_facet_focus = np.zeros((K, 8), dtype=np.float32)

        for k in range(K):
            if evidence_sufficient > 0.5:
                # Passages have strong relevance and entailment for at least one interpretation
                if prompt_clear > 0.5:
                    # Quadrant 0: Top passages consistently support the exact clear query
                    is_gold = k < 2
                    d_sim = rng.uniform(0.75, 0.96) if is_gold else rng.uniform(0.35, 0.65)
                    entail = rng.uniform(0.72, 0.95) if is_gold else rng.uniform(0.15, 0.45)
                    contra = rng.uniform(0.01, 0.10)
                    h_cov = rng.uniform(0.78, 0.98) if is_gold else rng.uniform(0.20, 0.50)
                    passage_facet_focus[k, dominant_facet] = 1.0 if is_gold else 0.2
                else:
                    # Quadrant 1 (ASK): Passages cover multiple competing interpretations of the ambiguous query!
                    interp_facet = dominant_facet if (k % 2 == 0) else ((dominant_facet + 1) % 8)
                    d_sim = rng.uniform(0.68, 0.90) if k < 4 else rng.uniform(0.30, 0.55)
                    entail = rng.uniform(0.60, 0.88) if k < 4 else rng.uniform(0.15, 0.40)
                    contra = rng.uniform(0.05, 0.18)
                    h_cov = rng.uniform(0.65, 0.90) if k < 4 else rng.uniform(0.20, 0.45)
                    passage_facet_focus[k, interp_facet] = rng.uniform(0.75, 0.98)
            else:
                # Quadrant 2 (SEARCH) or Quadrant 3 (ABSTAIN): Evidence is missing or contradictory
                if evid_failure_mode == 0:
                    # Missing multi-hop bridge or low relevance retrieval
                    d_sim = rng.uniform(0.25, 0.58)
                    entail = rng.uniform(0.08, 0.35)
                    contra = rng.uniform(0.02, 0.15)
                    h_cov = rng.uniform(0.10, 0.42)
                elif evid_failure_mode == 1:
                    # Contradictory sources! Passage 0 and Passage 1 directly conflict (high contra score)
                    d_sim = rng.uniform(0.65, 0.88) if k < 2 else rng.uniform(0.30, 0.55)
                    entail = rng.uniform(0.50, 0.78) if k == 0 else rng.uniform(0.10, 0.35)
                    contra = rng.uniform(0.68, 0.95) if k in (0, 1) else rng.uniform(0.05, 0.20)
                    h_cov = rng.uniform(0.45, 0.75)
                else:
                    # Quadrant 3 (ABSTAIN): Unresolvable corpus gap / noisy conflicting snippets
                    d_sim = rng.uniform(0.12, 0.45)
                    entail = rng.uniform(0.02, 0.22)
                    contra = rng.uniform(0.25, 0.75) if k < 2 else rng.uniform(0.05, 0.25)
                    h_cov = rng.uniform(0.05, 0.28)

                passage_facet_focus[k] = rng.uniform(0.0, 0.25, size=8)

            dense_sims[k] = d_sim
            entail_scores[k] = entail
            contra_scores[k] = contra
            hop_cov[k] = h_cov

            p_emb = d_sim * base_topic_emb + np.sqrt(max(0.0, 1.0 - d_sim**2)) * rng.normal(0, 1.0, size=D_emb)
            p_emb = (p_emb / (np.linalg.norm(p_emb) + 1e-8)).astype(np.float32)

            p_stats = np.concatenate([
                np.array([
                    d_sim,
                    d_sim * rng.uniform(0.85, 1.10),  # BM25 lexical score
                    entail,
                    contra,
                    h_cov,
                    rng.uniform(0.5, 1.0) if k == 0 else rng.uniform(0.3, 0.8),  # Source authority
                    float(k) / K,
                    1.0 if (entail > 0.6 and contra < 0.25) else 0.0,
                ], dtype=np.float32),
                passage_facet_focus[k],  # (8,) facet alignment of passage k
            ])
            passage_vecs[k] = np.concatenate([p_emb, p_stats])

        # 4. Construct Candidate Clarification Questions (Q=8, 74-dim each) and Compute Exact VOI vs. IG
        # Questions 0..5 target specific latent intent facets; Questions 6..7 are high-entropy surface distractors
        # (e.g., asking formatting style or spelling preference — high entropy reduction, zero decision improvement).
        question_vecs = np.zeros((Q, D_emb + QUESTION_STAT_DIM), dtype=np.float32)
        question_voi = np.zeros(Q, dtype=np.float32)
        question_ig = np.zeros(Q, dtype=np.float32)

        # Base answer regret R(s) if generating an answer right now without acting:
        if quadrant == 0:
            current_regret = float(rng.uniform(0.01, 0.06))
        elif quadrant == 1:
            # Ambiguous query: answering without clarifying risks picking the wrong interpretation (~0.35 - 0.50 regret)
            current_regret = float(rng.uniform(0.32, 0.48))
        elif quadrant == 2:
            # Missing or contradictory evidence: answering right now risks hallucination (~0.38 - 0.55 regret)
            current_regret = float(rng.uniform(0.36, 0.54))
        else:
            # Quadrant 3: Both ambiguous and unresolvable in corpus
            current_regret = float(rng.uniform(0.48, 0.65))

        passage_facet_dispersion = np.std(passage_facet_focus, axis=0)  # (8,) how much retrieved passages disagree on each facet

        for q_idx in range(Q):
            is_distractor = q_idx >= 6
            target_facet = q_idx if not is_distractor else (q_idx % 8)
            inconv_cost = 1.0 if not is_distractor else 0.85

            # Generic Information Gain (entropy reduction on user's unstated preferences/formatting):
            # Distractor questions (e.g. "bullet points or paragraph?") have HIGH unknown entropy (0.85+)
            # even though they do not improve factual RAG accuracy!
            if is_distractor:
                ig_val = float(rng.uniform(0.75, 0.95))
            else:
                ig_val = float((1.0 - prompt_clear * 0.7) * rng.uniform(0.45, 0.85))
            question_ig[q_idx] = ig_val

            # Counterfactual Value of Information VOI(q):
            # Asking a question ONLY reduces regret when:
            # (a) The prompt is genuinely ambiguous (prompt_clear == 0), AND
            # (b) The database actually contains evidence to answer once clarified (evidence_sufficient == 1), AND
            # (c) The question targets the facet where user intent and retrieved passages diverge (`target_facet == dominant_facet`).
            if not is_distractor and prompt_clear < 0.5 and evidence_sufficient > 0.5:
                facet_weight = float(latent_intent[target_facet])
                discrim_power = float(passage_facet_dispersion[target_facet]) * 2.5
                regret_reduction = current_regret * np.clip(0.55 * (facet_weight * 2.2 + discrim_power), 0.05, 0.92)
            elif not is_distractor and prompt_clear < 0.5 and evidence_sufficient < 0.5:
                # If evidence is missing/unresolvable in the database, asking the user does very little to fix the missing docs
                regret_reduction = current_regret * float(latent_intent[target_facet]) * 0.12
            else:
                # Either prompt is already clear, or question is a surface distractor -> near-zero regret reduction
                regret_reduction = float(rng.uniform(0.0, 0.015))

            net_voi = regret_reduction - self.question_cost_lambda * inconv_cost
            question_voi[q_idx] = float(net_voi)

            q_emb = base_topic_emb * 0.5 + rng.normal(0, 0.5, size=D_emb).astype(np.float32)
            q_emb /= np.linalg.norm(q_emb) + 1e-8
            facet_oh = np.zeros(8, dtype=np.float32)
            facet_oh[target_facet] = 1.0 if not is_distractor else 0.0
            q_stats = np.concatenate([
                facet_oh,  # (8,) which semantic facet this question probes (0 for pure formatting distractor)
                np.array([1.0 if is_distractor else 0.0, inconv_cost], dtype=np.float32),
            ])
            question_vecs[q_idx] = np.concatenate([q_emb, q_stats])

        # 5. Construct Candidate Search Query Rewrites (S=4, 72-dim each) and Compute Rewrite VOI
        rewrite_vecs = np.zeros((S, D_emb + REWRITE_STAT_DIM), dtype=np.float32)
        rewrite_voi = np.zeros(S, dtype=np.float32)
        target_best_rewrite = int(rng.integers(0, S))

        for s_idx in range(S):
            is_optimal_hop = s_idx == target_best_rewrite
            hop_match = rng.uniform(0.75, 0.96) if is_optimal_hop else rng.uniform(0.20, 0.55)
            authority_boost = rng.uniform(0.70, 0.95) if is_optimal_hop else rng.uniform(0.25, 0.60)

            if quadrant == 2:
                # Clear prompt, missing or conflicting evidence -> Searching with the right hop rewrite resolves the gap!
                regret_red = current_regret * (0.82 if is_optimal_hop else rng.uniform(0.25, 0.50))
            elif quadrant == 1:
                # Prompt is ambiguous -> Searching more without clarifying the user's intent yields very small gain
                regret_red = current_regret * rng.uniform(0.02, 0.10)
            elif quadrant == 0:
                # Already have sufficient evidence -> Searching is redundant
                regret_red = rng.uniform(0.0, 0.01)
            else:
                # Quadrant 3: Corpus gap is unresolvable (e.g. non-existent fact / out-of-scope) -> search fails to verify
                regret_red = rng.uniform(0.01, 0.06)

            rewrite_voi[s_idx] = float(regret_red - self.search_cost_gamma)
            s_emb = base_topic_emb * 0.6 + rng.normal(0, 0.4, size=D_emb).astype(np.float32)
            s_emb /= np.linalg.norm(s_emb) + 1e-8
            s_stats = np.array([
                hop_match,
                authority_boost,
                1.0 - float(np.max(hop_cov)),
                float(np.max(contra_scores)),
                float(s_idx) / S,
                self.search_cost_gamma,
                1.0 if is_optimal_hop else 0.0,
                rng.uniform(0.2, 0.8),
            ], dtype=np.float32)
            rewrite_vecs[s_idx] = np.concatenate([s_emb, s_stats])

        # 6. Compute Macro Action Values [ANSWER, ASK, SEARCH, ABSTAIN] (relative to stopping immediately)
        best_q_idx = int(np.argmax(question_voi))
        best_s_idx = int(np.argmax(rewrite_voi))
        is_unsupported = float(evidence_sufficient < 0.5)

        val_answer = -0.06 if is_unsupported > 0.5 else 0.0
        val_ask = float(question_voi[best_q_idx])
        val_search = float(rewrite_voi[best_s_idx])
        val_abstain = float(current_regret - self.abstain_regret)

        macro_action_values = np.array([val_answer, val_ask, val_search, val_abstain], dtype=np.float32)
        optimal_macro_action = int(np.argmax(macro_action_values))

        # 7. Flat Summary Vector (164-dim) for MLP Policy (Model C) & Heuristic Baselines
        passage_mean = np.mean(passage_vecs[:, D_emb:], axis=0)  # (16,)
        passage_std = np.std(passage_vecs[:, D_emb:], axis=0)    # (16,)
        passage_max = np.max(passage_vecs[:, D_emb:], axis=0)    # (16,)
        q_stat_mean = np.mean(question_vecs[:, D_emb:], axis=0)  # (10,)
        s_stat_mean = np.mean(rewrite_vecs[:, D_emb:], axis=0)   # (8,)
        emb_interaction = query_emb[:22] * np.mean(passage_vecs[:, :22], axis=0)  # (22,)

        flat_summary = np.concatenate([
            query_vec,          # (76,)
            passage_mean,       # (16,)
            passage_std,        # (16,)
            passage_max,        # (16,)
            q_stat_mean,        # (10,)
            s_stat_mean,        # (8,)
            emb_interaction,    # (22,)
        ]).astype(np.float32)   # Total = 164

        return RAGStateExample(
            query_vec=query_vec,
            passage_vecs=passage_vecs,
            passage_mask=passage_mask,
            question_vecs=question_vecs,
            rewrite_vecs=rewrite_vecs,
            flat_summary=flat_summary,
            prompt_certainty_label=float(prompt_clear),
            evidence_certainty_label=float(evidence_sufficient),
            quadrant_label=int(quadrant),
            optimal_macro_action=optimal_macro_action,
            current_answer_regret=float(current_regret),
            is_safe_to_answer=float(quadrant == 0),
            is_unsupported_evidence=is_unsupported,
            macro_action_values=macro_action_values,
            question_voi=question_voi,
            question_ig=question_ig,
            rewrite_voi=rewrite_voi,
            best_question_idx=best_q_idx,
            best_rewrite_idx=best_s_idx,
            latent_intent_target=latent_intent,
        )
