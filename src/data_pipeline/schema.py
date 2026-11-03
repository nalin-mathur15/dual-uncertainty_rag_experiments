"""Data schemas and constants for the 2x2 Dual Uncertainty General RAG Benchmark.

Defines the four fundamental epistemic quadrants of any RAG interaction:
1. Quadrant 0 (Prompt Clear=1, Evidence Sufficient=1) -> Optimal Action: ANSWER (0)
2. Quadrant 1 (Prompt Ambiguous=0, Evidence Sufficient=1) -> Optimal Action: ASK (1)
3. Quadrant 2 (Prompt Clear=1, Evidence Missing/Conflicting=0) -> Optimal Action: SEARCH (2)
4. Quadrant 3 (Prompt Ambiguous=0, Evidence Unresolvable=0) -> Optimal Action: ABSTAIN (3)
"""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


ACTION_ANSWER = 0
ACTION_ASK = 1
ACTION_SEARCH = 2
ACTION_ABSTAIN = 3
NUM_MACRO_ACTIONS = 4

ACTION_NAMES = {
    ACTION_ANSWER: "ANSWER",
    ACTION_ASK: "ASK",
    ACTION_SEARCH: "SEARCH",
    ACTION_ABSTAIN: "ABSTAIN",
}

QUADRANT_NAMES = {
    0: "Q1: Clear Prompt + Sufficient Evidence (ANSWER)",
    1: "Q2: Ambiguous Prompt + Multi-Interpretation Evidence (ASK)",
    2: "Q3: Clear Prompt + Missing/Conflicting Evidence (SEARCH)",
    3: "Q4: Ambiguous/Unverifiable + Unresolvable Corpus (ABSTAIN)",
}

# Dimensions for dense + structured feature representations
DEFAULT_EMBED_DIM = 64          # Compressed dense semantic projection dim (from 384-d MiniLM/BGE)
QUERY_STAT_DIM = 12             # Lexical & structural ambiguity indicators (entity specificity, temporal/scope markers, etc.)
PASSAGE_STAT_DIM = 16           # Relevance, BM25, dense sim, NLI entailment, NLI contradiction, hop coverage, source authority
QUESTION_STAT_DIM = 10          # Target facet coverage, interpretation discrimination, redundancy, inconvenience cost
REWRITE_STAT_DIM = 8            # Hop expansion score, missing entity coverage, retrieval cost


@dataclass
class RawQARecord:
    """Normalized record ingested from AmbigQA, HotpotQA, MS MARCO, or SciFact."""
    record_id: str
    source_dataset: str
    clear_query: str
    ambiguous_query: str
    disambiguating_questions: List[str]
    distractor_questions: List[str]
    search_rewrites: List[str]
    gold_passages: List[str]
    distractor_passages: List[str]
    contradictory_passages: List[str]
    interpretations: List[str]


@dataclass
class RAGStateExample:
    """A single general-purpose RAG decision state with counterfactual ground-truth VOI."""
    # Input features
    query_vec: np.ndarray               # (DEFAULT_EMBED_DIM + QUERY_STAT_DIM,) = (76,)
    passage_vecs: np.ndarray            # (K, DEFAULT_EMBED_DIM + PASSAGE_STAT_DIM) = (6, 80)
    passage_mask: np.ndarray            # (K,) boolean padding mask (False = valid, True = padded)
    question_vecs: np.ndarray           # (Q, DEFAULT_EMBED_DIM + QUESTION_STAT_DIM) = (8, 74)
    rewrite_vecs: np.ndarray            # (S, DEFAULT_EMBED_DIM + REWRITE_STAT_DIM) = (4, 72)
    flat_summary: np.ndarray            # (164,) aggregated summary features for MLP / heuristic baselines

    # 2x2 Dual Uncertainty Ground-Truth Labels
    prompt_certainty_label: float       # 1.0 if query intent is unambiguous, 0.0 if ambiguous/underspecified
    evidence_certainty_label: float     # 1.0 if retrieved passages verify answer, 0.0 if missing/contradictory
    quadrant_label: int                 # 0: ANSWER, 1: ASK, 2: SEARCH, 3: ABSTAIN
    optimal_macro_action: int           # Best action in {0, 1, 2, 3} under counterfactual VOI

    # Counterfactual Regret & VOI Targets
    current_answer_regret: float        # R(s): Expected error/hallucination cost if answering immediately
    is_safe_to_answer: float            # 1.0 if R(s) <= 0.08 and evidence is verified
    is_unsupported_evidence: float      # 1.0 if answering now relies on missing or contradictory passages
    macro_action_values: np.ndarray     # (4,) net expected utility of [ANSWER, ASK, SEARCH, ABSTAIN]
    question_voi: np.ndarray            # (Q,) net VOI(q) = R(s) - E[R(s|q,r)] - \lambda C(q)
    question_ig: np.ndarray             # (Q,) generic Shannon entropy reduction (Information Gain)
    rewrite_voi: np.ndarray             # (S,) net VOI(search_s) = R(s) - E[R(s|search_s)] - \gamma C(s)
    best_question_idx: int              # Oracle clarification question index in [0, Q-1]
    best_rewrite_idx: int               # Oracle search query rewrite index in [0, S-1]
    latent_intent_target: np.ndarray    # (8,) ground-truth distribution over latent query facets/intents
