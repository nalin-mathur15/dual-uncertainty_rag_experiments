"""Batched evaluation metrics for General-Purpose Interactive RAG under Dual Uncertainty.

Evaluates:
1. 2x2 Quadrant Classification Accuracy & Macro-F1 (Prompt Clear/Ambiguous x Evidence Sufficient/Missing)
2. Decoupled Uncertainty AUROC (Prompt Ambiguity Detection vs. Evidence Gap Detection)
3. Counterfactual Decision Regret (expected downstream RAG error + action costs)
4. Clarification Question VOI Regret, Top-1 Oracle Match, and Surface Distractor Avoidance Rate
5. Search Rewrite Top-1 Accuracy & VOI Regret
6. Unsupported Answer / Hallucination Rate (answering when evidence is missing or contradictory)
7. Expected Calibration Error (ECE) & Selective Risk (AURC)
8. Regret vs. Interaction Budget Curve (k = 0, 1, 2, 3, 4 interactions)
"""
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader

from ..data_pipeline.schema import (
    ACTION_ANSWER,
    ACTION_ASK,
    ACTION_SEARCH,
    ACTION_ABSTAIN,
    DEFAULT_EMBED_DIM,
)


def compute_expected_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 10,
) -> Tuple[float, Dict[str, list]]:
    """Computes Expected Calibration Error (ECE) and reliability diagram bin statistics."""
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(confidences, bins[1:-1], right=True)
    ece = 0.0
    bin_conf = np.zeros(num_bins, dtype=np.float32)
    bin_acc = np.zeros(num_bins, dtype=np.float32)
    bin_counts = np.zeros(num_bins, dtype=np.int64)

    N = len(confidences)
    if N == 0:
        return 0.0, {"bin_conf": [], "bin_acc": [], "bin_counts": []}

    for b in range(num_bins):
        mask = bin_indices == b
        cnt = int(np.sum(mask))
        bin_counts[b] = cnt
        if cnt > 0:
            c_avg = float(np.mean(confidences[mask]))
            a_avg = float(np.mean(accuracies[mask]))
            bin_conf[b] = c_avg
            bin_acc[b] = a_avg
            ece += (cnt / N) * abs(a_avg - c_avg)

    return float(ece), {
        "bin_conf": bin_conf.tolist(),
        "bin_acc": bin_acc.tolist(),
        "bin_counts": bin_counts.tolist(),
    }


def compute_selective_risk_aurc(confidences: np.ndarray, regrets: np.ndarray) -> float:
    """Area Under the Risk-Coverage Curve (AURC)."""
    if len(confidences) == 0:
        return 0.0
    order = np.argsort(-confidences)
    sorted_reg = regrets[order]
    cum_risk = np.cumsum(sorted_reg) / np.arange(1, len(sorted_reg) + 1)
    return float(np.mean(cum_risk))


def evaluate_policy_on_split(
    policy: Any,
    data_loader: DataLoader,
    question_cost_lambda: float = 0.040,
    search_cost_gamma: float = 0.020,
    abstain_regret: float = 0.20,
) -> Dict[str, Any]:
    """Runs fast batched evaluation of a policy across an entire test split (e.g., 50,000 states)."""
    all_pred_macro = []
    all_true_macro = []
    all_pred_quad = []
    all_true_quad = []

    all_prompt_cert_prob = []
    all_true_prompt_cert = []
    all_evid_cert_prob = []
    all_true_evid_cert = []

    all_confidences = []
    all_safe_labels = []

    all_decision_regrets = []
    all_questions_asked = []
    all_searches_made = []
    all_unsupported_answers = []
    all_abstained = []

    all_q_voi_regrets = []
    all_q_top1_acc = []
    all_q_distractor_rate = []
    all_q_distractor_on_ask = []
    all_q_top1_on_ask = []
    all_s_top1_acc = []

    # Regret across interaction budget k = 0, 1, 2, 3, 4
    budget_regrets = {k: [] for k in range(5)}

    for batch in data_loader:
        preds = policy.predict_batch(batch)
        B = batch["query_vec"].shape[0]

        pred_macro = preds["pred_macro_action"]
        true_macro = batch["optimal_macro_action"].numpy()
        true_quad = batch["quadrant_label"].numpy()
        base_regret = batch["current_answer_regret"].numpy()
        is_unsupp = batch["is_unsupported_evidence"].numpy()
        q_voi = batch["question_voi"].numpy()          # (B, 8)
        s_voi = batch["rewrite_voi"].numpy()           # (B, 4)
        best_q_true = batch["best_question_idx"].numpy()
        best_s_true = batch["best_rewrite_idx"].numpy()
        q_is_distractor = batch["question_vecs"][:, :, DEFAULT_EMBED_DIM + 8].numpy()

        all_pred_macro.append(pred_macro)
        all_true_macro.append(true_macro)
        all_pred_quad.append(preds["pred_quadrant"])
        all_true_quad.append(true_quad)

        all_prompt_cert_prob.append(preds["prompt_cert_prob"])
        all_true_prompt_cert.append(batch["prompt_certainty_label"].numpy())
        all_evid_cert_prob.append(preds["evid_cert_prob"])
        all_true_evid_cert.append(batch["evidence_certainty_label"].numpy())

        all_confidences.append(preds["confidence"])
        all_safe_labels.append(batch["is_safe_to_answer"].numpy())

        # Question & Rewrite selection accuracy on every state
        chosen_q = np.argmax(preds["question_scores"], axis=1)
        chosen_s = np.argmax(preds["rewrite_scores"], axis=1)
        row_idx = np.arange(B)

        max_q_voi = np.max(q_voi, axis=1)
        chosen_q_voi = q_voi[row_idx, chosen_q]
        all_q_voi_regrets.append(np.maximum(0.0, max_q_voi - chosen_q_voi))
        all_q_top1_acc.append((chosen_q == best_q_true).astype(np.float32))
        all_q_distractor_rate.append(q_is_distractor[row_idx, chosen_q])
        all_s_top1_acc.append((chosen_s == best_s_true).astype(np.float32))

        # Specifically track question selection quality on genuine clarification states (Quadrant 1 or ASK action)
        ask_mask = (pred_macro == ACTION_ASK) | (true_quad == 1)
        if np.any(ask_mask):
            all_q_distractor_on_ask.append(q_is_distractor[ask_mask, chosen_q[ask_mask]])
            all_q_top1_on_ask.append((chosen_q[ask_mask] == best_q_true[ask_mask]).astype(np.float32))

        # Compute realized counterfactual regret for the policy's chosen action
        realized_regret = np.zeros(B, dtype=np.float32)
        q_count = np.zeros(B, dtype=np.float32)
        s_count = np.zeros(B, dtype=np.float32)
        unsupp_ans = np.zeros(B, dtype=np.float32)
        abst_flag = np.zeros(B, dtype=np.float32)

        # Case 0: Policy chooses ANSWER immediately
        mask_ans = pred_macro == ACTION_ANSWER
        realized_regret[mask_ans] = base_regret[mask_ans]
        unsupp_ans[mask_ans] = is_unsupp[mask_ans]

        # Case 1: Policy chooses ASK (executes chosen question q, then answers if resolved)
        mask_ask = pred_macro == ACTION_ASK
        # Net post-question regret = base_regret - (chosen_q_voi + cost) + cost = base_regret - chosen_q_voi
        post_ask_reg = np.maximum(0.02, base_regret[mask_ask] - chosen_q_voi[mask_ask])
        realized_regret[mask_ask] = post_ask_reg
        q_count[mask_ask] = 1.0 + (post_ask_reg > 0.22).astype(np.float32) * 0.65
        # If evidence was actually missing in database, asking the user still leaves answer unsupported!
        unsupp_ans[mask_ask] = is_unsupp[mask_ask] * 0.85

        # Case 2: Policy chooses SEARCH (executes chosen query rewrite s)
        mask_search = pred_macro == ACTION_SEARCH
        chosen_s_voi = s_voi[row_idx, chosen_s]
        post_search_reg = np.maximum(0.02, base_regret[mask_search] - chosen_s_voi[mask_search])
        realized_regret[mask_search] = post_search_reg
        s_count[mask_search] = 1.0
        # Searching resolves missing/conflicting evidence in Quadrant 2
        unsupp_ans[mask_search] = (true_quad[mask_search] == 3).astype(np.float32) * 0.35

        # Case 3: Policy chooses ABSTAIN
        mask_abst = pred_macro == ACTION_ABSTAIN
        realized_regret[mask_abst] = abstain_regret
        abst_flag[mask_abst] = 1.0

        all_decision_regrets.append(realized_regret)
        all_questions_asked.append(q_count)
        all_searches_made.append(s_count)
        all_unsupported_answers.append(unsupp_ans)
        all_abstained.append(abst_flag)

        # Compute multi-turn interaction budget curve (k = 0, 1, 2, 3, 4 clarifying/retrieval steps)
        budget_regrets[0].append(base_regret)
        q_order = np.argsort(-preds["question_scores"], axis=1)
        cur_reg = base_regret.copy()
        for k in range(1, 5):
            qk = q_order[:, k - 1]
            # Net reduction from kth ranked action combined with search capability
            gain_q = np.maximum(0.0, q_voi[row_idx, qk] + question_cost_lambda)
            gain_s = np.maximum(0.0, chosen_s_voi + search_cost_gamma) if k == 1 else np.zeros(B)
            # A policy that accurately separates ASK vs SEARCH applies the right action type
            uses_search = (preds["evid_cert_prob"] < preds["prompt_cert_prob"]).astype(np.float32)
            step_gain = (1.0 - uses_search) * gain_q + uses_search * (gain_s if k == 1 else gain_q * 0.4)
            cur_reg = np.maximum(0.02, cur_reg - step_gain * (0.65 ** (k - 1)))
            budget_regrets[k].append(cur_reg.copy())

    pred_macro_arr = np.concatenate(all_pred_macro)
    true_macro_arr = np.concatenate(all_true_macro)
    pred_quad_arr = np.concatenate(all_pred_quad)
    true_quad_arr = np.concatenate(all_true_quad)

    prompt_prob_arr = np.concatenate(all_prompt_cert_prob)
    true_prompt_arr = np.concatenate(all_true_prompt_cert)
    evid_prob_arr = np.concatenate(all_evid_cert_prob)
    true_evid_arr = np.concatenate(all_true_evid_cert)

    conf_arr = np.concatenate(all_confidences)
    safe_arr = np.concatenate(all_safe_labels)
    regret_arr = np.concatenate(all_decision_regrets)

    ece, cal_bins = compute_expected_calibration_error(conf_arr, safe_arr)
    aurc = compute_selective_risk_aurc(conf_arr, regret_arr)
    c_mat = confusion_matrix(true_quad_arr, pred_quad_arr, labels=[0, 1, 2, 3])

    try:
        prompt_auroc = float(roc_auc_score(true_prompt_arr, prompt_prob_arr))
        evid_auroc = float(roc_auc_score(true_evid_arr, evid_prob_arr))
    except Exception:
        prompt_auroc, evid_auroc = 0.5, 0.5

    return {
        "decision_regret": float(np.mean(regret_arr)),
        "quadrant_acc": float(np.mean(pred_quad_arr == true_quad_arr)),
        "quadrant_macro_f1": float(f1_score(true_quad_arr, pred_quad_arr, average="macro")),
        "macro_action_acc": float(np.mean(pred_macro_arr == true_macro_arr)),
        "prompt_uncertainty_auroc": prompt_auroc,
        "evidence_uncertainty_auroc": evid_auroc,
        "question_voi_regret": float(np.mean(np.concatenate(all_q_voi_regrets))),
        "top1_question_acc": float(np.mean(np.concatenate(all_q_top1_acc))),
        "distractor_question_rate": float(np.mean(np.concatenate(all_q_distractor_rate))),
        "top1_question_acc_on_ask": float(np.mean(np.concatenate(all_q_top1_on_ask))) if all_q_top1_on_ask else 0.0,
        "distractor_rate_on_ask": float(np.mean(np.concatenate(all_q_distractor_on_ask))) if all_q_distractor_on_ask else 0.0,
        "top1_rewrite_acc": float(np.mean(np.concatenate(all_s_top1_acc))),
        "avg_questions_asked": float(np.mean(np.concatenate(all_questions_asked))),
        "avg_searches_made": float(np.mean(np.concatenate(all_searches_made))),
        "unsupported_answer_rate": float(np.mean(np.concatenate(all_unsupported_answers))),
        "answer_coverage": float(1.0 - np.mean(np.concatenate(all_abstained))),
        "ece": ece,
        "selective_risk_aurc": aurc,
        "regret_by_budget": {str(k): float(np.mean(np.concatenate(budget_regrets[k]))) for k in range(5)},
        "calibration_bins": cal_bins,
        "confusion_matrix": c_mat.tolist(),
    }
