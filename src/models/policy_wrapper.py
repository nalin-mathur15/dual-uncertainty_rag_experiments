"""Unified batch evaluation wrapper for neural and baseline General RAG policies."""
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn


class GeneralRAGPolicyAdapter:
    """Wraps PyTorch models (Models C, D, E, F and Section 8 Ablation variants) into a
    fast batched inference interface for evaluating hundreds of thousands of states.
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        device: str = "cpu",
        ablation_mode: Optional[str] = None,
        question_cost_lambda: float = 0.040,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.model_name = model_name
        self.device = device
        self.ablation_mode = ablation_mode
        self.question_cost_lambda = question_cost_lambda

    @torch.no_grad()
    def predict_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        dev_batch = {k: v.to(self.device) for k, v in batch.items()}
        out = self.model(dev_batch)

        macro_scores = out["macro_action_scores"].cpu().numpy().copy()
        q_scores = out["question_scores"].cpu().numpy().copy()
        s_scores = out["rewrite_scores"].cpu().numpy().copy()

        # Ablation 4: No Question Cost (\lambda = 0) -> artificially boosts ASK score by removing cost penalty
        if self.ablation_mode == "no_question_cost":
            macro_scores[:, 1] += self.question_cost_lambda * 2.2
            q_scores += self.question_cost_lambda

        # Epistemic uncertainty penalty on ANSWER when Model F detects high ensemble variance
        if "macro_epistemic_std" in out and self.ablation_mode != "no_calibration":
            ep_std = out["macro_epistemic_std"].cpu().numpy()
            macro_scores[:, 0] -= 0.25 * ep_std[:, 0]

        if self.ablation_mode == "no_calibration" and "raw_readiness_logit" in out:
            r_logit = out["raw_readiness_logit"].cpu().numpy()
        else:
            r_logit = out["readiness_logit"].cpu().numpy()

        confidence = 1.0 / (1.0 + np.exp(-np.clip(r_logit, -15.0, 15.0)))
        prompt_cert = 1.0 / (1.0 + np.exp(-np.clip(out["prompt_cert_logit"].cpu().numpy(), -15.0, 15.0)))
        evid_cert = 1.0 / (1.0 + np.exp(-np.clip(out["evid_cert_logit"].cpu().numpy(), -15.0, 15.0)))
        pred_quadrant = np.argmax(out["quadrant_logits"].cpu().numpy(), axis=1)

        return {
            "macro_action_scores": macro_scores,
            "pred_macro_action": np.argmax(macro_scores, axis=1),
            "question_scores": q_scores,
            "rewrite_scores": s_scores,
            "prompt_cert_prob": prompt_cert,
            "evid_cert_prob": evid_cert,
            "pred_quadrant": pred_quadrant,
            "confidence": confidence,
        }
