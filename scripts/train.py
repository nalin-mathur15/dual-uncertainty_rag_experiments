"""Step 2: Train Neural Architectures (Models C, D, E, F) and Section 8 Ablation Variants.

Supports multi-worker training over 500,000 to 1,000,000+ memory-mapped RAG states
with automatic mixed precision (AMP) on CUDA GPUs and post-hoc validation temperature scaling.
"""
import argparse
import os
import json
from typing import Dict
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data_pipeline.dataset import ShardedRAGDataset
from src.models import (
    MLPGeneralRAGNet,
    SetTransformerGeneralRAGNet,
    BayesianLatentIntentRAGNet,
    DualUncertaintySetTransformerRAGNet,
)


MODEL_REGISTRY = [
    "model_c_mlp",
    "model_d_set_transformer",
    "model_e_bayesian_voi",
    "model_f_dual_uncertainty_voi",
    # Section 8 Ablation variants:
    "ablation_no_voi_ig",             # Trained to select questions by Information Gain (entropy) instead of VOI
    "ablation_no_evidence_sep",       # Model F without separate database-evidence uncertainty branch
    "ablation_no_uncertainty",        # Set Transformer without Bayesian posterior / Deep Ensemble uncertainty
]


def build_model(model_name: str, cfg: Dict) -> nn.Module:
    t_cfg = cfg["training"]
    h_dim = t_cfg["hidden_dim"]
    n_heads = t_cfg["num_heads"]
    n_layers = t_cfg["num_layers"]
    dropout = t_cfg["dropout"]
    ens_size = t_cfg["ensemble_size"]

    if model_name == "model_c_mlp":
        return MLPGeneralRAGNet(hidden_dim=h_dim, dropout=dropout)
    if model_name in ("model_d_set_transformer", "ablation_no_uncertainty"):
        return SetTransformerGeneralRAGNet(hidden_dim=h_dim, num_heads=n_heads, num_layers=n_layers, dropout=dropout)
    if model_name == "model_e_bayesian_voi":
        return BayesianLatentIntentRAGNet(hidden_dim=h_dim, dropout=dropout)
    if model_name in ("model_f_dual_uncertainty_voi", "ablation_no_voi_ig"):
        return DualUncertaintySetTransformerRAGNet(
            hidden_dim=h_dim,
            num_heads=n_heads,
            num_layers=n_layers,
            ensemble_size=ens_size,
            dropout=dropout,
            separate_evidence_branch=True,
        )
    if model_name == "ablation_no_evidence_sep":
        return DualUncertaintySetTransformerRAGNet(
            hidden_dim=h_dim,
            num_heads=n_heads,
            num_layers=n_layers,
            ensemble_size=ens_size,
            dropout=dropout,
            separate_evidence_branch=False,
        )
    raise ValueError(f"Unknown model_name: {model_name}")


def compute_loss(model_name: str, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    target_macro_vals = batch["macro_action_values"]
    target_q_voi = batch["question_voi"].clone()
    target_s_voi = batch["rewrite_voi"]

    # In the "No VOI (Information Gain)" ablation, supervise question scoring with Shannon Information Gain
    if model_name == "ablation_no_voi_ig":
        target_q_voi = batch["question_ig"] * 0.25 - 0.04
    else:
        # Distractor Suppression: Clamp distractor question target VOI (indices 6 & 7)
        # so they are strictly lower than all valid semantic candidate questions across all states
        min_semantic = target_q_voi[:, :6].min(dim=-1, keepdim=True)[0]
        target_q_voi[:, 6:] = torch.minimum(target_q_voi[:, 6:], min_semantic - 0.05)

    # 1. Macro Action VOI Regression + Cross-Entropy
    loss_macro_reg = F.smooth_l1_loss(out["macro_action_scores"], target_macro_vals)
    loss_macro_ce = F.cross_entropy(out["macro_action_scores"] * 8.0, batch["optimal_macro_action"])

    # 2. Candidate Question VOI & Search Rewrite VOI Regression
    loss_q_reg = F.smooth_l1_loss(out["question_scores"], target_q_voi)
    loss_s_reg = F.smooth_l1_loss(out["rewrite_scores"], target_s_voi)
    loss_s_ce = F.cross_entropy(out["rewrite_scores"] * 10.0, batch["best_rewrite_idx"])

    # 3. Targeted Clarification Question Discrimination
    # Identify ASK states (Quadrant 1) where choosing the right semantic question matters
    is_ask_state = (batch["quadrant_label"] == 1).float()  # (B,)
    q_scores = out["question_scores"]                       # (B, Q)

    # Cross-entropy on best question: ONLY on actual ASK states where a clarification question is warranted!
    # (Avoids supervising non-ASK states where distractor questions had higher negative VOI due to lower inconvenience cost)
    loss_q_ce_all = F.cross_entropy(q_scores * 12.0, batch["best_question_idx"], reduction="none")
    loss_q_ce = (loss_q_ce_all * is_ask_state).sum() / (is_ask_state.sum() + 1e-8)

    # Distractor Suppression: Extract distractor indicator at index 72 in question_vecs
    # (DEFAULT_EMBED_DIM=64 + 8 = 72)
    is_distractor = batch["question_vecs"][:, :, 72]        # (B, Q), 1.0 for distractors (indices 6, 7)

    # Direct penalty if distractor questions are assigned positive scores
    distractor_penalty = F.relu(q_scores) * is_distractor
    loss_q_distractor = distractor_penalty.sum(dim=-1).mean()

    # Universal Margin Ranking: Semantic questions (0..5) must outscore distractor questions (6..7) by at least 0.20
    semantic_scores = q_scores[:, :6]
    max_semantic, _ = torch.max(semantic_scores, dim=-1, keepdim=True)  # (B, 1)
    distractor_margin_violations = F.relu(q_scores + 0.20 - max_semantic) * is_distractor
    loss_q_margin = distractor_margin_violations.sum(dim=-1).mean()

    loss = (
        2.5 * loss_macro_reg
        + 0.5 * loss_macro_ce
        + 2.5 * loss_q_reg
        + 2.0 * loss_q_ce
        + 2.0 * loss_q_distractor
        + 2.0 * loss_q_margin
        + 1.5 * loss_s_reg
        + 0.4 * loss_s_ce
    )

    # 3. 2x2 Epistemic Quadrant & Decoupled Prompt/Evidence Uncertainty Supervision
    loss_quad = F.cross_entropy(out["quadrant_logits"], batch["quadrant_label"])
    loss_p_cert = F.binary_cross_entropy_with_logits(out["prompt_cert_logit"], batch["prompt_certainty_label"])
    loss_e_cert = F.binary_cross_entropy_with_logits(out["evid_cert_logit"], batch["evidence_certainty_label"])
    loss_ready = F.binary_cross_entropy_with_logits(out["readiness_logit"], batch["is_safe_to_answer"])
    loss_regret = F.smooth_l1_loss(out["est_regret"], batch["current_answer_regret"])

    loss = loss + 0.5 * loss_quad + 0.5 * loss_p_cert + 0.5 * loss_e_cert + 0.5 * loss_ready + 1.0 * loss_regret

    # 4. Latent Intent Posterior Gaussian NLL (for Bayesian Model E & Dual-Uncertainty Model F)
    if "intent_mean" in out:
        i_mean = out["intent_mean"]
        i_std = out["intent_std"]
        true_intent = batch["latent_intent_target"]
        nll_intent = 0.5 * (((true_intent - i_mean) / i_std) ** 2 + 2.0 * torch.log(i_std)).mean()
        loss = loss + 0.4 * nll_intent

    # 5. Deep Ensemble Head Supervision (for Model F)
    if "ens_macro_stack" in out:
        K_ens = out["ens_macro_stack"].shape[0]
        ens_loss = 0.0
        for k in range(K_ens):
            ens_loss = (
                ens_loss
                + F.smooth_l1_loss(out["ens_macro_stack"][k], target_macro_vals)
                + F.smooth_l1_loss(out["ens_q_stack"][k], target_q_voi)
            )
        loss = loss + (ens_loss / K_ens) * 1.5

    return loss


def calibrate_temperature(model: nn.Module, val_loader: DataLoader, device: str) -> None:
    """Post-hoc temperature scaling on the validation split for calibrated selective readiness."""
    if not hasattr(model, "temperature"):
        return
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            dev_batch = {k: v.to(device) for k, v in batch.items()}
            out = model(dev_batch)
            logits_list.append(out["raw_readiness_logit"])
            labels_list.append(dev_batch["is_safe_to_answer"])

    logits_t = torch.cat(logits_list, dim=0)
    labels_t = torch.cat(labels_list, dim=0)
    opt = torch.optim.LBFGS([model.temperature], lr=0.05, max_iter=40)

    def _closure():
        opt.zero_grad()
        t = torch.clamp(model.temperature, min=0.2, max=5.0)
        l = F.binary_cross_entropy_with_logits(logits_t / t, labels_t)
        l.backward()
        return l

    opt.step(_closure)
    print(f"    Calibrated validation temperature T = {float(model.temperature.item()):.3f}")


def train_single_model(
    model_name: str,
    cfg: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
) -> Dict:
    print(f"\n=== Training {model_name} on {device} ({len(train_loader.dataset):,} states) ===")
    model = build_model(model_name, cfg).to(device)
    t_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=t_cfg["learning_rate"], weight_decay=t_cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_cfg["epochs"])
    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_dir = t_cfg["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}.pt")
    best_val = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, t_cfg["epochs"] + 1):
        model.train()
        total_train = 0.0
        for batch in train_loader:
            dev_batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(dev_batch)
                loss = compute_loss(model_name, out, dev_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_train += float(loss.item()) * len(dev_batch["optimal_macro_action"])

        scheduler.step()
        avg_train = total_train / len(train_loader.dataset)

        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for batch in val_loader:
                dev_batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(dev_batch)
                    loss = compute_loss(model_name, out, dev_batch)
                total_val += float(loss.item()) * len(dev_batch["optimal_macro_action"])
        avg_val = total_val / len(val_loader.dataset)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), ckpt_path)

        print(f"  Epoch {epoch:02d}/{t_cfg['epochs']:02d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    calibrate_temperature(model, val_loader, device)
    torch.save(model.state_dict(), ckpt_path)
    return history


def main():
    parser = argparse.ArgumentParser(description="Train General RAG Dual-Uncertainty models.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default="all")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = cfg["storage"]["data_dir"]
    train_ds = ShardedRAGDataset(os.path.join(data_dir, "train"))
    val_ds = ShardedRAGDataset(os.path.join(data_dir, "val"))

    b_size = cfg["training"]["batch_size"]
    n_workers = cfg["training"].get("num_workers", 0) if device.startswith("cuda") else 0

    train_loader = DataLoader(train_ds, batch_size=b_size, shuffle=True, num_workers=n_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=b_size, shuffle=False, num_workers=n_workers, pin_memory=True)

    models_to_run = MODEL_REGISTRY if args.model == "all" else [args.model]
    histories = {}
    for m_name in models_to_run:
        histories[m_name] = train_single_model(m_name, cfg, train_loader, val_loader, device)

    res_dir = cfg["training"]["results_dir"]
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(res_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(histories, f, indent=2)
    print("\n[Train] All models trained and saved to", cfg["training"]["checkpoint_dir"])


if __name__ == "__main__":
    main()
