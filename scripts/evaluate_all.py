"""Step 3: Comprehensive Evaluation across All Six Models, Three Test Splits, and Six Ablations.

Evaluates:
  - In-Distribution Test Split (50,000 states: AmbigQA / HotpotQA / MS MARCO)
  - OOD Domain-Shift Split (20,000 states: SciFact / Biomedical Claim Verification)
  - Adversarial Evidence Split (20,000 states: High Contradiction & Multi-Hop Retrieval Failures)
  - Section 8 Systematic Ablations (No Attention, No Uncertainty, No VOI / Information Gain,
    No Question Cost, No Evidence Separation, No Calibration).
"""
import argparse
import os
import json
from typing import Dict, Any
import yaml
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data_pipeline.dataset import ShardedRAGDataset
from src.models import (
    HeuristicThresholdRAGBaseline,
    PromptedLLMRAGBaseline,
    GeneralRAGPolicyAdapter,
)
from src.evaluation.metrics import evaluate_policy_on_split
from src.evaluation.plotting import generate_all_publication_figures
from scripts.train import build_model


def load_adapter(ckpt_name: str, cfg: Dict, device: str, ablation_mode: str = None) -> GeneralRAGPolicyAdapter:
    ckpt_path = os.path.join(cfg["training"]["checkpoint_dir"], f"{ckpt_name}.pt")
    model = build_model(ckpt_name, cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return GeneralRAGPolicyAdapter(
        model=model,
        model_name=ckpt_name,
        device=device,
        ablation_mode=ablation_mode,
        question_cost_lambda=cfg["rag_environment"]["question_cost_lambda"],
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate all 6 systems and 6 ablations on General RAG.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap per split for quick testing")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = cfg["storage"]["data_dir"]
    b_size = cfg["training"]["batch_size"]

    split_dirs = {
        "in_distribution": os.path.join(data_dir, "test_id"),
        "domain_shift": os.path.join(data_dir, "test_domain_shift"),
        "adversarial_evidence": os.path.join(data_dir, "test_adversarial_evid"),
    }

    loaders = {}
    for s_name, s_dir in split_dirs.items():
        ds = ShardedRAGDataset(s_dir, max_samples=args.max_samples)
        loaders[s_name] = DataLoader(ds, batch_size=b_size, shuffle=False)

    policies: Dict[str, Any] = {
        "model_a_heuristic": HeuristicThresholdRAGBaseline(),
        "model_b_llm": PromptedLLMRAGBaseline(),
        "model_c_mlp": load_adapter("model_c_mlp", cfg, device),
        "model_d_set_transformer": load_adapter("model_d_set_transformer", cfg, device),
        "model_e_bayesian_voi": load_adapter("model_e_bayesian_voi", cfg, device),
        "model_f_dual_uncertainty_voi": load_adapter("model_f_dual_uncertainty_voi", cfg, device),
    }

    env_cfg = cfg["rag_environment"]
    full_results: Dict[str, Any] = {}
    main_rows = []

    for split_name, loader in loaders.items():
        print(f"\n=======================================================")
        print(f" Evaluating Split: {split_name} ({len(loader.dataset):,} RAG states)")
        print(f"=======================================================")
        full_results[split_name] = {}

        for m_name, pol in policies.items():
            print(f"  -> Evaluating {m_name}...")
            res = evaluate_policy_on_split(
                pol,
                loader,
                question_cost_lambda=env_cfg["question_cost_lambda"],
                search_cost_gamma=env_cfg["search_cost_gamma"],
                abstain_regret=env_cfg["abstain_regret"],
            )
            full_results[split_name][m_name] = res
            main_rows.append({
                "Split": split_name,
                "Model": m_name,
                "Regret": round(res["decision_regret"], 4),
                "Quad_2x2_Acc": round(res["quadrant_acc"], 3),
                "Quad_F1": round(res["quadrant_macro_f1"], 3),
                "Prompt_AUROC": round(res["prompt_uncertainty_auroc"], 3),
                "Evid_AUROC": round(res["evidence_uncertainty_auroc"], 3),
                "Q_VOI_Regret": round(res["question_voi_regret"], 4),
                "Top1_Q_Acc": round(res["top1_question_acc"], 3),
                "Top1_Q_on_Ask": round(res["top1_question_acc_on_ask"], 3),
                "Distractor_Rate": round(res["distractor_question_rate"], 3),
                "Distractor_on_Ask": round(res["distractor_rate_on_ask"], 3),
                "Top1_Search_Acc": round(res["top1_rewrite_acc"], 3),
                "Unsupported_Rate": round(res["unsupported_answer_rate"], 4),
                "ECE": round(res["ece"], 4),
                "AURC": round(res["selective_risk_aurc"], 4),
                "Coverage": round(res["answer_coverage"], 3),
            })

    # -------------------------------------------------------------------------
    # Section 8: Run Six Systematic Ablations on In-Distribution Test Split
    # -------------------------------------------------------------------------
    print("\n=======================================================")
    print(" Running Section 8 Ablation Study")
    print("=======================================================")
    ablation_policies = {
        "Full_Model_F": policies["model_f_dual_uncertainty_voi"],
        "1_No_Attention (Pooled_MLP)": policies["model_c_mlp"],
        "2_No_Uncertainty (Direct_SetTrans)": load_adapter("ablation_no_uncertainty", cfg, device),
        "3_No_VOI (Information_Gain)": load_adapter("ablation_no_voi_ig", cfg, device),
        "4_No_Question_Cost (Lambda=0)": load_adapter(
            "model_f_dual_uncertainty_voi", cfg, device, ablation_mode="no_question_cost"
        ),
        "5_No_Evidence_Separation": load_adapter("ablation_no_evidence_sep", cfg, device),
        "6_No_Calibration (Raw_Confidence)": load_adapter(
            "model_f_dual_uncertainty_voi", cfg, device, ablation_mode="no_calibration"
        ),
    }

    ablation_rows = []
    id_loader = loaders["in_distribution"]
    adv_loader = loaders["adversarial_evidence"]
    for ab_name, ab_pol in ablation_policies.items():
        print(f"  -> Ablation: {ab_name}...")
        res_id = evaluate_policy_on_split(
            ab_pol,
            id_loader,
            question_cost_lambda=env_cfg["question_cost_lambda"],
            search_cost_gamma=env_cfg["search_cost_gamma"],
            abstain_regret=env_cfg["abstain_regret"],
        )
        res_adv = evaluate_policy_on_split(
            ab_pol,
            adv_loader,
            question_cost_lambda=env_cfg["question_cost_lambda"],
            search_cost_gamma=env_cfg["search_cost_gamma"],
            abstain_regret=env_cfg["abstain_regret"],
        )
        ablation_rows.append({
            "Ablation": ab_name,
            "ID_Regret": round(res_id["decision_regret"], 4),
            "Adv_Regret": round(res_adv["decision_regret"], 4),
            "ID_Top1_Q": round(res_id["top1_question_acc"], 3),
            "ID_Distractor_Rate": round(res_id["distractor_question_rate"], 3),
            "ID_Distractor_on_Ask": round(res_id["distractor_rate_on_ask"], 3),
            "ID_Avg_Questions": round(res_id["avg_questions_asked"], 2),
            "Adv_Unsupported_Rate": round(res_adv["unsupported_answer_rate"], 4),
            "ID_ECE": round(res_id["ece"], 4),
            "ID_AURC": round(res_id["selective_risk_aurc"], 4),
        })

    full_results["ablations"] = ablation_rows

    res_dir = cfg["training"]["results_dir"]
    fig_dir = cfg["training"]["figures_dir"]
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    df_main = pd.DataFrame(main_rows)
    df_abl = pd.DataFrame(ablation_rows)
    df_main.to_csv(os.path.join(res_dir, "main_comparison_table.csv"), index=False)
    df_abl.to_csv(os.path.join(res_dir, "ablation_study_table.csv"), index=False)
    with open(os.path.join(res_dir, "all_evaluation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    generate_all_publication_figures(full_results, fig_dir)

    print("\n=== MAIN EXPERIMENTAL RESULTS (GENERAL RAG) ===")
    print(df_main.to_string(index=False))
    print("\n=== SECTION 8 ABLATION STUDY ===")
    print(df_abl.to_string(index=False))
    print(f"\nSaved tables to {res_dir}/ and figures to {fig_dir}/")


if __name__ == "__main__":
    main()
