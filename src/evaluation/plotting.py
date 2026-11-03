"""Publication-ready matplotlib/seaborn figures for General-Purpose Dual-Uncertainty RAG."""
import os
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


MODEL_DISPLAY_NAMES = {
    "model_a_heuristic": "A. Retrieval Heuristic Gate",
    "model_b_llm": "B. Prompted LLM Judge",
    "model_c_mlp": "C. Pooled MLP Policy",
    "model_d_set_transformer": "D. Cross-Passage Set Transformer",
    "model_e_bayesian_voi": "E. Bayesian Intent + VOI",
    "model_f_dual_uncertainty_voi": "F. Dual-Uncertainty SetTrans + VOI",
}

MODEL_COLORS = {
    "model_a_heuristic": "#7f8c8d",
    "model_b_llm": "#e67e22",
    "model_c_mlp": "#3498db",
    "model_d_set_transformer": "#9b59b6",
    "model_e_bayesian_voi": "#16a085",
    "model_f_dual_uncertainty_voi": "#c0392b",
}


def generate_all_publication_figures(results: Dict[str, Any], output_dir: str) -> None:
    """Generates and saves the 4 core thesis plots to `output_dir`."""
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

    id_res = results.get("in_distribution", {})
    if not id_res:
        return

    # -------------------------------------------------------------------------
    # Figure 1: Main Research Graph — RAG Decision Regret vs. Interaction Budget
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    budgets = [0, 1, 2, 3, 4]
    for m_key, m_data in id_res.items():
        if m_key not in MODEL_DISPLAY_NAMES:
            continue
        curve = m_data["regret_by_budget"]
        y_vals = [curve[str(k)] * 100.0 for k in budgets]
        lw = 2.8 if m_key == "model_f_dual_uncertainty_voi" else 1.8
        marker = "o" if m_key == "model_f_dual_uncertainty_voi" else "s"
        ax.plot(
            budgets,
            y_vals,
            label=MODEL_DISPLAY_NAMES[m_key],
            color=MODEL_COLORS[m_key],
            linewidth=lw,
            marker=marker,
        )

    ax.set_xlabel("Clarification / Retrieval Interactions Allowed")
    ax.set_ylabel("Expected RAG Decision Regret (x100, Lower is Better)")
    ax.set_title("RAG Decision Regret vs. Interaction Budget (50,000 ID Test States)")
    ax.set_xticks(budgets)
    ax.legend(frameon=True, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig1_rag_regret_vs_interactions.png"), dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 2: 2x2 Quadrant Confusion Matrix (Prompted LLM vs. Model F)
    # -------------------------------------------------------------------------
    if "model_b_llm" in id_res and "model_f_dual_uncertainty_voi" in id_res:
        fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0))
        q_labels = ["ANSWER\n(Clear+Suff)", "ASK\n(Ambig+Suff)", "SEARCH\n(Clear+Gap)", "ABSTAIN\n(Unresolv)"]

        for ax, m_key, title in [
            (axes[0], "model_b_llm", "B. Prompted LLM Judge"),
            (axes[1], "model_f_dual_uncertainty_voi", "F. Dual-Uncertainty SetTrans + VOI"),
        ]:
            cm = np.array(id_res[m_key]["confusion_matrix"], dtype=np.float32)
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0) * 100.0
            sns.heatmap(
                cm_norm,
                annot=True,
                fmt=".1f",
                cmap="Blues",
                xticklabels=q_labels,
                yticklabels=q_labels,
                cbar=False,
                ax=ax,
            )
            ax.set_xlabel("Predicted RAG Action / Quadrant")
            ax.set_ylabel("True Epistemic Quadrant")
            ax.set_title(f"{title} (Row %)")

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "fig2_2x2_quadrant_confusion.png"), dpi=300)
        plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 3: Readiness Confidence Calibration / Reliability Diagram
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.8, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect Calibration")
    for m_key in ["model_b_llm", "model_c_mlp", "model_d_set_transformer", "model_f_dual_uncertainty_voi"]:
        if m_key not in id_res:
            continue
        bins = id_res[m_key]["calibration_bins"]
        conf = np.array(bins["bin_conf"])
        acc = np.array(bins["bin_acc"])
        counts = np.array(bins["bin_counts"])
        valid = counts > 0
        if np.any(valid):
            ece_val = id_res[m_key]["ece"]
            ax.plot(
                conf[valid],
                acc[valid],
                marker="o",
                linewidth=2.0,
                color=MODEL_COLORS[m_key],
                label=f"{MODEL_DISPLAY_NAMES[m_key]} (ECE={ece_val:.3f})",
            )

    ax.set_xlabel("Predicted Readiness Confidence P(Safe to Answer)")
    ax.set_ylabel("Empirical Verified Answer Rate")
    ax.set_title("Confidence Calibration / Reliability Diagram")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig3_calibration_reliability.png"), dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 4: Out-of-Distribution Robustness (ID vs. SciFact Domain Shift vs. Adversarial)
    # -------------------------------------------------------------------------
    splits = ["in_distribution", "domain_shift", "adversarial_evidence"]
    split_labels = ["In-Distribution\n(AmbigQA/Hotpot)", "OOD Domain Shift\n(SciFact/BioMed)", "Adversarial Evidence\n(Contradictions)"]
    avail_splits = [s for s in splits if s in results and results[s]]
    if len(avail_splits) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
        x = np.arange(len(avail_splits))
        width = 0.13
        m_keys = [k for k in MODEL_DISPLAY_NAMES if k in id_res]

        for idx, m_key in enumerate(m_keys):
            reg_vals = [results[s][m_key]["decision_regret"] * 100.0 for s in avail_splits]
            unsupp_vals = [results[s][m_key]["unsupported_answer_rate"] * 100.0 for s in avail_splits]
            offset = (idx - len(m_keys) / 2.0) * width + width / 2.0
            axes[0].bar(x + offset, reg_vals, width, label=MODEL_DISPLAY_NAMES[m_key], color=MODEL_COLORS[m_key])
            axes[1].bar(x + offset, unsupp_vals, width, label=MODEL_DISPLAY_NAMES[m_key], color=MODEL_COLORS[m_key])

        axes[0].set_xticks(x)
        axes[0].set_xticklabels([split_labels[splits.index(s)] for s in avail_splits])
        axes[0].set_ylabel("Decision Regret (x100)")
        axes[0].set_title("RAG Decision Regret Across Distribution Shifts")

        axes[1].set_xticks(x)
        axes[1].set_xticklabels([split_labels[splits.index(s)] for s in avail_splits])
        axes[1].set_ylabel("Unsupported / Hallucinated Answer Rate (%)")
        axes[1].set_title("Unsupported Answers Under Contradictory Evidence")
        axes[1].legend(frameon=True, fontsize=8.5)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "fig4_ood_and_adversarial_robustness.png"), dpi=300)
        plt.close(fig)
