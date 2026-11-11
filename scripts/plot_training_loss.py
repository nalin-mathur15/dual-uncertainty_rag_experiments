"""Plots train and validation loss curves from outputs/results/training_history.json."""
import os
import json
import matplotlib.pyplot as plt

def main():
    json_path = os.path.join("outputs", "results", "training_history.json")
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), dpi=300)
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

    colors = {
        "model_c_mlp": "#3498db",
        "model_d_set_transformer": "#9b59b6",
        "model_e_bayesian_voi": "#16a085",
        "model_f_dual_uncertainty_voi": "#c0392b",
        "ablation_no_voi_ig": "#e67e22",
        "ablation_no_evidence_sep": "#f39c12",
        "ablation_no_uncertainty": "#34495e",
    }
    labels = {
        "model_c_mlp": "Model C (Pooled MLP)",
        "model_d_set_transformer": "Model D (Set Transformer)",
        "model_e_bayesian_voi": "Model E (Bayesian VOI)",
        "model_f_dual_uncertainty_voi": "Model F (Full Dual-Uncertainty)",
        "ablation_no_voi_ig": "Ablation: No VOI (Info Gain)",
        "ablation_no_evidence_sep": "Ablation: No Evid Separation",
        "ablation_no_uncertainty": "Ablation: No Uncertainty",
    }

    # Left: Core Neural Architectures (Models C, D, E, F)
    ax1 = axes[0]
    core_models = ["model_c_mlp", "model_d_set_transformer", "model_e_bayesian_voi", "model_f_dual_uncertainty_voi"]
    for m in core_models:
        if m in history:
            train_l = history[m]["train_loss"]
            val_l = history[m]["val_loss"]
            epochs = range(1, len(train_l) + 1)
            ax1.plot(epochs, train_l, label=f"{labels[m]} (Train)", color=colors[m], linewidth=1.8)
            ax1.plot(epochs, val_l, label=f"{labels[m]} (Val)", color=colors[m], linestyle="--", linewidth=1.4, alpha=0.8)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Multi-Task Loss")
    ax1.set_title("(a) Core Neural Models Training Dynamics")
    ax1.legend(fontsize=8, loc="upper right")

    # Right: Section 8 Systematic Ablations
    ax2 = axes[1]
    abl_models = ["model_f_dual_uncertainty_voi", "ablation_no_voi_ig", "ablation_no_evidence_sep", "ablation_no_uncertainty"]
    for m in abl_models:
        if m in history:
            train_l = history[m]["train_loss"]
            val_l = history[m]["val_loss"]
            epochs = range(1, len(train_l) + 1)
            ax2.plot(epochs, train_l, label=f"{labels[m]} (Train)", color=colors[m], linewidth=1.8)
            ax2.plot(epochs, val_l, label=f"{labels[m]} (Val)", color=colors[m], linestyle="--", linewidth=1.4, alpha=0.8)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Multi-Task Loss")
    ax2.set_title("(b) Section 8 Ablations Training Dynamics")
    ax2.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    out_dir = os.path.join("outputs", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig5_training_loss.png")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved training loss plot to: {out_path}")

    # Also save inside report/figures/ for LaTeX
    rep_fig_dir = os.path.join("report", "figures")
    os.makedirs(rep_fig_dir, exist_ok=True)
    plt.savefig(os.path.join(rep_fig_dir, "fig5_training_loss.png"), bbox_inches="tight")
    print(f"Saved to report/figures/fig5_training_loss.png")

if __name__ == "__main__":
    main()
