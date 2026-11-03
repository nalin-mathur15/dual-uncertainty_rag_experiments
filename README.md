# Knowing When to Ask, Search, Answer, or Abstain: Decoupling Prompt Ambiguity from Evidence Uncertainty in General-Purpose RAG

This repository contains the large-scale (500,000 to 1,000,000+ state) benchmark pipeline, neural architectures, training scripts, evaluation suite, and ablation studies for **General-Purpose Interactive RAG under Dual Uncertainty**.

---

## 1. The $2 \times 2$ Dual-Uncertainty Formulation for General RAG

Every RAG query $q$ paired with a retrieved document set $D = \{d_1, \dots, d_K\}$ falls into one of four fundamental epistemic states:

| | **Database Evidence is Sufficient ($E=1$)** | **Database Evidence is Missing / Contradictory ($E=0$)** |
| :--- | :--- | :--- |
| **User Prompt is Clear ($P=1$)** | **Quadrant 0 $\rightarrow$ `ANSWER`**<br>Retrieved passages directly entail a verified answer. | **Quadrant 2 $\rightarrow$ `SEARCH`**<br>Missing multi-hop bridge or NLI contradiction across passages. Execute optimal query rewrite $s^* = \arg\max_s \operatorname{VOI}(s)$. |
| **User Prompt is Ambiguous ($P=0$)** | **Quadrant 1 $\rightarrow$ `ASK`**<br>Passages cover multiple competing interpretations. Ask the clarifying question $q^* = \arg\max_q \operatorname{VOI}(q)$ that discriminates between competing passages. | **Quadrant 3 $\rightarrow$ `ABSTAIN`**<br>Underspecified query and unresolvable corpus gap. Refuse to hallucinate. |

---

## 2. Project Directory Structure

```text
general_dual_uncertainty_rag/
├── configs/
│   └── default.yaml                             # Configures 500k–1M+ states, costs (\lambda, \gamma), and training
├── src/
│   ├── data_pipeline/
│   │   ├── schema.py                            # 2x2 Quadrant definitions, tensor dimensions, RAGStateExample
│   │   ├── hf_ingestor.py                       # Streams AmbigQA, HotpotQA, MS MARCO, and SciFact (OOD)
│   │   ├── perturbation_engine.py               # 2x2x2 Counterfactual Perturbation Engine & exact VOI vs. IG
│   │   └── dataset.py                           # Zero-copy memory-mapped (.npy) ShardedRAGDataset for 1M+ rows
│   ├── models/
│   │   ├── baselines.py                         # Model A (Retrieval Heuristic Gate) & Model B (Prompted LLM Judge)
│   │   ├── mlp_policy.py                        # Model C (Pooled Feedforward MLP Policy)
│   │   ├── set_transformer.py                   # Model D (Cross-Passage Set Transformer)
│   │   ├── bayesian_intent.py                   # Model E (Bayesian Latent Intent Posterior + Analytical VOI)
│   │   ├── uncertainty_voi.py                   # Model F (Dual-Uncertainty Set Transformer + Ensemble + Calibrated VOI)
│   │   └── policy_wrapper.py                    # Fast batched evaluation wrapper & Section 8 ablation switches
│   └── evaluation/
│       ├── metrics.py                           # Quadrant F1, Prompt/Evid AUROC, Regret, VOI Regret, ECE, AURC
│       └── plotting.py                          # Generates 4 publication-grade figures (Regret, 2x2 Heatmap, ECE, OOD)
├── scripts/
│   ├── build_large_scale_rag_data.py            # Step 1: Builds 595,000+ (or 1.19M with --scale 2.0) mmap states
│   ├── train.py                                 # Step 2: Trains Models C, D, E, F & ablations with AMP + Temp Scaling
│   ├── evaluate_all.py                          # Step 3: Evaluates all 6 models & 6 ablations across ID, SciFact, Adversarial
│   └── run_cluster_pipeline.sh                  # SLURM batch script for university GPU cluster
├── notebooks/
│   ├── 01_dual_uncertainty_rag_data_explorer.ipynb
│   └── 02_results_and_ablation_analysis.ipynb
├── requirements.txt
└── setup.py
```

---

## 3. Running on Your University GPU Cluster

### Installation
```bash
cd general_dual_uncertainty_rag
pip install -r requirements.txt
pip install -e .
```

### Option A: One-Command SLURM Submission
```bash
sbatch scripts/run_cluster_pipeline.sh
```

### Option B: Step-by-Step Execution Order

1. **Step 1 — Build the Large-Scale Dataset (`595,000` states at `--scale 1.0`, or `1,190,000` states at `--scale 2.0`):**
   ```bash
   python scripts/build_large_scale_rag_data.py --config configs/default.yaml --scale 1.0
   ```
   *(Tip: Use `--scale 0.01 --offline` for a 15-second sanity check before launching the full cluster run).*

2. **Step 2 — Train All Neural Architectures & Ablation Models:**
   ```bash
   python scripts/train.py --config configs/default.yaml --model all --device cuda
   ```

3. **Step 3 — Evaluate All 6 Models & 6 Ablations Across All Splits:**
   ```bash
   python scripts/evaluate_all.py --config configs/default.yaml --device cuda
   ```
