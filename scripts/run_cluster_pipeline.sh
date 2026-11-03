#!/bin/bash
#SBATCH --job-name=general_rag_dual_unc
#SBATCH --output=outputs/slurm_%x_%j.out
#SBATCH --error=outputs/slurm_%x_%j.err
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -e

echo "=================================================================="
echo " Starting Cluster Pipeline: General-Purpose Dual-Uncertainty RAG"
echo " Host: $(hostname) | Date: $(date)"
echo "=================================================================="

source ~/rag_v100_venv/bin/activate
cd ~/grp_relRAG

mkdir -p outputs/checkpoints outputs/results outputs/figures

export PYTHONPATH="$(pwd):$PYTHONPATH"

# Scale=1.0 builds 595,000 total RAG states (500k train, 25k val, 50k ID test, 20k SciFact OOD, 20k Adversarial)
# Set --scale 2.0 if you want 1,190,000 states!
echo "[1/3] Building Large-Scale General RAG Benchmark Splits..."
python scripts/build_large_scale_rag_data.py --config configs/default.yaml --scale 1.0

echo "[2/3] Training All Neural Models (Models C, D, E, F) and Section 8 Ablation Variants..."
python scripts/train.py --config configs/default.yaml --model all --device cuda

echo "[3/3] Evaluating All 6 Systems and 6 Ablations Across All Splits..."
python scripts/evaluate_all.py --config configs/default.yaml --device cuda

echo "=================================================================="
echo " Pipeline Complete! Results in outputs/results & outputs/figures"
echo "=================================================================="
