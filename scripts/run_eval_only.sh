#!/bin/bash
#SBATCH --job-name=rag_eval
#SBATCH --output=outputs/slurm_eval_%j.out
#SBATCH --error=outputs/slurm_eval_%j.err
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -e

echo "=================================================================="
echo " Evaluating RAG Models Across All Splits & Section 8 Ablations"
echo " Host: $(hostname) | Date: $(date)"
echo "=================================================================="

source ~/rag_v100_venv/bin/activate
cd ~/grp_relRAG

mkdir -p outputs/results outputs/figures
export PYTHONPATH="$(pwd):$PYTHONPATH"

python scripts/evaluate_all.py --config configs/default.yaml --device cuda

echo "=================================================================="
echo " Evaluation Complete! Check outputs/results/"
echo "=================================================================="
