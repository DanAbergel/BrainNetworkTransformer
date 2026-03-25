#!/bin/bash
# =====================================================================
# SLURM Job Script — Parcellate ADNI 4D volumes (812 subjects)
# =====================================================================
#
# HOW TO USE:
#   sbatch jobs/parcellate_adni.sh
#
# CPU only, ~64 GB RAM for the full tensor.
# =====================================================================

#SBATCH --job-name=parcellate-adni
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=logs/parcellate_adni_%j.out
#SBATCH --error=logs/parcellate_adni_%j.err

set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
PROJECT_DIR="$LAB_DIR/repos/BrainNetworkTransformer"
VENV_DIR="$LAB_DIR/torch_env"

mkdir -p "$PROJECT_DIR/logs"

echo "Parcellate ADNI — Job $SLURM_JOB_ID on $(hostname)"

source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"

pip install --quiet nilearn

python3 -u scripts/parcellate_adni.py

echo "Done: $(date)"
