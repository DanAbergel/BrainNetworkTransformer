#!/bin/bash
# =====================================================================
# SLURM Job Script — Verify ADNI data structure for BNT
# =====================================================================
#
# HOW TO USE:
#   sbatch jobs/verify_adni.sh
#
# CPU only, minimal resources.
# =====================================================================

#SBATCH --job-name=verify-adni
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/verify_adni_%j.out
#SBATCH --error=logs/verify_adni_%j.err

set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
PROJECT_DIR="$LAB_DIR/repos/BrainNetworkTransformer"
VENV_DIR="$LAB_DIR/torch_env"

mkdir -p "$PROJECT_DIR/logs"

echo "Verify ADNI data — Job $SLURM_JOB_ID on $(hostname)"

source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"

python3 -u scripts/verify_adni_data.py

echo "Done: $(date)"
