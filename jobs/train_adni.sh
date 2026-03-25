#!/bin/bash
# =====================================================================
# SLURM Job Script — Train BrainNetworkTransformer on ADNI
# =====================================================================
#
# HOW TO USE:
#   sbatch jobs/train_adni.sh
#
# Runs BNT with all 4 models on ADNI Sex_Binary (default).
# To change label:
#   sbatch jobs/train_adni.sh CDR_Binary
# =====================================================================

#SBATCH --job-name=bnt-adni
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/train_adni_%j.out
#SBATCH --error=logs/train_adni_%j.err

set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
PROJECT_DIR="$LAB_DIR/repos/BrainNetworkTransformer"
VENV_DIR="$LAB_DIR/torch_env"

export TMPDIR="$LAB_DIR/tmp"
export PIP_CACHE_DIR="$LAB_DIR/cache/pip"
export XDG_CACHE_HOME="$LAB_DIR/cache"
export WANDB_MODE=disabled
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$PROJECT_DIR/logs"

LABEL=${1:-Sex_Binary}

echo "BNT ADNI Training — Job $SLURM_JOB_ID on $(hostname)"
echo "Label: $LABEL"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"

pip install --quiet hydra-core omegaconf wandb scikit-learn

python -m source \
    dataset=ADNI \
    model=bnt_adni \
    dataset.label_column=$LABEL \
    repeat_time=5 \
    preprocess=mixup \
    datasz=100p \
    2>&1 | tee "logs/bnt_adni_${LABEL}_${SLURM_JOB_ID}.log"

echo "Done: $(date)"
