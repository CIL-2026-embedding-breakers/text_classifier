#!/bin/bash
#SBATCH --job-name=cil-text-cls
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=cil_jobs
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --gpus=1




set -euo pipefail

mkdir -p logs



# --- W&B -----------------------------------------------------------------
# Set USE_WANDB=0 to skip logging entirely. Otherwise make sure WANDB_API_KEY
# is exported (e.g. via ~/.netrc or a secret in your shell init).


mkdir -p "$WANDB_DIR"

# --- HuggingFace caches (optional but recommended on shared clusters) ----

export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"


# --- Info ----------------------------------------------------------------
echo "Job ID:        $SLURM_JOB_ID"
echo "Job name:      $SLURM_JOB_NAME"
echo "Node:          $(hostname)"
echo "Working dir:   $SLURM_SUBMIT_DIR"
echo "GPUs visible:  ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi || true

cd "$SLURM_SUBMIT_DIR"

# --- Run -----------------------------------------------------------------
uv run python run_dl.py