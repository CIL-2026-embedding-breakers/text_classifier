#!/bin/bash
#SBATCH --job-name=cil-text-cls
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=cil_jobs
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=3
#SBATCH --mem=32G
#SBATCH --gpus=5060ti:1

set -euo pipefail

mkdir -p logs

# --- W&B -----------------------------------------------------------------
# Set local cache/log directories if not already defined
export WANDB_DIR="${WANDB_DIR:-$SLURM_SUBMIT_DIR/wandb}"
mkdir -p "$WANDB_DIR"

# --- HuggingFace caches (optional but recommended on shared clusters) ----
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# --- Checkpoints to scratch, final models to home -----------------------
export CIL_CHECKPOINT_ROOT="${CIL_CHECKPOINT_ROOT:-/work/scratch/sgodey/text_classifier_cache/checkpoints}"
export SAVE_MODEL_DIR="${SAVE_MODEL_DIR:-$SLURM_SUBMIT_DIR/final_models}"
mkdir -p "$CIL_CHECKPOINT_ROOT" "$SAVE_MODEL_DIR"

# --- Info ----------------------------------------------------------------
echo "Job ID:        $SLURM_JOB_ID"
echo "Job name:      $SLURM_JOB_NAME"
echo "Node:          $(hostname)"
echo "Working dir:   $SLURM_SUBMIT_DIR"
echo "GPUs visible:  ${CUDA_VISIBLE_DEVICES:-none}"
echo "WANDB_DIR:     $WANDB_DIR"
echo "HF_HOME:       $HF_HOME"
nvidia-smi || true

cd "$SLURM_SUBMIT_DIR"

# --- Run -----------------------------------------------------------------
uv run python run_dl.py