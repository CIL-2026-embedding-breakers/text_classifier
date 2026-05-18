#!/bin/bash
#SBATCH --job-name=sweep-ft
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err
#SBATCH --account=cil_jobs
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1

set -euo pipefail
mkdir -p logs

# --- W&B ---
export WANDB_DIR="${SCRATCH:-$HOME}/wandb"
mkdir -p "$WANDB_DIR"

# SWEEP_ID and SWEEP_COUNT are passed via --export from launch_sweep.sh
if [[ -z "${SWEEP_ID:-}" ]]; then
    echo "ERROR: SWEEP_ID is not set. Submit via launch_sweep.sh."
    exit 1
fi

SWEEP_COUNT="${SWEEP_COUNT:-9}"

echo "Job: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Sweep: ${SWEEP_ID}"
echo "Runs:  ${SWEEP_COUNT}"

# Single agent runs all sweep configurations sequentially on this one GPU.
uv run wandb agent --count "${SWEEP_COUNT}" "${SWEEP_ID}"