#!/bin/bash
# Run this once on the login node, NOT via sbatch.
# It creates the sweep and prints the sweep ID, then submits the array.
#
# Usage: ./this_script.sh <sweep_config.yaml>
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <sweep_config.yaml>" >&2
    exit 1
fi

SWEEP_YAML="$1"

if [[ ! -f "$SWEEP_YAML" ]]; then
    echo "Error: sweep config file '$SWEEP_YAML' not found." >&2
    exit 1
fi

PROJECT="sentiment-analysis-CIL-sweeps"
ENTITY="${WANDB_ENTITY:-}"   # set WANDB_ENTITY env var if needed

# Create the sweep and capture the sweep ID

if [[ -n "$ENTITY" ]]; then
    SWEEP_ID=$(uv run wandb sweep --project "$PROJECT" --entity "$ENTITY" "$SWEEP_YAML" 2>&1 \
        | tee /dev/stderr \
        | grep -oE 'wandb agent [^ ]+/[^ ]+/[^ ]+' \
        | tail -1 \
        | awk '{print $3}')
else
    SWEEP_ID=$(uv run wandb sweep --project "$PROJECT" "$SWEEP_YAML" 2>&1 \
        | tee /dev/stderr \
        | grep -oE 'wandb agent [^ ]+/[^ ]+/[^ ]+' \
        | tail -1 \
        | awk '{print $3}')
fi

if [[ -z "$SWEEP_ID" ]]; then
    echo "ERROR: Failed to parse sweep ID from wandb output." >&2
    exit 1
fi

echo ""
echo "Sweep created: $SWEEP_ID"

# 9 runs total (3 lr × 3 epochs); array 0-8, max 4 concurrent
sbatch --export=ALL,SWEEP_ID="$SWEEP_ID" ./scripts/sweep_agent_batch.sh