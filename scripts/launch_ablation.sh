#!/bin/bash
# Submit the 5-run per-language ablation under a single W&B group.
set -euo pipefail

GROUP="per_language_ablation_$(date +%Y%m%d_%H%M)"
echo "W&B group: $GROUP"

sbatch --export=ALL,CIL_WANDB_GROUP=$GROUP,CIL_MODEL_NAME=xlm-roberta-base,CIL_DATA_PATH=./data/split/,CIL_RUN_TAG=multilingual ./scripts/launch_run.sh
sbatch --export=ALL,CIL_WANDB_GROUP=$GROUP,CIL_MODEL_NAME=xlm-roberta-base,CIL_DATA_PATH=./data/lang/EN/,CIL_RUN_TAG=xlmr_en ./scripts/launch_run.sh
sbatch --export=ALL,CIL_WANDB_GROUP=$GROUP,CIL_MODEL_NAME=xlm-roberta-base,CIL_DATA_PATH=./data/lang/DE/,CIL_RUN_TAG=xlmr_de ./scripts/launch_run.sh
sbatch --export=ALL,CIL_WANDB_GROUP=$GROUP,CIL_MODEL_NAME=microsoft/deberta-v3-base,CIL_DATA_PATH=./data/lang/EN/,CIL_RUN_TAG=deberta_en ./scripts/launch_run.sh
sbatch --export=ALL,CIL_WANDB_GROUP=$GROUP,CIL_MODEL_NAME=deepset/gbert-base,CIL_DATA_PATH=./data/lang/DE/,CIL_RUN_TAG=gbert_de ./scripts/launch_run.sh
