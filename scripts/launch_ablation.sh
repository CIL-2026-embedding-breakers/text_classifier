#!/bin/bash
# Submit the 5-run per-language ablation under a single W&B group.
#
# NOTE: the per-language runs used review-only inputs with "light" preprocessing.
# Before submitting, set input_format="review_only" in run_dl.py (the line that
# currently reads input_format="title_plus_review"). CIL_PREPROC=light is set below.
set -euo pipefail

GROUP="per_language_ablation_$(date +%Y%m%d_%H%M)"
LR=5e-4
EPOCHS=5
LOSS=ce
PREPROC=light
echo "W&B group: $GROUP   (lr=$LR epochs=$EPOCHS loss=$LOSS preproc=$PREPROC)"

COMMON="CIL_WANDB_GROUP=$GROUP,CIL_LR=$LR,CIL_EPOCHS=$EPOCHS,CIL_LOSS_TYPE=$LOSS,CIL_PREPROC=$PREPROC"

sbatch --export=ALL,$COMMON,CIL_MODEL_NAME=xlm-roberta-base,CIL_DATA_PATH=./data/split/,CIL_RUN_TAG=multilingual ./scripts/launch_run.sh
sbatch --export=ALL,$COMMON,CIL_MODEL_NAME=xlm-roberta-base,CIL_DATA_PATH=./data/lang/EN/,CIL_RUN_TAG=xlmr_en ./scripts/launch_run.sh
sbatch --export=ALL,$COMMON,CIL_MODEL_NAME=xlm-roberta-base,CIL_DATA_PATH=./data/lang/DE/,CIL_RUN_TAG=xlmr_de ./scripts/launch_run.sh
sbatch --export=ALL,$COMMON,CIL_MODEL_NAME=microsoft/deberta-v3-base,CIL_DATA_PATH=./data/lang/EN/,CIL_RUN_TAG=deberta_en ./scripts/launch_run.sh
sbatch --export=ALL,$COMMON,CIL_MODEL_NAME=deepset/gbert-base,CIL_DATA_PATH=./data/lang/DE/,CIL_RUN_TAG=gbert_de ./scripts/launch_run.sh
