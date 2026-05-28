# Text Classifier — CIL 2026

Multilingual sentiment classification (5-class, EN+DE) on Amazon-style reviews.
Team: Embedding_breakers. Score: `1 - MAE / 4`.

## Layout

```
deep_learning/      transformer fine-tuning (trainer, preprocessor, configs)
baselines/          TF-IDF + LR / SVC / NB baselines
data_analysis/      EDA notebooks + lingua-py language splitter
scripts/            SLURM launchers + result-collection scripts
data/               train.csv, test.csv (mixed EN/DE)
configs/, sweeps.py legacy W&B sweep entry (kept; not used in final results)
instructions/       course PDFs
```

## Environment

Local: `uv sync` then `uv run python ...`.

On the ETH student cluster, the launch scripts assume the project's conda env
(or `uv`) is already on `$PATH`. We use `uv run` inside `scripts/launch_run.sh`.

## Cluster connection (one-time)

```
ssh <eth-username>@student-cluster.inf.ethz.ch
```

Optional alias in `~/.ssh/config`:

```
Host eth-student-cluster
    HostName student-cluster.inf.ethz.ch
    User <eth-username>
    IdentityFile ~/.ssh/id_ed25519
```

W&B login: `wandb login` (paste API key from <https://wandb.ai/authorize>).

## Reproducing the results

All runs are launched from the repo root.

### 1. Data preparation (run once)

```
python scripts/split_train_val.py        # 90/10 split, seed=42 -> data/split/
python scripts/run_language_detection.py # lingua-py split    -> data/lang/{EN,DE,UNCERTAIN}/
```

### 2. Per-language model ablation (5 runs)

```
bash scripts/launch_ablation.sh
```

Submits one multilingual baseline + 4 specialists (xlmr_en, xlmr_de, deberta_en,
gbert_de) under a single W&B group. All 5 runs use LR=5e-4, epochs=5, loss=ce
(pinned inside the script).

Collect results from the saved `validation_results.csv` files:

```
python scripts/per_language_ablation.py
```

### 3. Loss-function ablation (4 runs)

```
for loss in ce oll emd soft; do
  sbatch --export=ALL,CIL_LOSS_TYPE=$loss,CIL_RUN_TAG=loss_$loss ./scripts/launch_run.sh
done
```

Collect from W&B by tag:

```
python scripts/loss_results_from_wandb.py loss_ce loss_oll loss_emd loss_soft
```

### 4. Single custom run

Everything is driven by environment variables, e.g.:

```
sbatch --export=ALL,CIL_MODEL_NAME=deepset/gbert-large,CIL_TRAINING_TYPE=lora,\
CIL_LR=3e-4,CIL_EPOCHS=4,CIL_RUN_TAG=gbert_large ./scripts/launch_run.sh
```

## Environment variables

Read by `run_dl.py` and the trainer. All have defaults — only set the ones you
want to override.

| Variable             | Default                  | Meaning                                          |
|----------------------|--------------------------|--------------------------------------------------|
| `CIL_JOB_TYPE`       | `fine_tuning`            | `fine_tuning` or `mlm`                           |
| `CIL_MODEL_NAME`     | `xlm-roberta-base`       | any HuggingFace model id                         |
| `CIL_TRAINING_TYPE`  | `lora`                   | `lora`, `full`, or `head`                        |
| `CIL_LR`             | `5e-4`                   | learning rate                                    |
| `CIL_EPOCHS`         | `5`                      | epochs                                           |
| `CIL_LOSS_TYPE`      | `ce`                     | `ce`, `oll`, `emd`, `soft`                       |
| `CIL_PREPROC`        | `none`                   | `none`, `light`, `medium`, `heavy`               |
| `CIL_DELIMITER_MODE` | `plain_concat`           | `plain_concat` or `with_tags`                    |
| `CIL_DATA_PATH`      | `./data/split/`          | dir containing `train.csv`, `val.csv`, `test.csv`|
| `CIL_RUN_TAG`        | `multilingual`           | string appended to the W&B run name              |
| `CIL_WANDB_GROUP`    | unset                    | optional W&B group                               |
| `CIL_CHECKPOINT_ROOT`| `/work/scratch/$USER/...`| where the HF Trainer writes checkpoints          |
| `SAVE_MODEL_DIR`     | `$SLURM_SUBMIT_DIR/final_models` | where the final model is saved           |

## Forcing a fresh run

The trainer auto-resumes from the latest checkpoint in `CIL_CHECKPOINT_ROOT/<run_name>/`.
To restart from scratch, delete that directory before launching:

```
rm -rf /work/scratch/$USER/text_classifier_cache/checkpoints/<run_name>
```

## Useful links

- Cluster docs: <https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentCluster>
- Kaggle: <https://www.kaggle.com/t/7aafec306ca648acacee5c4fe633c0fb>
