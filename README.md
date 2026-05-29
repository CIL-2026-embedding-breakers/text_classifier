# Text Classifier — CIL 2026

Multilingual sentiment classification (5-class, EN+DE) on Amazon-style reviews.
Team: Embedding_breakers. Score: `1 - MAE / 4`.

---

# Part 1 — Repository

## Directory tree

```
text_classifier/
├── run_dl.py                         # main entry point (env-var driven)
├── sweeps.py                         # legacy W&B sweep entry (kept, not used for final results)
├── pyproject.toml                    # uv project
│
├── deep_learning/
│   ├── code_configs.py               # dataclasses: Fine_Tuning_Configs, LoRa_Configs, ...
│   ├── preprocessing/
│   │   ├── preprocessor.py           # text cleaning, title/review splitting
│   │   └── preprocessing_sanity_check.ipynb # notebook with preprocessing examples 
│   ├── training/
│   │   ├── trainer.py                # CustomTrainer with CE / OLL / EMD / SOFT losses
│   │   └── training_pipeline.py      # LoRA / full / head fine-tuning pipeline
│   └── analysis/
│       └── validation_analysis.py    # per-label MAE + big-mistakes CSV
│
├── baselines/
│   ├── simplified_baselines.ipynb    # all baseline numbers in the paper (Tables 1, 2, 3)
│   └── src/                          # TF-IDF + LR/SVC/NB models, preprocessing, lemmatizer + notebook with baselines plots
│
├── data_analysis/
│   ├── data_visualisation.ipynb
│   ├── data_visualisation_negation.ipynb
│   ├── data_visualisation_of_duplicates.ipynb
│   └── utils/
│       ├── language_detection.py     # lingua-py EN/DE/UNCERTAIN splitter
│       ├── text_analysis.py
│       ├── text_preprocessing.py
│       └── label_distribution.py
│
├── scripts/
│   ├── launch_run.sh                 # SLURM wrapper for one run of run_dl.py
│   ├── launch_ablation.sh            # convenience: per-language 5-run ablation
│   ├── split_train_val.py            # 90/10 split, seed=42
│   ├── run_language_detection.py     # lingua-py EN/DE/UNCERTAIN split
│   ├── setup_gbert_large.py          # download gbert-large to scratch + patch config
│   ├── per_language_ablation.py      # collect per-language ablation from local CSVs
│   ├── loss_results_from_wandb.py    # collect loss ablation results from W&B by tag
│   ├── launch_sweeps.sh              # legacy sweep launcher
│   └── sweep_agent_batch.sh          # legacy sweep agent
│
├── configs/                          # legacy W&B sweep YAMLs (not used for final results)
├── data/                             # train.csv, test.csv (mixed EN/DE)
├── analysis.ipynb                    # corpus EDA notebook
├── text_classification_intro.ipynb   # course-provided starter (untouched)
└── instructions/                     # course PDFs
```

## Resources

| Resource | Location |
|---|---|
| Cluster docs | <https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentCluster> |
| Kaggle competition | <https://www.kaggle.com/t/7aafec306ca648acacee5c4fe633c0fb> |
| W&B project | `cil-project` |
| Cluster conda env | `/cluster/courses/cil/envs/envs/text-classification` |
| Cluster shared data | `/cluster/courses/cil/text-classification/data/` |

## Environment setup

The project uses `uv` for dependency management; `uv.lock` is committed, so
versions are pinned.

Install `uv` (one-time, local machine or cluster):

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

From the repo root, create the venv and install all dependencies:

```
uv sync
```

This creates `.venv/`. To run any Python entry point:

```
uv run python run_dl.py
```

or activate the venv directly:

```
source .venv/bin/activate
python run_dl.py
```

The SLURM launcher `scripts/launch_run.sh` already wraps the training entry
point with `uv run`, so once `uv sync` has succeeded on the cluster every
`sbatch` command below works as-is.

## Cluster connection

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

W&B login (once on the cluster): `wandb login`.

## Environment variables

`run_dl.py` reads these. Defaults are shown; only set the ones you want to
override when launching a run.

| Variable | Default | Meaning |
|---|---|---|
| `CIL_JOB_TYPE` | `fine_tuning` | `fine_tuning` or `mlm` |
| `CIL_MODEL_NAME` | `xlm-roberta-base` | HuggingFace model id |
| `CIL_TRAINING_TYPE` | `lora` | `lora`, `full`, or `head` |
| `CIL_LR` | `5e-4` | learning rate |
| `CIL_EPOCHS` | `5` | epochs |
| `CIL_LOSS_TYPE` | `ce` | `ce`, `oll`, `emd`, `soft` |
| `CIL_PREPROC` | `none` | `none`, `light`, `medium`, `heavy` |
| `CIL_DELIMITER_MODE` | `plain_concat` | `plain_concat` or `with_tags` |
| `CIL_DATA_PATH` | `./data/split/` | dir with `train.csv`, `val.csv`, `test.csv` |
| `CIL_RUN_TAG` | `multilingual` | string appended to the W&B run name |
| `CIL_WANDB_GROUP` | unset | optional W&B group |
| `CIL_BATCH_SIZE` | unset | overrides `per_device_train_batch_size` (use for big LoRA runs) |
| `CIL_EVAL_BATCH_SIZE` | unset | overrides `per_device_eval_batch_size` |
| `CIL_GRAD_ACCUM` | unset | overrides `gradient_accumulation_steps` |
| `CIL_CHECKPOINT_ROOT` | `/work/scratch/$USER/text_classifier_cache/checkpoints` | Trainer checkpoints |
| `SAVE_MODEL_DIR` | `$SLURM_SUBMIT_DIR/final_models` | final model location |

A run resumes from its checkpoint dir automatically. To start fresh, delete it:

```
rm -rf /work/scratch/$USER/text_classifier_cache/checkpoints/<run_name>
```

A few knobs are not exposed as env vars and must be edited in code:
- `input_format` (`review_only` vs `title_plus_review`) → `run_dl.py:41`
- LoRA `r`, `alpha`, `dropout` → `deep_learning/code_configs.py:LoRa_Configs`

---

# Part 2 — Reproducing the paper

Each block corresponds to a table in the paper. Runs inside a block are
independent and can be assigned to different teammates.

## Data preparation (run once, before any training)

```
python scripts/split_train_val.py
python scripts/run_language_detection.py
```

This produces `data/split/{train,val,test}.csv` (seed=42, 90/10) and
`data/lang/{EN,DE,UNCERTAIN}/{train,val,test}.csv` (lingua-py, threshold 0.63).

## Baselines — Tables 1, 2, 3

All baseline numbers (BoW/TF-IDF + logistic regression, naive Bayes, SVC,
XGBoost, random forest, GloVe, fastText; feature-extraction comparison;
per-language comparison) come from one notebook:

```
jupyter nbconvert --execute --to notebook baselines/simplified_baselines.ipynb
```

No SLURM needed.

## Preprocessing ablation — Table 4

XLM-RoBERTa base + LoRA, learning rate `5e-5`. Nine independent runs.

**Caveat:** `input_format` is hardcoded in `run_dl.py:41`. Before launching the
"Review only" runs, edit that line to `input_format="review_only"`. For all
other runs, leave it as `input_format="title_plus_review"`.

Review-only runs (set `input_format="review_only"` first):

```
sbatch --export=ALL,CIL_LR=5e-5,CIL_PREPROC=none,CIL_RUN_TAG=prep_revonly_none     ./scripts/launch_run.sh
sbatch --export=ALL,CIL_LR=5e-5,CIL_PREPROC=light,CIL_RUN_TAG=prep_revonly_light   ./scripts/launch_run.sh
sbatch --export=ALL,CIL_LR=5e-5,CIL_PREPROC=medium,CIL_RUN_TAG=prep_revonly_medium ./scripts/launch_run.sh
sbatch --export=ALL,CIL_LR=5e-5,CIL_PREPROC=heavy,CIL_RUN_TAG=prep_revonly_heavy   ./scripts/launch_run.sh
```

Title+review with delimiters (set `input_format="title_plus_review"`):

```
sbatch --export=ALL,CIL_LR=5e-5,CIL_DELIMITER_MODE=with_tags,CIL_PREPROC=none,CIL_RUN_TAG=prep_tags_none     ./scripts/launch_run.sh
sbatch --export=ALL,CIL_LR=5e-5,CIL_DELIMITER_MODE=with_tags,CIL_PREPROC=light,CIL_RUN_TAG=prep_tags_light   ./scripts/launch_run.sh
sbatch --export=ALL,CIL_LR=5e-5,CIL_DELIMITER_MODE=with_tags,CIL_PREPROC=medium,CIL_RUN_TAG=prep_tags_medium ./scripts/launch_run.sh
sbatch --export=ALL,CIL_LR=5e-5,CIL_DELIMITER_MODE=with_tags,CIL_PREPROC=heavy,CIL_RUN_TAG=prep_tags_heavy   ./scripts/launch_run.sh
```

Title+review plain concat (set `input_format="title_plus_review"`):

```
sbatch --export=ALL,CIL_LR=5e-5,CIL_DELIMITER_MODE=plain_concat,CIL_PREPROC=none,CIL_RUN_TAG=prep_concat_none ./scripts/launch_run.sh
```

Results: read `eval/score` per run on the W&B project, or open
`fine_tuning/<run_name>/validation_metrics.txt` written by each job.

## Per-language specialisation — Table 5

XLM-RoBERTa base + LoRA, LR `5e-4`, 5 epochs, CE loss. Five independent runs.

```
sbatch --export=ALL,CIL_MODEL_NAME=xlm-roberta-base,CIL_DATA_PATH=./data/split/,CIL_RUN_TAG=multilingual        ./scripts/launch_run.sh
sbatch --export=ALL,CIL_MODEL_NAME=xlm-roberta-base,CIL_DATA_PATH=./data/lang/EN/,CIL_RUN_TAG=xlmr_en           ./scripts/launch_run.sh
sbatch --export=ALL,CIL_MODEL_NAME=xlm-roberta-base,CIL_DATA_PATH=./data/lang/DE/,CIL_RUN_TAG=xlmr_de           ./scripts/launch_run.sh
sbatch --export=ALL,CIL_MODEL_NAME=microsoft/deberta-v3-base,CIL_DATA_PATH=./data/lang/EN/,CIL_RUN_TAG=deberta_en ./scripts/launch_run.sh
sbatch --export=ALL,CIL_MODEL_NAME=deepset/gbert-base,CIL_DATA_PATH=./data/lang/DE/,CIL_RUN_TAG=gbert_de          ./scripts/launch_run.sh
```

All five at once (puts them in one W&B group):

```
bash scripts/launch_ablation.sh
```

Collect:

```
python scripts/per_language_ablation.py
```

## Loss-function ablation — Table 6

XLM-RoBERTa base + LoRA, multilingual data. Four independent runs.

```
sbatch --export=ALL,CIL_LOSS_TYPE=ce,CIL_RUN_TAG=loss_ce     ./scripts/launch_run.sh
sbatch --export=ALL,CIL_LOSS_TYPE=oll,CIL_RUN_TAG=loss_oll   ./scripts/launch_run.sh
sbatch --export=ALL,CIL_LOSS_TYPE=emd,CIL_RUN_TAG=loss_emd   ./scripts/launch_run.sh
sbatch --export=ALL,CIL_LOSS_TYPE=soft,CIL_RUN_TAG=loss_soft ./scripts/launch_run.sh
```

Collect from W&B (one tag at a time gives a per-loss table):

```
python scripts/loss_results_from_wandb.py loss_ce
python scripts/loss_results_from_wandb.py loss_oll
python scripts/loss_results_from_wandb.py loss_emd
python scripts/loss_results_from_wandb.py loss_soft
```

## LoRA vs full fine-tuning — Table 7

XLM-RoBERTa base, multilingual data. Two independent runs.

```
sbatch --export=ALL,CIL_TRAINING_TYPE=lora,CIL_RUN_TAG=lora_vs_full_lora ./scripts/launch_run.sh
sbatch --export=ALL,CIL_TRAINING_TYPE=full,CIL_RUN_TAG=lora_vs_full_full ./scripts/launch_run.sh
```

The full run uses the batch-size override hardcoded in `run_dl.py:71-74`
(per-device batch 8, grad accumulation 8).

## XLM-RoBERTa scaling — Table 8 / Figure (scaling)

LoRA fine-tuning across the XLM-R family, multilingual data. Three independent
runs. The XL model needs a larger GPU; submit it on a high-memory partition.

```
sbatch --export=ALL,CIL_MODEL_NAME=xlm-roberta-base,CIL_RUN_TAG=scale_base       ./scripts/launch_run.sh
sbatch --export=ALL,CIL_MODEL_NAME=xlm-roberta-large,CIL_RUN_TAG=scale_large     ./scripts/launch_run.sh
sbatch --export=ALL,CIL_MODEL_NAME=facebook/xlm-roberta-xl,CIL_RUN_TAG=scale_xl  ./scripts/launch_run.sh
```


# AI use


## Baselines:
Tool used: [Claude code]
Files affected: [all the baseline folder]
Asked Claude code to implement the baselines I had thought about, gave Claude the following 
“In order to have a well constructed project I think we should compute robust baselines with basic methods in order to get intuitions about what are the things a more complex model would help on.
	Among the things to look at:
* What are the limitations of the BOW approach. Why does it fail.
* Look at how informative/noisy the title and text samples are 
* See if it helps to have a specialized model per language 
* See if classification is the way to go or we should implement a loss that accounts for the regression aspect of the loss
* See how some basic preprocessing (lowercasing, removing spelling mistakes, …) can help on our task.
* See the performance assess the performance of bigger models in a  non-finetuned setting in order to assess the potential improvement when moving to deeper models 
For all these tasks I think we should compute the resulting metrics as well as perform a quick data analysis of the results in order to understand why an approach can help/hinder our task.

Here is how I see this part:

1. Figure out if classification or regression (or some other metric) is better
Compute logistic regression vs linear regression vs weighted logistics regression  with small parameter sweep and compare
Look at the failure modes and guess if DL would help
For the rest of the baselines, use the best performing objective

2. Figure the impact of the n gram by looking at the evolution of performance when we increase n.
3. Look at ensemble of  language specific BOW model vs one multilanguage BOW model (simple parameter sweep but the result should be pretty clear)
			For the future use the best performing approach

3. See the impact of preprocessing. Set up a preprocessing pipeline (many of them online) and see the impact on performance. It would be good to analyse the results and compare them.

1. See the difference between title only and title + text.
“
I later added Glove and fasttext and asked Claude for an implementation of a vectorizer, all code in the baseline folder has been checked and understood

## Wandb:
Tool used: [Claude code]
Files affected: [scripts/launch_sweeps.sh sweeps.py]
Asked Claude code to help with the implementation of wandb sweeps 

## Trainer:
Tool used: [Claude Sonnet 4.6]
Files affected: []
Asked Claude what combination of hyper parameters made sense for LoRa and different sizes of XLM-Roberta (conducted the experiment myself, just asked for initial values that made sense)
