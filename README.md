# Text Classifier — CIL 2026

Multilingual sentiment classification (5-class, EN+DE) on Amazon-style reviews.
Team: Embedding_breakers. Score: `1 - MAE / 4`.

This README is split in two parts: **Part 1** explains the repo layout and how to
get an environment running; **Part 2** has one short section per element of the
paper (each table / figure) with the exact command(s) to reproduce it. Sections
in Part 2 follow the order of the paper, and runs inside a section are
independent, so they can be split between teammates.

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
│   ├── simplified_baselines.ipynb    # all baseline numbers in the paper (Tables II, III)
│   └── src/                          # TF-IDF + LR/SVC/NB models, preprocessing, lemmatizer
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
| W&B project (base-scale runs) | `cil-project` |
| W&B project (scaling runs) | `cil-project-scaling-laws` |
| Cluster conda env | `/cluster/courses/cil/envs/envs/text-classification` |
| Cluster shared data | `/cluster/courses/cil/text-classification/data/` |

## Environment setup (uv)

The project uses `uv` for dependency management; `uv.lock` is committed, so
versions are pinned.

Install `uv` (one-time, local machine or cluster):

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

From the repo root, create the venv and install everything from the lockfile:

```
uv sync
```

This creates `.venv/`. From then on, prefix any Python command with `uv run`:

```
uv run python run_dl.py
```

or activate the venv once and use plain `python`:

```
source .venv/bin/activate
python run_dl.py
```

The SLURM launcher `scripts/launch_run.sh` already wraps the training entry point
with `uv run`, so once `uv sync` succeeds on the cluster, every `sbatch` command
below works as-is.

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
| `CIL_BATCH_SIZE` | unset | overrides `per_device_train_batch_size` (use for big runs) |
| `CIL_EVAL_BATCH_SIZE` | unset | overrides `per_device_eval_batch_size` |
| `CIL_GRAD_ACCUM` | unset | overrides `gradient_accumulation_steps` |
| `CIL_CHECKPOINT_ROOT` | `/work/scratch/$USER/text_classifier_cache/checkpoints` | Trainer checkpoints |
| `SAVE_MODEL_DIR` | `$SLURM_SUBMIT_DIR/final_models` | final model location |

A run resumes from its checkpoint dir automatically. To start fresh, delete it:

```
rm -rf /work/scratch/$USER/text_classifier_cache/checkpoints/<run_name>
```

### Knobs that are not env vars (edit in code)

A couple of settings are intentionally kept as code defaults instead of env vars:

- `input_format` (`review_only` vs `title_plus_review`) → `run_dl.py:46`
- LoRA `r`, `alpha`, `dropout` → `deep_learning/code_configs.py`, `LoRa_Configs`
  (defaults `r=16, alpha=32, dropout=0.1`)

The full-finetuning batch sizes are hardcoded in `run_dl.py:76-79`
(per-device train 8, eval 16, grad accumulation 8).

### What a run writes

Each run writes its metrics to W&B and to a local folder named after the run, at
`fine_tuning/<run_name>/`:

- `validation_metrics.txt` — overall score / accuracy / per-label MAE
- `validation_results.csv` — per-example predictions on the val set
- `big_mistakes.csv` — examples off by ≥ 2 classes
- `submission.csv` — predictions on the Kaggle test set

So for most tables you can just read `eval/score` on W&B, or `cat` the
`validation_metrics.txt` of each run.

---

# Part 2 — Reproducing the paper

> **Conventions used below.** Every transformer run uses `light` preprocessing
> (`CIL_PREPROC=light`) — the preprocessing ablation (Table V) is the only place
> this varies, and it shows the level barely matters (< 0.001 score). LoRA uses
> `r=16, alpha=32` (the code default). Where a run needs a non-default
> `input_format`, it is called out explicitly.

## Data preparation (run once, before any training)

This builds the train/val split and the per-language buckets used everywhere
below.

```
uv run python scripts/split_train_val.py        # 90/10 split, seed=42
uv run python scripts/run_language_detection.py  # lingua-py EN / DE / UNCERTAIN split
```

This produces:

- `data/split/{train,val,test}.csv` — seed 42, 90/10 split
- `data/lang/{EN,DE,UNCERTAIN}/{train,val,test}.csv` — lingua-py, confidence
  threshold 0.63

## Baselines — Tables II & III

All baseline numbers come from a single notebook: BoW / TF-IDF with logistic
regression, naive Bayes, SVC, XGBoost, random forest, plus GloVe and fastText
features (Table II), and the n-gram feature-extraction comparison (Table III).
No SLURM needed — runs on a laptop.

```
uv run jupyter nbconvert --execute --to notebook baselines/simplified_baselines.ipynb
```

## LoRA vs full fine-tuning — Table IV

XLM-RoBERTa-base, multilingual data. Two independent runs. Full fine-tuning needs
the lower learning rate (`1e-5`); LoRA uses the default `5e-4`.

```
sbatch --export=ALL,CIL_PREPROC=light,CIL_TRAINING_TYPE=lora,CIL_LR=5e-4,CIL_RUN_TAG=lora_vs_full_lora ./scripts/launch_run.sh
sbatch --export=ALL,CIL_PREPROC=light,CIL_TRAINING_TYPE=full,CIL_LR=1e-5,CIL_RUN_TAG=lora_vs_full_full ./scripts/launch_run.sh
```

The full run automatically uses the small batch size hardcoded in `run_dl.py:76-79`.

## Preprocessing ablation — Table V

XLM-RoBERTa-base + LoRA, learning rate `5e-5`. Nine independent runs that vary
the input representation and the normalization level.

**Before launching:** `input_format` is a code default (`run_dl.py:46`). Set it to
`"review_only"` for the four review-only runs, and back to `"title_plus_review"`
for the rest.

Review-only runs (set `input_format="review_only"`):

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

Read `eval/score` per run on W&B, or open each `fine_tuning/<run_name>/validation_metrics.txt`.

## Per-language specialisation — Table VI

XLM-RoBERTa-base + LoRA, LR `5e-4`, 5 epochs, CE loss, `light` preprocessing.
Five independent runs: the multilingual baseline, two language-filtered XLM-R
controls (EN-only, DE-only), and two monolingual specialists (DeBERTa-v3-base for
English, gBERT-base for German).

**Before launching:** these runs use review-only inputs. Set
`input_format="review_only"` in `run_dl.py:46`.

```
sbatch --export=ALL,CIL_PREPROC=light,CIL_MODEL_NAME=xlm-roberta-base,CIL_DATA_PATH=./data/split/,CIL_RUN_TAG=multilingual          ./scripts/launch_run.sh
sbatch --export=ALL,CIL_PREPROC=light,CIL_MODEL_NAME=xlm-roberta-base,CIL_DATA_PATH=./data/lang/EN/,CIL_RUN_TAG=xlmr_en             ./scripts/launch_run.sh
sbatch --export=ALL,CIL_PREPROC=light,CIL_MODEL_NAME=xlm-roberta-base,CIL_DATA_PATH=./data/lang/DE/,CIL_RUN_TAG=xlmr_de             ./scripts/launch_run.sh
sbatch --export=ALL,CIL_PREPROC=light,CIL_MODEL_NAME=microsoft/deberta-v3-base,CIL_DATA_PATH=./data/lang/EN/,CIL_RUN_TAG=deberta_en ./scripts/launch_run.sh
sbatch --export=ALL,CIL_PREPROC=light,CIL_MODEL_NAME=deepset/gbert-base,CIL_DATA_PATH=./data/lang/DE/,CIL_RUN_TAG=gbert_de          ./scripts/launch_run.sh
```

All five at once (puts them in one W&B group; the script already sets
`CIL_PREPROC=light`, but you still need the `input_format="review_only"` edit):

```
bash scripts/launch_ablation.sh
```

Collect the aggregated per-bucket / routed-pair numbers from the local
`validation_results.csv` files:

```
uv run python scripts/per_language_ablation.py
```

## Loss-function ablation — Table VII

XLM-RoBERTa-base + LoRA, multilingual data, `light` preprocessing,
`title_plus_review` inputs (the code default). Only the training-time loss
changes. Four independent runs.

```
sbatch --export=ALL,CIL_PREPROC=light,CIL_LOSS_TYPE=ce,CIL_RUN_TAG=loss_ce     ./scripts/launch_run.sh
sbatch --export=ALL,CIL_PREPROC=light,CIL_LOSS_TYPE=oll,CIL_RUN_TAG=loss_oll   ./scripts/launch_run.sh
sbatch --export=ALL,CIL_PREPROC=light,CIL_LOSS_TYPE=emd,CIL_RUN_TAG=loss_emd   ./scripts/launch_run.sh
sbatch --export=ALL,CIL_PREPROC=light,CIL_LOSS_TYPE=soft,CIL_RUN_TAG=loss_soft ./scripts/launch_run.sh
```

Collect from W&B (one tag at a time gives a per-loss table):

```
uv run python scripts/loss_results_from_wandb.py loss_ce
uv run python scripts/loss_results_from_wandb.py loss_oll
uv run python scripts/loss_results_from_wandb.py loss_emd
uv run python scripts/loss_results_from_wandb.py loss_soft
```

(OLL uses `alpha=1.5` and SOFT uses `sigma=4`, set in the trainer.)

## Scaling XLM-RoBERTa — Table VIII & Figure 2

LoRA fine-tuning across the XLM-R family (base / large / XL), multilingual data,
`title_plus_review` inputs with delimiters, LoRA `r=16, alpha=32`. The learning
rate is decreased as the model grows. Three independent runs.

```
# base — lr 5e-4, 6 epochs
sbatch --export=ALL,CIL_PREPROC=light,CIL_MODEL_NAME=xlm-roberta-base,CIL_LR=5e-4,CIL_EPOCHS=6,CIL_DELIMITER_MODE=with_tags,CIL_RUN_TAG=scale_base   ./scripts/launch_run.sh
# large — lr 3e-4, 6 epochs
sbatch --export=ALL,CIL_PREPROC=light,CIL_MODEL_NAME=xlm-roberta-large,CIL_LR=3e-4,CIL_EPOCHS=6,CIL_DELIMITER_MODE=with_tags,CIL_RUN_TAG=scale_large ./scripts/launch_run.sh
# XL — lr 2e-4, 4 epochs (3.49B params: needs a big GPU + small batch + grad accum)
sbatch --export=ALL,CIL_PREPROC=light,CIL_MODEL_NAME=facebook/xlm-roberta-xl,CIL_LR=2e-4,CIL_EPOCHS=4,CIL_DELIMITER_MODE=with_tags,CIL_BATCH_SIZE=8,CIL_GRAD_ACCUM=8,CIL_RUN_TAG=scale_xl ./scripts/launch_run.sh
```

These scaling runs were originally launched from the `scaling_law` branch, which
is identical to `main` except that it logs to a separate W&B project; on `main`
they log to `cil-project`.

**Figure 2** (training-loss and eval-MAE curves) is plotted directly from the
`train/loss` and `eval/mae` curves of these three runs as logged to W&B. There is
no separate plotting script in the repo.

## Do base-scale choices transfer to XL? — Table IX

This checks whether the two base-scale wins (OLL loss, monolingual backbones)
still help at XL scale. Same LoRA protocol as the scaling section above. The
models compared:

| Row (Table IX) | Model |
|---|---|
| Loss: XLM-R XL + CE | `facebook/xlm-roberta-xl` |
| Loss: XLM-R XL + OLL | `facebook/xlm-roberta-xl` |
| Backbone (routed EN) | `microsoft/deberta-v2-xlarge` (~0.9B) |
| Backbone (routed DE) | `ikim-uk-essen/geberta-xlarge` (~0.9B) |

**Loss at XL (CE vs OLL).** The CE run is the scaling XL run above. For the OLL
run, swap the loss (the XL CE run already gives the CE row):

```
sbatch --export=ALL,CIL_MODEL_NAME=facebook/xlm-roberta-xl,CIL_LOSS_TYPE=oll,CIL_LR=2e-4,CIL_EPOCHS=4,CIL_PREPROC=light,CIL_BATCH_SIZE=8,CIL_GRAD_ACCUM=8,CIL_RUN_TAG=xlmr_lora_oll ./scripts/launch_run.sh
```

**Backbone at XL (routed monolingual pair).** Fine-tune the two ~0.9B monolingual
specialists on the EN / DE buckets, then route by language. Both are DeBERTa-v2
architecture, so the LoRA target modules are picked automatically (the pipeline
keys on `deberta-v`/`geberta` in the model name).

```
sbatch --export=ALL,CIL_MODEL_NAME=microsoft/deberta-v2-xlarge,CIL_DATA_PATH=./data/lang/EN/,CIL_PREPROC=light,CIL_LR=2e-4,CIL_EPOCHS=4,CIL_BATCH_SIZE=4,CIL_GRAD_ACCUM=16,CIL_RUN_TAG=deberta_xl_en ./scripts/launch_run.sh
sbatch --export=ALL,CIL_MODEL_NAME=ikim-uk-essen/geberta-xlarge,CIL_DATA_PATH=./data/lang/DE/,CIL_PREPROC=light,CIL_LR=2e-4,CIL_EPOCHS=4,CIL_BATCH_SIZE=4,CIL_GRAD_ACCUM=16,CIL_RUN_TAG=geberta_xl_de ./scripts/launch_run.sh
```

The batch / grad-accum values above are starting points for fitting the big
models on a single GPU. The routed-pair score is obtained by concatenating the EN
and DE validation predictions (same idea as `scripts/per_language_ablation.py`).

---

# AI use


## Baselines:
Tool used: [Claude code]
Files affected: [all the baseline folder]
Purpose:[
Asked Claude code to implement the baselines I had thought about, gave Claude the following 
“In order to have a well constructed project I think we should compute robust baselines with basic methods in order to get intuitions about what are the things a more complex model would help on.
	Among the things to look at:
* What are the limitations of the BOW approach. Why does it fail.
* Look at how informative/noisy the title and text samples are 
* See if it helps to have a specialized model per language 
* See if classification is the way to go or we should implement a loss that accounts for the regression aspect of the loss
* See how some basic preprocessing (lowercasing, removing spelling mistakes, …) can help on our task.
* See the performance assess the performance of bigger models in a  non-finetuned setting in order to assess the potential improvement when moving to deeper models 
For all these tasks I think we should compute the resulting metrics as well as perform a quick data analysis of the results in order to understand why an approach can help/hinder our task.

Here is how I see this part:

1. Figure out if classification or regression (or some other metric) is better
Compute logistic regression vs linear regression vs weighted logistics regression  with small parameter sweep and compare
Look at the failure modes and guess if DL would help
For the rest of the baselines, use the best performing objective

2. Figure the impact of the n gram by looking at the evolution of performance when we increase n.
3. Look at ensemble of  language specific BOW model vs one multilanguage BOW model (simple parameter sweep but the result should be pretty clear)
			For the future use the best performing approach

3. See the impact of preprocessing. Set up a preprocessing pipeline (many of them online) and see the impact on performance. It would be good to analyse the results and compare them.

1. See the difference between title only and title + text.
“
I later added Glove and fasttext and asked Claude for an implementation of a vectorizer, all code in the baseline folder has been checked and understood]

## Wandb:
Tool used: [Claude code]
Files affected: [scripts/launch_sweeps.sh sweeps.py]
Purpose: [Asked Claude code to help with the implementation of wandb sweeps]

## Trainer:
Tool used: [Claude Sonnet 4.6]
Files affected: []
Purpose: [Asked Claude what combination of hyper parameters made sense for LoRa and different sizes of XLM-Roberta (conducted the experiment myself, just asked for initial values that made sense)]
