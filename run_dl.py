import os
import wandb
from pathlib import Path
from types import SimpleNamespace
from deep_learning.training.training_pipeline import training_pipeline
from deep_learning.code_configs import (
    MLM_Configs,
    Fine_Tuning_Configs,
    LoRa_Configs,
    General_Trainer_Configs,
    Preprocessor_Configs,
)


def get_run_config():
    """Define your hyperparameters here instead of getting them from a sweep."""
    return SimpleNamespace(
        # Job / training type
        job_type=os.environ.get("CIL_JOB_TYPE", "fine_tuning"),  # "mlm" or "fine_tuning"

        # Model
        model_name=os.environ.get("CIL_MODEL_NAME", "xlm-roberta-base"),
        training_type=os.environ.get("CIL_TRAINING_TYPE", "lora"),  # "lora", "full", or "head"

        # Optimization
        lr=float(os.environ.get("CIL_LR", 5e-4)),
        num_epochs=int(os.environ.get("CIL_EPOCHS", 5)),
        warmup_steps=1000,
        weight_decay=0.01,
        mlm_probability=0.15,  # Only relevant for MLM further pretraining

        # Optional batch-size overrides (None -> keep code defaults)
        batch_size=int(os.environ["CIL_BATCH_SIZE"]) if "CIL_BATCH_SIZE" in os.environ else None,
        eval_batch_size=int(os.environ["CIL_EVAL_BATCH_SIZE"]) if "CIL_EVAL_BATCH_SIZE" in os.environ else None,
        grad_accum=int(os.environ["CIL_GRAD_ACCUM"]) if "CIL_GRAD_ACCUM" in os.environ else None,

        # Optional LoRA hyperparameter overrides (None -> keep LoRa_Configs defaults)
        lora_r=int(os.environ["CIL_LORA_R"]) if "CIL_LORA_R" in os.environ else None,
        lora_alpha=int(os.environ["CIL_LORA_ALPHA"]) if "CIL_LORA_ALPHA" in os.environ else None,
        lora_dropout=float(os.environ["CIL_LORA_DROPOUT"]) if "CIL_LORA_DROPOUT" in os.environ else None,

        # Data / task-specific
        data_path=os.environ.get("CIL_DATA_PATH", "./data/split/"),
        external_val_split=True,
        val_split=0.1,
        num_labels=5,
        run_tag=os.environ.get("CIL_RUN_TAG", "multilingual"),

        # Preprocessing / input format (defaults match youssef's known-good baseline)
        preprocessing_level=os.environ.get("CIL_PREPROC", "none"),       # "none", "light", "medium", "heavy"
        input_format="title_plus_review",        # "review_only", "title_plus_review"
        title_review_delimiter_mode=os.environ.get("CIL_DELIMITER_MODE", "plain_concat"),  # "with_tags" or "plain_concat"

        # W&B grouping for the runs table; None = ungrouped
        wandb_group=os.environ.get("CIL_WANDB_GROUP", None),

        # Optional custom W&B run name; if None, a descriptive one is built automatically
        run_name=None,
    )


def build_default_configs(cfg):
    # LoRA config (only if training_type == "lora")
    if cfg.training_type == "lora":
        lora_kwargs = {}
        if cfg.lora_r is not None:
            lora_kwargs["r"] = cfg.lora_r
        if cfg.lora_alpha is not None:
            lora_kwargs["alpha"] = cfg.lora_alpha
        if cfg.lora_dropout is not None:
            lora_kwargs["dropout"] = cfg.lora_dropout
        lora_cfg = LoRa_Configs(**lora_kwargs)
    else:
        lora_cfg = None

    training_cfg = General_Trainer_Configs(
        num_epochs=cfg.num_epochs,
        learning_rate=cfg.lr,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
    )

    if cfg.job_type == "mlm":
        training_cfg.per_device_eval_batch_size = 8
        training_cfg.per_device_train_batch_size = 8
        training_cfg.gradient_accumulation_steps = 4

    if cfg.training_type == "full":
        training_cfg.per_device_train_batch_size = 8
        training_cfg.per_device_eval_batch_size = 16
        training_cfg.gradient_accumulation_steps = 8

    # Env-var overrides win over the defaults above (use for large LoRA runs).
    if cfg.batch_size is not None:
        training_cfg.per_device_train_batch_size = cfg.batch_size
    if cfg.eval_batch_size is not None:
        training_cfg.per_device_eval_batch_size = cfg.eval_batch_size
    if cfg.grad_accum is not None:
        training_cfg.gradient_accumulation_steps = cfg.grad_accum


    preprocessor_cfg = Preprocessor_Configs(
        max_length=256,
        truncation=True,
        padding=False,
        preprocessing_level=cfg.preprocessing_level,
        input_format=cfg.input_format,
        title_review_delimiter_mode=cfg.title_review_delimiter_mode,
    )

    model_short = cfg.model_name.split("/")[-1]
    tag = (
        f"{model_short}_{cfg.training_type}"
        f"_{cfg.preprocessing_level}"
        f"_{cfg.input_format}"
        f"_{cfg.run_tag}"
        f"_lr{cfg.lr:.2e}_ep{cfg.num_epochs}"
    )

    if cfg.job_type == "mlm":
        return MLM_Configs(
            model_name=cfg.model_name,
            data_path=cfg.data_path,
            val_split=cfg.val_split,
            external_val_split=cfg.external_val_split,
            lora_configs=lora_cfg,
            training_configs=training_cfg,
            preprocessor_configs=preprocessor_cfg,
            mlm_probability=cfg.mlm_probability,
            training_type=cfg.training_type,
            saving_dir=Path(f"./mlm_further_pretraining/{tag}"),
            run_name=tag,
        )

    return Fine_Tuning_Configs(
        model_name=cfg.model_name,
        data_path=cfg.data_path,
        val_split=cfg.val_split,
        external_val_split=cfg.external_val_split,
        lora_configs=lora_cfg,
        training_configs=training_cfg,
        preprocessor_configs=preprocessor_cfg,
        training_type=cfg.training_type,
        saving_dir=Path(f"./fine_tuning/{tag}"),
        output_path=Path(f"./outputs/{tag}"),
        num_labels=cfg.num_labels,
    )


def main():
    cfg = get_run_config()

    model_short = cfg.model_name.split("/")[-1]
    full_run_name = (
        cfg.run_name
        if cfg.run_name is not None
        else (
            f"{model_short}_{cfg.training_type}"
            f"_{cfg.preprocessing_level}"
            f"_{cfg.input_format}"
            f"_{cfg.run_tag}"
            f"_lr{cfg.lr:.2e}_ep{cfg.num_epochs}"
        )
    )

    run = None
    os.environ["WANDB_WATCH"] = "gradients"

    run = wandb.init(
        project="cil-project",
        group=cfg.wandb_group,
        tags=[cfg.run_tag, model_short],
        name=full_run_name,
        config=vars(cfg),
        reinit=True,
    )

    try:
        configs = build_default_configs(cfg)

        # Ensure downstream pipeline uses the same descriptive run name
        configs.run_name = full_run_name

        training_pipeline(configs)
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
