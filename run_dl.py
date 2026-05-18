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
        job_type="fine_tuning",              # "mlm" or "fine_tuning"

        # Model
        model_name="xlm-roberta-base",
        
        training_type="lora",  # "lora", "full", or "head" (only for fine-tuning)
        # Optimization
        lr=2e-5,
        num_epochs=3,
        warmup_steps=1000,
        weight_decay=0.01,
        mlm_probability=0.15,  # Only relevant for MLM further pretraining
        # Data / task-specific
        val_split=0.1,
        num_labels=5,  

        run_name="trial_run_fine_tune_xlm_roberta",  # Optional: set a custom run name for W&B
    )



def build_default_configs(cfg):
    # LoRA config (only if training_type == 'lora')
    if cfg.training_type == "lora":
        lora_cfg = LoRa_Configs()
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
    
    preprocessor_cfg = Preprocessor_Configs(max_length=256, truncation=True, padding=False)

    model_short = cfg.model_name.split("/")[-1]
    tag = f"{model_short}_{cfg.training_type}_lr{cfg.lr:.2e}_ep{cfg.num_epochs}"

    if cfg.job_type == "mlm":
        return MLM_Configs(
            model_name=cfg.model_name,
            data_path="./data/",
            val_split=cfg.val_split,
            lora_configs=lora_cfg,
            training_configs=training_cfg,
            preprocessor_configs=preprocessor_cfg,
            mlm_probability=cfg.mlm_probability,
            training_type=cfg.training_type,
            saving_dir=Path(f"./mlm_further_pretraining/{tag}"),
            run_name=f"mlm_further_pretraining_{model_short}_{cfg.training_type}_run",
        )

    return Fine_Tuning_Configs(
        model_name=cfg.model_name,
        data_path="./data/",
        val_split=cfg.val_split,
        lora_configs=lora_cfg,
        training_configs=training_cfg,
        preprocessor_configs=preprocessor_cfg,
        training_type=cfg.training_type,
        saving_dir=Path(f"./fine_tuning/{tag}"),
        num_labels=cfg.num_labels
    )


def main():
    cfg = get_run_config()

    # Optional: keep wandb logging, just without sweeps.
    # Set USE_WANDB=0 to disable entirely, or run `wandb offline` / `wandb disabled`.
    run = None

    os.environ["WANDB_WATCH"] = "gradients"
    run = wandb.init(
        project="cil-text-classification",
        name=cfg.run_name,
        config=vars(cfg),
        reinit=True,
    )

    try:
        configs = build_default_configs(cfg)

        run_id = run.id if run is not None else "manual"
        configs.run_name = (
            f"{cfg.model_name.split('/')[-1]}_{cfg.training_type}_lr{cfg.lr:.2e}_ep{cfg.num_epochs}"
        )

        training_pipeline(configs)  
    finally:
        if run is not None:
            run.finish()

if __name__ == "__main__":
    main()