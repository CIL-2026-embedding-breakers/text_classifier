
import os

import wandb
from pathlib import Path
from deep_learning.training.training_pipeline import training_pipeline
from deep_learning.code_configs import MLM_Configs, Fine_Tuning_Configs, LoRa_Configs, General_Trainer_Configs, Preprocessor_Configs

def main():
    # wandb.init() is called by the agent; config is injected
    run = wandb.init(name="sweep_run", project="cil-text-classification-scaling", reinit=True)
    try:
        sweep_cfg = wandb.config

        os.environ["WANDB_WATCH"] = "gradients"

        configs = build_default_configs(sweep_cfg)  # however you normally construct it

        # Make run names unique per sweep run so they don't collide
        run.name = f"{sweep_cfg.model_name.split('/')[-1]}_{sweep_cfg.training_type}_lr{sweep_cfg.lr:.2e}_ep{sweep_cfg.num_epochs}"
        configs.run_name = run.name

        training_pipeline(configs)
    finally:
        run.finish()


def build_default_configs(sweep_cfg):
    # Build your default configs here
    if sweep_cfg.training_type == 'lora':
        lora_cfg = LoRa_Configs()
    else:
        lora_cfg = None
    training_cfg = General_Trainer_Configs(num_epochs=sweep_cfg.num_epochs, learning_rate=sweep_cfg.lr, warmup_steps=sweep_cfg.warmup_steps, weight_decay=sweep_cfg.weight_decay)
    preprocessor_cfg = Preprocessor_Configs(max_length=256)

    if sweep_cfg.job_type == 'mlm':
        mlm_cfg = MLM_Configs(
            model_name=sweep_cfg.model_name,
            data_path="./data/",
            val_split=0.1,
            lora_configs=lora_cfg,
            training_configs=training_cfg,
            preprocessor_configs=preprocessor_cfg,
            mlm_probability=sweep_cfg.mlm_probability,
            training_type=sweep_cfg.training_type,
            saving_dir=Path(f"./mlm_further_pretraining/{sweep_cfg.model_name.split('/')[-1]}_{sweep_cfg.training_type}_lr{sweep_cfg.lr:.2e}_ep{sweep_cfg.num_epochs}"),
            run_name=f"mlm_further_pretraining_{sweep_cfg.model_name.split('/')[-1]}_{sweep_cfg.training_type}_run"
        )
        return mlm_cfg


    fine_tuning_cfg = Fine_Tuning_Configs(
        model_name=sweep_cfg.model_name,
        data_path="./data/",
        val_split=sweep_cfg.val_split,
        lora_configs=lora_cfg,
        training_configs=training_cfg,
        preprocessor_configs=preprocessor_cfg,
        training_type=sweep_cfg.training_type,
        saving_dir=Path(f"./fine_tuning/{sweep_cfg.model_name.split('/')[-1]}_{sweep_cfg.training_type}_lr{sweep_cfg.lr:.2e}_ep{sweep_cfg.num_epochs}"),
        num_labels=sweep_cfg.num_labels
    )

    return fine_tuning_cfg


if __name__ == "__main__":
    main()