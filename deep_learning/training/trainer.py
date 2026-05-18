import os
import wandb
import numpy as np
from os import PathLike
from typing import Union
from dataclasses import dataclass, asdict
from transformers import PreTrainedModel, Trainer, TrainingArguments, DataCollatorWithPadding, DefaultDataCollator
from deep_learning.preprocessing.preprocessor import Preprocessor
from deep_learning.code_configs import MLM_Configs, Fine_Tuning_Configs
    

class General_Trainer():
    def __init__(self, configs: Union[MLM_Configs, Fine_Tuning_Configs], model: PreTrainedModel, data_collator: DefaultDataCollator, 
                 preprocessor= Preprocessor):
        self.configs = configs
        self.model = model
        self.data_collator = data_collator
        self.preprocessor = preprocessor


    def train(self, train_dataset, eval_dataset = None, output_dir="./results"):
        train_dataset = self.preprocessor(train_dataset)
        
        if eval_dataset is not None:
            eval_dataset = self.preprocessor(eval_dataset)

        tc = self.configs.training_configs
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=tc.num_epochs,
            per_device_train_batch_size=tc.per_device_train_batch_size,
            gradient_accumulation_steps=tc.gradient_accumulation_steps,
            per_device_eval_batch_size=tc.per_device_eval_batch_size,
            learning_rate=tc.learning_rate,
            weight_decay=tc.weight_decay,
            warmup_steps=tc.warmup_steps,
            lr_scheduler_type=tc.lr_scheduler_type,
            # --- saving / eval ---
            eval_strategy="steps",
            eval_steps=1000,  
            save_strategy="no",              # <-- don't save checkpoints
            # save_total_limit=1,            # <-- not needed when not saving
            load_best_model_at_end=False,    # <-- must be False when save_strategy="no"
            # metric_for_best_model='score', # <-- only relevant with load_best_model_at_end
            # greater_is_better=True,
            # --- logging ---
            logging_strategy="steps",
            logging_steps=getattr(tc, "logging_steps", 50),
            logging_first_step=True,
            run_name=getattr(tc, "run_name", None),
            report_to=["wandb"],
            bf16=True,
            dataloader_num_workers=tc.dataloader_num_workers,
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator,
            processing_class=self.preprocessor.tokenizer,
            compute_metrics=General_Trainer.compute_metrics_finetuning if isinstance(self.configs, Fine_Tuning_Configs) else General_Trainer.compute_metrics_mlm
        )
        self.trainer.train()

        self.trainer.save_model(f"{os.environ.get('SAVE_MODEL_DIR')}/{self.configs.run_name}")  # Save the final model after training



    def compute_metrics_finetuning(eval_pred):
        """
        Custom metric computation:
        L(y_hat, y) = 1 - (MAE / 4), plus standard exact-match accuracy.
        """
        predictions, labels = eval_pred
        if predictions.shape[-1] > 1:  # multi-class classification
            predictions = np.argmax(predictions, axis=-1)
        else:
            predictions = predictions.squeeze()  # regression or binary classification
        rounded_preds = np.clip(np.round(predictions), 0, 4)
        mae = np.mean(np.abs(rounded_preds - labels))
        score = 1.0 - (mae / 4.0)
        accuracy = np.mean(rounded_preds == labels)
        return {
            "mae": mae,
            "score": score,
            "accuracy": accuracy
        }

    def compute_metrics_mlm(eval_pred):
        """
        Standard MLM metric: perplexity.
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        mask = labels != -100
        masked_labels = labels[mask]
        masked_predictions = predictions[mask]
        accuracy = np.mean(masked_predictions == masked_labels)
        return {
            'score': accuracy  # for consistency with training args
        }

