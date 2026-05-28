import os
import wandb
import torch
import numpy as np
from os import PathLike
from typing import Union
from dataclasses import dataclass, asdict
from transformers import PreTrainedModel, Trainer, TrainingArguments, DataCollatorWithPadding, DefaultDataCollator
from transformers.trainer_utils import get_last_checkpoint
from deep_learning.preprocessing.preprocessor import Preprocessor
from deep_learning.code_configs import MLM_Configs, Fine_Tuning_Configs


class CustomTrainer(Trainer):
    def __init__(self, *args, loss_type="ce", **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.loss_type == "oll":
            probs = torch.softmax(logits, dim=-1)
            k = torch.arange(logits.shape[-1], device=logits.device).unsqueeze(0)
            dist = (k - labels.unsqueeze(-1)).abs().float() ** 1.5
            loss = (-torch.log(torch.clamp(1 - probs, min=1e-7)) * dist).sum(-1).mean()
        elif self.loss_type == "emd":
            probs = torch.softmax(logits, dim=-1)
            target = torch.zeros_like(probs).scatter_(1, labels.unsqueeze(-1), 1.0)
            loss = (probs.cumsum(-1) - target.cumsum(-1)).abs().sum(-1).mean()
        elif self.loss_type == "soft":
            sigma = 4.0
            k = torch.arange(logits.shape[-1], device=logits.device).unsqueeze(0).float()
            soft_target = torch.exp(-(k - labels.unsqueeze(-1).float()) ** 2 / (2 * sigma ** 2))
            soft_target = soft_target / soft_target.sum(-1, keepdim=True)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            loss = -(soft_target * log_probs).sum(-1).mean()
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss


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

        ckpt_root = os.environ.get("CIL_CHECKPOINT_ROOT")
        if ckpt_root:
            output_dir = f"{ckpt_root}/{self.configs.run_name}"

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
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="score",
            greater_is_better=True,
            gradient_checkpointing=getattr(self.configs, "training_type", "") in ("full", "lora"),
            # --- logging ---
            logging_strategy="steps",
            logging_steps=getattr(tc, "logging_steps", 50),
            logging_first_step=True,
            run_name=getattr(tc, "run_name", None),
            report_to=["wandb"],
            bf16=("deberta-v" not in self.configs.model_name.lower()) and ("geberta" not in self.configs.model_name.lower()),
            optim="adamw_torch_fused",
            dataloader_num_workers=tc.dataloader_num_workers,
        )

        is_mlm = isinstance(self.configs, MLM_Configs)
        self.trainer = CustomTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator,
            processing_class=self.preprocessor.tokenizer,
            compute_metrics=General_Trainer.compute_metrics_mlm if is_mlm else General_Trainer.compute_metrics_finetuning,
            preprocess_logits_for_metrics=(lambda logits, labels: logits.argmax(dim=-1)) if is_mlm else None,
            loss_type=os.environ.get("CIL_LOSS_TYPE", "ce") if not is_mlm else "ce",
        )
        last_ckpt = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
        self.trainer.train(resume_from_checkpoint=last_ckpt)

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
        # predictions already argmax'd by preprocess_logits_for_metrics
        mask = labels != -100
        masked_labels = labels[mask]
        masked_predictions = predictions[mask]
        accuracy = np.mean(masked_predictions == masked_labels)
        return {
            'score': accuracy  # for consistency with training args
        }

