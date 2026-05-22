from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


@dataclass
class General_Trainer_Configs:
    num_epochs: int = 3
    per_device_train_batch_size: int = 32
    gradient_accumulation_steps: int = 2
    per_device_eval_batch_size: int = 64
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    lr_scheduler_type: str = "cosine"
    metric_for_best_model: str = "score"
    greater_is_better: bool = True
    report_to: str = "wandb"
    dataloader_num_workers: int = 3

    # logging / wandb
    logging_steps: int = 50
    run_name: Optional[str] = None


@dataclass
class Preprocessor_Configs:
    preprocessing_level: str = "light"
    input_format: str = "review_only"
    max_length: int = 128
    truncation: bool = True
    padding: bool = False


@dataclass
class LoRa_Configs:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.1


@dataclass
class MLM_Configs:
    model_name: str     
    data_path: Path = Path("./data/")   
    val_split: float = 0.1     
    external_val_split: bool = False
    lora_configs: LoRa_Configs = field(default_factory=LoRa_Configs)
    training_configs: General_Trainer_Configs = field(default_factory=General_Trainer_Configs)
    preprocessor_configs: Preprocessor_Configs = field(default_factory=Preprocessor_Configs)
    mlm_probability: float = 0.15
    training_type: Literal['lora', 'full'] = 'lora'
    saving_dir: Path = Path("./mlm_further_pretraining")
    run_name: Optional[str] = 'mlm_further_pretraining_run'


@dataclass
class Fine_Tuning_Configs:
    model_name: str 
    data_path: Path = Path("./data/") 
    val_split: float = 0.1          
    external_val_split: bool = False
    lora_configs: LoRa_Configs = field(default_factory=LoRa_Configs)
    training_configs: General_Trainer_Configs = field(default_factory=General_Trainer_Configs)
    preprocessor_configs: Preprocessor_Configs = field(default_factory=Preprocessor_Configs)
    training_type: Literal['lora', 'full', 'head'] = 'lora'
    saving_dir: Path = Path("./fine_tuning")
    output_path: Path = Path("./outputs")  
    num_labels: int = 5
       
   