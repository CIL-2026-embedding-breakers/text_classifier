import pandas as pd
import torch
import os
from typing import Union
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification , DataCollatorForLanguageModeling, DataCollatorWithPadding
from datasets import Dataset, concatenate_datasets
from deep_learning.preprocessing.preprocessor import Preprocessor
from deep_learning.analysis.validation_analysis import validation_analysis
from deep_learning.code_configs import MLM_Configs, Fine_Tuning_Configs
from deep_learning.training.trainer import General_Trainer
from peft import LoraConfig, get_peft_model, TaskType



def training_pipeline(configs: Union[MLM_Configs, Fine_Tuning_Configs], susbample: int= None):
    
    if isinstance(configs, MLM_Configs):
        print("Starting MLM further pretraining pipeline...")
        train_type = "mlm"

    elif isinstance(configs, Fine_Tuning_Configs):
        print("Starting fine-tuning pipeline...")
        train_type = "fine_tuning"
    else:
        raise ValueError("Invalid configuration type. Please provide either MLM_Configs or Fine_Tuning_Configs.")
    
    test_df = pd.read_csv(f"{configs.data_path}/test.csv")

    if configs.external_val_split:
        train_df = pd.read_csv(f"{configs.data_path}/train.csv")
        val_df = pd.read_csv(f"{configs.data_path}/val.csv")
        if susbample is not None:
            train_df = train_df.sample(susbample, random_state=42).reset_index(drop=True)
            print(f"Subsampled training data to {susbample} examples.")
    else:
        df_all_train = pd.read_csv(f"{configs.data_path}/train.csv")
        if susbample is not None:
            df_all_train = df_all_train.sample(susbample, random_state=42).reset_index(drop=True)
            print(f"Subsampled training data to {susbample} examples.")
        train_df, val_df = train_test_split(
            df_all_train,
            test_size=configs.val_split,
            random_state=42,
        )

    if train_type == "fine_tuning" and configs.num_labels > 1:
        train_df["label"] = train_df["label"].astype(int)
        val_df["label"] = val_df["label"].astype(int)
    else:
        train_df["label"] = train_df["label"].astype(float)
        val_df["label"] = val_df["label"].astype(float)
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")


    # Load model and tokenizer
    model, tokenizer = initiate_model(configs, MLM= (train_type == "mlm"))

    # Initialize preprocessor
    preprocessor = Preprocessor(configs.preprocessor_configs, tokenizer,  return_special_tokens_mask=(train_type == "mlm"))  

    if train_type == "mlm":
        # Initialize data collator
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=configs.mlm_probability)
    else:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Initialize trainer
    general_trainer = General_Trainer(configs, model, data_collator, preprocessor)

    if train_type == "mlm":
        # Load datasets 
        fsplit = train_dataset.train_test_split(test_size=0.01, seed=42)
        train_dataset = concatenate_datasets([val_dataset, test_dataset, fsplit['train']])
        val_dataset = fsplit['test']    

        train_dataset= train_dataset.remove_columns(["id","label"])
        val_dataset= val_dataset.remove_columns(["id","label"])


    
    
    general_trainer.train(train_dataset, val_dataset, output_dir=configs.saving_dir)
    
    if train_type == "fine_tuning":

        os.makedirs(configs.saving_dir, exist_ok=True)
        val_dataset = preprocessor(val_dataset, test = True)
        eval_predictions = general_trainer.trainer.predict(val_dataset)
        predicted_labels_val = eval_predictions.predictions.argmax(-1)

        eval_results = pd.DataFrame({
            "id": val_df["id"],
            "true_label": val_df["label"],
            "predicted_label": predicted_labels_val
        })
        validation_analysis(eval_results, configs.saving_dir)

        # prediction on test set
        test_dataset = preprocessor(test_dataset, test = True)
        predictions_test = general_trainer.trainer.predict(test_dataset)
        predicted_labels_test = predictions_test.predictions.argmax(-1)

        submission = pd.DataFrame({
                "id": test_df["id"],
                "label": predicted_labels_test
            })


        submission.to_csv(f"{configs.saving_dir}/submission.csv", index=False)
        print(f"Submission successfully saved to {configs.saving_dir}/submission.csv")






def initiate_model(configs: Union[MLM_Configs, Fine_Tuning_Configs], MLM=False):

    if MLM:
        model = AutoModelForMaskedLM.from_pretrained(configs.model_name)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(configs.model_name, num_labels=configs.num_labels, torch_dtype=torch.float32,)

    tokenizer = AutoTokenizer.from_pretrained(configs.model_name)


    if configs.training_type == 'lora':
        name_lower = configs.model_name.lower()
        if "deberta-v" in name_lower:
            target_modules = ["query_proj", "value_proj"]
            bias = "none"
        elif "modernbert" in name_lower or "moderngbert" in name_lower:
            target_modules = ["Wqkv", "Wo"]
            bias = "none"
        else:
            target_modules = ["query", "value"]
            bias = "lora_only"
        lora_configs = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=configs.lora_configs.r,
                lora_alpha=configs.lora_configs.alpha,
                target_modules=target_modules,
                lora_dropout=configs.lora_configs.dropout,
                bias=bias,
            )
        model = get_peft_model(model, lora_configs)
    
       

    elif configs.training_type == 'head':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    print_trainable_parameters(model)

    return model, tokenizer


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )