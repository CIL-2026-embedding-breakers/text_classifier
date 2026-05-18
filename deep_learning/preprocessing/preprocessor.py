from transformers import AutoTokenizer
from deep_learning.code_configs import Preprocessor_Configs

class Preprocessor():
    def __init__(self, configs: Preprocessor_Configs, tokenizer: AutoTokenizer, return_special_tokens_mask):
        self.tokenizer = tokenizer
        self.configs = configs
        self.return_special_tokens_mask = return_special_tokens_mask


    def __call__(self, inputs, test = False):
        #TODO add potential preprocessing steps
        # duplicates handling
        # Title and corpus formatting
        # URL removing
        # ...
        
        self.tokenizing_function = lambda x: self.tokenizer(
                                                x["sentence"],
                                                return_special_tokens_mask = self.return_special_tokens_mask,
                                                truncation=self.configs.truncation,
                                                max_length=self.configs.max_length,
                                                padding=self.configs.padding
                                                )
        

        inputs = inputs.map(self.tokenizing_function, batched=True)

        if not test and "label" in inputs.column_names and "labels" not in inputs.column_names:
            inputs = inputs.rename_column("label", "labels")
        
        return inputs

        
