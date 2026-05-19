from transformers import AutoTokenizer
from deep_learning.code_configs import Preprocessor_Configs
import re
import pandas as pd


class Preprocessor:
    _TITLE_REVIEW_SPLIT_RE = re.compile(r"\n\s*\n+", flags=re.MULTILINE)
    _URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
    _REPEATED_PUNCT_RE = re.compile(r"([!?.,;:\-])\1{2,}")
    _WHITESPACE_RE = re.compile(r"\s+")
    _EMOJI_RE = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002600-\U000026FF"
        "\U00002700-\U000027BF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FAFF"
        "\U0000FE00-\U0000FE0F"
        "]+",
        flags=re.UNICODE,
    )

    def __init__(self, configs: Preprocessor_Configs, tokenizer: AutoTokenizer, return_special_tokens_mask):
        self.tokenizer = tokenizer
        self.configs = configs
        self.return_special_tokens_mask = return_special_tokens_mask
        print("Preprocessor config:")
        print("  preprocessing_level =", self.configs.preprocessing_level)
        print("  input_format =", self.configs.input_format)

    def split_title_review(self, text: object):
        raw = "" if pd.isna(text) else str(text)
        parts = self._TITLE_REVIEW_SPLIT_RE.split(raw, maxsplit=1)
        if len(parts) == 2:
            title, review = parts[0].strip(), parts[1].strip()
            if title and review:
                return {"title": title, "review": review, "had_title": True}
        return {"title": "", "review": raw.strip(), "had_title": False}

    def construct_input_text(self, text: object) -> str:
        parts = self.split_title_review(text)

        if self.configs.input_format == "review_only":
            return parts["review"]

        if self.configs.input_format == "title_plus_review":
            if parts["had_title"]:
                return f"Title: {parts['title']} Review: {parts['review']}"
            return f"Review: {parts['review']}"

        raise ValueError(f"Unknown input_format: {self.configs.input_format}")

    def replace_urls_with_token(self, text: str, token: str = "URLTOKEN") -> str:
        return self._URL_RE.sub(f" {token} ", text)

    def cap_extreme_punctuation(self, text: str, max_repeat: int = 2) -> str:
        return self._REPEATED_PUNCT_RE.sub(lambda m: m.group(1) * max_repeat, text)

    def normalize_whitespace(self, text: str) -> str:
        return self._WHITESPACE_RE.sub(" ", text).strip()

    def remove_emojis(self, text: str) -> str:
        return self._EMOJI_RE.sub("", text)

    def preprocess_one_text(self, text: object) -> str:
        text = self.construct_input_text(text)
        level = self.configs.preprocessing_level

        if level == "none":
            return self.normalize_whitespace(text)

        text = self.replace_urls_with_token(text, token="URLTOKEN")

        if level == "light":
            text = text.replace("\r", " ").replace("\n", " ")
            return self.normalize_whitespace(text)

        if level == "medium":
            text = text.replace("\r", " ").replace("\n", " ")
            text = self.cap_extreme_punctuation(text, max_repeat=3)
            return self.normalize_whitespace(text)

        if level == "heavy":
            text = text.replace("\r", " ").replace("\n", " ")
            text = self.cap_extreme_punctuation(text, max_repeat=2)
            text = self.remove_emojis(text)
            text = text.lower()
            return self.normalize_whitespace(text)

        raise ValueError(f"Unknown preprocessing_level: {level}")

    def tokenizing_function(self, batch):
        processed_texts = [self.preprocess_one_text(x) for x in batch["sentence"]]

        return self.tokenizer(
            processed_texts,
            return_special_tokens_mask=self.return_special_tokens_mask,
            truncation=self.configs.truncation,
            max_length=self.configs.max_length,
            padding=self.configs.padding,
        )

    def __call__(self, inputs, test=False):
        inputs = inputs.map(self.tokenizing_function, batched=True)

        if not test and "label" in inputs.column_names and "labels" not in inputs.column_names:
            inputs = inputs.rename_column("label", "labels")

        return inputs