import re
from typing import Literal

import emoji
from emosent import get_emoji_sentiment_rank

EmojiStrategy = Literal["keep", "strip", "demojize", "pos_neg_token"]
EntityStrategy = Literal["keep", "strip_descriptors", "replace_with_token"]


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
_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_HTML_RE = re.compile(r"<[^>]+>")
_BANG_RE = re.compile(r"!(?:[!\s]*!)*")
_QUEST_RE = re.compile(r"\?(?:[?\s]*\?)*")
_PUNCT_RUN_RE = re.compile(r"([.,;:\-])\1+")
_APOSTROPHES_RE = re.compile(r"[‘’ʼʹʻ`´]")
_WORD_RE = re.compile(r"\b\w+(?:'\w+)*\b")
_WHITESPACE_RE = re.compile(r"\s+")

# spaCy entity labels we act on, and the POS tags considered "descriptors"
# (adjectives/nouns/adverbs) inside an ORG/PRODUCT span.
_ENTITY_TYPES = {"PRODUCT", "ORG", "PERSON"}
_DESCRIPTOR_POS = {"ADJ", "NOUN", "ADV"}
_ENTITY_TOKEN = {"ORG": "<BRAND>", "PRODUCT": "<PRODUCT>", "PERSON": "<PERSON>"}

# Emoji sentiment thresholds for the pos_neg_token strategy.
_POS_THRESHOLD = 0.30
_NEG_THRESHOLD = -0.30
_EMOJI_TOKENS = {"pos": "<POS_EMOJI>", "neg": "<NEG_EMOJI>", "neu": "<NEU_EMOJI>"}


def _emoji_to_sentiment_tokens(text: str) -> str:
    """Replace each emoji with <POS_EMOJI>/<NEG_EMOJI>/<NEU_EMOJI> based on its
    emosent sentiment score."""
    def classify(match: re.Match) -> str:
        tokens = []
        for ch in match.group():
            rank = get_emoji_sentiment_rank(ch)
            score = float(rank["sentiment_score"]) if rank else 0.0
            if score >= _POS_THRESHOLD:
                tokens.append(_EMOJI_TOKENS["pos"])
            elif score <= _NEG_THRESHOLD:
                tokens.append(_EMOJI_TOKENS["neg"])
            else:
                tokens.append(_EMOJI_TOKENS["neu"])
        return " " + " ".join(tokens) + " "

    return _EMOJI_RE.sub(classify, text)


class TextPreprocessor:
    """Configurable text cleaner for the review corpus.

    With final_tokenize=True the output is a stream of lowercase word tokens
    (good for bag-of-words / language detection). With final_tokenize=False the
    casing, punctuation and any special tokens (<POS_EMOJI>, <BRAND>, ...) are
    kept, which is what we feed to the transformer models.
    """

    def __init__(
        self,
        lowercase: bool = True,
        emoji_strategy: EmojiStrategy = "strip",
        punct_max_repeat: int | None = 3,
        remove_urls: bool = True,
        strip_html: bool = True,
        entity_strategy: EntityStrategy = "strip_descriptors",
        person_org_proximity: int = 2,
        final_tokenize: bool = True,
    ) -> None:
        self.lowercase = lowercase
        self.emoji_strategy = emoji_strategy
        self.punct_max_repeat = punct_max_repeat
        self.remove_urls = remove_urls
        self.strip_html = strip_html
        self.entity_strategy = entity_strategy
        self.person_org_proximity = person_org_proximity
        self.final_tokenize = final_tokenize

    def normalize_punctuation(self, text: str) -> str:
        """Cap runs of repeated punctuation at punct_max_repeat (e.g. '!!!!!' -> '!!!')."""
        if self.punct_max_repeat is None:
            return text
        n = self.punct_max_repeat
        text = _BANG_RE.sub(lambda m: "!" * min(n, m.group().count("!")), text)
        text = _QUEST_RE.sub(lambda m: "?" * min(n, m.group().count("?")), text)
        return _PUNCT_RUN_RE.sub(r"\1", text)

    def handle_emojis(self, text: str, lang: str = "en") -> str:
        if self.emoji_strategy == "keep":
            return text
        if self.emoji_strategy == "strip":
            return _EMOJI_RE.sub("", text)
        if self.emoji_strategy == "demojize":
            return emoji.demojize(text, language=lang.lower(), delimiters=(" ", " "))
        if self.emoji_strategy == "pos_neg_token":
            return _emoji_to_sentiment_tokens(text)
        raise ValueError(f"Unknown emoji_strategy: {self.emoji_strategy!r}")

    def normalize_for_lexical_matching(self, text: str) -> str:
        """Lowercase, unify apostrophe variants and collapse whitespace so that
        word lists (e.g. negation words) match regardless of the apostrophe or
        spacing used in the review. Meant to be used with \\b...\\b regexes."""
        if not isinstance(text, str):
            return ""
        text = _APOSTROPHES_RE.sub("'", text.lower())
        return _WHITESPACE_RE.sub(" ", text).strip()

    def _fill(self, chars: list[str], start: int, end: int, fill: str | None = None) -> None:
        chars[start:end] = list(fill) if fill is not None else [" "] * (end - start)

    def _apply_ner(self, text: str, nlp) -> tuple[str, list[dict]]:
        """Strip or replace named entities. ORG/PRODUCT descriptors are removed
        (brand names and model codes survive); PERSON names are only removed when
        close to a brand/product mention."""
        doc = nlp(text)
        chars = list(text)
        removed: list[dict] = []

        brand_tokens = {
            tok.i for ent in doc.ents if ent.label_ in {"ORG", "PRODUCT"} for tok in ent
        }

        # Go right-to-left so token replacements don't shift earlier offsets.
        for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
            if ent.label_ not in _ENTITY_TYPES:
                continue

            if ent.label_ == "PERSON":
                if not brand_tokens:
                    continue
                min_dist = min(abs(tok.i - anchor) for tok in ent for anchor in brand_tokens)
                if min_dist > self.person_org_proximity:
                    continue
                fill = _ENTITY_TOKEN["PERSON"] if self.entity_strategy == "replace_with_token" else None
                self._fill(chars, ent.start_char, ent.end_char, fill)
                removed.append({"text": ent.text, "label": "PERSON", "removed_tokens": [ent.text]})
                continue

            if self.entity_strategy == "replace_with_token":
                self._fill(chars, ent.start_char, ent.end_char, _ENTITY_TOKEN[ent.label_])
                removed.append({"text": ent.text, "label": ent.label_, "removed_tokens": [ent.text]})
            else:
                span_removed = []
                for token in ent:
                    if token.pos_ in _DESCRIPTOR_POS:
                        self._fill(chars, token.idx, token.idx + len(token.text))
                        span_removed.append(token.text)
                if span_removed:
                    removed.append({"text": ent.text, "label": ent.label_, "removed_tokens": span_removed})

        return "".join(chars), removed

    def preprocess(self, text: str, nlp=None, lang: str = "en") -> str:
        return self.preprocess_with_ner_diff(text, nlp=nlp, lang=lang)[0]

    def preprocess_with_ner_diff(self, text: str, nlp=None, lang: str = "en") -> tuple[str, list[dict]]:
        """Run the full cleaning pipeline. Returns (cleaned_text, removed_entities)."""
        ner_removed: list[dict] = []
        text = text.replace("\n", " ").replace("\r", " ")
        if self.strip_html:
            text = _HTML_RE.sub(" ", text)
        if self.remove_urls:
            text = _URL_RE.sub(" ", text)
        text = self.handle_emojis(text, lang=lang)
        text = self.normalize_punctuation(text)
        if self.entity_strategy != "keep" and nlp is not None:
            text, ner_removed = self._apply_ner(text, nlp)

        text = _APOSTROPHES_RE.sub("'", text)
        if self.final_tokenize:
            words = _WORD_RE.findall(text.lower() if self.lowercase else text)
            cleaned = " ".join(words) if words else text.strip()
        else:
            cleaned = _WHITESPACE_RE.sub(" ", text).strip()
            if self.lowercase:
                cleaned = cleaned.lower()
        return cleaned, ner_removed
