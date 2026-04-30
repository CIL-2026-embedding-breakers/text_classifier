from __future__ import annotations

import re
from typing import Literal

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
_INTENSITY_BANG_RE = re.compile(r"!(?:[!\s]*!)*")
_INTENSITY_QUEST_RE = re.compile(r"\?(?:[?\s]*\?)*")
_OTHER_PUNCT_RUN_RE = re.compile(r"([.,;:\-])\1+")
_APOSTROPHES_RE = re.compile(r"[‘’ʼʹʻ`´]")
_WORD_RE = re.compile(r"\b\w+(?:'\w+)*\b")
_WHITESPACE_RE = re.compile(r"\s+")

_NER_ENTITY_TYPES: frozenset[str] = frozenset({"PRODUCT", "ORG", "PERSON"})
# POS tags inside ORG/PRODUCT spans considered "descriptor" (stripped under
# strip_descriptors). PROPN (brand names) and NUM/X/SYM (model codes) survive.
_DESCRIPTOR_POS: frozenset[str] = frozenset({"ADJ", "NOUN", "ADV"})

_EMOJI_POS_THRESHOLD = 0.30
_EMOJI_NEG_THRESHOLD = -0.30
_EMOJI_TOKENS = {"pos": "<POS_EMOJI>", "neg": "<NEG_EMOJI>", "neu": "<NEU_EMOJI>"}

_ENT_REPLACEMENT_TOKEN: dict[str, str] = {
    "ORG":     "<BRAND>",
    "PRODUCT": "<PRODUCT>",
    "PERSON":  "<PERSON>",
}


def _replace_emojis_with_sentiment_tokens(text: str) -> str:
    from emosent import get_emoji_sentiment_rank

    def _classify(match: re.Match) -> str:
        out = []
        for ch in match.group():
            try:
                rank = get_emoji_sentiment_rank(ch)
            except (KeyError, ValueError):
                rank = None
            score = float(rank["sentiment_score"]) if rank else 0.0
            if score >= _EMOJI_POS_THRESHOLD:
                out.append(_EMOJI_TOKENS["pos"])
            elif score <= _EMOJI_NEG_THRESHOLD:
                out.append(_EMOJI_TOKENS["neg"])
            else:
                out.append(_EMOJI_TOKENS["neu"])
        return " " + " ".join(out) + " "

    return _EMOJI_RE.sub(_classify, text)


class TextPreprocessor:
    """Configurable text preprocessor for review classification.

    Two output modes:

    - ``final_tokenize=True`` returns a stream of lowercase word tokens joined
      by spaces. Use for fasttext language detection (no punct, no emoji).
    - ``final_tokenize=False`` keeps casing, punctuation, and any special
      tokens (``<POS_EMOJI>``, ``<BRAND>``, …) intact. Use for BERT input.
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
        if self.punct_max_repeat is None:
            return text
        n = self.punct_max_repeat
        text = _INTENSITY_BANG_RE.sub(lambda m: "!" * min(n, m.group().count("!")), text)
        text = _INTENSITY_QUEST_RE.sub(lambda m: "?" * min(n, m.group().count("?")), text)
        return _OTHER_PUNCT_RUN_RE.sub(r"\1", text)

    def handle_emojis(self, text: str, lang: str = "en") -> str:
        strat = self.emoji_strategy
        if strat == "keep":
            return text
        if strat == "strip":
            return _EMOJI_RE.sub("", text)
        if strat == "demojize":
            import emoji
            return emoji.demojize(text, language=lang.lower(), delimiters=(" ", " "))
        if strat == "pos_neg_token":
            return _replace_emojis_with_sentiment_tokens(text)
        raise ValueError(f"Unknown emoji_strategy: {strat!r}")

    def _fill(self, chars: list[str], start: int, end: int, fill: str | None = None) -> None:
        chars[start:end] = list(fill) if fill is not None else [" "] * (end - start)

    def _apply_ner_to_doc(self, doc, text: str) -> tuple[str, list[dict]]:
        """Run entity-strategy logic on an already-parsed ``Doc``."""
        chars = list(text)
        removed: list[dict] = []

        brand_token_indices: set[int] = {
            tok.i for ent in doc.ents if ent.label_ in {"ORG", "PRODUCT"} for tok in ent
        }

        # Iterate in reverse char order so token-replacements (which can change
        # the string length) do not invalidate earlier offsets.
        for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
            if ent.label_ not in _NER_ENTITY_TYPES:
                continue

            if ent.label_ == "PERSON":
                if not brand_token_indices:
                    continue
                min_dist = min(
                    abs(tok.i - anchor) for tok in ent for anchor in brand_token_indices
                )
                if min_dist > self.person_org_proximity:
                    continue
                fill = _ENT_REPLACEMENT_TOKEN["PERSON"] if self.entity_strategy == "replace_with_token" else None
                self._fill(chars, ent.start_char, ent.end_char, fill)
                removed.append({"text": ent.text, "label": "PERSON", "removed_tokens": [ent.text]})
                continue

            if self.entity_strategy == "replace_with_token":
                self._fill(chars, ent.start_char, ent.end_char, _ENT_REPLACEMENT_TOKEN[ent.label_])
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

    def _apply_ner(self, text: str, nlp) -> tuple[str, list[dict]]:
        return self._apply_ner_to_doc(nlp(text), text)

    def _pre_ner(self, text: str, lang: str) -> str:
        text = text.replace("\n", " ").replace("\r", " ")
        if self.strip_html:
            text = _HTML_RE.sub(" ", text)
        if self.remove_urls:
            text = _URL_RE.sub(" ", text)
        text = self.handle_emojis(text, lang=lang)
        return self.normalize_punctuation(text)

    def _post_ner(self, text: str) -> str:
        text = _APOSTROPHES_RE.sub("'", text)
        if self.final_tokenize:
            words = _WORD_RE.findall(text.lower() if self.lowercase else text)
            return " ".join(words) if words else text.strip()
        cleaned = _WHITESPACE_RE.sub(" ", text).strip()
        return cleaned.lower() if self.lowercase else cleaned

    def preprocess(self, text: str, nlp=None, lang: str = "en") -> str:
        return self.preprocess_with_ner_diff(text, nlp=nlp, lang=lang)[0]

    def preprocess_with_ner_diff(
        self, text: str, nlp=None, lang: str = "en"
    ) -> tuple[str, list[dict]]:
        text = self._pre_ner(text, lang)
        ner_removed: list[dict] = []
        if self.entity_strategy != "keep" and nlp is not None:
            text, ner_removed = self._apply_ner(text, nlp)
        return self._post_ner(text), ner_removed

    def preprocess_batch(
        self, texts: list[str], nlp=None, lang: str = "en", ner_batch_size: int = 64
    ) -> list[str]:
        """Same pipeline as :meth:`preprocess` but uses ``nlp.pipe`` for the
        NER step — 5-10× faster on large inputs because spaCy can amortise
        model overhead across the batch.
        """
        pre = [self._pre_ner(t, lang) for t in texts]
        if self.entity_strategy != "keep" and nlp is not None:
            docs = nlp.pipe(pre, batch_size=ner_batch_size)
            after_ner = [self._apply_ner_to_doc(doc, raw)[0] for doc, raw in zip(docs, pre)]
        else:
            after_ner = pre
        return [self._post_ner(t) for t in after_ner]
