from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Compiled patterns — built once at import time
# ---------------------------------------------------------------------------

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
_URL_RE = re.compile(r'https?://\S+|www\.\S+', flags=re.IGNORECASE)
_HTML_RE = re.compile(r'<[^>]+>')
_PUNCT_RUN_RE = re.compile(r'([!?.,;:\-])\1+')
_WORD_RE = re.compile(r'\b\w+\b')

_DE_VARIANTS: frozenset[str] = frozenset({"NDS", "ALS", "BAR", "LB", "GSW"})
_NER_FILTER_TYPES: frozenset[str] = frozenset({"PRODUCT", "ORG", "PERSON"})

# Universal POS tags of tokens inside PRODUCT/ORG spans that carry language
# signal and should be stripped (adjectives, common nouns, adverbs).
# PROPN (brand names) and NUM/X/SYM (model codes) are deliberately kept.
_DESCRIPTOR_POS: frozenset[str] = frozenset({"ADJ", "NOUN", "ADV"})

# spaCy Matcher token-level patterns for spaced / obfuscated URLs that the
# regex pass would miss (e.g. "http : // example . com", "google dot com").
_SPACED_URL_PATTERNS: list[list[dict]] = [
    # "http : // word"
    [
        {"LOWER": {"IN": ["http", "https"]}},
        {"ORTH": ":"},
        {"ORTH": "//"},
        {"IS_ASCII": True},
    ],
    # "www . domain . tld"
    [
        {"LOWER": "www"},
        {"ORTH": "."},
        {"IS_ALPHA": True},
        {"ORTH": "."},
        {"IS_ALPHA": True},
    ],
    # "word dot word" / "word [dot] word"
    [
        {"IS_ALPHA": True},
        {"LOWER": {"IN": ["dot", "[dot]", "(dot)"]}},
        {"IS_ALPHA": True},
    ],
]

# Per-worker cache: vocab object id → Matcher.  Populated lazily on first use
# so that importing this module does not require spaCy to be installed.
_url_matcher_cache: dict[int, Any] = {}


def _get_url_matcher(nlp) -> Any:
    """Return (and cache per vocab) a spaCy Matcher for obfuscated URLs."""
    key = id(nlp.vocab)
    if key not in _url_matcher_cache:
        from spacy.matcher import Matcher  # type: ignore[import-untyped]
        m = Matcher(nlp.vocab)
        m.add("SPACED_URL", _SPACED_URL_PATTERNS)
        _url_matcher_cache[key] = m
    return _url_matcher_cache[key]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TextPreprocessor:
    """Two-stage text preprocessing pipeline.

    **Cleaning pipeline** (applied in :meth:`preprocess` /
    :meth:`preprocess_with_ner_diff`):

    1. Normalise newlines → spaces (fasttext requirement).
    2. Strip HTML/XML tags.
    3. Strip plain URLs with regex.
    4. Detect spaced/obfuscated URLs with spaCy Matcher and strip them.
    5. Strip emojis.
    6. Collapse repeated punctuation (``!!!!`` → ``!``).
    7. **Descriptor-word removal inside PRODUCT/ORG spans** (spaCy NER):
       - Find PRODUCT and ORG entity spans.
       - Within each span erase only tokens whose POS tag is in
         ``_DESCRIPTOR_POS`` (ADJ, NOUN, ADV) — generic language-bearing
         words such as "wireless", "keyboard", "stainless".
       - Brand names (PROPN) and model codes (NUM, X, SYM) are kept intact.
       - PERSON entities near a brand span are removed entirely (they are
         typically spaCy mislabellings of product model-code context).
    8. Extract word tokens (Unicode letters + digits) and optionally lowercase.

    Every step is independently toggleable for ablation studies.

    Parameters
    ----------
    lowercase
        Lower-case the output tokens.
    strip_emojis
        Remove emoji / pictograph characters.
    collapse_punctuation
        Collapse runs of identical punctuation.
    remove_urls
        Strip ``http``/``https``/``www`` URLs with regex.
    strip_html
        Strip HTML/XML tags.
    filter_entities
        Apply entity-aware descriptor removal (requires an *nlp* model at
        call time).  Falls back to fasttext per-token scoring when *nlp* is
        ``None`` and *model* is provided.
    use_spacy_ner
        Prefer spaCy NER over fasttext token scoring.
    entity_score_threshold
        Combined EN+DE probability threshold for the fasttext fallback.
    ner_entity_types
        spaCy entity label types to process.
        Default: ``{"PRODUCT", "ORG", "PERSON"}``.
    person_org_proximity
        Maximum token distance between a PERSON span and the nearest
        ORG/PRODUCT token for the PERSON to be removed entirely.
    """

    def __init__(
        self,
        lowercase: bool = True,
        strip_emojis: bool = True,
        collapse_punctuation: bool = True,
        remove_urls: bool = True,
        strip_html: bool = True,
        filter_entities: bool = True,
        use_spacy_ner: bool = True,
        entity_score_threshold: float = 0.10,
        ner_entity_types: frozenset[str] | None = None,
        person_org_proximity: int = 2,
    ) -> None:
        self.lowercase = lowercase
        self.strip_emojis = strip_emojis
        self.collapse_punctuation = collapse_punctuation
        self.remove_urls = remove_urls
        self.strip_html = strip_html
        self.filter_entities = filter_entities
        self.use_spacy_ner = use_spacy_ner
        self.entity_score_threshold = entity_score_threshold
        self.ner_entity_types = (
            ner_entity_types if ner_entity_types is not None else _NER_FILTER_TYPES
        )
        self.person_org_proximity = person_org_proximity

    # ------------------------------------------------------------------
    # Individual step methods
    # ------------------------------------------------------------------

    def remove_emojis(self, text: str) -> str:
        return _EMOJI_RE.sub("", text)

    def strip_html_tags(self, text: str) -> str:
        return _HTML_RE.sub(" ", text)

    def strip_urls(self, text: str) -> str:
        return _URL_RE.sub(" ", text)

    def normalize_punctuation(self, text: str) -> str:
        return _PUNCT_RUN_RE.sub(r"\1", text)

    def extract_words(self, text: str) -> list[str]:
        source = text.lower() if self.lowercase else text
        return _WORD_RE.findall(source)

    def is_linguistic_token(self, token: str, model) -> bool:
        """Return True if *token* carries EN/DE signal (fasttext fallback)."""
        if len(token) < 2:
            return False
        labels, probs = model.predict(token.lower(), k=5)
        en_de_langs = {"EN", "DE"} | _DE_VARIANTS
        score = sum(
            float(p)
            for lbl, p in zip(labels, probs)
            if lbl.replace("__label__", "").upper() in en_de_langs
        )
        return score >= self.entity_score_threshold

    # ------------------------------------------------------------------
    # Core NER + obfuscated-URL step
    # ------------------------------------------------------------------

    def _fill_text_array(self, text : list[str], idx_start, idx_end, fill_char =" ", fill_str= None):
        """Helper to replace a span of characters in *text* with spaces or a
        custom string."""
        if fill_str is not None:
            return text[:idx_start] + list(fill_str) + text[idx_end:]
        else:
            return text[:idx_start] + [fill_char] * (idx_end - idx_start) + text[idx_end:]

    def _apply_ner(self, text: str, nlp) -> tuple[str, list[dict]]:
        """Run NER-based descriptor removal and obfuscated-URL stripping.

        Returns ``(cleaned_text, removed_entries)`` where each entry is::

            {
                "text":           <full entity / URL span text>,
                "label":          <NER label or "SPACED_URL">,
                "removed_tokens": <list of token strings that were blanked>,
            }
        """
        doc = nlp(text)
        chars = list(text)
        removed: list[dict] = []

        # 1. Spaced / obfuscated URLs ----------------------------------------
        matcher = _get_url_matcher(nlp)
        for _, start, end in matcher(doc):
            span_text = doc[start:end].text
            span_tokens = [tok.text for tok in doc[start:end]]
            for tok in doc[start:end]:
                self._fill_text_array(chars, tok.idx, tok.idx + len(tok.text), fill_char=" ")
                #TODO : Here define a per language fillchar /.fill string to preserve info that an URL was here but 
                #       Make it Langugage neutral.
            
            removed.append({
                "text": span_text,
                "label": "SPACED_URL",
                "removed_tokens": span_tokens,
            })

        # 2. Brand-entity token indices (anchor for PERSON proximity) ---------
        brand_token_indices: set[int] = {
            tok.i
            for ent in doc.ents
            if ent.label_ in {"ORG", "PRODUCT"}
            for tok in ent
        }

        # 3. Entity-aware descriptor removal ----------------------------------
        for ent in doc.ents:
            if ent.label_ not in self.ner_entity_types:
                continue

            if ent.label_ == "PERSON":
                # Keep PERSON unless it sits within person_org_proximity tokens
                # of a brand entity (likely a mislabelled product-code token).
                if not brand_token_indices:
                    continue
                min_dist = min(
                    abs(tok.i - anchor)
                    for tok in ent
                    for anchor in brand_token_indices
                )
                if min_dist > self.person_org_proximity:
                    continue
                
                self._fill_text_array(chars, ent.start_char, ent.end_char)

                removed.append({
                    "text": ent.text,
                    "label": "PERSON",
                    "removed_tokens": [ent.text],
                })

            else:
                # PRODUCT / ORG: remove only descriptor tokens (ADJ, NOUN, ADV).
                # Brand names (PROPN) and model codes (NUM, X, SYM) are kept.
                span_removed: list[str] = []
                for token in ent:
                    if token.pos_ in _DESCRIPTOR_POS:
                        for i in range(token.idx, token.idx + len(token.text)):
                            chars[i] = " "
                        span_removed.append(token.text)
                if span_removed:
                    removed.append({
                        "text": ent.text,
                        "label": ent.label_,
                        "removed_tokens": span_removed,
                    })

        return "".join(chars), removed

    def remove_named_entities(self, text: str, nlp) -> str:
        """Convenience wrapper — returns only the cleaned string."""
        cleaned, _ = self._apply_ner(text, nlp)
        return cleaned

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def preprocess(self, text: str, model=None, nlp=None) -> str:
        """Clean *text* and return a single-line string for language detection.

        Pipeline: newline normalisation → HTML → URL (regex) → emoji →
        punctuation → NER descriptor removal (spaCy) or token scoring
        (fasttext fallback) → word extraction.
        """
        preprocessed, _ = self.preprocess_with_ner_diff(text, model, nlp)
        return preprocessed

    def preprocess_with_ner_diff(
        self, text: str, model=None, nlp=None
    ) -> tuple[str, list[dict]]:
        """Clean *text* and also return details of every NER removal.

        Returns
        -------
        preprocessed_text
            The cleaned, single-line string ready for fasttext.
        ner_removed
            List of removal entries (see :meth:`_apply_ner` for structure).
            Empty list when NER was not applied or nothing was removed.
        """
        ner_removed: list[dict] = []

        # fasttext rejects strings containing \\n
        text = text.replace("\n", " ").replace("\r", " ")
        if self.strip_html:
            text = self.strip_html_tags(text)
        if self.remove_urls:
            text = self.strip_urls(text)
        if self.strip_emojis:
            text = self.remove_emojis(text)
        if self.collapse_punctuation:
            text = self.normalize_punctuation(text)
        if self.filter_entities:
            if self.use_spacy_ner and nlp is not None:
                text, ner_removed = self._apply_ner(text, nlp)
            elif model is not None:
                tokens = _WORD_RE.findall(text)
                text = " ".join(
                    t for t in tokens if self.is_linguistic_token(t, model)
                )
        words = self.extract_words(text)
        preprocessed = (
            " ".join(words)
            if words
            else (text.lower().strip() if self.lowercase else text.strip())
        )
        return preprocessed, ner_removed
