"""Statistical analysis pipeline for the multilingual review corpus.

Designed to be used after language splitting.  All methods return plain
DataFrames so results can be piped directly into matplotlib/seaborn.

spaCy is optional: methods that need it raise ``ValueError`` when
``TextAnalyzer`` was constructed without an *nlp* model.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level compiled patterns (mirrors text_preprocessing but standalone)
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
    "]+",
    flags=re.UNICODE,
)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[A-Za-z]{2,}\b")
_NUMBER_RE = re.compile(r"\b\d+[.,]?\d*\b")
_HTML_RE = re.compile(r"<[^>]+>")
_PUNCT_RUN_RE = re.compile(r"([!?.,;:\-])\1+")
_WORD_RE = re.compile(r"\b\w+\b")

# Regex to parse the "_ner_removed" column written by language_detection.py
# Format: "span text [LABEL: removed=tok1,tok2]" or "span text [LABEL]"
_NER_REMOVED_DESCRIPTORS_RE = re.compile(r"removed=([^\]]+)\]")
_NER_REMOVED_LABEL_RE = re.compile(r"\[([A-Z_]+)(?::|])")


class TextAnalyzer:
    """Statistical analysis of a multilingual review corpus.

    Parameters
    ----------
    nlp
        Optional loaded spaCy ``Language`` pipeline.  Required for lemma,
        POS, morphology, and entity analysis.  Pass ``None`` to use only
        regex-based statistics.
    sample_size
        If set, methods that run spaCy on every row will work on a random
        sample of this size to keep runtimes reasonable.
    random_state
        Seed for reproducible sampling.
    """

    def __init__(
        self,
        nlp=None,
        sample_size: int | None = 5000,
        random_state: int = 42,
    ) -> None:
        self.nlp = nlp
        self.sample_size = sample_size
        self.random_state = random_state

    def _sample(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.sample_size and len(df) > self.sample_size:
            return df.sample(self.sample_size, random_state=self.random_state)
        return df

    def _require_nlp(self, method: str) -> None:
        if self.nlp is None:
            raise ValueError(
                f"{method} requires TextAnalyzer(nlp=<loaded spaCy model>)."
            )

    # ------------------------------------------------------------------
    # 1. Raw token stats
    # ------------------------------------------------------------------

    def token_stats(
        self,
        df: pd.DataFrame,
        text_col: str,
        group_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Token count statistics (total, unique, avg/median/p95) per group.

        Parameters
        ----------
        group_cols
            Column names to group by (e.g. ``["lang", "label"]``).
        """
        def _stats(texts: pd.Series) -> dict:
            lengths = texts.fillna("").str.split().str.len()
            all_toks = [t for s in texts.fillna("") for t in s.split()]
            return {
                "n_docs": len(texts),
                "total_tokens": len(all_toks),
                "unique_tokens": len({t.lower() for t in all_toks}),
                "avg_tokens": round(float(lengths.mean()), 2),
                "median_tokens": float(lengths.median()),
                "p95_tokens": float(lengths.quantile(0.95)),
            }

        if group_cols:
            rows = []
            for keys, grp in df.groupby(group_cols):
                row = _stats(grp[text_col])
                keys_iter = keys if isinstance(keys, tuple) else (keys,)
                for col, val in zip(group_cols, keys_iter):
                    row[col] = val
                rows.append(row)
            result = pd.DataFrame(rows)
            return result.set_index(group_cols) if len(group_cols) == 1 else result
        return pd.DataFrame([_stats(df[text_col])])

    # ------------------------------------------------------------------
    # 2. Top tokens
    # ------------------------------------------------------------------

    def top_tokens(
        self,
        df: pd.DataFrame,
        text_col: str,
        group_col: str | None = None,
        n: int = 20,
        stopwords: set[str] | None = None,
    ) -> pd.DataFrame:
        """Most frequent lowercase tokens per group (regex tokenisation).

        Returns a DataFrame with columns ``[group, token, count]`` when
        *group_col* is given, or ``[token, count]`` otherwise.
        """
        sw = stopwords or set()

        def _top(texts: Iterable[str]) -> list[tuple[str, int]]:
            c: Counter = Counter()
            for text in texts:
                if isinstance(text, str):
                    c.update(
                        t for t in _WORD_RE.findall(text.lower())
                        if len(t) > 1 and t not in sw
                    )
            return c.most_common(n)

        if group_col:
            rows = []
            for group, grp in df.groupby(group_col):
                for token, count in _top(grp[text_col]):
                    rows.append({"group": group, "token": token, "count": count})
            return pd.DataFrame(rows)
        return pd.DataFrame(_top(df[text_col]), columns=["token", "count"])

    # ------------------------------------------------------------------
    # 3. Top lemmas  (requires spaCy)
    # ------------------------------------------------------------------

    def top_lemmas(
        self,
        df: pd.DataFrame,
        text_col: str,
        group_col: str | None = None,
        n: int = 20,
        stopwords: set[str] | None = None,
        batch_size: int = 256,
    ) -> pd.DataFrame:
        """Most frequent lemmas per group using spaCy lemmatisation."""
        self._require_nlp("top_lemmas")
        sw = stopwords or set()
        sample = self._sample(df)

        def _lemmas(texts: list[str]) -> Counter:
            c: Counter = Counter()
            for doc in self.nlp.pipe(texts, batch_size=batch_size, disable=["ner"]):
                for tok in doc:
                    lem = tok.lemma_.lower()
                    if not tok.is_punct and not tok.is_space and not tok.is_stop \
                            and len(lem) > 1 and lem not in sw:
                        c[lem] += 1
            return c

        if group_col:
            rows = []
            for group, grp in sample.groupby(group_col):
                for lemma, count in _lemmas(grp[text_col].fillna("").tolist()).most_common(n):
                    rows.append({"group": group, "lemma": lemma, "count": count})
            return pd.DataFrame(rows)
        return pd.DataFrame(
            _lemmas(sample[text_col].fillna("").tolist()).most_common(n),
            columns=["lemma", "count"],
        )

    # ------------------------------------------------------------------
    # 4. Entity stats  (requires spaCy)
    # ------------------------------------------------------------------

    def entity_stats(
        self,
        df: pd.DataFrame,
        text_col: str,
        group_col: str | None = None,
        n_top_spans: int = 20,
        batch_size: int = 256,
    ) -> dict[str, pd.DataFrame]:
        """Entity type counts and top entity spans per group.

        Returns
        -------
        dict with keys ``"type_counts"`` and ``"top_spans"``.
        """
        self._require_nlp("entity_stats")
        sample = self._sample(df)

        def _collect(texts: list[str]) -> tuple[Counter, Counter]:
            types: Counter = Counter()
            spans: Counter = Counter()
            for doc in self.nlp.pipe(texts, batch_size=batch_size):
                for ent in doc.ents:
                    types[ent.label_] += 1
                    spans[ent.text.strip().lower()] += 1
            return types, spans

        if group_col:
            type_rows, span_rows = [], []
            for group, grp in sample.groupby(group_col):
                types, spans = _collect(grp[text_col].fillna("").tolist())
                for label, count in types.items():
                    type_rows.append({"group": group, "entity_type": label, "count": count})
                for span, count in spans.most_common(n_top_spans):
                    span_rows.append({"group": group, "span": span, "count": count})
            return {
                "type_counts": pd.DataFrame(type_rows),
                "top_spans": pd.DataFrame(span_rows),
            }
        types, spans = _collect(sample[text_col].fillna("").tolist())
        return {
            "type_counts": pd.DataFrame(
                sorted(types.items(), key=lambda x: -x[1]),
                columns=["entity_type", "count"],
            ),
            "top_spans": pd.DataFrame(spans.most_common(n_top_spans), columns=["span", "count"]),
        }

    # ------------------------------------------------------------------
    # 5. POS distribution  (requires spaCy)
    # ------------------------------------------------------------------

    def pos_distribution(
        self,
        df: pd.DataFrame,
        text_col: str,
        group_col: str | None = None,
        batch_size: int = 256,
    ) -> pd.DataFrame:
        """Universal POS tag distribution per group."""
        self._require_nlp("pos_distribution")
        sample = self._sample(df)

        def _pos(texts: list[str]) -> Counter:
            c: Counter = Counter()
            for doc in self.nlp.pipe(texts, batch_size=batch_size, disable=["ner"]):
                for tok in doc:
                    if not tok.is_space:
                        c[tok.pos_] += 1
            return c

        if group_col:
            rows = []
            for group, grp in sample.groupby(group_col):
                for pos, count in _pos(grp[text_col].fillna("").tolist()).items():
                    rows.append({"group": group, "pos": pos, "count": count})
            return pd.DataFrame(rows)
        counts = _pos(sample[text_col].fillna("").tolist())
        return pd.DataFrame(sorted(counts.items(), key=lambda x: -x[1]), columns=["pos", "count"])

    # ------------------------------------------------------------------
    # 6. Morphology distribution  (requires spaCy)
    # ------------------------------------------------------------------

    def morphology_distribution(
        self,
        df: pd.DataFrame,
        text_col: str,
        group_col: str | None = None,
        n_top: int = 20,
        batch_size: int = 256,
    ) -> pd.DataFrame:
        """Top morphological features (``Feature=Value``) per group."""
        self._require_nlp("morphology_distribution")
        sample = self._sample(df)

        def _morph(texts: list[str]) -> Counter:
            c: Counter = Counter()
            for doc in self.nlp.pipe(texts, batch_size=batch_size, disable=["ner"]):
                for tok in doc:
                    for feat, val in tok.morph.to_dict().items():
                        c[f"{feat}={val}"] += 1
            return c

        if group_col:
            rows = []
            for group, grp in sample.groupby(group_col):
                for feat, count in _morph(grp[text_col].fillna("").tolist()).most_common(n_top):
                    rows.append({"group": group, "feature": feat, "count": count})
            return pd.DataFrame(rows)
        counts = _morph(sample[text_col].fillna("").tolist())
        return pd.DataFrame(counts.most_common(n_top), columns=["feature", "count"])

    # ------------------------------------------------------------------
    # 7. Sentence-length distribution
    # ------------------------------------------------------------------

    def length_distribution(
        self,
        df: pd.DataFrame,
        text_col: str,
        group_col: str | None = None,
    ) -> pd.DataFrame:
        """Word-count per document, optionally tagged by group.

        Returns a tidy DataFrame with columns ``[group?, length]`` suitable
        for histogram / box-plot calls.
        """
        lengths = df[text_col].fillna("").str.split().str.len().rename("length")
        if group_col:
            return df[[group_col]].join(lengths)
        return lengths.to_frame()

    # ------------------------------------------------------------------
    # 8. Special-token rates
    # ------------------------------------------------------------------

    def special_rates(
        self,
        df: pd.DataFrame,
        text_col: str,
        group_col: str | None = None,
    ) -> pd.DataFrame:
        """Percentage of docs containing URLs, emails, numbers, or emojis."""
        texts = df[text_col].fillna("")
        flags = pd.DataFrame({
            "has_url":    texts.str.contains(_URL_RE, regex=True),
            "has_email":  texts.str.contains(_EMAIL_RE, regex=True),
            "has_number": texts.str.contains(_NUMBER_RE, regex=True),
            "has_emoji":  texts.str.contains(_EMOJI_RE, regex=True),
        })
        if group_col:
            combined = df[[group_col]].join(flags)
            return (
                combined.groupby(group_col).mean()
                .mul(100).round(2)
                .rename(columns=lambda c: c.replace("has_", "") + "_%")
            )
        return (
            flags.mean().mul(100).round(2)
            .rename(index=lambda c: c.replace("has_", "") + "_%")
            .rename("rate_%")
            .to_frame()
        )

    # ------------------------------------------------------------------
    # 9. Rule diagnostics
    # ------------------------------------------------------------------

    def rule_diagnostics(
        self,
        df: pd.DataFrame,
        text_col: str,
    ) -> pd.DataFrame:
        """How many (and what %) of documents each cleaning rule would affect.

        Runs each rule independently against the *raw* text so you can see
        how much each one changes before running the full pipeline.
        """
        texts = df[text_col].fillna("")
        n = len(texts)
        rules = [
            ("strip_html",           texts.str.contains(r"<[^>]+>", regex=True)),
            ("strip_urls",           texts.str.contains(_URL_RE, regex=True)),
            ("strip_emojis",         texts.str.contains(_EMOJI_RE, regex=True)),
            ("collapse_punctuation", texts.str.contains(r"([!?.,;:\-])\1+", regex=True)),
            ("has_newline",          texts.str.contains(r"\n", regex=False)),
        ]
        rows = [
            {
                "rule": name,
                "n_affected": int(mask.sum()),
                "pct_affected": round(float(mask.mean()) * 100, 2),
            }
            for name, mask in rules
        ]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 10. Descriptor-word stats (from NER debug column)
    # ------------------------------------------------------------------

    def descriptor_words_stats(
        self,
        df: pd.DataFrame,
        ner_removed_col: str,
        group_col: str | None = None,
        n: int = 20,
    ) -> pd.DataFrame:
        """Top descriptor tokens removed inside PRODUCT/ORG spans.

        Parses the ``_ner_removed`` column written by
        :func:`language_detection.split_languages_in_csv` when
        ``debug=True``.  Format expected:
        ``"span [LABEL: removed=tok1,tok2]"``.
        """
        def _descriptors(val: str) -> list[str]:
            if not isinstance(val, str):
                return []
            return [
                t.strip().lower()
                for m in _NER_REMOVED_DESCRIPTORS_RE.finditer(val)
                for t in m.group(1).split(",")
                if t.strip()
            ]

        if group_col:
            rows = []
            for group, grp in df.groupby(group_col):
                c: Counter = Counter()
                for val in grp[ner_removed_col].fillna(""):
                    c.update(_descriptors(val))
                for word, count in c.most_common(n):
                    rows.append({"group": group, "descriptor": word, "count": count})
            return pd.DataFrame(rows)
        c: Counter = Counter()
        for val in df[ner_removed_col].fillna(""):
            c.update(_descriptors(val))
        return pd.DataFrame(c.most_common(n), columns=["descriptor", "count"])

    # ------------------------------------------------------------------
    # 11. Before / after cleaning examples
    # ------------------------------------------------------------------

    def before_after_examples(
        self,
        df: pd.DataFrame,
        raw_col: str,
        preprocessed_col: str,
        ner_removed_col: str | None = None,
        n: int = 10,
        only_changed: bool = True,
    ) -> pd.DataFrame:
        """Sample rows showing how the cleaning pipeline changed each text.

        Parameters
        ----------
        only_changed
            When ``True`` (default), return only rows where the text was
            actually modified by preprocessing.
        """
        cols = {raw_col: "raw", preprocessed_col: "preprocessed"}
        if ner_removed_col:
            cols[ner_removed_col] = "ner_removed"
        result = df[list(cols)].rename(columns=cols).copy()
        if only_changed:
            changed = result["raw"].fillna("") != result["preprocessed"].fillna("")
            result = result[changed]
        return result.head(n).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 12. Ablation helpers
    # ------------------------------------------------------------------

    def ablation_configs(self) -> list[dict]:
        """Return a list of ``TextPreprocessor`` kwarg dicts for ablation.

        Each dict disables exactly one preprocessing step relative to the
        full pipeline, plus a ``"name"`` key for display.
        """
        full = {
            "strip_html": True,
            "remove_urls": True,
            "strip_emojis": True,
            "collapse_punctuation": True,
            "filter_entities": True,
            "use_spacy_ner": True,
        }
        configs = [{"name": "full_pipeline", **full}]
        for key in [k for k in full if k != "name"]:
            variant = {**full, key: False, "name": f"no_{key}"}
            configs.append(variant)
        # Fasttext fallback instead of spaCy NER
        configs.append({
            **full,
            "use_spacy_ner": False,
            "name": "fasttext_entity_filter",
        })
        # No entity filtering at all
        configs.append({
            **full,
            "filter_entities": False,
            "name": "no_entity_filter",
        })
        return configs
