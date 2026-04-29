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
_APOSTROPHES_RE = re.compile(r"[‘’ʼʹʻ`´]")
_WORD_RE = re.compile(r"\b\w+(?:'\w+)*\b")
_CAPS_TOKEN_RE = re.compile(r"\b\w{2,}\b")

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
        # nlp may be a single spaCy Language or a {lang_code: Language} dict.
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

    def _pipeline_for(self, lang: str | None = None):
        """Return the spaCy pipeline for *lang*, or the single configured pipeline."""
        if isinstance(self.nlp, dict):
            if lang is None:
                raise ValueError(
                    "TextAnalyzer was configured with a {lang: pipeline} dict; "
                    "the calling method must pass a language."
                )
            if lang not in self.nlp:
                raise ValueError(
                    f"No spaCy pipeline configured for language {lang!r}. "
                    f"Available: {sorted(self.nlp)}"
                )
            return self.nlp[lang]
        return self.nlp

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
        group_col: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """Percentage of docs containing URLs, emails, numbers, or emojis.

        ``group_col`` may be a column name or a list of column names for
        multi-key grouping (e.g. ``['_lang', 'label']``).
        """
        texts = df[text_col].fillna("")
        flags = pd.DataFrame({
            "has_url":    texts.str.contains(_URL_RE, regex=True),
            "has_email":  texts.str.contains(_EMAIL_RE, regex=True),
            "has_number": texts.str.contains(_NUMBER_RE, regex=True),
            "has_emoji":  texts.str.contains(_EMOJI_RE, regex=True),
        }, index=df.index)
        if group_col:
            keys = [group_col] if isinstance(group_col, str) else list(group_col)
            combined = df[keys].join(flags)
            return (
                combined.groupby(keys).mean()
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
    # 11a. Per-(lang, label) length stats + Spearman rho
    # ------------------------------------------------------------------

    def length_by_label(
        self,
        df: pd.DataFrame,
        text_col: str,
        lang_col: str,
        label_col: str,
    ) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
        """Per-(lang, label) length statistics + Spearman rho(length, label).

        Returns
        -------
        stats
            Tidy DataFrame with columns
            ``[lang, label, mean, median, p10, p90, n]``.
        spearman
            ``{lang: {"rho": float, "pvalue": float, "n": int}}``.
        """
        from scipy.stats import spearmanr

        lengths = df[text_col].fillna("").str.split().str.len()
        work = pd.DataFrame({
            "_lang_": df[lang_col].values,
            "_label_": df[label_col].values,
            "_len_": lengths.values,
        })

        stats_rows = []
        for (lang, label), grp in work.groupby(["_lang_", "_label_"]):
            stats_rows.append({
                "lang": lang,
                "label": label,
                "mean": float(grp["_len_"].mean()),
                "median": float(grp["_len_"].median()),
                "p10": float(grp["_len_"].quantile(0.10)),
                "p90": float(grp["_len_"].quantile(0.90)),
                "n": int(len(grp)),
            })
        stats = pd.DataFrame(stats_rows).sort_values(["lang", "label"]).reset_index(drop=True)

        spearman: dict[str, dict[str, float]] = {}
        for lang, grp in work.groupby("_lang_"):
            rho, pval = spearmanr(grp["_len_"], grp["_label_"])
            spearman[lang] = {"rho": float(rho), "pvalue": float(pval), "n": int(len(grp))}
        return stats, spearman

    # ------------------------------------------------------------------
    # 11b. ALL-CAPS-word rate (token-level, not char-level)
    # ------------------------------------------------------------------

    def caps_word_rate(
        self,
        df: pd.DataFrame,
        text_col: str,
        group_cols: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """Mean per-doc share of tokens (length >= 2) that are entirely uppercase.

        Operates on raw text — case is destroyed in ``_preprocessed_text``.
        """
        def _rate(text: str) -> float:
            if not isinstance(text, str):
                return 0.0
            toks = _CAPS_TOKEN_RE.findall(text)
            return sum(1 for t in toks if t.isupper()) / len(toks) if toks else 0.0

        rates = df[text_col].map(_rate)
        out = pd.DataFrame({"caps_token_rate_%": rates.mul(100).round(3)}, index=df.index)
        if group_cols:
            keys = [group_cols] if isinstance(group_cols, str) else list(group_cols)
            return df[keys].join(out).groupby(keys).mean().round(3)
        return out.mean().round(3).to_frame("caps_token_rate_%")

    # ------------------------------------------------------------------
    # 11c. Distinctive vocabulary per (lang, label) — log-odds with
    # informative Dirichlet prior (Monroe, Colaresi, Quinn 2008).
    # ------------------------------------------------------------------

    def distinctive_vocab(
        self,
        df: pd.DataFrame,
        text_col: str,
        lang_col: str,
        label_col: str,
        n: int = 15,
        prior_alpha: float = 0.01,
        min_count: int = 5,
        batch_size: int = 256,
    ) -> pd.DataFrame:
        """Top *n* lemmas per (lang, label) ranked by log-odds z-score.

        Uses each language's whole-corpus lemma counts as the informative
        Dirichlet prior. Lemmatisation requires a spaCy pipeline per
        language: pass ``nlp={"EN": en_pipe, "DE": de_pipe}`` to
        :class:`TextAnalyzer`.

        ``min_count`` filters out very rare lemmas (which otherwise dominate
        the |z| ranking due to small-count noise).
        """
        self._require_nlp("distinctive_vocab")
        sample = self._sample(df)
        rows: list[dict] = []

        for lang, lang_df in sample.groupby(lang_col):
            nlp = self._pipeline_for(lang)
            label_counts: dict[object, Counter] = {}
            for label, label_df in lang_df.groupby(label_col):
                c: Counter = Counter()
                texts = label_df[text_col].fillna("").tolist()
                for doc in nlp.pipe(texts, batch_size=batch_size, disable=["ner"]):
                    for tok in doc:
                        if tok.is_punct or tok.is_space or tok.is_stop:
                            continue
                        lem = tok.lemma_.lower()
                        if len(lem) <= 1:
                            continue
                        c[lem] += 1
                label_counts[label] = c

            corpus = Counter()
            for c in label_counts.values():
                corpus.update(c)
            total_corpus = sum(corpus.values()) or 1
            alpha = {w: prior_alpha * cnt for w, cnt in corpus.items()}
            alpha_total = sum(alpha.values())

            for label, c_in in label_counts.items():
                n_in = sum(c_in.values()) or 1
                c_out = Counter({w: corpus[w] - c_in.get(w, 0) for w in corpus})
                n_out = total_corpus - n_in or 1

                scored = []
                for w, count_in in c_in.items():
                    if count_in < min_count:
                        continue
                    a_w = alpha.get(w, prior_alpha)
                    count_out = c_out[w]
                    p_in = (count_in + a_w) / (n_in + alpha_total - count_in - a_w)
                    p_out = (count_out + a_w) / (n_out + alpha_total - count_out - a_w)
                    delta = np.log(p_in) - np.log(p_out)
                    var = 1.0 / (count_in + a_w) + 1.0 / (count_out + a_w)
                    z = delta / np.sqrt(var)
                    scored.append((w, z, count_in, corpus[w]))

                scored.sort(key=lambda t: -t[1])
                for lemma, z, c_label, c_total in scored[:n]:
                    rows.append({
                        "lang": lang,
                        "label": label,
                        "lemma": lemma,
                        "z": round(float(z), 3),
                        "count_in_label": int(c_label),
                        "count_total": int(c_total),
                    })

        return pd.DataFrame(rows)

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


# ---------------------------------------------------------------------------
# Module-level ablation evaluator (runs in-process, no ray needed)
# ---------------------------------------------------------------------------

def _load_fasttext_model(model_path: str):
    """Load fasttext with the NumPy>=2 compatibility shim used by the splitter."""
    import types as _types
    import numpy as _np
    import fasttext.FastText as _ft_module

    _shim = _types.SimpleNamespace(
        **{k: getattr(_np, k) for k in dir(_np) if not k.startswith("__")}
    )
    _shim.array = lambda obj, *a, **kw: (
        _np.asarray(obj, *a) if not kw.get("copy", True) else _np.array(obj, *a, **kw)
    )
    _ft_module.np = _shim
    _ft_module.eprint = lambda *a, **kw: None

    import fasttext
    return fasttext.load_model(model_path)


def evaluate_ablation_variants(
    sample_df: pd.DataFrame,
    variants: list[dict],
    text_column: str,
    fasttext_model_path: str,
    nlp=None,
    close_call_delta: float = 0.15,
    baseline_name: str = "full_pipeline",
) -> pd.DataFrame:
    """Run each ``TextPreprocessor`` variant on *sample_df*, classify with
    fasttext, and report per-variant ambiguity counts + disagreement vs
    *baseline_name*.

    No ray, no file I/O — all work happens in-process. Best run on a
    stratified sample of a few thousand rows.

    Returns a tidy DataFrame with columns:
    ``[variant, n_rows, n_other, n_ambiguous, n_forced_en_de, n_en_de_close,
       pct_disagree_vs_baseline]``.
    """
    from data_analysis.utils.text_preprocessing import TextPreprocessor
    from data_analysis.utils.language_detection import tag_ambiguity

    ft_model = _load_fasttext_model(fasttext_model_path)
    texts = sample_df[text_column].fillna("").tolist()

    def _run_variant(cfg: dict) -> pd.DataFrame:
        kwargs = {k: v for k, v in cfg.items() if k != "name"}
        pre = TextPreprocessor(**kwargs)
        rows = []
        for text in texts:
            cleaned = pre.preprocess(text, model=ft_model, nlp=nlp)
            cleaned = cleaned if cleaned else " "
            labels, probs = ft_model.predict(cleaned, k=10)
            langs = [lbl.replace("__label__", "").upper() for lbl in labels]
            preds = {l: float(p) for l, p in zip(langs, probs)}
            rows.append({
                "_lang": langs[0],
                "_lang_prob": float(probs[0]),
                "_en_prob": preds.get("EN", 0.0),
                "_de_prob": preds.get("DE", 0.0),
            })
        out = pd.DataFrame(rows)
        sum_en_de = out["_en_prob"] + out["_de_prob"]
        safe = sum_en_de.where(sum_en_de > 0, 1.0)
        out["_en_prob"] = out["_en_prob"] / safe
        out["_de_prob"] = out["_de_prob"] / safe
        other_mask = tag_ambiguity(out, close_call_delta=close_call_delta)
        out["_other"] = other_mask
        return out

    baseline = None
    for cfg in variants:
        if cfg.get("name") == baseline_name:
            baseline = _run_variant(cfg)
            break
    if baseline is None:
        raise ValueError(f"Baseline variant {baseline_name!r} not in variants list.")

    rows = []
    for cfg in variants:
        name = cfg["name"]
        out = baseline if name == baseline_name else _run_variant(cfg)
        amb = out["_ambiguous_reason"]
        rows.append({
            "variant": name,
            "n_rows": int(len(out)),
            "n_other": int(out["_other"].sum()),
            "n_ambiguous": int(amb.ne("").sum()),
            "n_forced_en_de": int((amb == "forced_en_de").sum()),
            "n_en_de_close": int((amb == "en_de_close").sum()),
            "pct_disagree_vs_baseline": round(
                100.0 * float((out["_lang"] != baseline["_lang"]).mean()), 3
            ),
        })
    return pd.DataFrame(rows)
