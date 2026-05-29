"""Statistical analysis helpers for the multilingual review corpus.

Used after the language split. The spaCy pipelines are only needed for
`distinctive_vocab`; the other methods work on raw text.
"""
import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


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
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[A-Za-z]{2,}\b")
_NUMBER_RE = re.compile(r"\b\d+[.,]?\d*\b")
_CAPS_TOKEN_RE = re.compile(r"\b\w{2,}\b")
_PUNCT_RUN_RE = re.compile(r"!{2,}|\?{2,}|\.{2,}|,{2,}|;{2,}|:{2,}|-{2,}")
_HTML_RE = re.compile(r"<[^>]+>")


class TextAnalyzer:
    """Statistical analysis of a multilingual review corpus.

    `nlp` is a {lang_code: spaCy pipeline} dict (e.g. {"EN": ..., "DE": ...})
    used to dispatch the right pipeline per language in `distinctive_vocab`.
    """

    def __init__(self, nlp=None, sample_size: int | None = 5000, random_state: int = 42) -> None:
        self.nlp = nlp
        self.sample_size = sample_size
        self.random_state = random_state

    def _sample(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.sample_size and len(df) > self.sample_size:
            return df.sample(self.sample_size, random_state=self.random_state)
        return df

    def length_by_label(
        self, df: pd.DataFrame, text_col: str, lang_col: str, label_col: str
    ) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
        """Per-(lang, label) review-length stats plus a per-language Spearman
        correlation between length and label."""
        lengths = df[text_col].fillna("").str.split().str.len()
        work = pd.DataFrame({
            "lang": df[lang_col].values,
            "label": df[label_col].values,
            "len": lengths.values,
        })

        stats = (
            work.groupby(["lang", "label"])["len"]
                .agg(mean="mean", median="median",
                     p10=lambda s: s.quantile(0.10),
                     p90=lambda s: s.quantile(0.90),
                     n="size")
                .reset_index()
        )

        spearman: dict[str, dict[str, float]] = {}
        for lang, grp in work.groupby("lang"):
            rho, p = spearmanr(grp["len"], grp["label"])
            spearman[lang] = {"rho": float(rho), "pvalue": float(p), "n": int(len(grp))}
        return stats, spearman

    def caps_word_rate(
        self, df: pd.DataFrame, text_col: str, group_cols: str | list[str] | None = None
    ) -> pd.DataFrame:
        """Share of all-caps words (>=2 letters) per document, optionally
        averaged over group_cols. Run on raw text (casing matters here)."""
        def rate(text):
            if not isinstance(text, str):
                return 0.0
            toks = _CAPS_TOKEN_RE.findall(text)
            return sum(1 for t in toks if t.isupper()) / len(toks) if toks else 0.0

        rates = df[text_col].map(rate).mul(100).round(3)
        out = pd.DataFrame({"caps_token_rate_%": rates}, index=df.index)
        if group_cols:
            keys = [group_cols] if isinstance(group_cols, str) else list(group_cols)
            return df[keys].join(out).groupby(keys).mean().round(3)
        return out.mean().round(3).to_frame("caps_token_rate_%")

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
        """Top-n lemmas per (lang, label) ranked by log-odds z-score (Monroe et
        al. 2008), using each language's whole-corpus counts as the prior."""
        if self.nlp is None:
            raise ValueError("distinctive_vocab requires nlp pipelines.")
        sample = self._sample(df)
        rows: list[dict] = []

        for lang, lang_df in sample.groupby(lang_col):
            nlp = self.nlp[lang]
            label_counts: dict[object, Counter] = {}
            for label, label_df in lang_df.groupby(label_col):
                counts: Counter = Counter()
                texts = label_df[text_col].fillna("").tolist()
                for doc in nlp.pipe(texts, batch_size=batch_size, disable=["ner"]):
                    for tok in doc:
                        if tok.is_punct or tok.is_space or tok.is_stop:
                            continue
                        lemma = tok.lemma_.lower()
                        if len(lemma) > 1:
                            counts[lemma] += 1
                label_counts[label] = counts

            corpus = Counter()
            for counts in label_counts.values():
                corpus.update(counts)
            total_corpus = sum(corpus.values()) or 1
            alpha = {w: prior_alpha * cnt for w, cnt in corpus.items()}
            alpha_total = sum(alpha.values())

            for label, c_in in label_counts.items():
                n_in = sum(c_in.values()) or 1
                n_out = (total_corpus - n_in) or 1
                scored = []
                for word, count_in in c_in.items():
                    if count_in < min_count:
                        continue
                    a_w = alpha.get(word, prior_alpha)
                    count_out = corpus[word] - count_in
                    p_in = (count_in + a_w) / (n_in + alpha_total - count_in - a_w)
                    p_out = (count_out + a_w) / (n_out + alpha_total - count_out - a_w)
                    delta = np.log(p_in) - np.log(p_out)
                    var = 1.0 / (count_in + a_w) + 1.0 / (count_out + a_w)
                    scored.append((word, delta / np.sqrt(var), count_in, corpus[word]))

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

    def special_rates(
        self, df: pd.DataFrame, text_col: str, group_col: str | list[str] | None = None
    ) -> pd.DataFrame:
        """Presence rate (%) of URLs, emails, numbers and emojis, optionally
        broken down by group_col."""
        texts = df[text_col].fillna("")
        flags = pd.DataFrame({
            "has_url":    texts.str.contains(_URL_RE, regex=True),
            "has_email":  texts.str.contains(_EMAIL_RE, regex=True),
            "has_number": texts.str.contains(_NUMBER_RE, regex=True),
            "has_emoji":  texts.str.contains(_EMOJI_RE, regex=True),
        }, index=df.index)
        if group_col:
            keys = [group_col] if isinstance(group_col, str) else list(group_col)
            return (
                df[keys].join(flags).groupby(keys).mean()
                       .mul(100).round(2)
                       .rename(columns=lambda c: c.replace("has_", "") + "_%")
            )
        return (
            flags.mean().mul(100).round(2)
                 .rename(index=lambda c: c.replace("has_", "") + "_%")
                 .rename("rate_%").to_frame()
        )

    def rule_diagnostics(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """How many documents each cleaning rule would touch."""
        texts = df[text_col].fillna("")
        rules = [
            ("strip_html",           texts.str.contains(_HTML_RE, regex=True)),
            ("strip_urls",           texts.str.contains(_URL_RE, regex=True)),
            ("strip_emojis",         texts.str.contains(_EMOJI_RE, regex=True)),
            ("collapse_punctuation", texts.str.contains(_PUNCT_RUN_RE, regex=True)),
            ("has_newline",          texts.str.contains("\n", regex=False)),
        ]
        return pd.DataFrame([
            {"rule": name, "n_affected": int(mask.sum()),
             "pct_affected": round(float(mask.mean()) * 100, 2)}
            for name, mask in rules
        ])
