"""Statistical analysis + ablation helpers for the multilingual review corpus.

Designed to be used after language splitting. spaCy is optional; only
``distinctive_vocab`` requires loaded pipelines.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Callable

import numpy as np
import pandas as pd


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
_WORD_RE = re.compile(r"\b\w+(?:'\w+)*\b")
_CAPS_TOKEN_RE = re.compile(r"\b\w{2,}\b")
_PUNCT_RUN_RE = re.compile(r"!{2,}|\?{2,}|\.{2,}|,{2,}|;{2,}|:{2,}|-{2,}")


class TextAnalyzer:
    """Statistical analysis of a multilingual review corpus.

    ``nlp`` may be a single spaCy pipeline or a ``{lang_code: pipeline}`` dict
    for per-language dispatch.
    """

    def __init__(self, nlp=None, sample_size: int | None = 5000, random_state: int = 42) -> None:
        self.nlp = nlp
        self.sample_size = sample_size
        self.random_state = random_state

    def _sample(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.sample_size and len(df) > self.sample_size:
            return df.sample(self.sample_size, random_state=self.random_state)
        return df

    def _pipeline_for(self, lang: str | None = None):
        if not isinstance(self.nlp, dict):
            return self.nlp
        if lang is None:
            raise ValueError("Multi-pipeline analyzer requires a language argument.")
        return self.nlp[lang]

    # ------------------------------------------------------------------
    # Length
    # ------------------------------------------------------------------

    def length_by_label(
        self, df: pd.DataFrame, text_col: str, lang_col: str, label_col: str
    ) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
        """Per-(lang, label) length stats and per-language Spearman rho."""
        from scipy.stats import spearmanr

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

    # ------------------------------------------------------------------
    # ALL-CAPS-word rate (raw text only — case is destroyed in `_preprocessed_text`)
    # ------------------------------------------------------------------

    def caps_word_rate(
        self, df: pd.DataFrame, text_col: str, group_cols: str | list[str] | None = None
    ) -> pd.DataFrame:
        def _rate(text):
            if not isinstance(text, str):
                return 0.0
            toks = _CAPS_TOKEN_RE.findall(text)
            return sum(1 for t in toks if t.isupper()) / len(toks) if toks else 0.0

        rates = df[text_col].map(_rate).mul(100).round(3)
        out = pd.DataFrame({"caps_token_rate_%": rates}, index=df.index)
        if group_cols:
            keys = [group_cols] if isinstance(group_cols, str) else list(group_cols)
            return df[keys].join(out).groupby(keys).mean().round(3)
        return out.mean().round(3).to_frame("caps_token_rate_%")

    # ------------------------------------------------------------------
    # Distinctive vocabulary — log-odds with informative Dirichlet prior
    # (Monroe, Colaresi, Quinn 2008).
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
        """Top-*n* lemmas per (lang, label) ranked by descending log-odds z-score.

        Uses each language's whole-corpus lemma counts as the informative prior.
        Requires ``TextAnalyzer(nlp={"EN": ..., "DE": ...})``.
        """
        if self.nlp is None:
            raise ValueError("distinctive_vocab requires nlp pipelines.")
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
                        if len(lem) > 1:
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
                n_out = (total_corpus - n_in) or 1
                scored = []
                for w, count_in in c_in.items():
                    if count_in < min_count:
                        continue
                    a_w = alpha.get(w, prior_alpha)
                    count_out = corpus[w] - count_in
                    p_in = (count_in + a_w) / (n_in + alpha_total - count_in - a_w)
                    p_out = (count_out + a_w) / (n_out + alpha_total - count_out - a_w)
                    delta = np.log(p_in) - np.log(p_out)
                    var = 1.0 / (count_in + a_w) + 1.0 / (count_out + a_w)
                    scored.append((w, delta / np.sqrt(var), count_in, corpus[w]))

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
    # Presence-rate of special tokens (URL, email, number, emoji)
    # ------------------------------------------------------------------

    def special_rates(
        self, df: pd.DataFrame, text_col: str, group_col: str | list[str] | None = None
    ) -> pd.DataFrame:
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

    # ------------------------------------------------------------------
    # Cleaning rule diagnostics — how many docs each rule would touch
    # ------------------------------------------------------------------

    def rule_diagnostics(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        texts = df[text_col].fillna("")
        rules = [
            ("strip_html",           texts.str.contains(r"<[^>]+>", regex=True)),
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

    # ------------------------------------------------------------------
    # Ablation config generators
    # ------------------------------------------------------------------

    def ablation_configs(self) -> list[dict]:
        """LID ablation variants (one preprocessing toggle per row)."""
        full = {
            "strip_html": True,
            "remove_urls": True,
            "emoji_strategy": "strip",
            "punct_max_repeat": 3,
            "entity_strategy": "strip_descriptors",
        }
        configs = [{"name": "full_pipeline", **full}]
        for key in ("strip_html", "remove_urls"):
            configs.append({**full, key: False, "name": f"no_{key}"})
        configs.append({**full, "emoji_strategy": "keep", "name": "no_strip_emojis"})
        configs.append({**full, "punct_max_repeat": None, "name": "no_collapse_punctuation"})
        configs.append({**full, "entity_strategy": "keep", "name": "no_entity_filter"})
        return configs

    def sentiment_ablation_configs(self) -> list[dict]:
        """Sentiment ablation variants — merged scopes (4 emoji × 3 punct × 3 entity).

        Each variant changes exactly one axis off the baseline. BERT-style
        output (casing kept, no final tokenisation).
        """
        baseline = {
            "lowercase": False,
            "strip_html": True,
            "remove_urls": True,
            "emoji_strategy": "strip",
            "punct_max_repeat": 3,
            "entity_strategy": "strip_descriptors",
            "final_tokenize": False,
        }
        configs: list[dict] = [{"name": "baseline", **baseline}]

        for strat in ("keep", "demojize", "pos_neg_token"):
            configs.append({**baseline, "emoji_strategy": strat, "name": f"emoji_{strat}"})

        for cap, label in [(None, "keep"), (1, "single")]:
            configs.append({**baseline, "punct_max_repeat": cap, "name": f"punct_{label}"})

        for strat in ("keep", "replace_with_token"):
            configs.append({**baseline, "entity_strategy": strat, "name": f"entity_{strat}"})

        return configs


# ---------------------------------------------------------------------------
# Generic ablation evaluator: preprocess once per variant, score with a
# pluggable evaluator function. Used for the sentiment ablation.
# ---------------------------------------------------------------------------

EvaluatorFn = Callable[[list[str], list[int]], dict[str, float]]


def evaluate_ablations(
    sample_df: pd.DataFrame,
    variants: list[dict],
    text_column: str,
    label_column: str,
    evaluator_fn: EvaluatorFn,
    nlp_by_lang: dict | None = None,
    text_lang_column: str | None = None,
) -> pd.DataFrame:
    """For each variant: instantiate ``TextPreprocessor``, preprocess every row
    (dispatching the spaCy pipeline by ``text_lang_column`` if given), then
    pass ``(cleaned_texts, labels)`` to ``evaluator_fn``.

    Returns a tidy DataFrame ``[variant, <metrics from evaluator_fn>]``.
    """
    import time
    from data_analysis.utils.text_preprocessing import TextPreprocessor

    labels = sample_df[label_column].tolist()
    rows = []
    for cfg in variants:
        name = cfg["name"]
        kwargs = {k: v for k, v in cfg.items() if k != "name"}
        pre = TextPreprocessor(**kwargs)
        t0 = time.time()
        print(f"  [{name}] preprocessing {len(sample_df):,} rows ...", end="", flush=True)
        cleaned: list[str] = []
        for _, row in sample_df.iterrows():
            lang_code = (row[text_lang_column] if text_lang_column else "en").lower()
            nlp = nlp_by_lang.get(lang_code.upper()) if nlp_by_lang else None
            cleaned.append(pre.preprocess(row[text_column], nlp=nlp, lang=lang_code))
        t_pre = time.time() - t0
        print(f" {t_pre:.1f}s   evaluating ...", end="", flush=True)
        t0 = time.time()
        metrics = evaluator_fn(cleaned, labels)
        t_eval = time.time() - t0
        print(f" {t_eval:.1f}s   mae={metrics.get('mae', float('nan')):.4f}  "
              f"score={metrics.get('score', float('nan')):.4f}")
        rows.append({"variant": name, **metrics})
    return pd.DataFrame(rows)


_DEFAULT_BOW_CONFIG: dict = {
    "type": "bow_lr",
    "name": "bow_lr_tfidf_bigram",
    "ngram_range": [1, 2],
    "max_features": 10000,
    "C": 1.0,
    "vectorizer_type": "tfidf",
    "solver": "lbfgs",
    "penalty": "l2",
}


def sentiment_evaluator(
    cleaned_texts: list[str],
    labels: list[int],
    model_config: dict | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, float]:
    """Stratified 80/20 split → fit a BOW model from ``baselines/`` → return
    held-out metrics (MAE, score = 1 - MAE/4, accuracy, macro-F1).

    Default model is ``BOWLogisticRegression`` with TF-IDF + bigrams + L2,
    matching the strongest baseline found in ``baselines/outputs_v2``.
    """
    from sklearn.model_selection import train_test_split
    from baselines.src.models import create_model_from_config
    from baselines.src.utils import MetricsComputer

    cfg = model_config or _DEFAULT_BOW_CONFIG
    X = np.asarray(cleaned_texts)
    y = np.asarray(labels)
    X_tr, X_dv, y_tr, y_dv = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    model = create_model_from_config(cfg)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_dv)
    return MetricsComputer.compute_all_metrics(y_dv, preds)


# ---------------------------------------------------------------------------
# LID ablation evaluator (baseline-comparison metric requires special handling
# so it stays as a dedicated function rather than going through evaluate_ablations).
# ---------------------------------------------------------------------------

def _patch_fasttext_for_numpy2() -> None:
    import types
    import numpy as _np
    import fasttext.FastText as _ft

    shim = types.SimpleNamespace(**{k: getattr(_np, k) for k in dir(_np) if not k.startswith("__")})
    shim.array = lambda obj, *a, **kw: _np.asarray(obj, *a) if not kw.get("copy", True) else _np.array(obj, *a, **kw)
    _ft.np = shim
    _ft.eprint = lambda *a, **kw: None


def _load_fasttext_model(model_path: str):
    _patch_fasttext_for_numpy2()
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
    """LID ablation. Per variant: preprocess + fasttext + tag_ambiguity, then
    compare ``_lang`` against the ``baseline_name`` variant for disagreement.
    """
    from data_analysis.utils.text_preprocessing import TextPreprocessor
    from data_analysis.utils.language_detection import tag_ambiguity

    ft_model = _load_fasttext_model(fasttext_model_path)
    texts = sample_df[text_column].fillna("").tolist()

    def _run_variant(cfg: dict) -> pd.DataFrame:
        pre = TextPreprocessor(**{k: v for k, v in cfg.items() if k != "name"})
        rows = []
        for text in texts:
            cleaned = pre.preprocess(text, nlp=nlp) or " "
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
        sum_en_de = (out["_en_prob"] + out["_de_prob"]).where(lambda s: s > 0, 1.0)
        out["_en_prob"] /= sum_en_de
        out["_de_prob"] /= sum_en_de
        out["_other"] = tag_ambiguity(out, close_call_delta=close_call_delta)
        return out

    baseline = next((_run_variant(c) for c in variants if c.get("name") == baseline_name), None)
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
