from __future__ import annotations

import types
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import ray

from data_analysis.utils.text_preprocessing import TextPreprocessor


_FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

# German dialects/variants that fasttext may return — reclassify as DE
_DE_VARIANTS = {"NDS", "ALS", "BAR", "LB", "GSW"}

_DEFAULT_PREPROCESSOR = TextPreprocessor()


def tag_ambiguity(df: pd.DataFrame, close_call_delta: float = 0.15) -> pd.Series:
    """Reclassify DE dialects, force non-EN/DE picks to the better of EN/DE
    when there is *any* EN/DE signal, and tag ``_ambiguous_reason`` on the
    DataFrame.

    Mutates ``df`` in place. Expects columns: ``_lang``, ``_lang_prob``,
    ``_en_prob``, ``_de_prob``. Returns the boolean mask of rows that have
    *no* EN/DE signal and should be routed to OTHER.
    """
    df.loc[df["_lang"].isin(_DE_VARIANTS), "_lang"] = "DE"

    df["_ambiguous_reason"] = ""

    forced_mask = ~df["_lang"].isin(("EN", "DE"))
    has_en_de_signal = df["_en_prob"].gt(0) | df["_de_prob"].gt(0)

    other_mask = forced_mask & ~has_en_de_signal
    override_mask = forced_mask & has_en_de_signal

    df.loc[override_mask, "_ambiguous_reason"] = "forced_en_de"
    df.loc[override_mask, "_lang"] = np.where(
        df.loc[override_mask, "_en_prob"] >= df.loc[override_mask, "_de_prob"],
        "EN",
        "DE",
    )
    df.loc[override_mask, "_lang_prob"] = df.loc[override_mask, ["_en_prob", "_de_prob"]].max(axis=1)

    close_mask = (
        df["_ambiguous_reason"].eq("")
        & df["_lang"].isin(("EN", "DE"))
        & (df["_en_prob"] - df["_de_prob"]).abs().lt(close_call_delta)
        & df[["_en_prob", "_de_prob"]].min(axis=1).gt(0.05)
    )
    df.loc[close_mask, "_ambiguous_reason"] = "en_de_close"

    return other_mask


def _get_model_path() -> Path:
    """Resolve path to the fasttext model cache inside .venv."""
    for parent in Path(__file__).resolve().parents:
        venv = parent / ".venv"
        if venv.exists():
            model_dir = venv / "models" / "fasttext"
            model_dir.mkdir(parents=True, exist_ok=True)
            return model_dir / "lid.176.bin"
    fallback = Path(__file__).parent / "models" / "fasttext"
    fallback.mkdir(exist_ok=True)
    return fallback / "lid.176.bin"


def _download_model(model_path: Path) -> None:
    if model_path.exists():
        return
    print(f"Downloading fasttext language ID model → {model_path} ...")
    urllib.request.urlretrieve(_FASTTEXT_MODEL_URL, model_path)
    print("Download complete.")


def _preprocess(text: str, model=None, nlp=None, preprocessor: TextPreprocessor = _DEFAULT_PREPROCESSOR) -> tuple[str, list[dict]]:
    """Normalize *text* and return ``(preprocessed_text, ner_removed_entities)``."""
    return preprocessor.preprocess_with_ner_diff(text, model, nlp)



@ray.remote(num_cpus=1)
def _detect_batch(texts: list[str], model_path: str) -> list[dict]:
    """Ray task: run fasttext language detection on a batch of texts."""
    import numpy as _np
    import fasttext.FastText as _ft_module

    # NumPy 2.x compatibility: fasttext calls np.array(probs, copy=False) which
    # raises ValueError in NumPy >= 2.0 — replace the np reference with a shim.
    _shim = types.SimpleNamespace(**{k: getattr(_np, k) for k in dir(_np) if not k.startswith("__")})
    _shim.array = lambda obj, *a, **kw: _np.asarray(obj, *a) if not kw.get("copy", True) else _np.array(obj, *a, **kw)
    _ft_module.np = _shim
    _ft_module.eprint = lambda *a, **kw: None  # suppress load_model warning

    import fasttext
    model = fasttext.load_model(model_path)

    import spacy
    try:
        nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])
    except OSError:
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])

    results = []
    for text in texts:
        clean, ner_removed = _preprocess(text, model, nlp)
        labels, probs = model.predict(clean, k=10)
        langs = [lbl.replace("__label__", "").upper() for lbl in labels]
        all_preds = {l: float(p) for l, p in zip(langs, probs)}
        results.append(
            {
                "top_lang": langs[0],
                "top_prob": float(probs[0]),
                "en_prob": all_preds.get("EN", 0.0),
                "de_prob": all_preds.get("DE", 0.0),
                "all_preds": all_preds,
                "preprocessed_text": clean,
                "ner_removed": ner_removed,
            }
        )
    return results


def split_languages_in_csv(
    input_filepath: str,
    text_column: str,
    output_dir: str,
    batch_size: int = 2000,
    num_workers: int = 4,
    close_call_delta: float = 0.15,
    debug: bool = False,
) -> dict[str, str]:
    """Detect language for every row in a CSV and split into per-language files.

    Outputs
    -------
    {output_dir}/EN/{filename}
        English rows (original columns only).
    {output_dir}/DE/{filename}
        German rows (original columns only).
    {output_dir}/OTHER/{filename}
        Rows where neither EN nor DE reached any meaningful probability.
        Includes full metadata columns.
    {output_dir}/AMBIGUOUS/{filename}
        Rows that needed a judgement call, kept for human review:
          - "forced_en_de"  : model predicted a 3rd language but we overrode
                              to the better of EN/DE (indicates noisy/short text).
          - "en_de_close"   : model picked EN or DE but the margin over the
                              other was < close_call_delta (hard to tell apart).
        These rows are ALSO written to EN / DE with the chosen label.
    {output_dir}/DEBUG/{filename}  (only when ``debug=True``)
        Full dataset with every row, containing:
          - all original columns
          - ``_raw_text``         : original value of *text_column*
          - ``_preprocessed_text``: text as actually fed to fasttext
          - all detection metadata (_lang, _lang_prob, _en_prob, _de_prob,
            _all_preds, _ambiguous_reason)

    Parameters
    ----------
    input_filepath   : path to the input CSV file
    text_column      : name of the column that contains the text to classify
    output_dir       : directory where output files will be written
    batch_size       : rows per Ray task
    num_workers      : max parallel Ray tasks (keep low to avoid system lag)
    close_call_delta : EN/DE probability difference below which a row is
                       flagged as "en_de_close" in AMBIGUOUS (default 0.15)
    debug            : write a DEBUG/{filename} CSV showing original vs.
                       preprocessed text alongside all detection metadata
    """
    model_path = _get_model_path()
    _download_model(model_path)

    df = pd.read_csv(input_filepath)
    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found. Available columns: {list(df.columns)}"
        )

    texts = df[text_column].fillna("").tolist()
    n = len(texts)

    if not ray.is_initialized():
        ray.init(num_cpus=num_workers, ignore_reinit_error=True)

    batches = [texts[i : i + batch_size] for i in range(0, n, batch_size)]
    print(f"Detecting languages for {n} rows in {len(batches)} batches (max {num_workers} parallel workers)...")

    model_path_str = str(model_path)
    flat_results: list[dict] = []

    for i in range(0, len(batches), num_workers):
        window = batches[i : i + num_workers]
        futures = [_detect_batch.remote(batch, model_path_str) for batch in window]
        for batch_result in ray.get(futures):
            flat_results.extend(batch_result)
        done = min(i + num_workers, len(batches))
        print(f"  {min(done * batch_size, n)}/{n} rows processed...")

    ray.shutdown()  # release workers immediately — stops system lag after the job


    sum_probs = [r["en_prob"] + r["de_prob"] for r in flat_results]
    prob_de_normalized = [r["de_prob"] / sum_prob if sum_prob > 0 else 0 for r, sum_prob in zip(flat_results, sum_probs)]
    prob_en_normalized = [r["en_prob"] / sum_prob if sum_prob > 0 else 0 for r, sum_prob in zip(flat_results, sum_probs)]

    df["_lang"]              = [r["top_lang"]          for r in flat_results]
    df["_lang_prob"]         = [r["top_prob"]           for r in flat_results]
    df["_en_prob"]           = prob_en_normalized
    df["_de_prob"]           = prob_de_normalized
    df["_all_preds"]         = [str(r["all_preds"])     for r in flat_results]
    df["_preprocessed_text"] = [r["preprocessed_text"] for r in flat_results]
    df["_ner_removed"] = [
        ", ".join(
            "{} [{}{}]".format(
                e["text"],
                e["label"],
                (": removed=" + ",".join(e["removed_tokens"])) if e.get("removed_tokens") else "",
            )
            for e in r["ner_removed"]
        )
        for r in flat_results
    ]

    other_mask = tag_ambiguity(df, close_call_delta=close_call_delta)

    # --- Write outputs ---
    orig_filename = Path(input_filepath).name
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "OTHER").mkdir(exist_ok=True)
    (out / "AMBIGUOUS").mkdir(exist_ok=True)
    (out / "EN").mkdir(exist_ok=True)
    (out / "DE").mkdir(exist_ok=True)

    output_paths: dict[str, str] = {}
    original_cols = [c for c in df.columns if not c.startswith("_")]
    meta_cols = ["_lang", "_lang_prob", "_en_prob", "_de_prob", "_all_preds", "_ambiguous_reason", "_ner_removed"]
    debug_cols = original_cols + meta_cols

    # EN / DE — clean rows only (ambiguous rows are included too, just also in AMBIGUOUS)
    for lang in ("EN", "DE"):
        subset = df[df["_lang"] == lang][original_cols]
        path = out / f"{lang}" / orig_filename
        subset.to_csv(path, index=False)
        output_paths[lang] = str(path)
        print(f"  {lang}: {len(subset):>6} rows  →  {path}")

    # AMBIGUOUS — forced overrides + close EN/DE calls
    ambiguous = df[df["_ambiguous_reason"].ne("")]
    amb_path = out / "AMBIGUOUS" / orig_filename
    ambiguous[debug_cols].to_csv(amb_path, index=False)
    output_paths["AMBIGUOUS"] = str(amb_path)
    print(f"  AMBIGUOUS: {len(ambiguous):>4} rows  →  {amb_path}")
    if len(ambiguous) > 0:
        print(f"  AMBIGUOUS reason breakdown:\n{ambiguous['_ambiguous_reason'].value_counts().to_string()}")

    # OTHER — truly foreign (no EN/DE signal at all)
    other = df[other_mask]
    other_path = out / "OTHER" / orig_filename
    other[debug_cols].to_csv(other_path, index=False)
    output_paths["OTHER"] = str(other_path)
    print(f"  OTHER: {len(other):>5} rows  →  {other_path}")
    if len(other) > 0:
        print(f"  OTHER language breakdown:\n{other['_lang'].value_counts().to_string()}")

    # DEBUG — original vs. preprocessed text for every row
    if debug:
        (out / "DEBUG").mkdir(exist_ok=True)

        # Full debug dump: original + preprocessed side-by-side + all metadata
        debug_path = out / "DEBUG" / orig_filename
        text_col_idx = original_cols.index(text_column) + 1
        other_orig_cols = [c for c in original_cols if c != text_column]
        ordered_cols = (
            other_orig_cols[:text_col_idx - 1]
            + [text_column, "_preprocessed_text"]
            + other_orig_cols[text_col_idx - 1:]
            + meta_cols
        )
        df[ordered_cols].to_csv(debug_path, index=False)
        output_paths["DEBUG"] = str(debug_path)
        print(f"  DEBUG: {len(df):>6} rows  →  {debug_path}")

        # NER changes: only rows where something was actually removed
        ner_mask = df["_ner_removed"].ne("")
        ner_affected = df[ner_mask]
        ner_path = out / "DEBUG" / f"ner_changes_{orig_filename}"
        ner_cols = (
            other_orig_cols[:text_col_idx - 1]
            + [text_column, "_ner_removed", "_preprocessed_text"]
            + other_orig_cols[text_col_idx - 1:]
            + ["_lang", "_lang_prob", "_en_prob", "_de_prob", "_ambiguous_reason"]
        )
        ner_affected[ner_cols].to_csv(ner_path, index=False)
        output_paths["NER_CHANGES"] = str(ner_path)
        print(f"  NER changes: {len(ner_affected):>5} rows affected  →  {ner_path}")

    return output_paths
