from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from lingua import Language, LanguageDetectorBuilder


_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_HTML_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

_LANG_TO_BUCKET = {Language.ENGLISH: "EN", Language.GERMAN: "DE"}


def _clean_for_lid(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ")
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text or " "


def split_languages_in_csv(
    input_filepath: str,
    text_column: str,
    output_dir: str,
    confidence_threshold: float = 0.7,
    debug: bool = False,
) -> dict[str, str]:
    """Detect EN vs DE per row with lingua-py, route into EN / DE / UNCERTAIN
    buckets. A row stays in EN or DE only if the top-language confidence is
    at least ``confidence_threshold``; everything else goes to UNCERTAIN.

    Writes ``{output_dir}/{EN,DE,UNCERTAIN[,DEBUG]}/{filename}``."""
    df = pd.read_csv(input_filepath)
    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found. Available columns: {list(df.columns)}"
        )

    detector = (
        LanguageDetectorBuilder
        .from_languages(Language.ENGLISH, Language.GERMAN)
        .build()
    )

    n = len(df)
    print(f"Detecting languages for {n} rows...")

    langs: list[str] = []
    probs: list[float] = []
    progress_step = max(1, n // 10)
    for i, raw in enumerate(df[text_column].fillna("").tolist()):
        clean = _clean_for_lid(raw)
        top = detector.compute_language_confidence_values(clean)[0]
        langs.append(_LANG_TO_BUCKET.get(top.language, "UNCERTAIN"))
        probs.append(top.value)
        if (i + 1) % progress_step == 0:
            print(f"  {i + 1}/{n} rows processed...")

    df["_lang"] = langs
    df["_lang_prob"] = probs

    confident = df["_lang"].isin(("EN", "DE")) & df["_lang_prob"].ge(confidence_threshold)
    bucket = df["_lang"].where(confident, "UNCERTAIN")

    orig_filename = Path(input_filepath).name
    out = Path(output_dir)
    for sub in ("EN", "DE", "UNCERTAIN"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    output_paths: dict[str, str] = {}
    original_cols = [c for c in df.columns if not c.startswith("_")]
    meta_cols = ["_lang", "_lang_prob"]

    for b in ("EN", "DE", "UNCERTAIN"):
        sub = df[bucket == b]
        cols = original_cols if b in ("EN", "DE") else (original_cols + meta_cols)
        path = out / b / orig_filename
        sub[cols].to_csv(path, index=False)
        output_paths[b] = str(path)
        pct = 100 * len(sub) / n if n else 0
        print(f"  {b}: {len(sub):>7} rows ({pct:5.2f}%)  →  {path}")

    if debug:
        (out / "DEBUG").mkdir(exist_ok=True)
        debug_path = out / "DEBUG" / orig_filename
        df[original_cols + meta_cols].to_csv(debug_path, index=False)
        output_paths["DEBUG"] = str(debug_path)
        print(f"  DEBUG: {n:>7} rows  →  {debug_path}")

    return output_paths
