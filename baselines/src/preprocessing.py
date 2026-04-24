"""
Preprocessing pipelines for text data.
"""
import re
import string
from typing import Callable, List, Optional, Tuple, Dict
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import urllib.request
import os
from pathlib import Path
import warnings


class PreprocessingPipeline(ABC):
    """Base class for preprocessing pipelines."""
    
    @abstractmethod
    def __call__(self, text: str) -> str:
        """Apply preprocessing to text."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get pipeline name."""
        pass


class NoOpPreprocessing(PreprocessingPipeline):
    """No preprocessing."""
    
    def __call__(self, text: str) -> str:
        return text
    
    def get_name(self) -> str:
        return "no_preprocessing"


class BasicPreprocessing(PreprocessingPipeline):
    """Basic preprocessing: lowercase, whitespace normalization."""
    
    def __call__(self, text: str) -> str:
        # Lowercase
        text = text.lower()
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def get_name(self) -> str:
        return "basic"


class AggresivePreprocessing(PreprocessingPipeline):
    """Aggressive preprocessing: lowercase, extra whitespace."""
    
    def __call__(self, text: str) -> str:
        # Lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\!\?-]', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def get_name(self) -> str:
        return "aggressive"

class AgressivePreproccessing_No_Punctuation(PreprocessingPipeline):
    """Aggressive preprocessing: lowercase, punctuation removal, extra whitespace."""
    
    def __call__(self, text: str) -> str:
        # Lowercase

        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9äöüß0 ]', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def get_name(self) -> str:
        return "aggressive_no_punctuation"


class ExtendedPreprocessing(PreprocessingPipeline):
    """Extended preprocessing with special character handling."""
    
    def __call__(self, text: str) -> str:
        # Lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        # Remove numbers (often not informative)
        text = re.sub(r'\d+', '', text)
        # Remove special unicode characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z\s\.\!\?\-äöüßÄÖÜ]', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def get_name(self) -> str:
        return "extended"


class LanguageAwarePreprocessing(PreprocessingPipeline):
    """Adaptive preprocessing that detects language and applies appropriate rules."""
    
    def __call__(self, text: str) -> str:
        """Detect language and apply language-specific preprocessing."""
        lang = LanguageDetector.detect_language(text)
        
        # Lowercase
        text = text.lower()
        
        # Remove URLs and emails universally
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        if lang == 'de':
            return self._preprocess_german(text)
        elif lang == 'en':
            return self._preprocess_english(text)
        else:
            # Fallback: minimal aggressive preprocessing
            text = re.sub(r'[^a-zA-Z0-9\s\.\!\?\-äöüßÄÖÜ]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
    
    @staticmethod
    def _preprocess_german(text: str) -> str:
        """German-specific preprocessing."""
        # Keep umlauts and special German characters
        text = re.sub(r'[^a-zA-Zäöüß0-9\s\.\!\?\-]', '', text)
        # Common German contractions (optional normalization)
        text = re.sub(r'\b(das|daß)\b', 'das', text)  # Normalize old spelling
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    @staticmethod
    def _preprocess_english(text: str) -> str:
        """English-specific preprocessing."""
        # Remove numbers but keep common contractions
        text = re.sub(r'[^a-zA-Z0-9\s\.\!\?\-\']', '', text)
        # Normalize common contractions
        text = re.sub(r"n't", ' not', text)
        text = re.sub(r"'s", ' is', text)
        text = re.sub(r"'re", ' are', text)
        text = re.sub(r"'ve", ' have', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def get_name(self) -> str:
        return "language_aware"


class PreprocessingFactory:
    """Factory for creating preprocessing pipelines."""
    
    _pipelines = {
        'none': NoOpPreprocessing,
        'basic': BasicPreprocessing,
        'aggressive': AggresivePreprocessing,
        'extended': ExtendedPreprocessing,
        'language_aware': LanguageAwarePreprocessing,
    }
    
    @classmethod
    def create(cls, name: str) -> PreprocessingPipeline:
        """Create a preprocessing pipeline by name."""
        if name not in cls._pipelines:
            raise ValueError(f"Unknown preprocessing: {name}. Available: {list(cls._pipelines.keys())}")
        return cls._pipelines[name]()
    
    @classmethod
    def get_available(cls) -> List[str]:
        """Get list of available preprocessing pipelines."""
        return list(cls._pipelines.keys())


def apply_preprocessing_to_df(df: pd.DataFrame, text_col: str, 
                              preprocessor: PreprocessingPipeline,
                              output_col: Optional[str] = None) -> pd.DataFrame:
    """Apply preprocessing to a dataframe column."""
    result_df = df.copy()
    target_col = output_col or text_col
    result_df[target_col] = df[text_col].apply(preprocessor)
    return result_df


def compare_preprocessing_results(df: pd.DataFrame, text_col: str = 'sentence',
                                 num_samples: int = 5) -> pd.DataFrame:
    """Compare different preprocessing pipelines on sample texts."""
    samples = df[text_col].head(num_samples).values
    pipelines = PreprocessingFactory.get_available()
    
    results = []
    for i, text in enumerate(samples):
        row = {'sample_id': i, 'original': text[:100]}  # Just show first 100 chars
        for pipeline_name in pipelines:
            preprocessor = PreprocessingFactory.create(pipeline_name)
            processed = preprocessor(text)
            row[pipeline_name] = processed[:100]
        results.append(row)
    
    return pd.DataFrame(results)


# Language detection utilities
class LanguageDetector:
    """
    Language detection using FastText's pre-trained language identification model.
    Accurate detection for German and English without heavy dependencies.
    
    Model URL: https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
    """
    
    _FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    _MODEL_PATH = None
    _PREPROCESSOR = AgressivePreproccessing_No_Punctuation()
    _MODEL = None
    _FALLBACK_WORDS = {
        'de': {'der', 'die', 'das', 'und', 'in', 'zu', 'ich', 'das', 'nicht', 'von', 'ein', 'für',
               'ä', 'ö', 'ü', 'ß'},
        'en': {'the', 'and', 'a', 'to', 'of', 'in', 'is', 'that', 'for', 'you', 'it', 'be'},
    }
    
    @staticmethod
    def _get_model_path() -> Path:
        """Get or create the model cache directory."""
        if LanguageDetector._MODEL_PATH is None:
            cache_dir = Path.home() / '.cache' / 'fasttext_models'
            cache_dir.mkdir(parents=True, exist_ok=True)
            LanguageDetector._MODEL_PATH = cache_dir
        return LanguageDetector._MODEL_PATH
    
    @staticmethod
    def _download_model() -> Path:
        """Download FastText language identification model if not cached."""
        model_path = LanguageDetector._get_model_path() / 'lid.176.bin'
        
        if model_path.exists():
            return model_path
        
        print(f"Downloading FastText language identification model (~355 MB)...")
        print(f"This may take a few minutes on first run.")
        
        try:
            import urllib.request
            urllib.request.urlretrieve(
                LanguageDetector._FASTTEXT_MODEL_URL,
                str(model_path),
                reporthook=LanguageDetector._download_progress
            )
            print(f"\n✓ Model downloaded to {model_path}")
            return model_path
        except Exception as e:
            print(f"⚠️  Failed to download model: {e}")
            print(f"Falling back to simple word-based detection")
            return None
    
    @staticmethod
    def _download_progress(block_num: int, block_size: int, total_size: int) -> None:
        """Report download progress."""
        downloaded = block_num * block_size
        percent = min(100, int((downloaded / total_size) * 100))
        print(f"\rDownload progress: {percent}%", end='', flush=True)
    
    @staticmethod
    def _load_fasttext_model():
        """Lazy load FastText model."""
        if LanguageDetector._MODEL is not None:
            return LanguageDetector._MODEL
        
        try:
            import fasttext
            # Suppress FastText warnings
            fasttext.FastText.eprint = lambda x: None
            
            model_path = LanguageDetector._download_model()
            if model_path and model_path.exists():
                LanguageDetector._MODEL = fasttext.load_model(str(model_path))
                return LanguageDetector._MODEL
        except ImportError:
            print("⚠️  fasttext not installed. Install with: pip install fasttext")
        except Exception as e:
            print(f"⚠️  Error loading FastText model: {e}")
        
        return None
    
    @staticmethod
    def detect_language(text: str, use_fasttext: bool = True) -> str:
        """
        Detect language of text (German or English).
        
        Args:
            text: Text to detect language for
            use_fasttext: Try FastText first, fall back to simple detection
        
        Returns:
            Language code: 'de', 'en', or 'unknown'
        """
        if not text or len(text.strip()) < 3:
            return 'unknown'
        
        # Try FastText first

        if use_fasttext:
            model = LanguageDetector._MODEL
            if model is not None:
                try:
                    # FastText returns predictions with __label__ prefix
                    predictions = model.predict(text.replace('\n', ' '), k=10)
                    if predictions and predictions[0]:
                        for i in range(10):
                            lang_label = predictions[0][i].replace('__label__', '')
                            
                            # Map to our format (de/en)
                            if lang_label.endswith('de'):
                                return 'de'
                            elif lang_label.endswith('en'):
                                return 'en'
                except Exception as e:
                    print(f"⚠️  FastText detection failed: {e}. Falling back to simple detection.")
        
        # Fallback: simple word matching
        print(text)
        return LanguageDetector._detect_language_fallback(text)
    
    @staticmethod
    def _detect_language_fallback(text: str) -> str:
        """Fallback language detection using word frequency."""
        words = set(text.lower().split())
        
        scores = {}
        for lang, lang_words in LanguageDetector._FALLBACK_WORDS.items():
            scores[lang] = len(words & lang_words)
        
        # Check for German umlauts (strong indicator)
        umlaut_count = sum(1 for c in text.lower() if c in {'ä', 'ö', 'ü', 'ß'})
        if umlaut_count >= 2:
            return 'de'
        
        if scores['de'] > scores['en']:
            return 'de'
        elif scores['en'] > scores['de']:
            return 'en'
        else:
            return 'unknown'
    
    @staticmethod
    def detect_language_batch(texts: pd.DataFrame, use_fasttext: bool = True) -> pd.Series:
        """Detect language for a batch of texts."""
        texts['sentence'] = texts['sentence'].apply(LanguageDetector._PREPROCESSOR)
        return texts['sentence'].apply(lambda x: LanguageDetector.detect_language(x, use_fasttext=use_fasttext))
    
    @staticmethod
    def get_language_confidence(text: str) -> Dict[str, float]:
        """
        Get confidence scores for each language.
        
        Returns:
            Dict with 'de' and 'en' confidence scores (0-1)
        """
        try:
            model = LanguageDetector._load_fasttext_model()
            if model is not None:
                predictions = model.predict(text.replace('\n', ' '), k=2)
                if predictions and len(predictions[0]) >= 2:
                    scores = {}
                    for label, conf in zip(predictions[0], predictions[1]):
                        lang = label.replace('__label__', '')
                        if lang in ['de', 'en']:
                            scores[lang] = float(conf)
                    
                    # Ensure both languages are present
                    if 'de' not in scores:
                        scores['de'] = 0.0
                    if 'en' not in scores:
                        scores['en'] = 0.0
                    
                    total = scores['de'] + scores['en']
                    if total > 0:
                        return {
                            'de': scores['de'] / total if total > 0 else 0.5,
                            'en': scores['en'] / total if total > 0 else 0.5,
                        }
        except Exception as e:
            print(f"⚠️  Error getting confidence: {e}")
        
        # Default fallback
        return {'de': 0.5, 'en': 0.5}
    
    @staticmethod
    def clear_cache() -> None:
        """Clear cached model from memory (useful for memory-limited systems)."""
        LanguageDetector._MODEL = None
