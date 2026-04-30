# Language-Aware Lemmatization Module
# Add this to the end of preprocessing.py

# Language-Aware Lemmatizer for Sentiment Analysis
# Supports German and English lemmatization using NLTK and spaCy

from typing import List, Optional
import pandas as pd


class LanguageAwareLemmatizer:
    """
    Language-aware lemmatizer that handles both English and German text.
    Automatically detects language and applies appropriate lemmatization.
    """
    
    _EN_LEMMATIZER = None
    _DE_LEMMATIZER = None
    
    # Basic German lemmatization rules (for cases where advanced lemmatizers are unavailable)
    _DE_LEMMA_RULES = {
        # Common German suffixes
        'en': '',  # verbs ending in -en
        'n': '',   # verbs ending in -n
        'e': '',   # adjectives/nouns ending in -e
        'er': '',  # comparative
        'est': '',  # superlative
        'heit': 'heit',  # keep -heit
        'keit': 'keit',  # keep -keit
        'ung': 'ung',  # keep -ung
        'lich': 'lich',  # keep -lich
    }
    
    @staticmethod
    def _load_english_lemmatizer():
        """Lazy load English lemmatizer (Spacy English model)."""
        if LanguageAwareLemmatizer._EN_LEMMATIZER is not None:
            return LanguageAwareLemmatizer._EN_LEMMATIZER
        try:
            import spacy
            # Try to load English model
            try:
                nlp = spacy.load('en_core_web_sm', disable=["parser", "ner", "senter"])
            except OSError:
                print("⚠️  English spaCy model not found. Installing...")
                import subprocess
                import sys
                subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm', '-q'])
                nlp = spacy.load('en_core_web_sm', disable=["parser", "ner", "senter"])
            
            LanguageAwareLemmatizer._EN_LEMMATIZER = nlp
            return LanguageAwareLemmatizer._EN_LEMMATIZER
        except ImportError:
            # Silently fail - will use fallback
            print('Using english fallback')
        except Exception as e:
            print('Using english fallback')
            return None
    
    @staticmethod
    def _load_german_lemmatizer():
        """Lazy load German lemmatizer (spaCy German model)."""
        if LanguageAwareLemmatizer._DE_LEMMATIZER is not None:
            return LanguageAwareLemmatizer._DE_LEMMATIZER
        
        try:
            import spacy
            
            # Try to load German model
            try:
                nlp = spacy.load('de_core_news_sm', disable=["parser", "ner", "senter"])
            except OSError:
                print("⚠️  German spaCy model not found. Installing...")
                import subprocess
                import sys
                subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'de_core_news_sm', '-q'])
                nlp = spacy.load('de_core_news_sm', disable=["parser", "ner", "senter"])
            
            LanguageAwareLemmatizer._DE_LEMMATIZER = nlp
            return LanguageAwareLemmatizer._DE_LEMMATIZER
        except ImportError:
            print('Using german fallback')
            return None
        except Exception as e:
            print('Using german fallback')
            return None
    
    @staticmethod
    def _lemmatize_english(text: str) -> List[str]:
        """Lemmatize English text using NLTK."""
        nlp = LanguageAwareLemmatizer._load_english_lemmatizer()
        if nlp is None:
            return text.split()
        
        doc = nlp(text)
        lemmatized = [token.lemma_ for token in doc]
        return lemmatized

    
    @staticmethod
    def _lemmatize_german(text: str) -> List[str]:
        """Lemmatize German text using spaCy."""
        nlp = LanguageAwareLemmatizer._load_german_lemmatizer()
        
        if nlp is None:
            # Fallback to simple rule-based lemmatization
            return LanguageAwareLemmatizer._lemmatize_german_fallback(text)
        
        doc = nlp(text)
        lemmatized = [token.lemma_ for token in doc]
        return lemmatized
    
    @staticmethod
    def _lemmatize_german_fallback(text: str) -> List[str]:
        """
        Fallback German lemmatization using simple suffix-stripping rules.
        Not as accurate as spaCy, but works without ML models.
        """
        words = text.split()
        lemmatized = []
        
        for word in words:
            lemma = word
            
            # Try to remove common suffixes
            if len(word) > 4:
                for suffix, replacement in LanguageAwareLemmatizer._DE_LEMMA_RULES.items():
                    if word.endswith(suffix) and len(word) > len(suffix) + 2:
                        lemma = word[:-len(suffix)] + replacement
                        break
            
            lemmatized.append(lemma)
        
        return lemmatized
    
    @staticmethod
    def lemmatize(text: str, language: Optional[str] = None) -> str:
        """
        Lemmatize text in either German or English.
        
        Args:
            text: Text to lemmatize
            language: Language code ('en', 'de', or None for auto-detection)
        
        Returns:
            Lemmatized text (space-separated lemmas)
        """
        if language is None:
            try:
                from .preprocessing import LanguageDetector
                language = LanguageDetector.detect_language(text)
            except Exception:
                # Fallback to simple heuristic
                language = 'en'  # Default to English
        
        if language == 'de':
            lemmatized = LanguageAwareLemmatizer._lemmatize_german(text)
        elif language == 'en':
            lemmatized = LanguageAwareLemmatizer._lemmatize_english(text)
        else:
            # Unknown language, return as-is
            lemmatized = text.split()
        
        return ' '.join(lemmatized)
    
    @staticmethod
    def lemmatize_batch(texts: pd.Series, language_series: Optional[pd.Series] = None) -> pd.Series:
        """
        Lemmatize a batch of texts.
        
        Args:
            texts: Series of texts to lemmatize
            language_series: Optional series of language codes
        
        Returns:
            Series of lemmatized texts
        """
        result = []
        
        for idx, text in texts.items():
            lang = language_series.iloc[idx] if language_series is not None else None
            lemmatized = LanguageAwareLemmatizer.lemmatize(text, language=lang)
            result.append(lemmatized)
        
        return pd.Series(result, index=texts.index)


def get_lemmatizer() -> LanguageAwareLemmatizer:
    """Get the language-aware lemmatizer instance."""
    return LanguageAwareLemmatizer
