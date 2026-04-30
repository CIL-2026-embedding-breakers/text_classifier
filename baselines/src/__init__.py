"""Baseline implementations for sentiment analysis."""

from .utils import MetricsComputer, ExperimentLogger, DataAnalyzer, save_pickle, load_pickle
from .preprocessing import (
    PreprocessingPipeline, PreprocessingFactory, LanguageDetector,
    apply_preprocessing_to_df, compare_preprocessing_results
)
from .models import (
    BaselineModel, BOWLogisticRegression, BOWLogisticRegressionWeighted,
    BOWLinearRegression, BOWNaiveBayes, BOWLinearSVC,
    create_model_from_config
)
from .experiments import ExperimentRunner, BaselineComparison

__all__ = [
    'MetricsComputer', 'ExperimentLogger', 'DataAnalyzer',
    'PreprocessingFactory', 'LanguageDetector',
    'BaselineModel', 'BOWLogisticRegression', 'BOWLogisticRegressionWeighted',
    'BOWLinearRegression', 'BOWNaiveBayes', 'BOWLinearSVC',
    'ExperimentRunner', 'BaselineComparison',
    'save_pickle', 'load_pickle'
]
