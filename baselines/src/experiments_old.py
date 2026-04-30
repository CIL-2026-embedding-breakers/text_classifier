"""
Experiment runner and orchestration.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from .models import BaselineModel, create_model_from_config
from .preprocessing import PreprocessingFactory, LanguageDetector
from .utils import MetricsComputer, ExperimentLogger, DataAnalyzer


class ExperimentRunner:
    """Run baseline experiments systematically."""
    
    def __init__(self, output_dir: str = 'outputs'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = ExperimentLogger(output_dir)
        self.results = []
    
    def run_experiment(self, model: BaselineModel, train_texts: np.ndarray, 
                      train_labels: np.ndarray, val_texts: np.ndarray,
                      val_labels: np.ndarray, config: Dict[str, Any],
                      test_texts: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run a single experiment."""
        
        print(f"\nRunning: {config.get('name', 'unnamed')}")
        print(f"Config: {config}")
        
        # Fit model
        model.fit(train_texts, train_labels)
        
        # Evaluate on train and validation
        train_preds = model.predict(train_texts)
        val_preds = model.predict(val_texts)
        
        task = config.get('task', 'classification')
        train_metrics = MetricsComputer.compute_all_metrics(train_labels, train_preds, task=task)
        val_metrics = MetricsComputer.compute_all_metrics(val_labels, val_preds, task=task)
        
        print(f"  Train MAE: {train_metrics['mae']:.4f}, Test MAE: {val_metrics['mae']:.4f}")
        
        # Log results
        self.logger.log_experiment(config, val_metrics, train_metrics)
        
        result = {
            'config': config,
            'model': model,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'train_preds': train_preds,
            'val_preds': val_preds,
        }
        
        # Make test predictions if provided
        if test_texts is not None:
            test_preds = model.predict(test_texts)
            result['test_preds'] = test_preds
        
        self.results.append(result)
        return result
    
    def run_multiple_experiments(self, configs: List[Dict[str, Any]],
                                train_texts: np.ndarray, train_labels: np.ndarray,
                                val_texts: np.ndarray, val_labels: np.ndarray,
                                test_texts: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """Run multiple experiments."""
        results = []
        for config in tqdm(configs, desc="Running experiments"):
            try:
                model = create_model_from_config(config)
                result = self.run_experiment(model, train_texts, train_labels,
                                            val_texts, val_labels, config, test_texts)
                results.append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
        
        return results
    
    def save_results(self) -> None:
        """Save all results."""
        self.logger.save_results_csv('results.csv')
        self.logger.save_json('results.json')
    
    def get_best_result(self, metric: str = 'mae', minimize: bool = True) -> Dict[str, Any]:
        """Get best performing experiment."""
        if not self.results:
            raise ValueError("No results to compare")
        
        best = min(self.results, key=lambda r: r['val_metrics'][metric]) if minimize else \
               max(self.results, key=lambda r: r['val_metrics'][metric])
        return best


class BaselineComparison:
    """Compare different baseline approaches."""
    
    def __init__(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                 test_df: Optional[pd.DataFrame] = None,
                 text_col: str = 'sentence', label_col: str = 'label',
                 output_dir: str = 'outputs'):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.text_col = text_col
        self.label_col = label_col
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def compare_objectives(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Compare classification vs regression objectives."""
        print("\n" + "="*60)
        print("COMPARING OBJECTIVES: Classification vs Regression")
        print("="*60)
        
        configs = [
            {
                'name': 'obj_classification',
                'type': 'bow_mlp',
                'task': 'classification',
                'ngram_range': [1, 2],
                'max_features': 10000,
                'hidden_layers': [256, 128],
                'vectorizer_type': 'count'
            },
            {
                'name': 'obj_regression',
                'type': 'bow_mlp',
                'task': 'regression',
                'ngram_range': [1, 2],
                'max_features': 10000,
                'hidden_layers': [256, 128],
                'vectorizer_type': 'count'
            }
        ]
        
        train_texts = self.train_df[self.text_col].values
        train_labels = self.train_df[self.label_col].values
        val_texts = self.val_df[self.text_col].values
        val_labels = self.val_df[self.label_col].values
        test_texts = self.test_df[self.text_col].values if self.test_df is not None else None
        
        runner = ExperimentRunner(str(self.output_dir / 'objectives'))
        results = runner.run_multiple_experiments(configs, train_texts, train_labels,
                                                 val_texts, val_labels, test_texts)


        runner.save_results()
        
        return results[0], results[1]
    
    def compare_preprocessing(self) -> List[Dict[str, Any]]:
        """Compare different preprocessing strategies."""
        print("\n" + "="*60)
        print("COMPARING PREPROCESSING STRATEGIES")
        print("="*60)
        
        preprocessings = ['none', 'basic', 'aggressive', 'extended']
        configs = []
        
        for prep in preprocessings:
            configs.append({
                'name': f'prep_{prep}',
                'type': 'bow_mlp',
                'task': 'classification',
                'preprocessing': prep,
                'ngram_range': [1, 2],
                'max_features': 10000,
                'hidden_layers': [256, 128],
                'vectorizer_type': 'count'
            })
        
        results = []
        for prep, config in zip(preprocessings, configs):
            # Apply preprocessing
            prep_factory = PreprocessingFactory.create(prep)
            train_df_prep = self.train_df.copy()
            train_df_prep[self.text_col] = train_df_prep[self.text_col].apply(prep_factory)
            
            val_df_prep = self.val_df.copy()
            val_df_prep[self.text_col] = val_df_prep[self.text_col].apply(prep_factory)
            
            train_texts = train_df_prep[self.text_col].values
            train_labels = train_df_prep[self.label_col].values
            val_texts = val_df_prep[self.text_col].values
            val_labels = val_df_prep[self.label_col].values
            test_texts = None
            if self.test_df is not None:
                test_df_prep = self.test_df.copy()
                test_df_prep[self.text_col] = test_df_prep[self.text_col].apply(prep_factory)
                test_texts = test_df_prep[self.text_col].values
            
            runner = ExperimentRunner(str(self.output_dir / f'preprocessing_{prep}'))
            result = runner.run_experiment(
                create_model_from_config(config),
                train_texts, train_labels, val_texts, val_labels,
                config, test_texts
            )
            runner.save_results()
            results.append(result)
        
        return results
    
    def compare_text_components(self) -> Dict[str, Dict[str, Any]]:
        """Compare using title only vs title+text."""
        print("\n" + "="*60)
        print("COMPARING TEXT COMPONENTS (Title only vs Title+Text)")
        print("="*60)
        
        # Check if data has separate title column
        if 'title' not in self.train_df.columns:
            print("  WARNING: No 'title' column found. Using text_col as is.")
            return {}
        
        components = {
            'title_only': 'title',
            'text_only': self.text_col,
        }
        
        results = {}
        
        for comp_name, text_col in components.items():
            config = {
                'name': f'component_{comp_name}',
                'type': 'bow_mlp',
                'task': 'classification',
                'ngram_range': [1, 2],
                'max_features': 10000,
                'hidden_layers': [256, 128],
                'vectorizer_type': 'count'
            }
            
            train_texts = self.train_df[text_col].values
            train_labels = self.train_df[self.label_col].values
            val_texts = self.val_df[text_col].values
            val_labels = self.val_df[self.label_col].values
            test_texts = self.test_df[text_col].values if self.test_df is not None else None
            
            runner = ExperimentRunner(str(self.output_dir / f'component_{comp_name}'))
            result = runner.run_experiment(
                create_model_from_config(config),
                train_texts, train_labels, val_texts, val_labels,
                config, test_texts
            )
            runner.save_results()
            results[comp_name] = result
        
        return results
    
    def compare_language_specific(self) -> Dict[str, Dict[str, Any]]:
        """Compare language-specific models vs multilingual."""
        print("\n" + "="*60)
        print("COMPARING LANGUAGE-SPECIFIC vs MULTILINGUAL MODELS")
        print("="*60)
        
        # Detect languages
        self.train_df['language'] = LanguageDetector.detect_language_batch(
            self.train_df[self.text_col]
        )
        self.val_df['language'] = LanguageDetector.detect_language_batch(
            self.val_df[self.text_col]
        )
        
        print(f"  Train language distribution:\n{self.train_df['language'].value_counts()}")
        print(f"  Val language distribution:\n{self.val_df['language'].value_counts()}")
        
        results = {}
        
        # Multilingual model
        config_multi = {
            'name': 'lang_multilingual',
            'type': 'bow_mlp',
            'task': 'classification',
            'ngram_range': [1, 2],
            'max_features': 10000,
            'hidden_layers': [256, 128],
            'vectorizer_type': 'count'
        }
        
        train_texts = self.train_df[self.text_col].values
        train_labels = self.train_df[self.label_col].values
        val_texts = self.val_df[self.text_col].values
        val_labels = self.val_df[self.label_col].values
        test_texts = self.test_df[self.text_col].values if self.test_df is not None else None
        
        runner_multi = ExperimentRunner(str(self.output_dir / 'lang_multilingual'))
        result_multi = runner_multi.run_experiment(
            create_model_from_config(config_multi),
            train_texts, train_labels, val_texts, val_labels,
            config_multi, test_texts
        )
        runner_multi.save_results()
        results['multilingual'] = result_multi
        
        # Language-specific models
        for lang in self.train_df['language'].unique():
            if lang == 'unknown':
                continue
            
            train_mask = self.train_df['language'] == lang
            val_mask = self.val_df['language'] == lang
            
            if (~train_mask).all() or (~val_mask).all():
                print(f"  Skipping language {lang} (insufficient samples)")
                continue
            
            config_lang = {
                'name': f'lang_{lang}',
                'type': 'bow_mlp',
                'task': 'classification',
                'language': lang,
                'ngram_range': [1, 2],
                'max_features': 10000,
                'hidden_layers': [256, 128],
                'vectorizer_type': 'count'
            }
            
            train_texts_lang = self.train_df[train_mask][self.text_col].values
            train_labels_lang = self.train_df[train_mask][self.label_col].values
            val_texts_lang = self.val_df[val_mask][self.text_col].values
            val_labels_lang = self.val_df[val_mask][self.label_col].values
            
            runner_lang = ExperimentRunner(str(self.output_dir / f'lang_{lang}'))
            result_lang = runner_lang.run_experiment(
                create_model_from_config(config_lang),
                train_texts_lang, train_labels_lang, val_texts_lang, val_labels_lang,
                config_lang, None
            )
            runner_lang.save_results()
            results[f'lang_{lang}'] = result_lang
        
        return results
