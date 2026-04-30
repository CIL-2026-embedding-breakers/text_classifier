"""
Utility functions for baseline experiments.
"""
import json
import pickle
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, confusion_matrix, 
    classification_report
)


class MetricsComputer:
    """Compute comprehensive metrics for classification and regression."""
    
    @staticmethod
    def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                      labels: Optional[List[int]] = None) -> Dict[str, float]:
        """Compute classification metrics."""
        if labels is None:
            labels = sorted(np.unique(y_true))
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', labels=labels, zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', labels=labels, zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0),
        }
    
    @staticmethod
    def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute regression metrics."""
        from sklearn.metrics import r2_score
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'score': 1.0 - (mae / 4.0),  # Custom competition metric
        }
    
    @staticmethod
    def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                           task: str = 'classification') -> Dict[str, float]:
        """Compute both classification and regression metrics for flexibility."""
        metrics = {}
        
        # Always compute regression metrics
        metrics.update(MetricsComputer.compute_regression_metrics(y_true, y_pred))
        
        # For classification task, also include classification metrics
        if task == 'classification':
            metrics.update(MetricsComputer.compute_classification_metrics(y_true, y_pred))
        
        return metrics


class ExperimentLogger:
    """Log and track experiment results."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def log_experiment(self, config: Dict[str, Any], metrics: Dict[str, float], 
                      train_metrics: Optional[Dict[str, float]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a single experiment result."""
        result = {
            'config': config,
            'test_metrics': metrics,
            'train_metrics': train_metrics,
            'metadata': metadata or {},
        }
        self.results.append(result)
    
    def save_results_csv(self, filename: str = 'results.csv') -> None:
        """Save flattened results to CSV for easy comparison."""
        rows = []
        for result in self.results:
            row = {'exp_name': result['config'].get('name', 'unknown')}
            row.update({f"test_{k}": v for k, v in result['test_metrics'].items()})
            if result['train_metrics']:
                row.update({f"train_{k}": v for k, v in result['train_metrics'].items()})
            row.update(result['config'])
            rows.append(row)
        
        df = pd.DataFrame(rows)
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    
    def save_json(self, filename: str = 'results.json') -> None:
        """Save full results to JSON."""
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Full results saved to {output_path}")


class DataAnalyzer:
    """Analyze model failures and patterns."""
    
    @staticmethod
    def analyze_failures(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
                         text_col: str = 'sentence', threshold: int = 2) -> pd.DataFrame:
        """Identify and analyze failure cases."""
        analysis_df = df.copy()
        analysis_df['true_label'] = y_true
        analysis_df['pred_label'] = y_pred
        analysis_df['error'] = np.abs(y_true - y_pred)
        analysis_df['text_length'] = analysis_df[text_col].str.len()
        analysis_df['text_tokens'] = analysis_df[text_col].str.split().str.len()
        
        high_error = analysis_df[analysis_df['error'] >= threshold].copy()
        return high_error.sort_values('error', ascending=False)
    
    @staticmethod
    def compute_error_statistics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Compute detailed error statistics."""
        errors = np.abs(y_true - y_pred)
        
        stats = {
            'mean_error': float(errors.mean()),
            'median_error': float(np.median(errors)),
            'std_error': float(errors.std()),
            'max_error': int(errors.max()),
            'min_error': int(errors.min()),
            'error_distribution': {},
            'perfect_predictions': int((errors == 0).sum()),
            'large_errors': int((errors >= 2).sum()),
        }
        
        # Error distribution
        for i in range(int(errors.max()) + 1):
            count = int((errors == i).sum())
            pct = 100.0 * count / len(errors)
            stats['error_distribution'][f'error_{i}'] = {'count': count, 'percentage': pct}
        
        return stats
    
    @staticmethod
    def compute_label_distribution_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Compute per-label metrics."""
        metrics_by_label = {}
        
        for label in sorted(np.unique(y_true)):
            mask = y_true == label
            if mask.sum() == 0:
                continue
            
            label_true = y_true[mask]
            label_pred = y_pred[mask]
            label_errors = np.abs(label_true - label_pred)
            
            metrics_by_label[f'label_{int(label)}'] = {
                'count': int(mask.sum()),
                'mae': float(label_errors.mean()),
                'accuracy': float((label_errors == 0).sum() / len(label_errors)),
                'mean_prediction': float(label_pred.mean()),
                'std_prediction': float(label_pred.std()),
            }
        
        return metrics_by_label
    
    @staticmethod
    def analyze_language_performance(df: pd.DataFrame, y_true: np.ndarray, 
                                     y_pred: np.ndarray, language_col: str = 'language') -> Dict[str, Any]:
        """Analyze performance by language."""
        if language_col not in df.columns:
            return {}
        
        lang_metrics = {}
        for lang in df[language_col].unique():
            mask = df[language_col] == lang
            if mask.sum() == 0:
                continue
            
            lang_true = y_true[mask]
            lang_pred = y_pred[mask]
            errors = np.abs(lang_true - lang_pred)
            
            lang_metrics[f'lang_{lang}'] = {
                'count': int(mask.sum()),
                'mae': float(errors.mean()),
                'accuracy': float((errors == 0).sum() / len(errors)),
            }
        
        return lang_metrics
    
    @staticmethod
    def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray, 
                               output_path: Optional[str] = None) -> None:
        """Plot error distribution."""
        errors = np.abs(y_true - y_pred)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].hist(errors, bins=20, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Absolute Error')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Error Distribution')
        
        axes[1].scatter(y_true, y_pred, alpha=0.3, s=10)
        axes[1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[1].set_xlabel('True Label')
        axes[1].set_ylabel('Predicted Label')
        axes[1].set_title('Predictions vs Ground Truth')
        
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                             labels: Optional[List[int]] = None,
                             output_path: Optional[str] = None) -> None:
        """Plot confusion matrix."""
        if labels is None:
            labels = sorted(np.unique(y_true))
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, 
                   yticklabels=labels, cbar=True)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        if output_path:
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()


def save_pickle(obj: Any, filepath: str) -> None:
    """Save object to pickle file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Saved to {filepath}")


def load_pickle(filepath: str) -> Any:
    """Load object from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
