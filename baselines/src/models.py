"""
Baseline models for sentiment analysis using simple sklearn models.
Focuses on Logistic Regression (classification and regression variants) for fast, interpretable baselines.
"""
import numpy as np
from typing import Dict, Optional, Tuple, Any
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
import pandas as pd


class BaselineModel:
    """Base class for all baseline models."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.vectorizer = None
        self.is_fitted = False
    
    def fit(self, X_text: np.ndarray, y: np.ndarray) -> None:
        """Fit the model."""
        raise NotImplementedError
    
    def predict(self, X_text: np.ndarray) -> np.ndarray:
        """Make predictions."""
        raise NotImplementedError
    
    def predict_proba(self, X_text: np.ndarray) -> Optional[np.ndarray]:
        """Get prediction probabilities (if available)."""
        return None


class BOWLogisticRegression(BaselineModel):
    """Bag-of-Words + Logistic Regression baseline (Classification)."""
    
    def __init__(self, name: str = "bow_lr", ngram_range: Tuple[int, int] = (1, 2),
                 max_features: int = 10000, C: float = 1.0, max_iter: int = 1000,
                 vectorizer_type: str = 'tfidf', solver: str = 'lbfgs',
                 penalty: str = 'l2', lemmatize: bool = False):
        super().__init__(name)
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.C = C
        self.max_iter = max_iter
        self.vectorizer_type = vectorizer_type
        self.solver = solver
        self.penalty = penalty
        self.lemmatize = lemmatize
        
        # Validate penalty and solver compatibility
        if penalty == 'l1' and solver not in ['liblinear', 'saga']:
            print(f"Warning: L1 penalty requires 'liblinear' or 'saga' solver. Changing from '{solver}' to 'liblinear'")
            self.solver = 'liblinear'
    
    def fit(self, X_text: np.ndarray, y: np.ndarray) -> None:
        """Fit BOW vectorizer and logistic regression."""
        
        # Create vectorizer with optional lemmatization
        analyzer = None
        if self.lemmatize:
            try:
                from .lemmatizer import LanguageAwareLemmatizer
                lemmatizer = LanguageAwareLemmatizer
                # Define custom analyzer function for lemmatization
                def lem_analyzer(text):
                    lemmatized = lemmatizer.lemmatize(text)
                    return lemmatized.lower().split()
                analyzer = lem_analyzer
            except Exception as e:
                print(f"Warning: Could not load lemmatizer ({e}), using default analyzer")
        
        if self.vectorizer_type == 'count':
            if analyzer:
                self.vectorizer = CountVectorizer(
                    ngram_range=self.ngram_range,
                    max_features=self.max_features,
                    analyzer=analyzer
                )
            else:
                self.vectorizer = CountVectorizer(
                    ngram_range=self.ngram_range,
                    max_features=self.max_features
                )
        elif self.vectorizer_type == 'tfidf':
            if analyzer:
                self.vectorizer = TfidfVectorizer(
                    ngram_range=self.ngram_range,
                    max_features=self.max_features,
                    analyzer=analyzer
                )
            else:
                self.vectorizer = TfidfVectorizer(
                    ngram_range=self.ngram_range,
                    max_features=self.max_features
                )
        else:
            raise ValueError(f"Unknown vectorizer type: {self.vectorizer_type}")
        
        X_vectors = self.vectorizer.fit_transform(X_text)
        
        # Use 'ovr' (one-vs-rest) for liblinear compatibility with multiclass
        
        self.model = LogisticRegression(
            C=self.C, 
            max_iter=self.max_iter,
            solver=self.solver,
            penalty=self.penalty,
            n_jobs=-1,
            random_state=42
        )
        self.model.fit(X_vectors, y)
        self.is_fitted = True
    
    def predict(self, X_text: np.ndarray) -> np.ndarray:
        """Predict labels."""
        assert self.is_fitted, "Model must be fitted first"
        X_vectors = self.vectorizer.transform(X_text)
        return self.model.predict(X_vectors)
    
    def predict_proba(self, X_text: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        assert self.is_fitted, "Model must be fitted first"
        X_vectors = self.vectorizer.transform(X_text)
        return self.model.predict_proba(X_vectors)
    
    def get_feature_names(self):
        """Get feature names (vocabulary)."""
        return self.vectorizer.get_feature_names_out()
    
    def get_top_features(self, class_idx: int, top_n: int = 10):
        """Get top features for a specific class."""
        coef = self.model.coef_[class_idx]
        top_indices = np.argsort(np.abs(coef))[-top_n:]
        feature_names = self.get_feature_names()
        return [(feature_names[i], coef[i]) for i in reversed(top_indices)]
    
    def get_sparsity(self):
        """Get sparsity of coefficients (percentage of zero coefficients)."""
        coef = self.model.coef_
        num_nonzero = np.count_nonzero(coef)
        total = coef.size
        sparsity = 100.0 * (1 - num_nonzero / total)
        return sparsity


class BOWLogisticRegressionWeighted(BaselineModel):
    """Logistic Regression outputting continuous scores via weighted class average."""
    
    def __init__(self, name: str = "bow_lr_weighted", ngram_range: Tuple[int, int] = (1, 2),
                 max_features: int = 10000, C: float = 1.0, max_iter: int = 100,
                 vectorizer_type: str = 'tfidf', solver: str = 'lbfgs',
                 penalty: str = 'l2', lemmatize: bool = False):
        super().__init__(name)
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.C = C
        self.max_iter = max_iter
        self.vectorizer_type = vectorizer_type
        self.solver = solver
        self.penalty = penalty
        self.lemmatize = lemmatize
        
        # Validate penalty and solver compatibility
        if penalty == 'l1' and solver not in ['liblinear', 'saga']:
            print(f"Warning: L1 penalty requires 'liblinear' or 'saga' solver. Changing from '{solver}' to 'liblinear'")
            self.solver = 'liblinear'
    
    def fit(self, X_text: np.ndarray, y: np.ndarray) -> None:
        """Fit BOW vectorizer and logistic regression."""
        # Create vectorizer with optional lemmatization
        analyzer = None
        if self.lemmatize:
            try:
                from .lemmatizer import LanguageAwareLemmatizer
                lemmatizer = LanguageAwareLemmatizer
                # Define custom analyzer function for lemmatization
                def lem_analyzer(text):
                    lemmatized = lemmatizer.lemmatize(text)
                    return lemmatized.lower().split()
                analyzer = lem_analyzer
            except Exception as e:
                print(f"Warning: Could not load lemmatizer ({e}), using default analyzer")
        
        if self.vectorizer_type == 'count':
            if analyzer:
                self.vectorizer = CountVectorizer(
                    ngram_range=self.ngram_range,
                    max_features=self.max_features,
                    analyzer=analyzer
                )
            else:
                self.vectorizer = CountVectorizer(
                    ngram_range=self.ngram_range,
                    max_features=self.max_features
                )
        elif self.vectorizer_type == 'tfidf':
            if analyzer:
                self.vectorizer = TfidfVectorizer(
                    ngram_range=self.ngram_range,
                    max_features=self.max_features,
                    analyzer=analyzer
                )
            else:
                self.vectorizer = TfidfVectorizer(
                    ngram_range=self.ngram_range,
                    max_features=self.max_features
                )
        else:
            raise ValueError(f"Unknown vectorizer type: {self.vectorizer_type}")
        
        X_vectors = self.vectorizer.fit_transform(X_text)
        
        
        # Train logistic regression for classification
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver=self.solver,
            penalty=self.penalty,
            n_jobs=-1,
            random_state=42
        )
        self.model.fit(X_vectors, y)
        self.is_fitted = True
    
    def predict(self, X_text: np.ndarray) -> np.ndarray:
        """Predict continuous scores via weighted class average."""
        assert self.is_fitted, "Model must be fitted first"
        X_vectors = self.vectorizer.transform(X_text)
        proba = self.model.predict_proba(X_vectors)
        
        # Weighted average: sum of (class_label * probability)
        classes = self.model.classes_
        predictions = np.dot(proba, classes)
        return np.clip(np.round(predictions), 1, 5).astype(int)
    
    def get_sparsity(self):
        """Get sparsity of coefficients (percentage of zero coefficients)."""
        coef = self.model.coef_
        num_nonzero = np.count_nonzero(coef)
        total = coef.size
        sparsity = 100.0 * (1 - num_nonzero / total)
        return sparsity


class BOWLinearRegression(BaselineModel):
    """Bag-of-Words + Linear Regression baseline."""
    
    def __init__(self, name: str = "bow_linear_reg", ngram_range: Tuple[int, int] = (1, 2),
                 max_features: int = 10000, vectorizer_type: str = 'tfidf', 
                 nearest_label: bool = False):
        super().__init__(name)
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vectorizer_type = vectorizer_type
        self.nearest_label = nearest_label  # If True, round to nearest label
    
    def fit(self, X_text: np.ndarray, y: np.ndarray) -> None:
        """Fit BOW vectorizer and linear regression."""
        if self.vectorizer_type == 'count':
            self.vectorizer = CountVectorizer(ngram_range=self.ngram_range,
                                             max_features=self.max_features)
        elif self.vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range,
                                             max_features=self.max_features)
        else:
            raise ValueError(f"Unknown vectorizer type: {self.vectorizer_type}")
        
        X_vectors = self.vectorizer.fit_transform(X_text)
        
        self.model = LinearRegression(n_jobs=-1)
        self.model.fit(X_vectors, y)
        self.is_fitted = True
    
    def predict(self, X_text: np.ndarray) -> np.ndarray:
        """Predict labels."""
        assert self.is_fitted, "Model must be fitted first"
        X_vectors = self.vectorizer.transform(X_text)
        predictions = self.model.predict(X_vectors)
        
        if self.nearest_label:
            # Assign to nearest valid label (1-5)
            predictions = np.clip(np.round(predictions), 1, 5).astype(int)
        else:
            predictions = np.clip(predictions, 1, 5).astype(int)
        
        return predictions


class BOWNaiveBayes(BaselineModel):
    """Bag-of-Words + Multinomial Naive Bayes baseline (fast)."""
    
    def __init__(self, name: str = "bow_nb", ngram_range: Tuple[int, int] = (1, 2),
                 max_features: int = 10000, alpha: float = 1.0):
        super().__init__(name)
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.alpha = alpha
    
    def fit(self, X_text: np.ndarray, y: np.ndarray) -> None:
        """Fit BOW vectorizer and Naive Bayes."""
        self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range,
                                         max_features=self.max_features)
        X_vectors = self.vectorizer.fit_transform(X_text)
        
        self.model = MultinomialNB(alpha=self.alpha)
        self.model.fit(X_vectors, y)
        self.is_fitted = True
    
    def predict(self, X_text: np.ndarray) -> np.ndarray:
        """Predict labels."""
        assert self.is_fitted, "Model must be fitted first"
        X_vectors = self.vectorizer.transform(X_text)
        return self.model.predict(X_vectors)
    
    def predict_proba(self, X_text: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        assert self.is_fitted, "Model must be fitted first"
        X_vectors = self.vectorizer.transform(X_text)
        return self.model.predict_proba(X_vectors)


class BOWLinearSVC(BaselineModel):
    """Bag-of-Words + Linear SVC baseline (one-vs-rest multi-class)."""
    
    def __init__(self, name: str = "bow_svc", ngram_range: Tuple[int, int] = (1, 2),
                 max_features: int = 10000, C: float = 1.0, vectorizer_type: str = 'tfidf'):
        super().__init__(name)
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.C = C
        self.vectorizer_type = vectorizer_type
    
    def fit(self, X_text: np.ndarray, y: np.ndarray) -> None:
        """Fit BOW vectorizer and Linear SVC."""
        if self.vectorizer_type == 'count':
            self.vectorizer = CountVectorizer(ngram_range=self.ngram_range,
                                             max_features=self.max_features)
        elif self.vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range,
                                             max_features=self.max_features)
        else:
            raise ValueError(f"Unknown vectorizer type: {self.vectorizer_type}")
        
        X_vectors = self.vectorizer.fit_transform(X_text)
        
        self.model = LinearSVC(
            C=self.C,
            max_iter=2000,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_vectors, y)
        self.is_fitted = True
    
    def predict(self, X_text: np.ndarray) -> np.ndarray:
        """Predict labels."""
        assert self.is_fitted, "Model must be fitted first"
        X_vectors = self.vectorizer.transform(X_text)
        return self.model.predict(X_vectors)



def create_model_from_config(config: Dict[str, Any]) -> BaselineModel:
    """Create a model from configuration dict."""
    model_type = config['type']
    
    if model_type == 'bow_lr':
        return BOWLogisticRegression(
            name=config.get('name', 'bow_lr'),
            ngram_range=tuple(config.get('ngram_range', [1, 2])),
            max_features=config.get('max_features', 10000),
            C=config.get('C', 1.0),
            max_iter=config.get('max_iter', 100),
            vectorizer_type=config.get('vectorizer_type', 'tfidf'),
            solver=config.get('solver', 'lbfgs'),
            penalty=config.get('penalty', 'l2'),
            lemmatize=config.get('lemmatize', False)
        )
    elif model_type == 'bow_lr_weighted':
        return BOWLogisticRegressionWeighted(
            name=config.get('name', 'bow_lr_weighted'),
            ngram_range=tuple(config.get('ngram_range', [1, 2])),
            max_features=config.get('max_features', 10000),
            C=config.get('C', 1.0),
            max_iter=config.get('max_iter', 100),
            vectorizer_type=config.get('vectorizer_type', 'tfidf'),
            solver=config.get('solver', 'lbfgs'),
            penalty=config.get('penalty', 'l2'),
            lemmatize=config.get('lemmatize', False)
        )
    elif model_type == 'bow_linear_reg':
        return BOWLinearRegression(
            name=config.get('name', 'bow_linear_reg'),
            ngram_range=tuple(config.get('ngram_range', [1, 2])),
            max_features=config.get('max_features', 10000),
            vectorizer_type=config.get('vectorizer_type', 'tfidf'),
            nearest_label=config.get('nearest_label', True)
        )
    elif model_type == 'bow_nb':
        return BOWNaiveBayes(
            name=config.get('name', 'bow_nb'),
            ngram_range=tuple(config.get('ngram_range', [1, 2])),
            max_features=config.get('max_features', 10000),
            alpha=config.get('alpha', 1.0)
        )
    elif model_type == 'bow_svc':
        return BOWLinearSVC(
            name=config.get('name', 'bow_svc'),
            ngram_range=tuple(config.get('ngram_range', [1, 2])),
            max_features=config.get('max_features', 10000),
            C=config.get('C', 1.0),
            vectorizer_type=config.get('vectorizer_type', 'tfidf')
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
