"""
Baseline models for sentiment analysis using simple sklearn models.
Focuses on Logistic Regression (classification and regression variants) for fast, interpretable baselines.
"""
import numpy as np
from typing import Dict, Optional, Tuple, Any
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.ensemble import RandomForestRegressor 
import os

import numpy as np

class GloveVectorizer:
    def __init__(self, glove_path, dim=100):
        self.glove_path = glove_path
        self.dim = dim

    def fit(self, X, y=None):
        self.embeddings_ = {}
        with open(self.glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.rstrip().split(' ')
                word = values[0]
                try:
                    vector = np.asarray(values[1:], dtype='float32')
                except ValueError:
                    continue
                self.embeddings_[word] = vector
        # infer dim from the loaded vectors
        self.dim = len(next(iter(self.embeddings_.values())))
        return self

    def transform(self, X):
        # X is an iterable of documents (strings)
        out = np.zeros((len(X), self.dim), dtype='float32')
        for i, doc in enumerate(X):
            tokens = doc.split() if isinstance(doc, str) else doc
            vecs = [self.embeddings_[t] for t in tokens if t in self.embeddings_]
            if vecs:
                out[i] = np.mean(vecs, axis=0)
        return out

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)



class FastTextVectorizer():
    def __init__(self, vectors_path, dim=300):
        self.vectors_path = vectors_path
        self.dim = dim

    def fit(self, X, y=None):
        self.embeddings_ = {}
        with open(self.vectors_path, 'r', encoding='utf-8') as f:
            next(f)  # skip header line: "<vocab_size> <dim>"
            for line in f:
                values = line.rstrip().split(' ')
                word = values[0]
                try:
                    vector = np.asarray(values[1:], dtype='float32')
                except ValueError:
                    continue
                self.embeddings_[word] = vector
        self.dim = len(next(iter(self.embeddings_.values())))
        return self

    def transform(self, X):
        out = np.zeros((len(X), self.dim), dtype='float32')
        for i, doc in enumerate(X):
            tokens = doc.lower().split() if isinstance(doc, str) else doc
            vecs = [self.embeddings_[t] for t in tokens if t in self.embeddings_]
            if vecs:
                out[i] = np.mean(vecs, axis=0)
        return out

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


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
        
        self.vectorizer = get_vectorizer_class(self.vectorizer_type, self.ngram_range, self.max_features)
        
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

class XGBoostClassifier(BaselineModel):
    """Bag-of-Words + XGBoost Classifier baseline."""
    
    def __init__(self, name: str = "bow_xgb", ngram_range: Tuple[int, int] = (1, 2),
                 max_features: int = 10000, max_depth: int = 3, n_estimators: int = 1000,
                 learning_rate: float = 0.05, vectorizer_type: str = 'tfidf'):
        super().__init__(name)
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.vectorizer_type = vectorizer_type
    
    def fit(self, X_text: np.ndarray, y: np.ndarray) -> None:
        """Fit BOW vectorizer and XGBoost classifier."""
        
        self.vectorizer = get_vectorizer_class(self.vectorizer_type, self.ngram_range, self.max_features)
        
        X_vectors = self.vectorizer.fit_transform(X_text)

        self.model = XGBClassifier(
                max_depth=5,                
                n_estimators=1000,
                learning_rate=0.05,
                min_child_weight=10,         
                gamma=0.1,
                reg_alpha=1.0,
                reg_lambda=1.0,
                tree_method='hist',        
                random_state=42,
                n_jobs=-1,
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
    
class RandomForestClassifier(BaselineModel):
    """Bag-of-Words + Random Forest Classifier baseline."""
    
    def __init__(self, name: str = "bow_rf", ngram_range: Tuple[int, int] = (1, 2),
                 max_features: int = 10000, n_estimators: int = 500, max_depth: Optional[int] = None,
                 vectorizer_type: str = 'tfidf'):
        super().__init__(name)
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.vectorizer_type = vectorizer_type
    
    def fit(self, X_text: np.ndarray, y: np.ndarray) -> None:
        """Fit BOW vectorizer and Random Forest classifier."""
        
        self.vectorizer = get_vectorizer_class(self.vectorizer_type, self.ngram_range, self.max_features)
        
        X_vectors = self.vectorizer.fit_transform(X_text)
        
        self.model = SklearnRandomForestClassifier(
            n_estimators=300,            # 500+ is fine but slow at 200k
            max_depth=20,                # can go deeper than 10 with this much data
            min_samples_leaf=20,        # ~0.05% of data — scales with N
            min_samples_split=10,
            max_features=0.5,
            max_samples=0.5,             # row subsampling cuts training time too
            random_state=42,
            n_jobs=-1,
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

class XGBoostRegressor(BaselineModel):
    """Bag-of-Words + XGBoost Regressor baseline."""
    
    def __init__(self, name: str = "bow_xgb_reg", ngram_range: Tuple[int, int] = (1, 2),
                 max_features: int = 10000, max_depth: int = 3, n_estimators: int = 1000,
                 learning_rate: float = 0.05, vectorizer_type: str = 'tfidf'):
        super().__init__(name)
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.vectorizer_type = vectorizer_type
    
    def fit(self, X_text: np.ndarray, y: np.ndarray) -> None:
        """Fit BOW vectorizer and XGBoost regressor."""
        
        self.vectorizer = get_vectorizer_class(self.vectorizer_type, self.ngram_range, self.max_features)
        
        X_vectors = self.vectorizer.fit_transform(X_text)
        

        self.model = XGBRegressor(
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            colsample_bytree=0.3,
            tree_method='hist',
            min_child_weight=1,
            random_state=42
        )
        self.model.fit(X_vectors, y)
        self.is_fitted = True
    
    def predict(self, X_text: np.ndarray) -> np.ndarray:
        """Predict continuous scores."""
        assert self.is_fitted, "Model must be fitted first"
        X_vectors = self.vectorizer.transform(X_text)
        predictions = self.model.predict(X_vectors)
        return np.clip(predictions, 1, 5).astype(int)
    
class RandomForest_Regressor(BaselineModel):
    """Bag-of-Words + Random Forest Regressor baseline."""
    
    def __init__(self, name: str = "bow_rf_reg", ngram_range: Tuple[int, int] = (1, 2),
                 max_features: int = 10000, n_estimators: int = 500, max_depth: Optional[int] = None,
                 vectorizer_type: str = 'tfidf'):
        super().__init__(name)
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.vectorizer_type = vectorizer_type
    
    def fit(self, X_text: np.ndarray, y: np.ndarray) -> None:
        """Fit BOW vectorizer and Random Forest regressor."""
        
        self.vectorizer = get_vectorizer_class(self.vectorizer_type, self.ngram_range, self.max_features)
        
        X_vectors = self.vectorizer.fit_transform(X_text)
        
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=50,
            min_samples_leaf=5,   
            max_features='sqrt',
            n_jobs=-1
        )
        self.model.fit(X_vectors, y)
        self.is_fitted = True
    
    def predict(self, X_text: np.ndarray) -> np.ndarray:
        """Predict continuous scores."""
        assert self.is_fitted, "Model must be fitted first"
        X_vectors = self.vectorizer.transform(X_text)
        predictions = self.model.predict(X_vectors)
        return np.clip(predictions, 1, 5).astype(int)


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
        
        self.vectorizer = get_vectorizer_class(self.vectorizer_type, self.ngram_range, self.max_features)
        
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
        self.vectorizer = get_vectorizer_class(self.vectorizer_type, self.ngram_range, self.max_features)
        
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
        self.vectorizer = get_vectorizer_class(self.vectorizer_type, self.ngram_range, self.max_features)
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
        self.vectorizer = get_vectorizer_class(self.vectorizer_type, self.ngram_range, self.max_features)

        
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


def get_vectorizer_class(vectorizer_type: str, n_gram_range: Tuple[int, int], max_features: int):
    """Get vectorizer class based on type."""
    if vectorizer_type == 'count':
        return CountVectorizer(ngram_range=n_gram_range, max_features=max_features)
    elif vectorizer_type == 'tfidf':
        return TfidfVectorizer(ngram_range=n_gram_range, max_features=max_features)
    elif vectorizer_type == 'glove':
        return GloveVectorizer(glove_path=os.path.expanduser('~/Downloads/wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined.txt'), 
                               dim=max_features)
    elif vectorizer_type == 'fasttext':
        return FastTextVectorizer(vectors_path=os.path.expanduser('~/Downloads/cc.en.300.vec'), 
                                  dim=max_features)
    else:
        raise ValueError(f"Unknown vectorizer type: {vectorizer_type}")

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
    elif model_type == 'bow_xgb':
        return XGBoostClassifier(
            name=config.get('name', 'bow_xgb'),
            ngram_range=tuple(config.get('ngram_range', [1, 2])),
            max_features=config.get('max_features', 10000),
            max_depth=config.get('max_depth', 6),
            n_estimators=config.get('n_estimators', 100),
            learning_rate=config.get('learning_rate', 0.1),
            vectorizer_type=config.get('vectorizer_type', 'tfidf')
        )
    elif model_type == 'bow_xgb_reg':
        return XGBoostRegressor(
            name=config.get('name', 'bow_xgb_reg'),
            ngram_range=tuple(config.get('ngram_range', [1, 2])),
            max_features=config.get('max_features', 10000),
            max_depth=config.get('max_depth', 6),
            n_estimators=config.get('n_estimators', 100),
            learning_rate=config.get('learning_rate', 0.1),
            vectorizer_type=config.get('vectorizer_type', 'tfidf')
        )
    elif model_type == 'bow_rf_reg':
        return RandomForest_Regressor(
            name=config.get('name', 'bow_rf_reg'),
            ngram_range=tuple(config.get('ngram_range', [1, 2])),
            max_features=config.get('max_features', 10000),
            n_estimators=config.get('n_estimators', 100),
            max_depth=config.get('max_depth', None),
            vectorizer_type=config.get('vectorizer_type', 'tfidf')
        )
    elif model_type == 'bow_rf':
        return RandomForestClassifier(
            name=config.get('name', 'bow_rf'),
            ngram_range=tuple(config.get('ngram_range', [1, 2])),
            max_features=config.get('max_features', 10000),
            n_estimators=config.get('n_estimators', 100),
            max_depth=config.get('max_depth', None),
            vectorizer_type=config.get('vectorizer_type', 'tfidf')
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
