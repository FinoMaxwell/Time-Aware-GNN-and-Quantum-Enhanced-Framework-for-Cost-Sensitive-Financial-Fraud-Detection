"""
Classical Machine Learning Models for Fraud Detection
Baseline models for comparison with quantum approaches
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


class ClassicalBaseline:
    """Wrapper for classical ML models"""
    
    def __init__(self, model_type: str = 'random_forest', **kwargs):
        """
        Initialize classical model
        
        Args:
            model_type: 'random_forest', 'xgboost', 'svm', 'logistic', 'mlp'
            **kwargs: Additional parameters for the model
        """
        self.model_type = model_type
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                **kwargs
            )
        elif model_type == 'xgboost':
            self.model = XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                **kwargs
            )
        elif model_type == 'svm':
            self.model = SVC(
                probability=True,
                random_state=42,
                **kwargs
            )
        elif model_type == 'logistic':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                **kwargs
            )
        elif model_type == 'mlp':
            self.model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                random_state=42,
                max_iter=500,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model"""
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        return self.model.predict_proba(X)
    
    def get_model_name(self) -> str:
        """Get model name"""
        return self.model_type.replace('_', ' ').title()
