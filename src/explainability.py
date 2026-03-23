"""
Explainability and Feature Sensitivity Analysis
Provides interpretability for quantum and classical models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')


class ModelExplainability:
    """
    Provides explainability for fraud detection models
    """
    
    def __init__(self, model, feature_names: List[str]):
        """
        Initialize explainability analyzer
        
        Args:
            model: Trained model
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
    
    def permutation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_repeats: int = 10,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Compute permutation importance
        
        Args:
            X: Feature matrix
            y: Target labels
            n_repeats: Number of permutation repeats
            random_state: Random seed
            
        Returns:
            DataFrame with importance scores
        """
        result = permutation_importance(
            self.model,
            X, y,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring='f1'
        )
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        return importance_df
    
    def feature_sensitivity_analysis(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Analyze sensitivity of predictions to feature changes
        
        Args:
            X: Feature matrix
            y: Target labels
            n_samples: Number of samples to analyze
            
        Returns:
            Dictionary with sensitivity metrics
        """
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X[indices]
            y_sample = y[indices]
        else:
            X_sample = X
            y_sample = y
        
        # Get baseline predictions
        baseline_pred = self.model.predict_proba(X_sample)
        
        sensitivities = {}
        
        for i, feature_name in enumerate(self.feature_names):
            # Perturb feature
            X_perturbed = X_sample.copy()
            X_perturbed[:, i] += np.std(X_sample[:, i]) * 0.1  # 10% perturbation
            
            # Get new predictions
            perturbed_pred = self.model.predict_proba(X_perturbed)
            
            # Compute sensitivity (change in prediction)
            sensitivity = np.abs(perturbed_pred - baseline_pred).mean(axis=0)
            sensitivities[feature_name] = sensitivity
        
        return sensitivities
    
    def shap_values_analysis(
        self,
        X: np.ndarray,
        n_samples: int = 100
    ) -> Optional[np.ndarray]:
        """
        Compute SHAP values (if available)
        
        Args:
            X: Feature matrix
            n_samples: Number of samples
            
        Returns:
            SHAP values or None if SHAP not available
        """
        try:
            import shap
            
            if len(X) > n_samples:
                X_sample = X[:n_samples]
            else:
                X_sample = X
            
            # Create SHAP explainer
            if hasattr(self.model, 'predict_proba'):
                explainer = shap.TreeExplainer(self.model.model) if hasattr(self.model, 'model') else shap.Explainer(self.model)
                shap_values = explainer.shap_values(X_sample)
                return shap_values
            else:
                return None
                
        except ImportError:
            print("SHAP not available. Install with: pip install shap")
            return None
        except Exception as e:
            print(f"Error computing SHAP values: {e}")
            return None
    
    def compare_model_stability(
        self,
        X: np.ndarray,
        models: Dict[str, any],
        noise_level: float = 0.01
    ) -> pd.DataFrame:
        """
        Compare stability of different models to input noise
        
        Args:
            X: Feature matrix
            models: Dictionary of {model_name: model}
            noise_level: Noise level to add
            
        Returns:
            DataFrame with stability metrics
        """
        stability_results = []
        
        # Baseline predictions
        baseline_preds = {}
        for name, model in models.items():
            baseline_preds[name] = model.predict_proba(X)
        
        # Add noise and measure change
        for _ in range(10):  # Multiple noise realizations
            X_noisy = X + np.random.normal(0, noise_level * np.std(X, axis=0), X.shape)
            
            for name, model in models.items():
                noisy_pred = model.predict_proba(X_noisy)
                stability = np.mean(np.abs(noisy_pred - baseline_preds[name]))
                stability_results.append({
                    'model': name,
                    'stability': stability
                })
        
        return pd.DataFrame(stability_results).groupby('model')['stability'].agg(['mean', 'std']).reset_index()
