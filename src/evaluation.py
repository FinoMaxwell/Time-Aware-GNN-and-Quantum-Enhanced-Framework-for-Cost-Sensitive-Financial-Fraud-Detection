"""
Evaluation Metrics and Model Comparison
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from typing import Dict, Any
import pandas as pd


class ModelEvaluator:
    """Evaluate and compare models"""
    
    @staticmethod
    def evaluate_model(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Evaluate a model's performance
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            model_name: Name of the model
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'model': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            except:
                metrics['roc_auc'] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['tn'] = int(cm[0, 0])
        metrics['fp'] = int(cm[0, 1])
        metrics['fn'] = int(cm[1, 0])
        metrics['tp'] = int(cm[1, 1])
        
        return metrics
    
    @staticmethod
    def print_evaluation_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None,
        model_name: str = "Model"
    ):
        """Print detailed evaluation report"""
        print(f"\n{'='*60}")
        print(f"Evaluation Report: {model_name}")
        print(f"{'='*60}")
        
        metrics = ModelEvaluator.evaluate_model(y_true, y_pred, y_proba, model_name)
        
        print(f"\nMetrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        if 'roc_auc' in metrics:
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {metrics['tn']}")
        print(f"  False Positives: {metrics['fp']}")
        print(f"  False Negatives: {metrics['fn']}")
        print(f"  True Positives:  {metrics['tp']}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Normal', 'Fraud']))
    
    @staticmethod
    def compare_models(results: list) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            results: List of metric dictionaries
            
        Returns:
            DataFrame with comparison
        """
        df = pd.DataFrame(results)
        df = df.set_index('model')
        return df
