"""
Cost-Sensitive Fraud Detection
Implements financial loss metrics and threshold optimization
"""

import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Dict, Tuple, Optional
import pandas as pd


class CostSensitiveFraudDetection:
    """
    Cost-sensitive fraud detection with financial loss optimization
    """
    
    def __init__(
        self,
        false_negative_cost: float = 1000.0,  # Cost of missing a fraud
        false_positive_cost: float = 10.0,     # Cost of false alarm
        transaction_cost: float = 0.0           # Cost of processing transaction
    ):
        """
        Initialize cost-sensitive detector
        
        Args:
            false_negative_cost: Financial loss per missed fraud (default: NGN 1000)
            false_positive_cost: Cost per false alarm (default: NGN 10)
            transaction_cost: Cost per transaction processed (default: 0)
        """
        self.fn_cost = false_negative_cost
        self.fp_cost = false_positive_cost
        self.tx_cost = transaction_cost
    
    def calculate_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        transaction_amounts: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate total financial loss
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            transaction_amounts: Transaction amounts (for weighted loss)
            
        Returns:
            Dictionary with loss breakdown
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Base costs
        fn_loss = fn * self.fn_cost
        fp_loss = fp * self.fp_cost
        processing_cost = len(y_true) * self.tx_cost
        
        # Weighted loss if transaction amounts provided
        if transaction_amounts is not None:
            # False negatives: lose the fraud amount
            fn_indices = (y_true == 1) & (y_pred == 0)
            fn_loss = transaction_amounts[fn_indices].sum()
            
            # False positives: cost of investigation
            fp_loss = fp * self.fp_cost
        
        total_loss = fn_loss + fp_loss + processing_cost
        
        return {
            'total_loss': float(total_loss),
            'false_negative_loss': float(fn_loss),
            'false_positive_loss': float(fp_loss),
            'processing_cost': float(processing_cost),
            'false_negatives': int(fn),
            'false_positives': int(fp),
            'true_positives': int(tp),
            'true_negatives': int(tn)
        }
    
    def optimize_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        transaction_amounts: Optional[np.ndarray] = None,
        threshold_range: Tuple[float, float] = (0.0, 1.0),
        n_points: int = 100
    ) -> Dict[str, float]:
        """
        Find optimal threshold that minimizes financial loss
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            transaction_amounts: Transaction amounts (optional)
            threshold_range: Range of thresholds to test
            n_points: Number of threshold points to test
            
        Returns:
            Dictionary with optimal threshold and metrics
        """
        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_points)
        losses = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            loss_dict = self.calculate_loss(y_true, y_pred, transaction_amounts)
            losses.append({
                'threshold': threshold,
                'loss': loss_dict['total_loss'],
                'fn': loss_dict['false_negatives'],
                'fp': loss_dict['false_positives']
            })
        
        losses_df = pd.DataFrame(losses)
        optimal_idx = losses_df['loss'].idxmin()
        optimal = losses_df.loc[optimal_idx]
        
        # Calculate metrics at optimal threshold
        optimal_threshold = optimal['threshold']
        y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
        optimal_loss = self.calculate_loss(y_true, y_pred_optimal, transaction_amounts)
        
        return {
            'optimal_threshold': float(optimal_threshold),
            'minimal_loss': float(optimal['loss']),
            'false_negatives': int(optimal['fn']),
            'false_positives': int(optimal['fp']),
            'loss_breakdown': optimal_loss,
            'threshold_curve': losses_df
        }
    
    def evaluate_with_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float,
        transaction_amounts: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate model with specific threshold
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            threshold: Classification threshold
            transaction_amounts: Transaction amounts (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = (y_proba >= threshold).astype(int)
        loss_dict = self.calculate_loss(y_true, y_pred, transaction_amounts)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        return {
            'threshold': threshold,
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
            **loss_dict
        }
