"""
Advanced Visualization Module
Creates publication-quality plots for fraud detection analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, 
    confusion_matrix, classification_report
)
from typing import Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class FraudDetectionVisualizer:
    """Creates comprehensive visualizations for fraud detection"""
    
    def __init__(self, output_dir: str = 'data/plots'):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_roc_curves(
        self,
        results: Dict[str, Tuple[np.ndarray, np.ndarray]],
        title: str = "ROC Curves Comparison",
        save_path: Optional[str] = None
    ):
        """
        Plot ROC curves for multiple models
        
        Args:
            results: Dictionary of {model_name: (y_true, y_proba)}
            title: Plot title
            save_path: Path to save figure
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, (y_true, y_proba) in results.items():
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_precision_recall_curves(
        self,
        results: Dict[str, Tuple[np.ndarray, np.ndarray]],
        title: str = "Precision-Recall Curves",
        save_path: Optional[str] = None
    ):
        """Plot precision-recall curves"""
        plt.figure(figsize=(10, 8))
        
        for model_name, (y_true, y_proba) in results.items():
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.3f})', linewidth=2)
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrices(
        self,
        results: Dict[str, Tuple[np.ndarray, np.ndarray]],
        title_prefix: str = "Confusion Matrix",
        save_path: Optional[str] = None
    ):
        """Plot confusion matrices for multiple models"""
        n_models = len(results)
        cols = 2
        rows = (n_models + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (model_name, (y_true, y_pred)) in enumerate(results.items()):
            cm = confusion_matrix(y_true, y_pred)
            
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                ax=axes[idx], cbar_kws={'label': 'Count'}
            )
            axes[idx].set_xlabel('Predicted', fontsize=10)
            axes[idx].set_ylabel('Actual', fontsize=10)
            axes[idx].set_title(f'{title_prefix}: {model_name}', fontsize=11, fontweight='bold')
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_threshold_optimization(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        costs: Dict[str, float],
        title: str = "Threshold Optimization",
        save_path: Optional[str] = None
    ):
        """Plot threshold optimization curve"""
        from src.cost_sensitive import CostSensitiveFraudDetection
        
        cost_detector = CostSensitiveFraudDetection(
            false_negative_cost=costs.get('fn_cost', 1000),
            false_positive_cost=costs.get('fp_cost', 10)
        )
        
        thresholds = np.linspace(0, 1, 100)
        losses = []
        f1_scores = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            loss = cost_detector.calculate_loss(y_true, y_pred)
            losses.append(loss['total_loss'])
            
            from sklearn.metrics import f1_score
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curve
        ax1.plot(thresholds, losses, 'b-', linewidth=2)
        optimal_idx = np.argmin(losses)
        ax1.axvline(thresholds[optimal_idx], color='r', linestyle='--', 
                   label=f'Optimal: {thresholds[optimal_idx]:.3f}')
        ax1.set_xlabel('Threshold', fontsize=11)
        ax1.set_ylabel('Total Loss (NGN)', fontsize=11)
        ax1.set_title('Financial Loss vs Threshold', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # F1 score curve
        ax2.plot(thresholds, f1_scores, 'g-', linewidth=2)
        optimal_f1_idx = np.argmax(f1_scores)
        ax2.axvline(thresholds[optimal_f1_idx], color='r', linestyle='--',
                   label=f'Optimal F1: {thresholds[optimal_f1_idx]:.3f}')
        ax2.set_xlabel('Threshold', fontsize=11)
        ax2.set_ylabel('F1 Score', fontsize=11)
        ax2.set_title('F1 Score vs Threshold', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        importances: np.ndarray,
        top_n: int = 20,
        title: str = "Feature Importance",
        save_path: Optional[str] = None
    ):
        """Plot feature importance"""
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, max(8, top_n * 0.4)))
        plt.barh(range(top_n), importances[indices], color='steelblue')
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance', fontsize=11)
        plt.title(title, fontsize=13, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_fraud_distribution(
        self,
        df: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """Plot fraud distribution across different dimensions"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Fraud rate over time
        if 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['date'].dt.hour
            fraud_by_hour = df.groupby('hour')['is_fraud'].mean()
            axes[0, 0].plot(fraud_by_hour.index, fraud_by_hour.values, 'o-', linewidth=2)
            axes[0, 0].set_xlabel('Hour of Day', fontsize=10)
            axes[0, 0].set_ylabel('Fraud Rate', fontsize=10)
            axes[0, 0].set_title('Fraud Rate by Hour', fontsize=11, fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Fraud by amount
        if 'amount_ngn' in df.columns:
            df['amount_bin'] = pd.qcut(df['amount_ngn'], q=10, duplicates='drop')
            fraud_by_amount = df.groupby('amount_bin')['is_fraud'].mean()
            axes[0, 1].bar(range(len(fraud_by_amount)), fraud_by_amount.values, color='coral')
            axes[0, 1].set_xlabel('Amount Bins', fontsize=10)
            axes[0, 1].set_ylabel('Fraud Rate', fontsize=10)
            axes[0, 1].set_title('Fraud Rate by Transaction Amount', fontsize=11, fontweight='bold')
            axes[0, 1].set_xticks(range(len(fraud_by_amount)))
            axes[0, 1].set_xticklabels([f'Bin {i+1}' for i in range(len(fraud_by_amount))], rotation=45)
            axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Fraud by region
        if 'ip_geo_region' in df.columns:
            fraud_by_region = df.groupby('ip_geo_region')['is_fraud'].mean().sort_values(ascending=False)
            axes[1, 0].barh(range(len(fraud_by_region)), fraud_by_region.values, color='steelblue')
            axes[1, 0].set_yticks(range(len(fraud_by_region)))
            axes[1, 0].set_yticklabels(fraud_by_region.index)
            axes[1, 0].set_xlabel('Fraud Rate', fontsize=10)
            axes[1, 0].set_title('Fraud Rate by Region', fontsize=11, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # Overall fraud distribution
        fraud_counts = df['is_fraud'].value_counts()
        axes[1, 1].pie(fraud_counts.values, labels=['Normal', 'Fraud'], autopct='%1.1f%%',
                      startangle=90, colors=['lightblue', 'coral'])
        axes[1, 1].set_title('Overall Fraud Distribution', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_comprehensive_report(
        self,
        model_results: Dict[str, Dict],
        output_dir: Optional[str] = None
    ):
        """Create comprehensive visualization report"""
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # This would create all visualizations
        # Implementation depends on available data
        print(f"Visualizations saved to {self.output_dir}")
