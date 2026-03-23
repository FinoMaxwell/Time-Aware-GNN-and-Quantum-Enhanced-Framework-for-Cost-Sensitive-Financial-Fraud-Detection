"""
Robustness Experiments
Tests model performance under various conditions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')


class RobustnessTester:
    """
    Tests model robustness under various conditions
    """
    
    def __init__(self):
        """Initialize robustness tester"""
        pass
    
    def test_small_dataset_scenario(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model,
        sizes: List[int] = [100, 500, 1000, 5000],
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Test performance with varying dataset sizes
        
        Args:
            X: Feature matrix
            y: Target labels
            model: Model to test
            sizes: List of dataset sizes to test
            random_state: Random seed
            
        Returns:
            DataFrame with results
        """
        results = []
        
        for size in sizes:
            if size >= len(X):
                continue
            
            # Sample data
            indices = np.random.choice(len(X), size, replace=False)
            X_sample = X[indices]
            y_sample = y[indices]
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X_sample, y_sample, test_size=0.2, random_state=random_state, stratify=y_sample
            )
            
            # Train and evaluate
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                results.append({
                    'dataset_size': size,
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1_score': f1_score(y_test, y_pred, zero_division=0)
                })
            except Exception as e:
                results.append({
                    'dataset_size': size,
                    'error': str(e)
                })
        
        return pd.DataFrame(results)
    
    def test_noisy_labels(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model,
        noise_rates: List[float] = [0.0, 0.05, 0.1, 0.2, 0.3],
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Test performance with noisy labels
        
        Args:
            X: Feature matrix
            y: Target labels
            model: Model to test
            noise_rates: List of label noise rates
            random_state: Random seed
            
        Returns:
            DataFrame with results
        """
        results = []
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
        
        for noise_rate in noise_rates:
            # Add noise to labels
            y_train_noisy = y_train.copy()
            n_noise = int(len(y_train) * noise_rate)
            noise_indices = np.random.choice(len(y_train), n_noise, replace=False)
            y_train_noisy[noise_indices] = 1 - y_train_noisy[noise_indices]  # Flip labels
            
            # Train and evaluate
            try:
                model.fit(X_train, y_train_noisy)
                y_pred = model.predict(X_test)
                
                results.append({
                    'noise_rate': noise_rate,
                    'noisy_samples': n_noise,
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1_score': f1_score(y_test, y_pred, zero_division=0)
                })
            except Exception as e:
                results.append({
                    'noise_rate': noise_rate,
                    'error': str(e)
                })
        
        return pd.DataFrame(results)
    
    def test_imbalanced_classes(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model,
        fraud_ratios: List[float] = [0.01, 0.05, 0.1, 0.15, 0.2],
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Test performance with varying class imbalance
        
        Args:
            X: Feature matrix
            y: Target labels
            model: Model to test
            fraud_ratios: List of target fraud ratios
            random_state: Random seed
            
        Returns:
            DataFrame with results
        """
        results = []
        
        for target_ratio in fraud_ratios:
            # Resample to achieve target ratio
            fraud_indices = np.where(y == 1)[0]
            normal_indices = np.where(y == 0)[0]
            
            n_fraud = len(fraud_indices)
            n_normal_needed = int(n_fraud / target_ratio * (1 - target_ratio))
            
            if n_normal_needed > len(normal_indices):
                n_normal_needed = len(normal_indices)
            
            # Sample
            sampled_normal = np.random.choice(normal_indices, n_normal_needed, replace=False)
            sampled_indices = np.concatenate([fraud_indices, sampled_normal])
            
            X_balanced = X[sampled_indices]
            y_balanced = y[sampled_indices]
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X_balanced, y_balanced, test_size=0.2, random_state=random_state, stratify=y_balanced
            )
            
            # Train and evaluate
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                results.append({
                    'fraud_ratio': target_ratio,
                    'actual_ratio': y_train.mean(),
                    'train_size': len(X_train),
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1_score': f1_score(y_test, y_pred, zero_division=0)
                })
            except Exception as e:
                results.append({
                    'fraud_ratio': target_ratio,
                    'error': str(e)
                })
        
        return pd.DataFrame(results)
    
    def comprehensive_robustness_test(
        self,
        X: np.ndarray,
        y: np.ndarray,
        models: Dict[str, any],
        output_dir: str = 'data/robustness'
    ) -> Dict[str, pd.DataFrame]:
        """
        Run comprehensive robustness tests
        
        Args:
            X: Feature matrix
            y: Target labels
            models: Dictionary of models to test
            output_dir: Directory to save results
            
        Returns:
            Dictionary with all test results
        """
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        
        for model_name, model in models.items():
            print(f"\nTesting robustness of {model_name}...")
            
            # Small dataset test
            print("  Testing small dataset scenario...")
            small_results = self.test_small_dataset_scenario(X, y, model)
            small_results['model'] = model_name
            all_results[f'{model_name}_small'] = small_results
            small_results.to_csv(output_path / f'{model_name}_small_dataset.csv', index=False)
            
            # Noisy labels test
            print("  Testing noisy labels...")
            noisy_results = self.test_noisy_labels(X, y, model)
            noisy_results['model'] = model_name
            all_results[f'{model_name}_noisy'] = noisy_results
            noisy_results.to_csv(output_path / f'{model_name}_noisy_labels.csv', index=False)
            
            # Imbalanced classes test
            print("  Testing imbalanced classes...")
            imbalanced_results = self.test_imbalanced_classes(X, y, model)
            imbalanced_results['model'] = model_name
            all_results[f'{model_name}_imbalanced'] = imbalanced_results
            imbalanced_results.to_csv(output_path / f'{model_name}_imbalanced.csv', index=False)
        
        return all_results
