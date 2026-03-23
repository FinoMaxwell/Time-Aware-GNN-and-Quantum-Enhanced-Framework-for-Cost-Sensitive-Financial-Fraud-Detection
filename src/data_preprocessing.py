"""
Data Preprocessing Module for Fraud Detection
Handles data loading, cleaning, feature engineering, and scaling
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional


class FraudDataPreprocessor:
    """Preprocesses financial transaction data for fraud detection"""
    
    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize preprocessor
        
        Args:
            scaler_type: 'standard' or 'robust' scaling
        """
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'robust'")
        
        self.feature_names = None
        self.is_fitted = False
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit scaler and transform data"""
        X_scaled = self.scaler.fit_transform(X)
        self.feature_names = X.columns.tolist()
        self.is_fitted = True
        return X_scaled
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted scaler"""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transforming")
        return self.scaler.transform(X)
    
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'is_fraud',
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col].values
        
        # Scale features
        X_scaled = self.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_count(self) -> int:
        """Get number of features"""
        if self.feature_names is None:
            raise ValueError("Preprocessor not fitted yet")
        return len(self.feature_names)
