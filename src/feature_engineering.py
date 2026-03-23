"""
Advanced Feature Engineering
Derives additional features from the Nigerian Financial Transactions dataset
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for fraud detection
    """
    
    def __init__(self):
        """Initialize feature engineer"""
        pass
    
    def engineer_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer temporal features from timestamp
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            DataFrame with added temporal features
        """
        df = df.copy()
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Time-based features
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_month'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
            df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
            
            # Cyclical encoding for time features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def engineer_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer transaction-based features
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with added transaction features
        """
        df = df.copy()
        
        # Amount-based features
        if 'amount_ngn' in df.columns:
            df['amount_log'] = np.log1p(df['amount_ngn'])
            df['amount_sqrt'] = np.sqrt(df['amount_ngn'])
            df['amount_zscore'] = (df['amount_ngn'] - df['amount_ngn'].mean()) / df['amount_ngn'].std()
            
            # Amount bins
            df['amount_bin'] = pd.qcut(df['amount_ngn'], q=5, labels=False, duplicates='drop')
        
        # Velocity features (if not already present)
        if 'velocity_score' not in df.columns and 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
            df['time_since_last'] = df.groupby('sender_account')['timestamp'].diff().dt.total_seconds() / 3600
            df['time_since_last'] = df['time_since_last'].fillna(df['time_since_last'].median())
        
        return df
    
    def engineer_user_behavior_features(
        self,
        df: pd.DataFrame,
        user_id_col: str = 'sender_account'
    ) -> pd.DataFrame:
        """
        Engineer user behavior features
        
        Args:
            df: DataFrame with transaction data
            user_id_col: Column name for user identifier
            
        Returns:
            DataFrame with added user behavior features
        """
        df = df.copy()
        
        if user_id_col not in df.columns:
            return df
        
        # User-level aggregations
        user_stats = df.groupby(user_id_col).agg({
            'amount_ngn': ['count', 'mean', 'std', 'min', 'max', 'sum'],
            'is_fraud': 'mean'
        }).reset_index()
        
        user_stats.columns = [
            user_id_col,
            'user_txn_count',
            'user_avg_amount',
            'user_std_amount',
            'user_min_amount',
            'user_max_amount',
            'user_total_amount',
            'user_fraud_rate'
        ]
        
        # Merge back
        df = df.merge(user_stats, on=user_id_col, how='left')
        
        # Rolling window features (if timestamp available)
        if 'timestamp' in df.columns:
            df = df.sort_values(['sender_account', 'timestamp'])
            
            # Last N transactions
            for window in [1, 3, 7]:
                df[f'amount_last_{window}'] = (
                    df.groupby(user_id_col)['amount_ngn']
                    .shift(1)
                    .rolling(window=window, min_periods=1)
                    .mean()
                )
        
        return df
    
    def engineer_merchant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer merchant-related features
        
        Args:
            df: DataFrame with merchant data
            
        Returns:
            DataFrame with added merchant features
        """
        df = df.copy()
        
        if 'merchant_category' in df.columns:
            # Merchant fraud rate (using expanding window to avoid leakage)
            df = df.sort_values('timestamp' if 'timestamp' in df.columns else df.index)
            df['merchant_fraud_rate'] = (
                df.groupby('merchant_category')['is_fraud']
                .expanding()
                .mean()
                .shift(1)
                .fillna(0.1)
                .reset_index(0, drop=True)
            )
            
            # Merchant transaction count
            merchant_counts = df['merchant_category'].value_counts().to_dict()
            df['merchant_popularity'] = df['merchant_category'].map(merchant_counts)
        
        return df
    
    def engineer_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer geographic features
        
        Args:
            df: DataFrame with location data
            
        Returns:
            DataFrame with added geographic features
        """
        df = df.copy()
        
        if 'ip_geo_region' in df.columns:
            # Region fraud rate
            region_fraud = df.groupby('ip_geo_region')['is_fraud'].mean().to_dict()
            df['region_fraud_rate'] = df['ip_geo_region'].map(region_fraud)
            
            # Region transaction count
            region_counts = df['ip_geo_region'].value_counts().to_dict()
            df['region_transaction_count'] = df['ip_geo_region'].map(region_counts)
        
        if 'location' in df.columns:
            # Location fraud rate
            location_fraud = df.groupby('location')['is_fraud'].mean().to_dict()
            df['location_fraud_rate'] = df['location'].map(location_fraud)
        
        return df
    
    def engineer_all_features(
        self,
        df: pd.DataFrame,
        user_id_col: str = 'sender_account'
    ) -> pd.DataFrame:
        """
        Engineer all features
        
        Args:
            df: Input DataFrame
            user_id_col: User identifier column
            
        Returns:
            DataFrame with all engineered features
        """
        print("Engineering temporal features...")
        df = self.engineer_temporal_features(df)
        
        print("Engineering transaction features...")
        df = self.engineer_transaction_features(df)
        
        print("Engineering user behavior features...")
        df = self.engineer_user_behavior_features(df, user_id_col)
        
        print("Engineering merchant features...")
        df = self.engineer_merchant_features(df)
        
        print("Engineering geographic features...")
        df = self.engineer_geographic_features(df)
        
        print(f"Feature engineering complete. Total features: {df.shape[1]}")
        
        return df
