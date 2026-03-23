"""
Nigerian Financial Transactions Dataset Loader
Loads and preprocesses the Hugging Face dataset
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class NigerianFraudDatasetLoader:
    """
    Loader for Nigerian Financial Transactions and Fraud Detection Dataset
    from Hugging Face: electricsheepafrica/Nigerian-Financial-Transactions-and-Fraud-Detection-Dataset
    """
    
    def __init__(self, cache_dir: Optional[str] = None, local_path: Optional[str] = None):
        """
        Initialize dataset loader
        
        Args:
            cache_dir: Directory to cache downloaded dataset
            local_path: Path to local dataset file (CSV or Parquet)
        """
        self.cache_dir = cache_dir
        self.local_path = local_path
        self.dataset = None
        self.df = None
    
    def load_dataset(
        self,
        split: str = 'train',
        sample_size: Optional[int] = None,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Load dataset from local file or Hugging Face
        
        Args:
            split: Dataset split ('train' or 'test')
            sample_size: Number of samples to load (None = all)
            random_state: Random seed for sampling
            
        Returns:
            DataFrame with transaction data
        """
        # First, try to load from local file(s)
        local_files = self._find_local_file()
        
        if local_files:
            if len(local_files) == 1:
                print(f"Loading dataset from local file: {local_files[0]}")
            else:
                print(f"Loading dataset from {len(local_files)} local files")
            return self._load_from_file(local_files, sample_size, random_state)
        
        # If no local file, load from Hugging Face
        print(f"Loading Nigerian Financial Transactions dataset from Hugging Face...")
        print("This may take a few minutes on first download...")
        print("Tip: Download locally first using: python scripts/download_dataset.py")
        
        try:
            # Load dataset from Hugging Face
            dataset = load_dataset(
                'electricsheepafrica/Nigerian-Financial-Transactions-and-Fraud-Detection-Dataset',
                split=split,
                cache_dir=self.cache_dir
            )
            
            # Convert to pandas DataFrame
            print("Converting to pandas DataFrame...")
            df = dataset.to_pandas()
            
            # Sample if requested
            if sample_size and len(df) > sample_size:
                print(f"Sampling {sample_size:,} transactions from {len(df):,} total...")
                df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
            
            self.dataset = dataset
            self.df = df
            
            print(f"Loaded {len(df):,} transactions")
            print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
            print(f"Features: {df.shape[1]}")
            
            return df
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("\nTrying to load from local file...")
            # Fallback: try loading from local file if available
            local_files = self._find_local_file()
            if local_files:
                return self._load_from_file(local_files, sample_size, random_state)
            else:
                raise FileNotFoundError(
                    "Could not load dataset from Hugging Face or local file.\n"
                    "Please ensure dataset files are in the data folder with name containing 'Nigerian-Financial-Transactions-and-Fraud-Detection-Dataset'"
                )
    
    def _find_local_file(self) -> Optional[List[str]]:
        """
        Find local dataset files in common locations
        Returns list of file paths (handles split CSV files)
        """
        from pathlib import Path
        
        # Check user-specified path first
        if self.local_path:
            if Path(self.local_path).exists():
                return [self.local_path]
        
        # Check data folder for Nigerian dataset files
        data_dir = Path('data')
        if data_dir.exists():
            # Check subdirectory: data/Nigerian-Financial-Transactions-and-Fraud-Detection-Dataset/
            nigerian_dir = data_dir / 'Nigerian-Financial-Transactions-and-Fraud-Detection-Dataset'
            if nigerian_dir.exists():
                # Find all CSV files in this directory
                csv_files = sorted(nigerian_dir.glob('*.csv'))
                if csv_files:
                    return sorted([str(f) for f in csv_files])
            
            # Look for files matching the pattern in data folder
            pattern_files = list(data_dir.glob('*Nigerian-Financial-Transactions-and-Fraud-Detection-Dataset*.csv'))
            if pattern_files:
                return sorted([str(f) for f in pattern_files])
            
            # Also check for numbered CSV files (e.g., dataset_1.csv, dataset_2.csv)
            numbered_files = sorted(data_dir.glob('*Nigerian*.csv'))
            if numbered_files:
                return sorted([str(f) for f in numbered_files])
        
        # Check other common locations
        possible_paths = [
            'data/raw/nigerian_fraud_dataset.csv',
            'data/nigerian_fraud_dataset.csv',
            'nigerian_fraud_dataset.csv',
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return [path]
        
        return None
    
    def _load_from_file(
        self,
        file_paths: Optional[List[str]] = None,
        sample_size: Optional[int] = None,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Load from local CSV or Parquet file(s)
        Handles multiple CSV files (combines them)
        """
        from pathlib import Path
        
        if file_paths is None:
            file_paths = self._find_local_file()
        
        if file_paths is None or len(file_paths) == 0:
            raise FileNotFoundError(
                "No local dataset file found in data folder.\n"
                "Expected files matching: *Nigerian-Financial-Transactions-and-Fraud-Detection-Dataset*.csv"
            )
        
        try:
            if len(file_paths) == 1:
                # Single file
                file_path = Path(file_paths[0])
                print(f"Loading from: {file_path}")
                
                if file_path.suffix == '.parquet':
                    df = pd.read_parquet(file_path)
                else:
                    # For very large CSVs, avoid loading everything if sampling is requested
                    if sample_size is not None and sample_size > 0:
                        print(f"Sampling {sample_size:,} rows from CSV using chunked reading...")
                        df = self._sample_csv_in_chunks(
                            file_path=str(file_path),
                            sample_size=sample_size,
                            random_state=random_state
                        )
                    else:
                        print("Reading CSV file (this may take a few minutes for large files)...")
                        df = pd.read_csv(file_path, low_memory=False)
                
                print(f"Loaded {len(df):,} transactions from local file")
            else:
                # Multiple files - combine them
                print(f"Found {len(file_paths)} CSV files. Combining...")
                dfs = []
                
                for i, file_path in enumerate(file_paths, 1):
                    file_path = Path(file_path)
                    print(f"  Loading file {i}/{len(file_paths)}: {file_path.name}...")
                    
                    if file_path.suffix == '.parquet':
                        df_part = pd.read_parquet(file_path)
                    else:
                        if sample_size is not None and sample_size > 0:
                            # Sample from each file and then re-sample globally below.
                            # This keeps memory usage bounded even for multi-million row datasets.
                            per_file = max(1, int(sample_size / max(len(file_paths), 1) * 2))
                            df_part = self._sample_csv_in_chunks(
                                file_path=str(file_path),
                                sample_size=per_file,
                                random_state=random_state + i
                            )
                        else:
                            df_part = pd.read_csv(file_path, low_memory=False)
                    
                    dfs.append(df_part)
                    print(f"    Loaded {len(df_part):,} transactions")
                
                # Combine all dataframes
                print("Combining all files...")
                df = pd.concat(dfs, ignore_index=True)
                print(f"Combined dataset: {len(df):,} total transactions")

            # Sample if requested (for parquet or combined multi-file sampling)
            if sample_size and len(df) > sample_size:
                print(f"Sampling {sample_size:,} transactions from {len(df):,} total...")
                df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
            
            self.df = df
            
            print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
            print(f"Features: {df.shape[1]}")
            
            return df
            
        except Exception as e:
            raise FileNotFoundError(
                f"Error loading local file(s): {e}\n"
                "Please ensure the files exist and are not corrupted."
            )

    def _sample_csv_in_chunks(
        self,
        file_path: str,
        sample_size: int,
        random_state: int = 42,
        chunksize: int = 200_000
    ) -> pd.DataFrame:
        """
        Sample rows from a CSV file without loading the full file into memory.

        Strategy:
        - Read the CSV in chunks
        - Take a small random sample from each chunk
        - Keep a fixed-size global sample by re-sampling down to sample_size

        This is not a perfect reservoir sampler, but it is practical and reproducible
        for large datasets and keeps memory bounded (~2*sample_size rows).
        """
        rng = np.random.RandomState(random_state)
        kept: Optional[pd.DataFrame] = None
        chunk_i = 0
        total_rows = 0

        for chunk in pd.read_csv(file_path, low_memory=False, chunksize=chunksize):
            chunk_i += 1
            if len(chunk) == 0:
                continue
            total_rows += len(chunk)
            if chunk_i == 1 or chunk_i % 5 == 0:
                print(f"  ... processed ~{total_rows:,} rows (chunks read: {chunk_i})")

            # Sample from chunk (up to sample_size rows)
            k = min(sample_size, len(chunk))
            chunk_sample = chunk.sample(n=k, random_state=rng.randint(0, 2**31 - 1))

            if kept is None:
                kept = chunk_sample
            else:
                kept = pd.concat([kept, chunk_sample], ignore_index=True)
                if len(kept) > sample_size:
                    kept = kept.sample(n=sample_size, random_state=rng.randint(0, 2**31 - 1)).reset_index(drop=True)

        if kept is None:
            raise ValueError(f"No data read from CSV: {file_path}")

        # Ensure exact size if possible
        if len(kept) > sample_size:
            kept = kept.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

        return kept
    
    def get_feature_columns(self, exclude_cols: Optional[List[str]] = None) -> List[str]:
        """
        Get list of feature columns (excluding target and metadata)
        
        Args:
            exclude_cols: Additional columns to exclude
            
        Returns:
            List of feature column names
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        default_exclude = [
            'transaction_id', 'timestamp', 'is_fraud', 'fraud_type',
            'sender_account', 'receiver_account', 'ip_address', 'device_hash'
        ]
        
        if exclude_cols:
            default_exclude.extend(exclude_cols)
        
        feature_cols = [col for col in self.df.columns if col not in default_exclude]
        return feature_cols
    
    def prepare_features(
        self,
        feature_selection: str = 'all',
        exclude_categorical: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for modeling
        
        Args:
            feature_selection: 'all', 'numeric', 'engineered', or 'core'
            exclude_categorical: Whether to exclude categorical columns
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        # Get feature columns
        feature_cols = self.get_feature_columns()
        
        # Filter by type
        if exclude_categorical:
            # Keep only numeric columns
            numeric_cols = self.df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in feature_cols if col in numeric_cols]
        
        # Feature selection strategies
        if feature_selection == 'core':
            # Core fraud detection features
            core_features = [
                'amount_ngn', 'time_since_last_transaction', 'spending_deviation_score',
                'velocity_score', 'geo_anomaly_score', 'channel_risk_score',
                'persona_fraud_risk', 'location_fraud_risk', 'merchant_fraud_rate',
                'bvn_linked', 'new_device_transaction', 'is_device_shared',
                'is_ip_shared', 'geospatial_velocity_anomaly', 'is_weekend',
                'is_night_txn', 'txn_hour'
            ]
            feature_cols = [col for col in core_features if col in self.df.columns]
        
        elif feature_selection == 'engineered':
            # Use engineered features only
            engineered_features = [
                col for col in feature_cols
                if any(keyword in col for keyword in ['score', 'risk', 'rate', 'count', 'avg', 'std', 'frequency'])
            ]
            feature_cols = engineered_features
        
        # Prepare data
        X = self.df[feature_cols].copy()
        y = self.df['is_fraud'].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Convert boolean to int
        bool_cols = X.select_dtypes(include=[bool]).columns
        X[bool_cols] = X[bool_cols].astype(int)
        
        print(f"Prepared {len(feature_cols)} features for modeling")
        print(f"Feature columns: {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}")
        
        return X, y
    
    def get_dataset_info(self) -> dict:
        """Get dataset information and statistics"""
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        info = {
            'total_transactions': len(self.df),
            'fraud_count': self.df['is_fraud'].sum(),
            'fraud_rate': self.df['is_fraud'].mean(),
            'total_features': self.df.shape[1],
            'numeric_features': len(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(self.df.select_dtypes(exclude=[np.number]).columns),
            'missing_values': self.df.isnull().sum().sum(),
            'date_range': {
                'start': self.df['timestamp'].min() if 'timestamp' in self.df.columns else None,
                'end': self.df['timestamp'].max() if 'timestamp' in self.df.columns else None
            }
        }
        
        # Fraud type distribution
        if 'fraud_type' in self.df.columns:
            info['fraud_types'] = self.df[self.df['is_fraud']]['fraud_type'].value_counts().to_dict()
        
        return info
