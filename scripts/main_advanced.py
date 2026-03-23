"""
Advanced Main Script for Quantum-Enhanced Fraud Detection
Uses Nigerian Financial Transactions dataset with research-grade features
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from src.dataset_loader import NigerianFraudDatasetLoader
from src.data_preprocessing import FraudDataPreprocessor
from src.feature_engineering import AdvancedFeatureEngineer
from src.quantum_models import QuantumVariationalClassifier, QuantumKernelSVM
from src.classical_models import ClassicalBaseline
from src.evaluation import ModelEvaluator
from src.cost_sensitive import CostSensitiveFraudDetection
from src.visualization import FraudDetectionVisualizer
from src.gnn_models import (
    GCNFraudClassifier,
    build_transaction_entity_graph,
)
import warnings
warnings.filterwarnings('ignore')


def main():
    """Main execution function with advanced features"""
    print("="*80)
    print("Quantum-Enhanced Fraud Detection - Advanced Research Version")
    print("Using Nigerian Financial Transactions Dataset")
    print("="*80)
    
    # Step 1: Load Nigerian dataset from local files
    print("\n[1/6] Loading Nigerian Financial Transactions dataset from local files...")
    # Prefer the smaller "for-model-training" file to keep CPU/RAM usage practical.
    dataset_dir = project_root / 'data' / 'Nigerian-Financial-Transactions-and-Fraud-Detection-Dataset'
    preferred_file = dataset_dir / 'V2-nigerian-financial-transactions-and-fraud-detection-dataset-for-model-training.csv'
    if preferred_file.exists():
        loader = NigerianFraudDatasetLoader(local_path=str(preferred_file))
        print(f"Using preferred dataset file: {preferred_file.name}")
    else:
        loader = NigerianFraudDatasetLoader()
    
    # Load from local CSV files (will automatically find and combine multiple files)
    df = loader.load_dataset(
        split='train',  # Not used when loading from local files
        sample_size=50000,  # Start with 50k for faster processing (None = all)
        random_state=42
    )
    
    # Display dataset info
    info = loader.get_dataset_info()
    print(f"\nDataset Information:")
    print(f"  Total transactions: {info['total_transactions']:,}")
    print(f"  Fraud count: {info['fraud_count']:,}")
    print(f"  Fraud rate: {info['fraud_rate']:.2%}")
    print(f"  Total features: {info['total_features']}")
    
    # Step 2: Feature engineering
    print("\n[2/6] Engineering advanced features...")
    feature_engineer = AdvancedFeatureEngineer()
    df = feature_engineer.engineer_all_features(df)
    # IMPORTANT: ensure loader uses the engineered dataframe
    loader.df = df
    
    # Step 3: Prepare features
    print("\n[3/6] Preparing features for modeling...")
    X, y = loader.prepare_features(feature_selection='core', exclude_categorical=True)
    # Keep as DataFrame/Series for index alignment
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    
    # Time-aware train/test split based on transaction timestamp
    # (simulate realistic deployment: train on past, evaluate on future)
    if 'timestamp' in df.columns:
        # Use the engineered dataframe's timestamp (already converted to datetime)
        sorted_idx = np.argsort(pd.to_datetime(df['timestamp']).values)
    else:
        # Fallback: index order (if timestamp missing)
        sorted_idx = np.arange(len(X))
    
    split_point = int(0.8 * len(sorted_idx))
    train_idx = sorted_idx[:split_point]
    test_idx = sorted_idx[split_point:]
    
    # Preprocess
    preprocessor = FraudDataPreprocessor(scaler_type='robust')
    X_train = preprocessor.fit_transform(X.iloc[train_idx])
    X_test = preprocessor.transform(X.iloc[test_idx])
    y_train = y.iloc[train_idx].values
    y_test = y.iloc[test_idx].values
    # Also scale all transactions (for GNN transaction node features)
    X_all_scaled = preprocessor.transform(X)
    
    print(f"Training set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    print(f"Features: {X_train.shape[1]}")
    
    # Step 4: Train models
    print("\n[4/6] Training models...")
    
    # Classical models
    print("\n  Training classical models...")
    classical_models = {
        'Random Forest': ClassicalBaseline('random_forest'),
        'XGBoost': ClassicalBaseline('xgboost'),
        'Logistic Regression': ClassicalBaseline('logistic'),
    }
    
    classical_results = {}
    for name, model in classical_models.items():
        print(f"    Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        classical_results[name] = {
            'y_pred': y_pred,
            'y_proba': y_proba[:, 1],
            'model': model
        }
    
    # GNN model (transaction-entity graph)
    print("\n  Training GNN model (GCN on transaction-entity graph)...")
    entity_cols = [
        'sender_account',
        'receiver_account',
        'device_hash',
        'ip_address',
        'merchant_category',
        'location',
        'ip_geo_region',
    ]
    entity_cols_present = [c for c in entity_cols if c in df.columns]
    if len(entity_cols_present) < 2:
        print("    Skipping GNN: not enough entity columns found in dataset.")
        gnn_results = {}
    else:
        # Map columns to entity types (reduces node type explosion)
        entity_type_by_col = {
            'sender_account': 'account',
            'receiver_account': 'account',
            'device_hash': 'device',
            'ip_address': 'ip',
            'merchant_category': 'merchant',
            'location': 'location',
            'ip_geo_region': 'region',
        }
        
        entity_table = df[entity_cols_present].astype(str).values

        # Time-aware validation: last 10% of the training window acts as validation,
        # earlier 90% is used for fitting the GNN.
        split_point_val = int(0.9 * len(train_idx))
        train_idx_gnn = train_idx[:split_point_val]
        val_idx_gnn = train_idx[split_point_val:]

        # Cap to keep graph size CPU-friendly (tune as needed).
        # IMPORTANT: entity risk statistics are computed ONLY from train_txn_gnn
        # to avoid leakage from the validation window.
        graph = build_transaction_entity_graph(
            transaction_features=X_all_scaled,
            transaction_labels=y.values,
            entity_table=entity_table,
            entity_col_names=entity_cols_present,
            entity_type_by_col=entity_type_by_col,
            train_txn_idx_for_entity_stats=train_idx_gnn,
            max_unique_entities_per_col=5000,
        )
        
        gnn = GCNFraudClassifier(
            hidden_dim=64,
            dropout=0.2,
            lr=1e-2,
            weight_decay=5e-4,
            max_epochs=20,
            random_state=42,
            verbose=True,
            device="cpu",
        )
        gnn.fit(graph, train_txn_idx=train_idx_gnn, val_txn_idx=val_idx_gnn)

        # Cost-aware thresholding for GNN decisions (align with monetary loss objective)
        false_negative_cost = 1000.0
        false_positive_cost = 10.0

        y_proba_val = gnn.predict_proba(graph, txn_idx=val_idx_gnn)[:, 1]
        y_val_true = y.iloc[val_idx_gnn].values.astype(int)

        candidate_thresholds = np.linspace(0.05, 0.95, 19)
        best_t = 0.5
        best_cost = float("inf")
        for t in candidate_thresholds:
            y_val_pred = (y_proba_val >= t).astype(int)
            fn = int(((y_val_true == 1) & (y_val_pred == 0)).sum())
            fp = int(((y_val_true == 0) & (y_val_pred == 1)).sum())
            total_cost = fn * false_negative_cost + fp * false_positive_cost
            if total_cost < best_cost:
                best_cost = total_cost
                best_t = float(t)

        y_proba_gnn = gnn.predict_proba(graph, txn_idx=test_idx)[:, 1]
        y_pred_gnn = (y_proba_gnn >= best_t).astype(int)
        gnn_results = {
            'GNN (GCN)': {
                'y_pred': y_pred_gnn,
                'y_proba': y_proba_gnn,
                'model': gnn,
            }
        }
    
    # Quantum models (smaller subset for speed)
    print("\n  Training quantum models...")
    quantum_results = {}
    
    # Use subset for quantum training
    n_quantum_train = min(2000, len(X_train))
    n_quantum_test = min(500, len(X_test))
    quantum_train_idx = np.random.choice(len(X_train), n_quantum_train, replace=False)
    quantum_test_idx = np.random.choice(len(X_test), n_quantum_test, replace=False)
    
    X_train_q = X_train[quantum_train_idx]
    y_train_q = y_train[quantum_train_idx]
    X_test_q = X_test[quantum_test_idx]
    y_test_q = y_test[quantum_test_idx]
    
    # Quantum VQC
    print("    Training Quantum VQC...")
    try:
        qvc = QuantumVariationalClassifier(
            n_qubits=min(4, X_train.shape[1]),
            n_layers=2,
            learning_rate=0.05,
            max_iter=30,
            batch_size=32,
            optimizer='adam',
            random_state=42
        )
        qvc.fit(X_train_q, y_train_q)
        y_pred_qvc = qvc.predict(X_test_q)
        y_proba_qvc = qvc.predict_proba(X_test_q)
        quantum_results['Quantum VQC'] = {
            'y_pred': y_pred_qvc,
            'y_proba': y_proba_qvc[:, 1],
            'model': qvc
        }
    except Exception as e:
        print(f"    Error training QVC: {e}")
    
    # Step 5: Cost-sensitive evaluation
    # ------------------------------------------------------------------
    # Save model comparison metrics (includes GNN)
    # ------------------------------------------------------------------
    evaluation_results = []
    for name, results in classical_results.items():
        y_proba_1d = np.asarray(results["y_proba"], dtype=float)
        y_proba_2d = np.vstack([1 - y_proba_1d, y_proba_1d]).T
        metrics = ModelEvaluator.evaluate_model(
            y_test,
            results["y_pred"],
            y_proba_2d,
            name,
        )
        evaluation_results.append(metrics)

    for name, results in gnn_results.items():
        y_proba_1d = np.asarray(results["y_proba"], dtype=float)
        y_proba_2d = np.vstack([1 - y_proba_1d, y_proba_1d]).T
        metrics = ModelEvaluator.evaluate_model(
            y_test,
            results["y_pred"],
            y_proba_2d,
            name,
        )
        evaluation_results.append(metrics)

    for name, results in quantum_results.items():
        y_proba_1d = np.asarray(results["y_proba"], dtype=float)
        y_proba_2d = np.vstack([1 - y_proba_1d, y_proba_1d]).T
        metrics = ModelEvaluator.evaluate_model(
            y_test_q,
            results["y_pred"],
            y_proba_2d,
            name,
        )
        evaluation_results.append(metrics)

    if evaluation_results:
        comparison_df = ModelEvaluator.compare_models(evaluation_results)
        output_file = project_root / "data" / "model_comparison.csv"
        output_file.parent.mkdir(exist_ok=True)
        comparison_df.to_csv(output_file)
        print(f"\nModel metrics saved to '{output_file.name}'")

    print("\n[5/6] Cost-sensitive evaluation...")
    
    # Get transaction amounts for cost calculation
    transaction_amounts = None
    if 'amount_ngn' in df.columns:
        test_indices = preprocessor.scaler.inverse_transform(X_test) if hasattr(preprocessor.scaler, 'inverse_transform') else None
        # This is simplified - in practice, you'd preserve amount_ngn separately
    
    cost_detector = CostSensitiveFraudDetection(
        false_negative_cost=1000.0,  # NGN 1000 per missed fraud
        false_positive_cost=10.0     # NGN 10 per false alarm
    )
    
    all_results = {**classical_results, **gnn_results, **quantum_results}
    cost_results = {}
    
    for name, results in all_results.items():
        y_pred = results['y_pred']
        y_proba = results['y_proba']
        
        # Use appropriate test set
        if 'Quantum' in name:
            y_true_eval = y_test_q
        else:
            y_true_eval = y_test
        
        loss = cost_detector.calculate_loss(y_true_eval, y_pred, transaction_amounts)
        cost_results[name] = loss
        
        print(f"\n  {name}:")
        print(f"    Total Loss: NGN {loss['total_loss']:,.2f}")
        print(f"    False Negatives: {loss['false_negatives']} (Loss: NGN {loss['false_negative_loss']:,.2f})")
        print(f"    False Positives: {loss['false_positives']} (Loss: NGN {loss['false_positive_loss']:,.2f})")
    
    # Step 6: Visualizations
    print("\n[6/6] Creating visualizations...")
    visualizer = FraudDetectionVisualizer(output_dir='data/plots')
    
    # ROC curves
    roc_data = {}
    for name, results in all_results.items():
        if 'Quantum' in name:
            y_true_eval = y_test_q
        else:
            y_true_eval = y_test
        roc_data[name] = (y_true_eval, results['y_proba'])
    
    visualizer.plot_roc_curves(
        roc_data,
        title="ROC Curves - Nigerian Financial Transactions",
        save_path='data/plots/roc_curves.png'
    )
    
    # Precision-Recall curves
    visualizer.plot_precision_recall_curves(
        roc_data,
        title="Precision-Recall Curves",
        save_path='data/plots/pr_curves.png'
    )
    
    # Confusion matrices
    cm_data = {}
    for name, results in all_results.items():
        if 'Quantum' in name:
            y_true_eval = y_test_q
        else:
            y_true_eval = y_test
        cm_data[name] = (y_true_eval, results['y_pred'])
    
    visualizer.plot_confusion_matrices(
        cm_data,
        title_prefix="Confusion Matrix",
        save_path='data/plots/confusion_matrices.png'
    )
    
    # Threshold optimization (for best model)
    best_model_name = min(cost_results.keys(), key=lambda k: cost_results[k]['total_loss'])
    best_proba = all_results[best_model_name]['y_proba']
    if 'Quantum' in best_model_name:
        y_true_eval = y_test_q
    else:
        y_true_eval = y_test
    
    visualizer.plot_threshold_optimization(
        y_true_eval,
        best_proba,
        costs={'fn_cost': 1000.0, 'fp_cost': 10.0},
        title="Threshold Optimization",
        save_path='data/plots/threshold_optimization.png'
    )
    
    # Fraud distribution
    visualizer.plot_fraud_distribution(
        df.sample(min(10000, len(df))),
        save_path='data/plots/fraud_distribution.png'
    )
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print(f"Results saved to data/plots/")
    print("="*80)


if __name__ == "__main__":
    main()
