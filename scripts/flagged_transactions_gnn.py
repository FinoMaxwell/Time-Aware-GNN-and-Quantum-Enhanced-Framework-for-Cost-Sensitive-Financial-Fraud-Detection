"""
Generate a 'flagged transactions' report using the GNN model.

This script:
- Loads the Nigerian dataset (sampled for laptop-friendly size)
- Performs advanced feature engineering and time-aware splitting
- Builds a transaction–entity graph and trains the GCN-based GNN
- Scores the *future* test transactions
- Produces:
  - A CSV of all test transactions with GNN predictions
  - A CSV of only high-risk / flagged transactions
  - A console summary table of the top flagged transactions
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from src.dataset_loader import NigerianFraudDatasetLoader
from src.feature_engineering import AdvancedFeatureEngineer
from src.data_preprocessing import FraudDataPreprocessor
from src.gnn_models import (
    GCNFraudClassifier,
    build_transaction_entity_graph,
)
import warnings

warnings.filterwarnings("ignore")


def _risk_level(prob: float) -> str:
    if prob >= 0.8:
        return "CRITICAL"
    if prob >= 0.6:
        return "HIGH"
    if prob >= 0.4:
        return "MEDIUM"
    if prob >= 0.2:
        return "LOW"
    return "MINIMAL"


def _recommended_action(prob: float) -> str:
    if prob >= 0.8:
        return "BLOCK"
    if prob >= 0.6:
        return "FLAG_FOR_REVIEW"
    if prob >= 0.4:
        return "MONITOR"
    return "APPROVE"


def main():
    print("=" * 80)
    print("Flagged Transactions Report using GNN (GCN)")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1) Load dataset (same sampling settings as main_advanced)
    # ------------------------------------------------------------------
    print("\n[1/4] Loading Nigerian dataset (sampled)...")
    dataset_dir = project_root / "data" / "Nigerian-Financial-Transactions-and-Fraud-Detection-Dataset"
    preferred_file = dataset_dir / "V2-nigerian-financial-transactions-and-fraud-detection-dataset-for-model-training.csv"
    if preferred_file.exists():
        loader = NigerianFraudDatasetLoader(local_path=str(preferred_file))
        print(f"Using preferred dataset file: {preferred_file.name}")
    else:
        loader = NigerianFraudDatasetLoader()

    df = loader.load_dataset(
        split="train",
        sample_size=50_000,
        random_state=42,
    )

    info = loader.get_dataset_info()
    print(f"  Total transactions (sampled): {info['total_transactions']:,}")
    print(f"  Fraud count: {info['fraud_count']:,}")
    print(f"  Fraud rate: {info['fraud_rate']:.2%}")

    # ------------------------------------------------------------------
    # 2) Feature engineering and time-aware split
    # ------------------------------------------------------------------
    print("\n[2/4] Feature engineering and time-aware split...")
    feature_engineer = AdvancedFeatureEngineer()
    df = feature_engineer.engineer_all_features(df)
    loader.df = df  # ensure loader uses engineered frame

    # Prepare core numeric features and labels
    X, y = loader.prepare_features(feature_selection="core", exclude_categorical=True)
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    # Time-aware split: train on earlier 80%, test on later 20%
    if "timestamp" in df.columns:
        sorted_idx = np.argsort(pd.to_datetime(df["timestamp"]).values)
    else:
        sorted_idx = np.arange(len(X))

    split_point = int(0.8 * len(sorted_idx))
    train_idx = sorted_idx[:split_point]
    test_idx = sorted_idx[split_point:]

    print(f"  Train size: {len(train_idx):,}")
    print(f"  Test size : {len(test_idx):,}")

    # Scale features
    preprocessor = FraudDataPreprocessor(scaler_type="robust")
    X_train = preprocessor.fit_transform(X.iloc[train_idx])
    X_test = preprocessor.transform(X.iloc[test_idx])
    y_train = y.iloc[train_idx].values
    y_test = y.iloc[test_idx].values
    X_all_scaled = preprocessor.transform(X)

    # ------------------------------------------------------------------
    # 3) Build graph and train GNN
    # ------------------------------------------------------------------
    print("\n[3/4] Building transaction–entity graph and training GNN...")
    entity_cols = [
        "sender_account",
        "receiver_account",
        "device_hash",
        "ip_address",
        "merchant_category",
        "location",
        "ip_geo_region",
    ]
    entity_cols_present = [c for c in entity_cols if c in df.columns]
    if len(entity_cols_present) < 2:
        raise RuntimeError("Not enough entity columns available to build a useful graph.")

    entity_type_by_col = {
        "sender_account": "account",
        "receiver_account": "account",
        "device_hash": "device",
        "ip_address": "ip",
        "merchant_category": "merchant",
        "location": "location",
        "ip_geo_region": "region",
    }

    entity_table = df[entity_cols_present].astype(str).values

    # Time-aware validation: last 10% of training window
    split_point_val = int(0.9 * len(train_idx))
    train_idx_gnn = train_idx[:split_point_val]
    val_idx_gnn = train_idx[split_point_val:]

    # Build graph with ONLY train_txn_gnn used for entity risk statistics
    graph = build_transaction_entity_graph(
        transaction_features=X_all_scaled,
        transaction_labels=y.values,
        entity_table=entity_table,
        entity_col_names=entity_cols_present,
        entity_type_by_col=entity_type_by_col,
        train_txn_idx_for_entity_stats=train_idx_gnn,
        max_unique_entities_per_col=5_000,
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

    # Calibrated probabilities are produced automatically by gnn.predict_proba()
    y_proba_val = gnn.predict_proba(graph, txn_idx=val_idx_gnn)[:, 1]
    y_val_true = y.iloc[val_idx_gnn].values.astype(int)

    # ------------------------------------------------------------------
    # Threshold optimization for monetary cost on the validation window
    # ------------------------------------------------------------------
    false_negative_cost = 1000.0  # cost per missed fraud
    false_positive_cost = 10.0    # cost per false alarm

    candidate_thresholds = np.linspace(0.05, 0.95, 19)  # coarse grid (fast)
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

    t_monitor = best_t

    y_proba = gnn.predict_proba(graph, txn_idx=test_idx)[:, 1]
    y_pred = (y_proba >= t_monitor).astype(int)

    # ------------------------------------------------------------------
    # 4) Build report and save flagged transactions
    # ------------------------------------------------------------------
    print("\n[4/4] Building flagged-transactions report...")
    results_df = df.iloc[test_idx].copy()

    # Create multi-tier risk levels using quantiles within flagged set.
    flagged_mask = y_proba >= t_monitor
    flagged_probs = y_proba[flagged_mask]

    # Defaults if something unexpected happens
    t_flag = t_monitor
    t_block = t_monitor

    if len(flagged_probs) >= 20:
        # CRITICAL = top 10% of flagged, HIGH = next 20% of flagged
        t_block = float(np.quantile(flagged_probs, 0.90))
        t_flag = float(np.quantile(flagged_probs, 0.70))
    elif len(flagged_probs) > 0:
        # Small flagged set: use min/max split
        t_block = float(np.max(flagged_probs))
        t_flag = float(np.min(flagged_probs))

    def risk_and_action(prob: float) -> tuple[str, str]:
        if prob >= t_block:
            return ("CRITICAL", "BLOCK")
        if prob >= t_flag:
            return ("HIGH", "FLAG_FOR_REVIEW")
        if prob >= t_monitor:
            return ("MEDIUM", "MONITOR")
        return ("LOW", "APPROVE")

    risk_action = [risk_and_action(float(p)) for p in y_proba]
    risk_level = [ra[0] for ra in risk_action]
    recommended_action = [ra[1] for ra in risk_action]

    results_df = results_df.assign(
        predicted_fraud=y_pred.astype(int),
        fraud_probability=y_proba,
        risk_level=risk_level,
        recommended_action=recommended_action,
    )

    # Choose which columns to expose in the report
    cols_for_report = [
        c
        for c in [
            "timestamp",
            "amount_ngn",
            "sender_account",
            "receiver_account",
            "device_hash",
            "ip_address",
            "merchant_category",
            "location",
            "ip_geo_region",
            "is_fraud",
            "predicted_fraud",
            "fraud_probability",
            "risk_level",
            "recommended_action",
        ]
        if c in results_df.columns
    ]
    results_df = results_df[cols_for_report]

    # Save full predictions
    output_dir = project_root / "data"
    output_dir.mkdir(exist_ok=True)
    all_pred_path = output_dir / "gnn_predictions.csv"
    results_df.to_csv(all_pred_path, index=False)

    # Filter flagged transactions (fraud_probability >= t_monitor)
    flagged_df = results_df[results_df["fraud_probability"] >= t_monitor].copy()
    flagged_path = output_dir / "flagged_transactions_gnn.csv"
    flagged_df.to_csv(flagged_path, index=False)

    print(f"\nTotal test transactions: {len(results_df):,}")
    print(f"Flagged (prob >= optimized threshold {t_monitor:.2f}):  {len(flagged_df):,}")
    print(f"\nFull GNN predictions saved to : {all_pred_path}")
    print(f"Flagged transactions saved to : {flagged_path}")

    # ------------------------------------------------------------------
    # Cluster evidence: show whether top-risk transactions concentrate on
    # specific shared entities (device/account/IP/merchant/etc.)
    # ------------------------------------------------------------------
    print("\n[Cluster Evidence] Concentration among top-risk transactions...")
    top_k_txn = min(200, len(results_df))
    top_risk = results_df.sort_values("fraud_probability", ascending=False).head(top_k_txn)

    cluster_rows = []
    entity_cols_cluster = [
        "device_hash",
        "sender_account",
        "receiver_account",
        "ip_address",
        "merchant_category",
        "location",
        "ip_geo_region",
    ]
    for col in entity_cols_cluster:
        if col not in top_risk.columns:
            continue
        vc = top_risk[col].value_counts().head(20)
        for ent_val, cnt in vc.items():
            sub = top_risk[top_risk[col] == ent_val]
            # is_fraud may be bool; convert to int for mean
            fraud_rate = sub["is_fraud"].astype(int).mean() if len(sub) else 0.0
            mean_prob = sub["fraud_probability"].mean() if len(sub) else 0.0
            cluster_rows.append(
                {
                    "entity_column": col,
                    "entity_value": ent_val,
                    "top_risk_count": int(cnt),
                    "fraud_rate_in_top_risk": float(fraud_rate),
                    "mean_probability_in_top_risk": float(mean_prob),
                }
            )

    cluster_df = pd.DataFrame(cluster_rows)
    if not cluster_df.empty:
        # Sort to make it easy to read: highest fraud rate first
        cluster_df = cluster_df.sort_values("fraud_rate_in_top_risk", ascending=False).head(100)
        cluster_path = output_dir / "cluster_evidence_gnn.csv"
        cluster_df.to_csv(cluster_path, index=False)
        print(f"Cluster evidence saved to: {cluster_path}")
        # Console preview
        preview_cols = [
            "entity_column",
            "entity_value",
            "top_risk_count",
            "fraud_rate_in_top_risk",
            "mean_probability_in_top_risk",
        ]
        print("\nTop cluster evidence rows:\n")
        print(cluster_df[preview_cols].head(10).to_string(index=False))
    else:
        print("No cluster evidence output (missing columns in dataset).")

    # Show top-N flagged in console, sorted by risk
    if not flagged_df.empty:
        print("\nTop 20 flagged transactions by fraud probability:\n")
        display_cols = [c for c in ["timestamp", "amount_ngn", "sender_account", "receiver_account",
                                    "fraud_probability", "risk_level", "recommended_action", "is_fraud"]
                        if c in flagged_df.columns]
        top20 = flagged_df.sort_values("fraud_probability", ascending=False).head(20)[display_cols]
        print(top20.to_string(index=False))
    else:
        print("\nNo transactions were flagged at the chosen threshold.")

    print("\n" + "=" * 80)
    print("Flagged transactions report complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()

