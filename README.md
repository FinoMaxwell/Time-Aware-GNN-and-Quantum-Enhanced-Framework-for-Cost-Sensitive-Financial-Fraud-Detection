# Quantum-Enhanced Fraud Detection

A research-grade fraud detection pipeline comparing **Graph Neural Networks (GCN)**, **Variational Quantum Circuits (VQC)**, and classical ML baselines on real Nigerian financial transaction data.


---

## Overview

This project detects financial fraud by combining three paradigms:

- **Classical ML** — Random Forest, XGBoost, Logistic Regression
- **Graph Neural Network** — GCN trained on a heterogeneous transaction-entity graph (devices, accounts, IPs, merchants, regions)
- **Quantum ML** — Variational Quantum Circuit (VQC) and Quantum Kernel SVM via PennyLane

A key focus is **cost-sensitive evaluation**: rather than optimizing accuracy on imbalanced data, models are evaluated by financial loss (₦1,000 per missed fraud vs. ₦10 per false alarm).

---

## Results

| Model | Accuracy | Recall | F1-Score | ROC-AUC | Total Loss (₦) |
|---|---|---|---|---|---|
| Random Forest | 96.4% | 0.0% | 0.000 | 0.582 | 361,000 |
| XGBoost | 96.3% | 0.0% | 0.000 | 0.580 | 361,050 |
| Logistic Regression | 96.4% | 0.0% | 0.000 | 0.598 | 361,000 |
| **GCN (GNN)** | 4.0% | **99.7%** | 0.070 | 0.524 | **97,000** |
| Quantum VQC | 79.6% | 21.4% | 0.056 | 0.541 | 11,910* |

*Evaluated on a 500-sample subset due to simulation cost.

**Key finding:** Classical models achieve high accuracy by predicting every transaction as legitimate — detecting zero fraud. The GCN catches 360/361 fraudulent transactions through relational risk propagation, reducing total financial loss by 73%.

---

## Project Structure

```
├── src/
│   ├── dataset_loader.py          # Nigerian dataset loading & sampling
│   ├── feature_engineering.py     # Temporal, behavioral, merchant & geo features
│   ├── data_preprocessing.py      # RobustScaler, time-aware splitting
│   ├── classical_models.py        # RF, XGBoost, Logistic Regression wrappers
│   ├── gnn_models.py              # Heterogeneous graph construction + GCN classifier
│   ├── quantum_models.py          # VQC and Quantum Kernel SVM (PennyLane)
│   ├── quantum_feature_maps.py    # Angle, amplitude & re-uploading encoding
│   ├── cost_sensitive.py          # Asymmetric loss & threshold optimization
│   ├── evaluation.py              # Metrics computation & model comparison
│   ├── explainability.py          # SHAP, permutation importance, sensitivity
│   ├── robustness.py              # Dataset size, label noise, imbalance tests
│   └── visualization.py           # ROC, PR curves, confusion matrices, plots
├── scripts/
│   ├── main_advanced.py           # Full pipeline: train all models & evaluate
│   ├── flagged_transactions_gnn.py  # Generate flagged transaction report (GNN)
│   └── compute_fraud_probability_demo.py  # Step-by-step probability walkthrough
├── data/
│   ├── gnn_predictions.csv        # Full test set predictions with probabilities
│   ├── flagged_transactions_gnn.csv  # High-risk flagged transactions
│   ├── cluster_evidence_gnn.csv   # Fraud ring cluster analysis
│   └── model_comparison.csv       # Metrics for all models
└── README.md
```

---

## Installation

```bash
git clone https://github.com/FinoMaxwel/Time-Aware-GNN-and-Quantum-Enhanced-Framework-for-Cost-Sensitive-Financial-Fraud-Detection.git
cd quantum-fraud-detection
pip install -r requirements.txt
```

**Requirements:**
```
numpy
pandas
scikit-learn
xgboost
torch
torch-geometric
pennylane
matplotlib
seaborn
```

---

## Dataset

This project uses the **Nigerian Financial Transactions and Fraud Detection Dataset (V2)** from Kaggle.

1. Download from Kaggle: `Nigerian-Financial-Transactions-and-Fraud-Detection-Dataset`
2. Place the CSV file in:
```
data/Nigerian-Financial-Transactions-and-Fraud-Detection-Dataset/
```

The pipeline samples 50,000 transactions by default (configurable via `sample_size` in `main_advanced.py`).

---

## Usage

**Run the full pipeline (all models):**
```bash
python scripts/main_advanced.py
```

**Generate flagged transactions report (GNN only):**
```bash
python scripts/flagged_transactions_gnn.py
```

**Understand how fraud probability is computed:**
```bash
python scripts/compute_fraud_probability_demo.py
```

Outputs are saved to `data/` (CSVs) and `data/plots/` (visualizations).

---

## Methodology

### Time-Aware Splitting
Transactions are split chronologically (80/20) to simulate real deployment — the model trains on the past and predicts the future, preventing data leakage.

### Graph Construction
A heterogeneous bipartite graph connects each transaction to shared entity nodes across 7 columns: `sender_account`, `receiver_account`, `device_hash`, `ip_address`, `merchant_category`, `location`, `ip_geo_region`. Entity nodes are initialized with historical fraud rate statistics computed from training data only.

### Cost-Sensitive Threshold Optimization
The GCN decision threshold is tuned on a validation window by minimizing:

```
Cost = (False Negatives × ₦1,000) + (False Positives × ₦10)
```

This reflects the real asymmetry between missing fraud and generating false alarms.

---

## Related Work

> Innan, N., et al. (2024). Financial Fraud Detection using Quantum Graph Neural Networks. *Quantum Machine Intelligence, 6*(1), 7. https://doi.org/10.1007/s42484-024-00143-6

---

## License

This project is for academic research purposes. See [LICENSE](LICENSE) for details.
