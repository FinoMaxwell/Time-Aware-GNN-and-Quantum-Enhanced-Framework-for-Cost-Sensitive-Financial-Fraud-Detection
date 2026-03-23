"""
Demonstration: How Fraud Probability is Computed from Features

This script shows step-by-step how each model type computes fraud probability
and how individual features impact the final prediction.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from src.classical_models import ClassicalBaseline
from src.gnn_models import build_transaction_entity_graph, GCNFraudClassifier
from src.quantum_models import QuantumVariationalClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def demonstrate_logistic_regression():
    """Show how Logistic Regression computes fraud probability"""
    print("="*80)
    print("DEMONSTRATION 1: Logistic Regression - Exact Formula")
    print("="*80)
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(100, 17)  # 100 transactions, 17 features
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple rule: fraud if feature 0 + feature 1 > 0
    
    # Train model
    model = ClassicalBaseline('logistic')
    model.fit(X, y)
    
    # Get learned weights
    lr_model = model.model
    weights = np.concatenate([[lr_model.intercept_[0]], lr_model.coef_[0]])
    
    print("\nLearned Weights:")
    print(f"  Bias (w₀): {weights[0]:.4f}")
    print(f"  Feature 1 (w₁): {weights[1]:.4f}")
    print(f"  Feature 2 (w₂): {weights[2]:.4f}")
    print(f"  ... (15 more features)")
    
    # Example transaction
    x_example = X[0]
    print(f"\nExample Transaction Features:")
    print(f"  x₁ (feature 1): {x_example[0]:.4f}")
    print(f"  x₂ (feature 2): {x_example[1]:.4f}")
    print(f"  ... (15 more features)")
    
    # Manual computation
    z = weights[0] + np.dot(weights[1:], x_example)
    P_manual = 1 / (1 + np.exp(-z))
    
    # Model prediction
    P_model = model.predict_proba(x_example.reshape(1, -1))[0, 1]
    
    print(f"\nStep-by-Step Computation:")
    print(f"  1. Linear combination: z = w₀ + Σ(wᵢ·xᵢ)")
    print(f"     z = {weights[0]:.4f} + {np.dot(weights[1:], x_example):.4f}")
    print(f"     z = {z:.4f}")
    print(f"  2. Sigmoid: P(fraud) = 1 / (1 + exp(-z))")
    print(f"     P(fraud) = 1 / (1 + exp(-{z:.4f}))")
    print(f"     P(fraud) = {P_manual:.4f}")
    print(f"\nModel Prediction: {P_model:.4f}")
    print(f"Match: {'✓' if abs(P_manual - P_model) < 1e-6 else '✗'}")
    
    # Feature contributions
    print(f"\nFeature Contributions (wᵢ · xᵢ):")
    contributions = weights[1:] * x_example
    top_5_idx = np.argsort(np.abs(contributions))[-5:][::-1]
    for idx in top_5_idx:
        print(f"  Feature {idx+1}: {contributions[idx]:+.4f} (weight: {weights[idx+1]:+.4f}, value: {x_example[idx]:+.4f})")


def demonstrate_random_forest():
    """Show how Random Forest computes fraud probability"""
    print("\n" + "="*80)
    print("DEMONSTRATION 2: Random Forest - Ensemble Voting")
    print("="*80)
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(100, 17)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Train model
    model = ClassicalBaseline('random_forest', n_estimators=10)  # Use 10 trees for demo
    model.fit(X, y)
    
    # Example transaction
    x_example = X[0]
    
    print(f"\nExample Transaction:")
    print(f"  Features: {x_example[:5]} ... (12 more)")
    
    # Get individual tree predictions
    rf_model = model.model
    tree_predictions = []
    for tree in rf_model.estimators_:
        pred = tree.predict(x_example.reshape(1, -1))[0]
        tree_predictions.append(pred)
    
    tree_predictions = np.array(tree_predictions)
    fraud_votes = np.sum(tree_predictions == 1)
    normal_votes = np.sum(tree_predictions == 0)
    P_ensemble = fraud_votes / len(tree_predictions)
    
    print(f"\nTree-by-Tree Voting:")
    for i, pred in enumerate(tree_predictions):
        print(f"  Tree {i+1}: {'FRAUD' if pred == 1 else 'NORMAL'}")
    
    print(f"\nVote Count:")
    print(f"  FRAUD votes: {fraud_votes}")
    print(f"  NORMAL votes: {normal_votes}")
    print(f"  Total trees: {len(tree_predictions)}")
    
    print(f"\nFinal Probability:")
    print(f"  P(fraud) = FRAUD_votes / Total_trees")
    print(f"  P(fraud) = {fraud_votes} / {len(tree_predictions)}")
    print(f"  P(fraud) = {P_ensemble:.4f}")
    
    # Model prediction
    P_model = model.predict_proba(x_example.reshape(1, -1))[0, 1]
    print(f"\nModel Prediction: {P_model:.4f}")
    print(f"Match: {'✓' if abs(P_ensemble - P_model) < 0.01 else '✗'}")
    
    # Feature importance
    print(f"\nFeature Importance (how often feature is used in splits):")
    importances = rf_model.feature_importances_
    top_5_idx = np.argsort(importances)[-5:][::-1]
    for idx in top_5_idx:
        print(f"  Feature {idx+1}: {importances[idx]:.4f}")


def demonstrate_gnn():
    """Show how GNN computes fraud probability (simplified)"""
    print("\n" + "="*80)
    print("DEMONSTRATION 3: GNN - Graph-Based Aggregation")
    print("="*80)
    
    print("\nGNN Computation Process:")
    print("  1. Build graph: Transactions ↔ Entities (accounts, devices, IPs, etc.)")
    print("  2. Layer 1: Each transaction aggregates features from connected entities")
    print("  3. Layer 2: Further refinement using 2-hop neighbors")
    print("  4. Output: Fraud probability from final node representation")
    
    print("\nExample:")
    print("  Transaction T₁:")
    print("    Own features: [amount=2.0, velocity=1.0, ...]")
    print("    Connected to Device D (used in 8 fraud cases)")
    print("    Connected to Account A (used in 3 fraud cases)")
    print("    Connected to Merchant M (fraud rate = 0.15)")
    
    print("\n  GNN Message Passing:")
    print("    Step 1: T₁ receives 'risk signals' from Device D, Account A, Merchant M")
    print("    Step 2: Aggregates: T₁_features + Device_D_risk + Account_A_risk + Merchant_M_risk")
    print("    Step 3: Learns: 'Device D is suspicious' → increases T₁'s fraud probability")
    
    print("\n  Result:")
    print("    Without graph: P(fraud) = 0.30 (based on T₁'s own features)")
    print("    With graph: P(fraud) = 0.75 (Device D's history increases risk)")
    
    print("\n  Key Insight:")
    print("    GNN captures RELATIONAL patterns:")
    print("    - Fraudsters often share devices/IPs")
    print("    - High-risk merchants attract fraud")
    print("    - Account history matters")


def demonstrate_quantum_vqc():
    """Show how Quantum VQC computes fraud probability (simplified)"""
    print("\n" + "="*80)
    print("DEMONSTRATION 4: Quantum VQC - Quantum Circuit")
    print("="*80)
    
    print("\nQuantum VQC Computation Process:")
    print("  1. Feature Encoding: Encode first 4 features as quantum rotations")
    print("  2. Variational Layers: Apply entangling gates + learnable rotations")
    print("  3. Measurement: Measure expectation value ⟨Z₀⟩")
    print("  4. Probability: Convert measurement to probability via sigmoid")
    
    print("\nExample:")
    print("  Input Features (first 4 only):")
    print("    x₁ = amount_ngn = 3.2")
    print("    x₂ = velocity_score = 1.8")
    print("    x₃ = geo_anomaly_score = 0.9")
    print("    x₄ = channel_risk_score = 0.7")
    
    print("\n  Step 1: Quantum Encoding")
    print("    Qubit 0: RY(3.2) → |ψ₀⟩ = cos(1.6)|0⟩ + sin(1.6)|1⟩")
    print("    Qubit 1: RY(1.8) → |ψ₁⟩ = cos(0.9)|0⟩ + sin(0.9)|1⟩")
    print("    Qubit 2: RY(0.9) → |ψ₂⟩ = cos(0.45)|0⟩ + sin(0.45)|1⟩")
    print("    Qubit 3: RY(0.7) → |ψ₃⟩ = cos(0.35)|0⟩ + sin(0.35)|1⟩")
    print("    Combined: |ψ⟩ = |ψ₀⟩ ⊗ |ψ₁⟩ ⊗ |ψ₂⟩ ⊗ |ψ₃⟩")
    
    print("\n  Step 2: Variational Layers")
    print("    Layer 1: Entangle qubits (CNOT gates) + Rotate (learnable parameters)")
    print("    Layer 2: Further entangle + rotate")
    print("    Result: |ψ_final⟩")
    
    print("\n  Step 3: Measurement")
    print("    output = ⟨ψ_final| Z₀ |ψ_final⟩")
    print("    output = +0.8 (example)")
    
    print("\n  Step 4: Probability")
    print("    P(fraud) = sigmoid(0.8)")
    print("    P(fraud) = 1 / (1 + exp(-0.8))")
    print("    P(fraud) = 0.69")
    
    print("\n  Key Insight:")
    print("    Quantum ENTANGLEMENT captures non-linear correlations")
    print("    between the 4 encoded features")
    print("    (Hard to interpret, but captures complex patterns)")


def demonstrate_feature_impact():
    """Show how to analyze feature impact"""
    print("\n" + "="*80)
    print("DEMONSTRATION 5: Feature Impact Analysis")
    print("="*80)
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(100, 17)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Train Random Forest
    model = ClassicalBaseline('random_forest')
    model.fit(X, y)
    
    # Feature importance
    importances = model.model.feature_importances_
    
    print("\nMethod 1: Feature Importance (Random Forest)")
    print("  Shows how often each feature is used in decision tree splits")
    print("\n  Top 5 Most Important Features:")
    top_5_idx = np.argsort(importances)[-5:][::-1]
    for i, idx in enumerate(top_5_idx, 1):
        print(f"    {i}. Feature {idx+1}: {importances[idx]:.4f}")
    
    # Train Logistic Regression for coefficient analysis
    lr_model = ClassicalBaseline('logistic')
    lr_model.fit(X, y)
    coefficients = lr_model.model.coef_[0]
    
    print("\nMethod 2: Coefficient Analysis (Logistic Regression)")
    print("  Shows direct impact: positive = increases fraud prob, negative = decreases")
    print("\n  Top 5 Features by |coefficient|:")
    top_5_coef_idx = np.argsort(np.abs(coefficients))[-5:][::-1]
    for i, idx in enumerate(top_5_coef_idx, 1):
        sign = "+" if coefficients[idx] > 0 else "-"
        print(f"    {i}. Feature {idx+1}: {sign}{abs(coefficients[idx]):.4f}")
        print(f"       Impact: {'Increases' if coefficients[idx] > 0 else 'Decreases'} fraud probability")
    
    print("\nMethod 3: SHAP Values (for detailed analysis)")
    print("  Install: pip install shap")
    print("  Usage:")
    print("    import shap")
    print("    explainer = shap.Explainer(model, X_train)")
    print("    shap_values = explainer(X_test)")
    print("  Shows contribution of each feature to each prediction")


def main():
    """Run all demonstrations"""
    print("\n" + "="*80)
    print("HOW FRAUD PROBABILITY IS COMPUTED FROM FEATURES")
    print("="*80)
    print("\nThis script demonstrates the exact computation process for each model type.")
    print("Run each demonstration to see step-by-step how features become fraud probabilities.\n")
    
    try:
        demonstrate_logistic_regression()
        demonstrate_random_forest()
        demonstrate_gnn()
        demonstrate_quantum_vqc()
        demonstrate_feature_impact()
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print("\n1. Classical ML (Logistic Regression):")
        print("   P(fraud) = sigmoid(w₀ + w₁·x₁ + w₂·x₂ + ... + w₁₇·x₁₇)")
        print("   → Direct formula, fully interpretable")
        
        print("\n2. Classical ML (Random Forest):")
        print("   P(fraud) = (votes_for_fraud) / (total_trees)")
        print("   → Ensemble voting, feature importance available")
        
        print("\n3. GNN (Graph Neural Network):")
        print("   P(fraud) = sigmoid(GCN_layers(transaction_features + neighbor_features))")
        print("   → Relational learning, captures fraud rings")
        
        print("\n4. Quantum VQC:")
        print("   P(fraud) = sigmoid(⟨quantum_circuit(encoded_features)⟩)")
        print("   → Quantum entanglement, non-linear correlations")
        
        print("\n5. Feature Impact:")
        print("   - Feature Importance: How often feature is used")
        print("   - Coefficients: Direct impact (positive/negative)")
        print("   - SHAP Values: Contribution to each prediction")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Some demonstrations may require trained models or additional data.")


if __name__ == "__main__":
    main()
