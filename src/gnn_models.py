"""
Graph Neural Network (GNN) models for fraud detection (CPU-friendly).

This module implements a simple GCN-style message passing network in pure PyTorch
without relying on PyTorch Geometric (so it runs on a standard laptop CPU).

We model fraud detection as TRANSACTION NODE CLASSIFICATION on a graph:
- Each transaction is a node with tabular features (scaled numeric features).
- Each entity (account/device/ip/merchant/location/region) is a node.
- Edges connect a transaction node to the entity nodes involved in it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _TORCH_IMPORT_ERROR = e
else:
    _TORCH_IMPORT_ERROR = None


@dataclass
class BuiltGraph:
    """Container for a constructed transaction-entity graph."""

    x: "torch.Tensor"  # (N, F) node features
    adj: "torch.Tensor"  # sparse (N, N) normalized adjacency
    transaction_node_idx: np.ndarray  # indices of transaction nodes (0..n_txn-1)
    y: "torch.Tensor"  # (n_txn,) labels for transaction nodes


def _require_torch():
    if torch is None:
        raise ImportError(
            "PyTorch is required for the GNN model but is not installed.\n"
            "Install it with: pip install torch\n"
            f"Original import error: {_TORCH_IMPORT_ERROR}"
        )


def _as_str(val) -> str:
    # Make entity values hashable/consistent (avoid float NaN issues)
    if val is None:
        return ""
    return str(val)


def build_transaction_entity_graph(
    *,
    transaction_features: np.ndarray,
    transaction_labels: np.ndarray,
    entity_table: "np.ndarray",
    entity_col_names: List[str],
    entity_type_by_col: Optional[Dict[str, str]] = None,
    train_txn_idx_for_entity_stats: Optional[np.ndarray] = None,
    entity_stat_smoothing_alpha: float = 1.0,
    entity_stat_smoothing_beta: float = 1.0,
    max_unique_entities_per_col: Optional[int] = None,
) -> BuiltGraph:
    """
    Build a transaction-entity graph suitable for CPU GNN training.

    Args:
        transaction_features: (n_txn, d) numeric features for each transaction (already scaled).
        transaction_labels: (n_txn,) 0/1 labels.
        entity_table: (n_txn, n_entity_cols) values for entity columns (strings/ids).
        entity_col_names: names matching entity_table columns.
        entity_type_by_col: map col_name -> type name (e.g. sender_account/receiver_account -> "account").
        train_txn_idx_for_entity_stats: optional indices (in [0..n_txn-1]) used to compute
            per-entity fraud rate features (count + smoothed fraud rate) to help the GNN learn rings.
        entity_stat_smoothing_alpha: Laplace-style smoothing prior alpha for fraud rate.
        entity_stat_smoothing_beta: Laplace-style smoothing prior beta for fraud rate.
        max_unique_entities_per_col: optional cap to limit graph size (keeps the most frequent entities).

    Returns:
        BuiltGraph containing node features, sparse normalized adjacency, and labels.
    """
    _require_torch()

    X = np.asarray(transaction_features, dtype=np.float32)
    y = np.asarray(transaction_labels, dtype=np.int64)
    if X.ndim != 2:
        raise ValueError("transaction_features must be 2D (n_txn, d)")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("transaction_labels must be 1D and aligned to transaction_features")
    if entity_table.shape[0] != X.shape[0]:
        raise ValueError("entity_table rows must match number of transactions")
    if entity_table.shape[1] != len(entity_col_names):
        raise ValueError("entity_table columns must match entity_col_names length")

    n_txn, d_txn = X.shape

    if entity_type_by_col is None:
        entity_type_by_col = {c: c for c in entity_col_names}

    # Types: transaction + unique entity types
    entity_types = sorted(set(entity_type_by_col.get(c, c) for c in entity_col_names))
    types = ["transaction"] + entity_types
    type_to_idx = {t: i for i, t in enumerate(types)}
    type_dim = len(types)

    # Build mapping from (type, value) -> node index.
    # Transaction nodes are 0..n_txn-1. Entity nodes start at offset.
    entity_offset = n_txn
    entity_node_map: Dict[Tuple[str, str], int] = {}

    # Optional size cap: keep only the most frequent entities per column
    allowed_values_by_col: Optional[Dict[str, set]] = None
    if max_unique_entities_per_col is not None and max_unique_entities_per_col > 0:
        allowed_values_by_col = {}
        for j, col in enumerate(entity_col_names):
            col_vals = [_as_str(v) for v in entity_table[:, j]]
            col_vals = [v for v in col_vals if v not in ("", "nan", "NaN", "None")]
            if not col_vals:
                allowed_values_by_col[col] = set()
                continue
            # most frequent
            uniq, counts = np.unique(np.asarray(col_vals, dtype=object), return_counts=True)
            top_idx = np.argsort(-counts)[:max_unique_entities_per_col]
            allowed_values_by_col[col] = set(str(uniq[i]) for i in top_idx)

    edges_src: List[int] = []
    edges_dst: List[int] = []

    next_entity_node = entity_offset

    # Optional: compute per-entity stats from the training window only.
    # Key: (entity_type, entity_value_as_str) -> [count_in_train, fraud_count_in_train]
    entity_stats: Dict[Tuple[str, str], List[float]] = {}
    train_mask = None
    if train_txn_idx_for_entity_stats is not None:
        train_mask = np.zeros(n_txn, dtype=bool)
        train_txn_idx_for_entity_stats = np.asarray(train_txn_idx_for_entity_stats, dtype=np.int64)
        train_mask[train_txn_idx_for_entity_stats] = True

    for i in range(n_txn):
        for j, col in enumerate(entity_col_names):
            raw_val = entity_table[i, j]
            val = _as_str(raw_val)
            if val in ("", "nan", "NaN", "None"):
                continue
            if allowed_values_by_col is not None and val not in allowed_values_by_col.get(col, set()):
                continue

            ent_type = entity_type_by_col.get(col, col)
            key = (ent_type, val)
            node_id = entity_node_map.get(key)
            if node_id is None:
                node_id = next_entity_node
                entity_node_map[key] = node_id
                next_entity_node += 1

            # Update stats only when the transaction is inside the training window.
            if train_mask is not None and train_mask[i]:
                if key not in entity_stats:
                    entity_stats[key] = [0.0, 0.0]
                entity_stats[key][0] += 1.0
                entity_stats[key][1] += float(y[i])

            # Undirected edge: txn <-> entity
            edges_src.append(i)
            edges_dst.append(node_id)
            edges_src.append(node_id)
            edges_dst.append(i)

    n_nodes = next_entity_node

    # Build node feature matrix:
    # - transaction nodes: [txn_features, onehot(transaction)]
    # - entity nodes: [zeros, onehot(entity_type)]
    # We append 2 entity-stat features to help the GNN learn risky entities:
    #   entity_log_count, entity_fraud_rate_smoothed
    stat_dim = 2
    x = np.zeros((n_nodes, d_txn + type_dim + stat_dim), dtype=np.float32)
    x[:n_txn, :d_txn] = X
    x[:n_txn, d_txn + type_to_idx["transaction"]] = 1.0

    for (ent_type, _val), node_id in entity_node_map.items():
        x[node_id, d_txn + type_to_idx[ent_type]] = 1.0
        if train_mask is None:
            # No train-based stats requested; keep zeros.
            continue
        # Fill entity stats from transactions in the training window.
        key = (ent_type, _val)
        count_in_train = entity_stats.get(key, [0.0, 0.0])[0]
        fraud_count_in_train = entity_stats.get(key, [0.0, 0.0])[1]
        # Smoothed fraud rate: (fraud+alpha)/(count+alpha+beta)
        fraud_rate_smoothed = (fraud_count_in_train + entity_stat_smoothing_alpha) / (
            count_in_train + entity_stat_smoothing_alpha + entity_stat_smoothing_beta
        )
        log_count = np.log1p(count_in_train)
        x[node_id, d_txn + type_dim + 0] = float(log_count)
        x[node_id, d_txn + type_dim + 1] = float(fraud_rate_smoothed)

    # Add self-loops (important for GCN stability)
    for i in range(n_nodes):
        edges_src.append(i)
        edges_dst.append(i)

    # Build sparse adjacency, then symmetric normalization:
    # A_norm = D^{-1/2} A D^{-1/2}
    idx = torch.tensor([edges_src, edges_dst], dtype=torch.int64)
    val = torch.ones(idx.shape[1], dtype=torch.float32)
    adj = torch.sparse_coo_tensor(idx, val, (n_nodes, n_nodes)).coalesce()

    # Degree
    deg = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1.0)
    deg_inv_sqrt = torch.pow(deg, -0.5)

    row, col = adj.indices()
    norm_vals = deg_inv_sqrt[row] * adj.values() * deg_inv_sqrt[col]
    adj_norm = torch.sparse_coo_tensor(adj.indices(), norm_vals, adj.shape).coalesce()

    return BuiltGraph(
        x=torch.from_numpy(x),
        adj=adj_norm,
        transaction_node_idx=np.arange(n_txn, dtype=np.int64),
        y=torch.from_numpy(y),
    )


class _GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x: "torch.Tensor", adj: "torch.Tensor") -> "torch.Tensor":
        # GCN layer 1: A_norm X W
        h = torch.sparse.mm(adj, x)
        h = self.fc1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        # GCN layer 2
        h = torch.sparse.mm(adj, h)
        logits = self.fc2(h).squeeze(1)  # (N,)
        return logits


class GCNFraudClassifier:
    """
    CPU-friendly GCN for fraud detection (transaction-node classification).

    API is intentionally similar to scikit-learn models used elsewhere in this repo.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        lr: float = 1e-2,
        weight_decay: float = 5e-4,
        max_epochs: int = 20,
        random_state: int = 42,
        verbose: bool = True,
        device: str = "cpu",
    ):
        _require_torch()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.random_state = random_state
        self.verbose = verbose
        self.device = device

        self.model: Optional[_GCN] = None
        self.is_fitted: bool = False
        self.temperature: float = 1.0

    def fit(
        self,
        graph: BuiltGraph,
        train_txn_idx: np.ndarray,
        val_txn_idx: Optional[np.ndarray] = None,
    ) -> "GCNFraudClassifier":
        """
        Train GCN.

        Args:
            graph: BuiltGraph from build_transaction_entity_graph().
            train_txn_idx: indices in [0..n_txn-1] for training transactions.
            val_txn_idx: optional validation indices for simple reporting.
        """
        _require_torch()
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        x = graph.x.to(self.device)
        adj = graph.adj.to(self.device)
        y = graph.y.to(self.device).float()  # (n_txn,)

        # Masks over ALL nodes (we only label transaction nodes)
        n_txn = y.shape[0]
        node_y = torch.full((x.shape[0],), -1.0, device=self.device)
        node_y[:n_txn] = y

        train_txn_idx_t = torch.tensor(train_txn_idx, dtype=torch.int64, device=self.device)
        train_mask = torch.zeros((x.shape[0],), dtype=torch.bool, device=self.device)
        train_mask[train_txn_idx_t] = True

        val_mask = None
        if val_txn_idx is not None:
            val_txn_idx_t = torch.tensor(val_txn_idx, dtype=torch.int64, device=self.device)
            val_mask = torch.zeros((x.shape[0],), dtype=torch.bool, device=self.device)
            val_mask[val_txn_idx_t] = True

        self.model = _GCN(in_dim=x.shape[1], hidden_dim=self.hidden_dim, dropout=self.dropout).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Handle class imbalance with pos_weight
        y_train = node_y[train_mask]
        pos = (y_train == 1).sum().clamp(min=1)
        neg = (y_train == 0).sum().clamp(min=1)
        pos_weight = (neg / pos).detach()
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_loss = float("inf")
        best_state = None
        patience = 5
        patience_left = patience

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            opt.zero_grad()
            logits = self.model(x, adj)
            loss = loss_fn(logits[train_mask], node_y[train_mask])
            loss.backward()
            opt.step()

            if self.verbose and (epoch == 1 or epoch % 5 == 0 or epoch == self.max_epochs):
                with torch.no_grad():
                    self.model.eval()
                    train_prob = torch.sigmoid(logits[train_mask])
                    train_pred = (train_prob >= 0.5).long()
                    train_acc = (train_pred == node_y[train_mask].long()).float().mean().item()

                    msg = f"      Epoch {epoch:02d}/{self.max_epochs} | loss={loss.item():.4f} | train_acc={train_acc:.3f}"
                    if val_mask is not None:
                        val_prob = torch.sigmoid(logits[val_mask])
                        val_pred = (val_prob >= 0.5).long()
                        val_acc = (val_pred == node_y[val_mask].long()).float().mean().item()
                        msg += f" | val_acc={val_acc:.3f}"
                    print(msg)

            # Early stopping + best checkpoint based on validation loss
            if val_mask is not None:
                with torch.no_grad():
                    self.model.eval()
                    logits_full = self.model(x, adj)
                    val_logits = logits_full[val_mask]
                    val_labels = node_y[val_mask]
                    val_loss = loss_fn(val_logits, val_labels).item()
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_left = patience
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        if self.verbose:
                            print(f"      Early stopping at epoch {epoch} (best_val_loss={best_val_loss:.4f})")
                        break

        # Restore best checkpoint (if early stopping ran)
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Temperature scaling calibration on validation logits
        self.temperature = 1.0
        if val_mask is not None:
            with torch.no_grad():
                self.model.eval()
                logits_full = self.model(x, adj)
                val_logits = logits_full[val_mask]
                val_labels = node_y[val_mask]

            # Grid search over temperature values to minimize BCE loss
            # Using a small grid keeps runtime reasonable on CPU.
            candidate_temps = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
            best_t = 1.0
            best_temp_loss = float("inf")
            bce = nn.BCEWithLogitsLoss()
            for T in candidate_temps:
                T = float(T)
                temp_logits = val_logits / T
                temp_loss = bce(temp_logits, val_labels).item()
                if temp_loss < best_temp_loss:
                    best_temp_loss = temp_loss
                    best_t = T
            self.temperature = best_t
            if self.verbose:
                print(f"      Calibrated temperature (T) = {self.temperature:.3f} (val BCE={best_temp_loss:.4f})")

        self.is_fitted = True
        return self

    def predict_proba(self, graph: BuiltGraph, txn_idx: np.ndarray) -> np.ndarray:
        """Predict probabilities for provided transaction indices."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
        _require_torch()

        self.model.eval()
        with torch.no_grad():
            x = graph.x.to(self.device)
            adj = graph.adj.to(self.device)
            logits = self.model(x, adj)
            probs = torch.sigmoid(logits / max(self.temperature, 1e-7)).cpu().numpy()  # (N,)

        txn_probs = probs[txn_idx]
        txn_probs = np.clip(txn_probs, 1e-7, 1 - 1e-7)
        return np.vstack([1 - txn_probs, txn_probs]).T

    def predict(self, graph: BuiltGraph, txn_idx: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict 0/1 labels for provided transaction indices."""
        proba = self.predict_proba(graph, txn_idx)
        return (proba[:, 1] >= threshold).astype(int)

