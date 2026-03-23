"""
Quantum Machine Learning Models for Fraud Detection
Uses PennyLane for hybrid quantum-classical training
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
import warnings
warnings.filterwarnings('ignore')


class QuantumVariationalClassifier(BaseEstimator, ClassifierMixin):
    """
    Variational Quantum Classifier (VQC) for fraud detection
    Uses parameterized quantum circuits with classical optimization
    """
    
    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        learning_rate: float = 0.01,
        max_iter: int = 100,
        batch_size: int = 32,
        optimizer: str = 'adam',
        device: str = 'default.qubit',
        random_state: int = 42
    ):
        """
        Initialize Quantum Variational Classifier
        
        Args:
            n_qubits: Number of qubits (should match or be less than n_features)
            n_layers: Number of variational layers
            learning_rate: Learning rate for optimizer
            max_iter: Maximum training iterations
            batch_size: Batch size for training
            optimizer: 'adam', 'sgd', or 'rmsprop'
            device: PennyLane device name
            random_state: Random seed
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.device = device
        self.random_state = random_state
        
        self.dev = None
        self.qnode = None
        self.weights = None
        self.n_features = None
        self.is_fitted = False
        
        np.random.seed(random_state)
    
    def _create_circuit(self, n_features: int):
        """Create parameterized quantum circuit"""
        # Initialize device
        self.dev = qml.device(self.device, wires=self.n_qubits)
        
        # Feature encoding: amplitude encoding or angle encoding
        def feature_encoding(x):
            """Encode classical features into quantum state"""
            # Use angle encoding for simplicity
            for i in range(min(len(x), self.n_qubits)):
                qml.RY(x[i], wires=i)
        
        # Variational layers
        def variational_layer(weights):
            """Parameterized quantum circuit layers"""
            for layer in range(self.n_layers):
                # Entangling layer
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if self.n_qubits > 2:
                    qml.CNOT(wires=[self.n_qubits - 1, 0])
                
                # Rotation gates
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
        
        # Quantum node
        @qml.qnode(self.dev, interface='autograd')
        def circuit(x, weights):
            """Full quantum circuit"""
            feature_encoding(x[:self.n_qubits])
            variational_layer(weights)
            # Measure expectation value of Z on first qubit
            return qml.expval(qml.PauliZ(0))
        
        self.qnode = circuit
        
        # Initialize weights
        weight_shape = (self.n_layers, self.n_qubits, 2)
        self.weights = pnp.random.normal(0, 0.1, weight_shape, requires_grad=True)
    
    def _predict_proba_single(self, x: np.ndarray) -> float:
        """Predict probability for a single sample"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Normalize output to [0, 1] using sigmoid
        output = self.qnode(x, self.weights)
        # Convert to regular numpy for sigmoid
        if hasattr(output, 'numpy'):
            output = output.numpy()
        prob = 1 / (1 + np.exp(-output))
        return float(prob)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the quantum classifier
        
        Args:
            X: Training features
            y: Training labels
        """
        X, y = check_X_y(X, y)
        self.n_features = X.shape[1]
        
        # Adjust n_qubits if needed
        if self.n_qubits > self.n_features:
            self.n_qubits = self.n_features
        
        # Create circuit
        self._create_circuit(self.n_features)
        
        # Setup optimizer
        if self.optimizer == 'adam':
            opt = qml.AdamOptimizer(stepsize=self.learning_rate)
        elif self.optimizer == 'sgd':
            opt = qml.GradientDescentOptimizer(stepsize=self.learning_rate)
        else:
            opt = qml.AdamOptimizer(stepsize=self.learning_rate)
        
        # Training loop
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        for iteration in range(self.max_iter):
            # Mini-batch training
            np.random.shuffle(indices)
            batch_indices = indices[:self.batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            # Compute loss and gradients
            def cost(weights):
                # Convert batch to PennyLane numpy arrays
                y_batch_pl = pnp.array(y_batch, requires_grad=False)
                
                # Compute predictions - build list then convert to array
                pred_list = []
                for x in X_batch:
                    pred_list.append(self.qnode(x, weights))
                # Stack predictions using qml.math for autograd compatibility
                predictions = qml.math.stack(pred_list)
                
                # Binary cross-entropy loss with sigmoid
                # Use qml.math functions for autograd compatibility
                probs = 1 / (1 + qml.math.exp(-predictions))
                # Add small epsilon for numerical stability in log
                eps = 1e-10
                loss = -qml.math.mean(y_batch_pl * qml.math.log(probs + eps) + 
                                     (1 - y_batch_pl) * qml.math.log(1 - probs + eps))
                return loss
            
            # Update weights
            self.weights = opt.step(cost, self.weights)
            
            if (iteration + 1) % 20 == 0:
                loss_val = cost(self.weights)
                # Convert to float if it's an autograd array
                if hasattr(loss_val, 'numpy'):
                    loss_val = float(loss_val.numpy())
                print(f"Iteration {iteration + 1}/{self.max_iter}, Loss: {loss_val:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        X = check_array(X)
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        probs = []
        for x in X:
            prob = self._predict_proba_single(x)
            probs.append([1 - prob, prob])
        
        return np.array(probs)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)


class QuantumKernelSVM(BaseEstimator, ClassifierMixin):
    """
    Quantum Support Vector Machine using quantum kernel
    Uses quantum feature maps for kernel computation
    """
    
    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        device: str = 'default.qubit',
        random_state: int = 42
    ):
        """
        Initialize Quantum Kernel SVM
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of feature map layers
            device: PennyLane device name
            random_state: Random seed
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = device
        self.random_state = random_state
        
        self.dev = None
        self.kernel = None
        self.X_train = None
        self.y_train = None
        self.n_features = None
        self.is_fitted = False
        
        np.random.seed(random_state)
    
    def _create_kernel(self, n_features: int):
        """Create quantum kernel circuit"""
        self.dev = qml.device(self.device, wires=self.n_qubits)
        
        @qml.qnode(self.dev)
        def kernel(x1, x2):
            """Quantum kernel function"""
            # Feature encoding for first data point
            for i in range(min(len(x1), self.n_qubits)):
                qml.RY(x1[i], wires=i)
            
            # Adjoint feature encoding for second data point
            for i in range(min(len(x2), self.n_qubits)):
                qml.RY(-x2[i], wires=i)
            
            # Entangling layers
            for layer in range(self.n_layers):
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if self.n_qubits > 2:
                    qml.CNOT(wires=[self.n_qubits - 1, 0])
            
            # Measure overlap
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        
        self.kernel = kernel
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the quantum kernel SVM"""
        X, y = check_X_y(X, y)
        self.n_features = X.shape[1]
        
        if self.n_qubits > self.n_features:
            self.n_qubits = self.n_features
        
        self._create_kernel(self.n_features)
        self.X_train = X
        self.y_train = y
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using quantum kernel"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = check_array(X)
        predictions = []
        
        for x in X:
            # Compute kernel with all training samples
            kernel_values = [self.kernel(x, x_train) for x_train in self.X_train]
            kernel_values = np.array(kernel_values)
            
            # Simple majority vote weighted by kernel similarity
            weights = kernel_values
            weighted_vote = np.sum(weights * (2 * self.y_train - 1))
            pred = 1 if weighted_vote > 0 else 0
            predictions.append(pred)
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = check_array(X)
        probs = []
        
        for x in X:
            kernel_values = [self.kernel(x, x_train) for x_train in self.X_train]
            kernel_values = np.array(kernel_values)
            
            # Normalize kernel values to probabilities
            pos_weights = np.sum(kernel_values[self.y_train == 1])
            neg_weights = np.sum(kernel_values[self.y_train == 0])
            total = pos_weights + neg_weights + 1e-10
            
            prob_fraud = pos_weights / total
            probs.append([1 - prob_fraud, prob_fraud])
        
        return np.array(probs)
