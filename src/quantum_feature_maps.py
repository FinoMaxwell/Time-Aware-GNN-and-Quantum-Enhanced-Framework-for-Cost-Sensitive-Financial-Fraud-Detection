"""
Quantum Feature Map Analysis
Implements and compares different quantum encoding schemes
"""

import numpy as np
import pennylane as qml
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class QuantumFeatureMapAnalyzer:
    """
    Analyzes and compares different quantum feature encoding schemes
    """
    
    def __init__(self, n_qubits: int = 4, device: str = 'default.qubit'):
        """
        Initialize analyzer
        
        Args:
            n_qubits: Number of qubits
            device: PennyLane device name
        """
        self.n_qubits = n_qubits
        self.device = device
        self.dev = qml.device(device, wires=n_qubits)
    
    def angle_encoding_circuit(self, x: np.ndarray) -> qml.QNode:
        """
        Angle encoding: Encode features as rotation angles
        
        Args:
            x: Input features
            
        Returns:
            Quantum node with angle encoding
        """
        @qml.qnode(self.dev)
        def circuit(x):
            for i in range(min(len(x), self.n_qubits)):
                qml.RY(x[i], wires=i)
            return qml.expval(qml.PauliZ(0))
        
        return circuit
    
    def data_reuploading_circuit(
        self,
        x: np.ndarray,
        n_layers: int = 2
    ) -> qml.QNode:
        """
        Data re-uploading: Re-encode data in multiple layers
        
        Args:
            x: Input features
            n_layers: Number of re-uploading layers
            
        Returns:
            Quantum node with data re-uploading
        """
        @qml.qnode(self.dev)
        def circuit(x, weights):
            # Re-upload data in each layer
            for layer in range(n_layers):
                # Encode data
                for i in range(min(len(x), self.n_qubits)):
                    qml.RY(x[i], wires=i)
                
                # Variational layer
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i], wires=i)
            
            return qml.expval(qml.PauliZ(0))
        
        return circuit
    
    def amplitude_encoding_circuit(self, x: np.ndarray) -> qml.QNode:
        """
        Amplitude encoding: Encode features in quantum state amplitudes
        
        Args:
            x: Input features (will be normalized)
            
        Returns:
            Quantum node with amplitude encoding
        """
        @qml.qnode(self.dev)
        def circuit(x):
            # Normalize for amplitude encoding
            x_norm = x / np.linalg.norm(x)
            # Pad to 2^n_qubits
            n_features = 2 ** self.n_qubits
            if len(x_norm) < n_features:
                x_padded = np.pad(x_norm, (0, n_features - len(x_norm)))
            else:
                x_padded = x_norm[:n_features]
            
            qml.AmplitudeEmbedding(x_padded, wires=range(self.n_qubits), normalize=True)
            return qml.expval(qml.PauliZ(0))
        
        return circuit
    
    def compute_kernel_matrix(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        encoding_type: str = 'angle'
    ) -> np.ndarray:
        """
        Compute quantum kernel matrix between two datasets
        
        Args:
            X1: First dataset
            X2: Second dataset
            encoding_type: 'angle', 'reupload', or 'amplitude'
            
        Returns:
            Kernel matrix
        """
        if encoding_type == 'angle':
            circuit = self.angle_encoding_circuit(X1[0])
        elif encoding_type == 'reupload':
            circuit = self.data_reuploading_circuit(X1[0])
            # Initialize random weights for reuploading
            weights = np.random.normal(0, 0.1, (2, self.n_qubits))
        elif encoding_type == 'amplitude':
            circuit = self.amplitude_encoding_circuit(X1[0])
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
        
        kernel_matrix = np.zeros((len(X1), len(X2)))
        
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                if encoding_type == 'reupload':
                    # For reuploading, we need to create a kernel that compares states
                    # This is a simplified version
                    state1 = self._get_state(x1, encoding_type)
                    state2 = self._get_state(x2, encoding_type)
                    kernel_matrix[i, j] = np.abs(np.dot(state1, state2)) ** 2
                else:
                    # Create kernel by comparing encoded states
                    state1 = self._get_state(x1, encoding_type)
                    state2 = self._get_state(x2, encoding_type)
                    kernel_matrix[i, j] = np.abs(np.dot(state1, state2)) ** 2
        
        return kernel_matrix
    
    def _get_state(self, x: np.ndarray, encoding_type: str) -> np.ndarray:
        """Get quantum state representation (simplified)"""
        if encoding_type == 'angle':
            # Simplified: return normalized feature vector
            return x[:self.n_qubits] / (np.linalg.norm(x[:self.n_qubits]) + 1e-10)
        elif encoding_type == 'amplitude':
            x_norm = x / (np.linalg.norm(x) + 1e-10)
            n_features = 2 ** self.n_qubits
            if len(x_norm) < n_features:
                return np.pad(x_norm, (0, n_features - len(x_norm)))
            return x_norm[:n_features]
        else:
            return x[:self.n_qubits] / (np.linalg.norm(x[:self.n_qubits]) + 1e-10)
    
    def compare_encodings(
        self,
        X: np.ndarray,
        sample_size: int = 100
    ) -> dict:
        """
        Compare different encoding schemes
        
        Args:
            X: Dataset
            sample_size: Number of samples to use for comparison
            
        Returns:
            Dictionary with comparison results
        """
        if len(X) > sample_size:
            X_sample = X[:sample_size]
        else:
            X_sample = X
        
        results = {}
        
        # Test each encoding
        for encoding in ['angle', 'amplitude']:
            try:
                circuit = getattr(self, f'{encoding}_encoding_circuit')(X_sample[0])
                
                # Compute outputs
                outputs = []
                for x in X_sample:
                    try:
                        if encoding == 'amplitude':
                            output = circuit(x)
                        else:
                            output = circuit(x)
                        outputs.append(float(output))
                    except:
                        outputs.append(0.0)
                
                results[encoding] = {
                    'mean': float(np.mean(outputs)),
                    'std': float(np.std(outputs)),
                    'min': float(np.min(outputs)),
                    'max': float(np.max(outputs)),
                    'outputs': outputs
                }
            except Exception as e:
                results[encoding] = {'error': str(e)}
        
        return results
