import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class NeuralNetwork:
    """
    Multilayer Perceptron Neural Network implementation.
    
    Attributes:
        layers (List[int]): List of layer sizes [input, hidden1, hidden2, ..., output]
        learning_rate (float): Learning rate for gradient descent
        epochs (int): Number of training epochs
        weights (List[np.ndarray]): Network weights
        biases (List[np.ndarray]): Network biases
        loss_history (List[float]): Training loss history
    """
    
    def __init__(self, layers: List[int], learning_rate: float = 0.01, epochs: int = 1000):
        self.layers = layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.biases = []
        self.loss_history = []
        
        # Initialize weights and biases
        self._initialize_parameters()
    
    def _initialize_parameters(self) -> None:
        """Initialize weights using Xavier initialization and biases to zero."""
        for i in range(len(self.layers) - 1):
            # Xavier initialization
            weight = np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2.0 / self.layers[i])
            bias = np.zeros((1, self.layers[i + 1]))
            
            self.weights.append(weight)
            self.biases.append(bias)
    
    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid function."""
        s = NeuralNetwork.sigmoid(z)
        return s * (1 - s)
    
    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, z)
    
    @staticmethod
    def relu_derivative(z: np.ndarray) -> np.ndarray:
        """Derivative of ReLU function."""
        return (z > 0).astype(float)
    
    def forward_propagation(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Forward propagation through the network.
        
        Args:
            X (np.ndarray): Input data
            
        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: Activations and z-values
        """
        activations = [X]
        z_values = []
        
        current_input = X
        
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(current_input, weight) + bias
            z_values.append(z)
            
            # Use sigmoid for all layers (can be modified for different activations)
            activation = self.sigmoid(z)
            activations.append(activation)
            current_input = activation
        
        return activations, z_values
    
    def backward_propagation(self, X: np.ndarray, y: np.ndarray, 
                           activations: List[np.ndarray], 
                           z_values: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backward propagation to compute gradients.
        
        Args:
            X (np.ndarray): Input data
            y (np.ndarray): True labels
            activations (List[np.ndarray]): Forward pass activations
            z_values (List[np.ndarray]): Forward pass z-values
            
        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: Weight and bias gradients
        """
        m = X.shape[0]  # Number of samples
        
        # Initialize gradient lists
        weight_gradients = [np.zeros_like(w) for w in self.weights]
        bias_gradients = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error
        delta = activations[-1] - y
        
        # Backward pass
        for i in reversed(range(len(self.weights))):
            # Compute gradients
            weight_gradients[i] = np.dot(activations[i].T, delta) / m
            bias_gradients[i] = np.sum(delta, axis=0, keepdims=True) / m
            
            if i > 0:  # Not the first layer
                # Propagate error to previous layer
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(z_values[i-1])
        
        return weight_gradients, bias_gradients
    
    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Tuple = None) -> None:
        """
        Train the neural network.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training targets
            validation_data (Tuple): Optional validation data (X_val, y_val)
        """
        for epoch in range(self.epochs):
            # Forward propagation
            activations, z_values = self.forward_propagation(X)
            
            # Compute loss (binary cross-entropy)
            predictions = activations[-1]
            loss = self._compute_loss(y, predictions)
            self.loss_history.append(loss)
            
            # Backward propagation
            weight_gradients, bias_gradients = self.backward_propagation(X, y, activations, z_values)
            
            # Update parameters
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * weight_gradients[i]
                self.biases[i] -= self.learning_rate * bias_gradients[i]
            
            # Print progress
            if epoch % 100 == 0:
                accuracy = self.accuracy(X, y)
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        activations, _ = self.forward_propagation(X)
        return (activations[-1] > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return prediction probabilities."""
        activations, _ = self.forward_propagation(X)
        return activations[-1]
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy on given data."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute binary cross-entropy loss."""
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def plot_training_history(self) -> None:
        """Plot the training loss history."""
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.loss_history, 'b-', linewidth=2)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.loss_history[-100:], 'r-', linewidth=2)
        plt.title('Training Loss (Last 100 Epochs)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def create_sample_data():
    """Create sample binary classification data."""
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                             n_informative=2, n_clusters_per_class=1, random_state=42)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Reshape y for neural network
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    return X_train, X_test, y_train, y_test

def main():
    """Demonstrate the neural network implementation."""
    print("Neural Network from Scratch")
    print("=" * 30)
    
    # Create sample data
    X_train, X_test, y_train, y_test = create_sample_data()
    
    # Create neural network
    # Architecture: input_size -> 10 -> 5 -> 1
    nn = NeuralNetwork(layers=[2, 10, 5, 1], learning_rate=0.1, epochs=1000)
    
    # Train the network
    print("Training neural network...")
    nn.fit(X_train, y_train)
    
    # Evaluate the network
    train_accuracy = nn.accuracy(X_train, y_train)
    test_accuracy = nn.accuracy(X_test, y_test)
    
    print(f"\nFinal Results:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Plot training history
    nn.plot_training_history()

if __name__ == "__main__":
    main()
