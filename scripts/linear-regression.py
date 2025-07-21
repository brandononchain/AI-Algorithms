import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

class LinearRegression:
    """
    Linear Regression implementation using gradient descent.
    
    Attributes:
        learning_rate (float): Step size for gradient descent
        n_iterations (int): Number of training iterations
        weights (np.ndarray): Model parameters [bias, weight]
        cost_history (list): History of cost function values
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.cost_history = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the linear regression model.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training targets
        """
        # Add bias term (intercept)
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        
        # Initialize weights randomly
        self.weights = np.random.normal(0, 0.01, X_with_bias.shape[1])
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            y_pred = X_with_bias.dot(self.weights)
            
            # Calculate cost (Mean Squared Error)
            cost = np.mean((y_pred - y) ** 2)
            self.cost_history.append(cost)
            
            # Calculate gradients
            gradients = (2/len(y)) * X_with_bias.T.dot(y_pred - y)
            
            # Update weights
            self.weights -= self.learning_rate * gradients
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predictions
        """
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        return X_with_bias.dot(self.weights)
    
    def plot_results(self, X: np.ndarray, y: np.ndarray) -> None:
        """Plot the regression line and training data."""
        plt.figure(figsize=(12, 4))
        
        # Plot 1: Data and regression line
        plt.subplot(1, 2, 1)
        plt.scatter(X, y, alpha=0.6, color='blue', label='Data points')
        
        X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred_range = self.predict(X_range)
        plt.plot(X_range, y_pred_range, color='red', linewidth=2, label='Regression line')
        
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Linear Regression Results')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Cost function over iterations
        plt.subplot(1, 2, 2)
        plt.plot(self.cost_history, color='green', linewidth=2)
        plt.xlabel('Iterations')
        plt.ylabel('Cost (MSE)')
        plt.title('Training Progress')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def generate_sample_data(n_samples: int = 100, noise: float = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sample data for demonstration."""
    np.random.seed(42)
    X = np.random.uniform(-50, 50, n_samples).reshape(-1, 1)
    y = 2 * X.flatten() + 5 + np.random.normal(0, noise, n_samples)
    return X, y

def main():
    """Demonstrate the linear regression implementation."""
    print("Linear Regression from Scratch")
    print("=" * 30)
    
    # Generate sample data
    X, y = generate_sample_data(n_samples=100, noise=10)
    
    # Create and train model
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Calculate metrics
    mse = np.mean((predictions - y) ** 2)
    r2 = 1 - (np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2))
    
    print(f"Final weights: {model.weights}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.3f}")
    
    # Plot results
    model.plot_results(X, y)

if __name__ == "__main__":
    main()
