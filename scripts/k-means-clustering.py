import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from sklearn.datasets import make_blobs
import random

class KMeans:
    """
    K-Means clustering algorithm implementation.
    
    Attributes:
        k (int): Number of clusters
        max_iters (int): Maximum number of iterations
        tolerance (float): Convergence tolerance
        centroids (np.ndarray): Cluster centroids
        labels (np.ndarray): Data point cluster assignments
        inertia (float): Within-cluster sum of squares
        history (List): History of centroids for visualization
    """
    
    def __init__(self, k: int = 3, max_iters: int = 100, tolerance: float = 1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.centroids = None
        self.labels = None
        self.inertia = None
        self.history = []
    
    def _initialize_centroids(self, X: np.ndarray, method: str = 'random') -> np.ndarray:
        """
        Initialize cluster centroids.
        
        Args:
            X (np.ndarray): Input data
            method (str): Initialization method ('random' or 'kmeans++')
            
        Returns:
            np.ndarray: Initial centroids
        """
        n_samples, n_features = X.shape
        
        if method == 'random':
            # Random initialization
            centroids = np.random.uniform(X.min(axis=0), X.max(axis=0), (self.k, n_features))
        
        elif method == 'kmeans++':
            # K-means++ initialization
            centroids = np.zeros((self.k, n_features))
            
            # Choose first centroid randomly
            centroids[0] = X[random.randint(0, n_samples - 1)]
            
            # Choose remaining centroids
            for i in range(1, self.k):
                distances = np.array([min([np.linalg.norm(x - c)**2 for c in centroids[:i]]) for x in X])
                probabilities = distances / distances.sum()
                cumulative_probabilities = probabilities.cumsum()
                r = random.random()
                
                for j, p in enumerate(cumulative_probabilities):
                    if r < p:
                        centroids[i] = X[j]
                        break
        
        else:
            raise ValueError("Method must be 'random' or 'kmeans++'")
        
        return centroids
    
    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """
        Assign each data point to the nearest centroid.
        
        Args:
            X (np.ndarray): Input data
            
        Returns:
            np.ndarray: Cluster assignments
        """
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Update centroids based on current cluster assignments.
        
        Args:
            X (np.ndarray): Input data
            labels (np.ndarray): Current cluster assignments
            
        Returns:
            np.ndarray: Updated centroids
        """
        new_centroids = np.zeros((self.k, X.shape[1]))
        
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                # Keep old centroid if no points assigned
                new_centroids[i] = self.centroids[i]
        
        return new_centroids
    
    def _calculate_inertia(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate within-cluster sum of squares (inertia).
        
        Args:
            X (np.ndarray): Input data
            labels (np.ndarray): Cluster assignments
            
        Returns:
            float: Inertia value
        """
        inertia = 0
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[i])**2)
        return inertia
    
    def fit(self, X: np.ndarray, init_method: str = 'kmeans++') -> 'KMeans':
        """
        Fit K-means clustering to data.
        
        Args:
            X (np.ndarray): Input data
            init_method (str): Centroid initialization method
            
        Returns:
            KMeans: Self for method chaining
        """
        # Initialize centroids
        self.centroids = self._initialize_centroids(X, init_method)
        self.history = [self.centroids.copy()]
        
        for iteration in range(self.max_iters):
            # Assign points to clusters
            new_labels = self._assign_clusters(X)
            
            # Update centroids
            new_centroids = self._update_centroids(X, new_labels)
            
            # Check for convergence
            centroid_shift = np.linalg.norm(new_centroids - self.centroids)
            
            self.centroids = new_centroids
            self.labels = new_labels
            self.history.append(self.centroids.copy())
            
            if centroid_shift < self.tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        # Calculate final inertia
        self.inertia = self._calculate_inertia(X, self.labels)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster assignments for new data.
        
        Args:
            X (np.ndarray): New data points
            
        Returns:
            np.ndarray: Cluster assignments
        """
        if self.centroids is None:
            raise ValueError("Model must be fitted before making predictions")
        
        return self._assign_clusters(X)
    
    def fit_predict(self, X: np.ndarray, init_method: str = 'kmeans++') -> np.ndarray:
        """
        Fit the model and return cluster assignments.
        
        Args:
            X (np.ndarray): Input data
            init_method (str): Centroid initialization method
            
        Returns:
            np.ndarray: Cluster assignments
        """
        self.fit(X, init_method)
        return self.labels
    
    def plot_clusters(self, X: np.ndarray, title: str = "K-Means Clustering Results") -> None:
        """
        Plot clustering results (works for 2D data).
        
        Args:
            X (np.ndarray): Input data
            title (str): Plot title
        """
        if X.shape[1] != 2:
            print("Plotting only supported for 2D data")
            return
        
        plt.figure(figsize=(12, 4))
        
        # Plot 1: Final clustering result
        plt.subplot(1, 2, 1)
        colors = plt.cm.Set1(np.linspace(0, 1, self.k))
        
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=[colors[i]], alpha=0.6, s=50, label=f'Cluster {i+1}')
        
        # Plot centroids
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='Centroids')
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Convergence history
        plt.subplot(1, 2, 2)
        
        # Plot centroid movement
        for i in range(self.k):
            centroid_path = np.array([hist[i] for hist in self.history])
            plt.plot(centroid_path[:, 0], centroid_path[:, 1], 
                    'o-', color=colors[i], alpha=0.7, label=f'Centroid {i+1}')
        
        # Plot data points lightly
        plt.scatter(X[:, 0], X[:, 1], c='lightgray', alpha=0.3, s=20)
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Centroid Movement During Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def elbow_method(X: np.ndarray, max_k: int = 10) -> None:
    """
    Apply elbow method to find optimal number of clusters.
    
    Args:
        X (np.ndarray): Input data
        max_k (int): Maximum number of clusters to try
    """
    inertias = []
    k_values = range(1, max_k + 1)
    
    for k in k_values:
        kmeans = KMeans(k=k, max_iters=100)
        kmeans.fit(X)
        inertias.append(kmeans.inertia)
    
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Within-cluster sum of squares)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True, alpha=0.3)
    
    # Highlight the "elbow" point
    if len(inertias) >= 3:
        # Simple elbow detection using second derivative
        second_derivatives = []
        for i in range(1, len(inertias) - 1):
            second_deriv = inertias[i-1] - 2*inertias[i] + inertias[i+1]
            second_derivatives.append(second_deriv)
        
        elbow_idx = np.argmax(second_derivatives) + 1
        plt.axvline(x=k_values[elbow_idx], color='red', linestyle='--', 
                   alpha=0.7, label=f'Suggested k = {k_values[elbow_idx]}')
        plt.legend()
    
    plt.show()

def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate silhouette score for clustering evaluation.
    
    Args:
        X (np.ndarray): Input data
        labels (np.ndarray): Cluster assignments
        
    Returns:
        float: Silhouette score
    """
    n_samples = X.shape[0]
    silhouette_scores = []
    
    for i in range(n_samples):
        # Same cluster distances (a)
        same_cluster = X[labels == labels[i]]
        if len(same_cluster) > 1:
            a = np.mean([np.linalg.norm(X[i] - point) for point in same_cluster if not np.array_equal(X[i], point)])
        else:
            a = 0
        
        # Different cluster distances (b)
        b = float('inf')
        for cluster_id in np.unique(labels):
            if cluster_id != labels[i]:
                other_cluster = X[labels == cluster_id]
                if len(other_cluster) > 0:
                    avg_dist = np.mean([np.linalg.norm(X[i] - point) for point in other_cluster])
                    b = min(b, avg_dist)
        
        # Silhouette score for point i
        if max(a, b) > 0:
            silhouette_scores.append((b - a) / max(a, b))
        else:
            silhouette_scores.append(0)
    
    return np.mean(silhouette_scores)

def create_sample_datasets() -> List[Tuple[np.ndarray, str]]:
    """Create various sample datasets for demonstration."""
    datasets = []
    
    # Dataset 1: Well-separated blobs
    X1, _ = make_blobs(n_samples=300, centers=4, n_features=2, 
                       random_state=42, cluster_std=1.5)
    datasets.append((X1, "Well-separated Blobs"))
    
    # Dataset 2: Close blobs
    X2, _ = make_blobs(n_samples=300, centers=3, n_features=2, 
                       random_state=42, cluster_std=2.5)
    datasets.append((X2, "Close Blobs"))
    
    # Dataset 3: Different sized blobs
    X3_1, _ = make_blobs(n_samples=100, centers=1, n_features=2, 
                         random_state=42, cluster_std=1.0)
    X3_2, _ = make_blobs(n_samples=200, centers=1, n_features=2, 
                         random_state=43, cluster_std=2.0)
    X3_2 += 10  # Shift second blob
    X3 = np.vstack([X3_1, X3_2])
    datasets.append((X3, "Different Sized Clusters"))
    
    return datasets

def main():
    """Demonstrate K-means clustering with various examples."""
    print("K-Means Clustering Demonstrations")
    print("=" * 40)
    
    # Create sample datasets
    datasets = create_sample_datasets()
    
    for i, (X, name) in enumerate(datasets, 1):
        print(f"\nDataset {i}: {name}")
        print(f"Data shape: {X.shape}")
        
        # Apply elbow method
        print("Finding optimal k using elbow method...")
        elbow_method(X, max_k=8)
        
        # Fit K-means with k=3
        print("Fitting K-means with k=3...")
        kmeans = KMeans(k=3, max_iters=100)
        labels = kmeans.fit_predict(X)
        
        # Calculate metrics
        silhouette = silhouette_score(X, labels)
        
        print(f"Inertia: {kmeans.inertia:.2f}")
        print(f"Silhouette Score: {silhouette:.3f}")
        
        # Plot results
        kmeans.plot_clusters(X, f"K-Means Results: {name}")
    
    # Demonstrate different initialization methods
    print("\nComparing Initialization Methods:")
    X, _ = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)
    
    methods = ['random', 'kmeans++']
    plt.figure(figsize=(12, 5))
    
    for i, method in enumerate(methods, 1):
        kmeans = KMeans(k=4, max_iters=100)
        labels = kmeans.fit_predict(X, init_method=method)
        
        plt.subplot(1, 2, i)
        colors = plt.cm.Set1(np.linspace(0, 1, 4))
        
        for j in range(4):
            cluster_points = X[labels == j]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=[colors[j]], alpha=0.6, s=50)
        
        plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
                   c='red', marker='x', s=200, linewidths=3)
        
        plt.title(f'{method.title()} Initialization\nInertia: {kmeans.inertia:.2f}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()