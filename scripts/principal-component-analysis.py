import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
from sklearn.datasets import load_iris, make_classification, load_wine
from sklearn.preprocessing import StandardScaler
import seaborn as sns

class PCA:
    """
    Principal Component Analysis implementation.
    
    Attributes:
        n_components (int): Number of principal components to keep
        components_ (np.ndarray): Principal components (eigenvectors)
        explained_variance_ (np.ndarray): Explained variance for each component
        explained_variance_ratio_ (np.ndarray): Proportion of variance explained
        singular_values_ (np.ndarray): Singular values corresponding to components
        mean_ (np.ndarray): Mean of the training data
        n_features_ (int): Number of features in the training data
        n_samples_ (int): Number of samples in the training data
    """
    
    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.n_features_ = None
        self.n_samples_ = None
    
    def fit(self, X: np.ndarray) -> 'PCA':
        """
        Fit PCA to the data.
        
        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features)
            
        Returns:
            PCA: Self for method chaining
        """
        self.n_samples_, self.n_features_ = X.shape
        
        # Set default number of components
        if self.n_components is None:
            self.n_components = min(self.n_samples_, self.n_features_)
        
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store the first n_components
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        
        # Calculate explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        # Calculate singular values
        self.singular_values_ = np.sqrt(self.explained_variance_ * (self.n_samples_ - 1))
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to principal component space.
        
        Args:
            X (np.ndarray): Data to transform
            
        Returns:
            np.ndarray: Transformed data
        """
        if self.components_ is None:
            raise ValueError("PCA must be fitted before transforming data")
        
        # Center the data using training mean
        X_centered = X - self.mean_
        
        # Project onto principal components
        return X_centered.dot(self.components_.T)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA and transform the data.
        
        Args:
            X (np.ndarray): Training data
            
        Returns:
            np.ndarray: Transformed data
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Transform data back to original space.
        
        Args:
            X_transformed (np.ndarray): Data in principal component space
            
        Returns:
            np.ndarray: Data in original space
        """
        if self.components_ is None:
            raise ValueError("PCA must be fitted before inverse transforming")
        
        # Project back to original space
        X_reconstructed = X_transformed.dot(self.components_)
        
        # Add back the mean
        return X_reconstructed + self.mean_
    
    def explained_variance_cumsum(self) -> np.ndarray:
        """Get cumulative explained variance ratio."""
        if self.explained_variance_ratio_ is None:
            raise ValueError("PCA must be fitted first")
        
        return np.cumsum(self.explained_variance_ratio_)
    
    def plot_explained_variance(self, max_components: Optional[int] = None) -> None:
        """
        Plot explained variance and cumulative explained variance.
        
        Args:
            max_components (Optional[int]): Maximum number of components to show
        """
        if self.explained_variance_ratio_ is None:
            raise ValueError("PCA must be fitted first")
        
        if max_components is None:
            max_components = len(self.explained_variance_ratio_)
        
        components_range = range(1, min(max_components + 1, len(self.explained_variance_ratio_) + 1))
        variance_ratios = self.explained_variance_ratio_[:max_components]
        cumulative_variance = self.explained_variance_cumsum()[:max_components]
        
        plt.figure(figsize=(12, 5))
        
        # Individual explained variance
        plt.subplot(1, 2, 1)
        plt.bar(components_range, variance_ratios, alpha=0.7, color='skyblue')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance by Component')
        plt.grid(True, alpha=0.3)
        
        # Cumulative explained variance
        plt.subplot(1, 2, 2)
        plt.plot(components_range, cumulative_variance, 'ro-', linewidth=2, markersize=6)
        plt.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% threshold')
        plt.axhline(y=0.90, color='orange', linestyle='--', alpha=0.7, label='90% threshold')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Cumulative Explained Variance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print("Explained Variance Summary:")
        print("-" * 40)
        for i, (var_ratio, cum_var) in enumerate(zip(variance_ratios, cumulative_variance)):
            print(f"PC{i+1:2}: {var_ratio:.4f} ({var_ratio*100:.2f}%) | "
                  f"Cumulative: {cum_var:.4f} ({cum_var*100:.2f}%)")
    
    def plot_components_2d(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                          feature_names: Optional[List[str]] = None,
                          class_names: Optional[List[str]] = None) -> None:
        """
        Plot data in 2D principal component space.
        
        Args:
            X (np.ndarray): Original data
            y (Optional[np.ndarray]): Target labels for coloring
            feature_names (Optional[List[str]]): Names of original features
            class_names (Optional[List[str]]): Names of classes
        """
        if self.n_components < 2:
            raise ValueError("Need at least 2 components for 2D visualization")
        
        # Transform data
        X_transformed = self.transform(X)
        
        plt.figure(figsize=(15, 5))
        
        # Plot 1: 2D scatter plot
        plt.subplot(1, 3, 1)
        if y is not None:
            unique_classes = np.unique(y)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))
            
            for i, class_val in enumerate(unique_classes):
                mask = y == class_val
                label = class_names[i] if class_names else f'Class {class_val}'
                plt.scatter(X_transformed[mask, 0], X_transformed[mask, 1], 
                           c=[colors[i]], alpha=0.7, s=50, label=label)
            plt.legend()
        else:
            plt.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.7, s=50)
        
        plt.xlabel(f'PC1 ({self.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({self.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('Data in Principal Component Space')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Biplot (if we have feature names)
        plt.subplot(1, 3, 2)
        if feature_names and len(feature_names) == self.n_features_:
            # Plot data points
            if y is not None:
                for i, class_val in enumerate(unique_classes):
                    mask = y == class_val
                    label = class_names[i] if class_names else f'Class {class_val}'
                    plt.scatter(X_transformed[mask, 0], X_transformed[mask, 1], 
                               c=[colors[i]], alpha=0.5, s=30, label=label)
            else:
                plt.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.5, s=30)
            
            # Plot feature vectors
            scale_factor = 3.0
            for i, feature in enumerate(feature_names):
                plt.arrow(0, 0, 
                         self.components_[0, i] * scale_factor,
                         self.components_[1, i] * scale_factor,
                         head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.8)
                plt.text(self.components_[0, i] * scale_factor * 1.1,
                        self.components_[1, i] * scale_factor * 1.1,
                        feature, fontsize=10, ha='center', va='center')
            
            plt.xlabel(f'PC1 ({self.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({self.explained_variance_ratio_[1]:.2%} variance)')
            plt.title('PCA Biplot')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Feature names\nnot provided\nfor biplot', 
                    ha='center', va='center', transform=plt.gca().transAxes,
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            plt.title('Biplot (Feature names needed)')
        
        # Plot 3: Component loadings heatmap
        plt.subplot(1, 3, 3)
        if feature_names:
            # Show loadings for first few components
            n_comp_show = min(5, self.n_components)
            loadings = self.components_[:n_comp_show, :]
            
            im = plt.imshow(loadings, cmap='RdBu_r', aspect='auto')
            plt.colorbar(im, shrink=0.8)
            plt.xlabel('Features')
            plt.ylabel('Principal Components')
            plt.title('Component Loadings')
            
            # Set tick labels
            plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
            plt.yticks(range(n_comp_show), [f'PC{i+1}' for i in range(n_comp_show)])
        else:
            plt.text(0.5, 0.5, 'Feature names\nnot provided\nfor loadings', 
                    ha='center', va='center', transform=plt.gca().transAxes,
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            plt.title('Component Loadings')
        
        plt.tight_layout()
        plt.show()
    
    def reconstruction_error(self, X: np.ndarray) -> float:
        """
        Calculate reconstruction error.
        
        Args:
            X (np.ndarray): Original data
            
        Returns:
            float: Mean squared reconstruction error
        """
        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)
        
        return np.mean((X - X_reconstructed) ** 2)

def compare_n_components(X: np.ndarray, y: np.ndarray, max_components: int = 10) -> None:
    """
    Compare reconstruction error for different numbers of components.
    
    Args:
        X (np.ndarray): Data
        y (np.ndarray): Labels
        max_components (int): Maximum number of components to test
    """
    n_components_range = range(1, min(max_components + 1, X.shape[1] + 1))
    reconstruction_errors = []
    explained_variances = []
    
    for n_comp in n_components_range:
        pca = PCA(n_components=n_comp)
        pca.fit(X)
        
        error = pca.reconstruction_error(X)
        explained_var = pca.explained_variance_cumsum()[-1]
        
        reconstruction_errors.append(error)
        explained_variances.append(explained_var)
    
    plt.figure(figsize=(12, 5))
    
    # Plot reconstruction error
    plt.subplot(1, 2, 1)
    plt.plot(n_components_range, reconstruction_errors, 'o-', linewidth=2, markersize=6)
    plt.xlabel('Number of Components')
    plt.ylabel('Reconstruction Error (MSE)')
    plt.title('Reconstruction Error vs Components')
    plt.grid(True, alpha=0.3)
    
    # Plot explained variance
    plt.subplot(1, 2, 2)
    plt.plot(n_components_range, explained_variances, 's-', linewidth=2, markersize=6, color='orange')
    plt.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% threshold')
    plt.axhline(y=0.90, color='green', linestyle='--', alpha=0.7, label='90% threshold')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs Components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Find optimal number of components
    optimal_95 = next((i for i, var in enumerate(explained_variances) if var >= 0.95), len(explained_variances))
    optimal_90 = next((i for i, var in enumerate(explained_variances) if var >= 0.90), len(explained_variances))
    
    print(f"Components needed for 90% variance: {optimal_90 + 1}")
    print(f"Components needed for 95% variance: {optimal_95 + 1}")

def demonstrate_dimensionality_reduction(X: np.ndarray, y: np.ndarray, 
                                       feature_names: List[str], class_names: List[str]) -> None:
    """Demonstrate dimensionality reduction capabilities."""
    print(f"Original data shape: {X.shape}")
    
    # Fit PCA with different numbers of components
    target_variances = [0.90, 0.95, 0.99]
    
    for target_var in target_variances:
        # Find number of components needed
        pca_full = PCA()
        pca_full.fit(X)
        
        cumsum = pca_full.explained_variance_cumsum()
        n_comp = np.argmax(cumsum >= target_var) + 1
        
        # Fit PCA with selected components
        pca = PCA(n_components=n_comp)
        X_reduced = pca.fit_transform(X)
        
        # Calculate compression ratio
        original_size = X.shape[0] * X.shape[1]
        reduced_size = X_reduced.shape[0] * X_reduced.shape[1]
        compression_ratio = reduced_size / original_size
        
        print(f"\nTarget {target_var:.0%} variance:")
        print(f"  Components needed: {n_comp}/{X.shape[1]}")
        print(f"  Actual variance explained: {cumsum[n_comp-1]:.4f}")
        print(f"  Reduced shape: {X_reduced.shape}")
        print(f"  Compression ratio: {compression_ratio:.3f}")
        print(f"  Space saving: {(1-compression_ratio)*100:.1f}%")

def create_sample_datasets():
    """Create sample datasets for demonstration."""
    datasets = []
    
    # Dataset 1: Iris
    iris = load_iris()
    datasets.append((
        iris.data, iris.target, 
        iris.feature_names, iris.target_names,
        "Iris Dataset"
    ))
    
    # Dataset 2: Wine
    wine = load_wine()
    datasets.append((
        wine.data, wine.target,
        wine.feature_names, wine.target_names,
        "Wine Dataset"
    ))
    
    # Dataset 3: Synthetic high-dimensional data
    X_synthetic, y_synthetic = make_classification(
        n_samples=500, n_features=20, n_informative=10, n_redundant=5,
        n_clusters_per_class=1, random_state=42
    )
    
    feature_names_synth = [f'Feature_{i+1}' for i in range(X_synthetic.shape[1])]
    class_names_synth = ['Class_A', 'Class_B']
    
    datasets.append((
        X_synthetic, y_synthetic,
        feature_names_synth, class_names_synth,
        "Synthetic High-Dimensional Dataset"
    ))
    
    return datasets

def main():
    """Demonstrate PCA implementation."""
    print("Principal Component Analysis (PCA) Demonstration")
    print("=" * 60)
    
    # Create datasets
    datasets = create_sample_datasets()
    
    for i, (X, y, feature_names, class_names, name) in enumerate(datasets, 1):
        print(f"\nDataset {i}: {name}")
        print(f"Shape: {X.shape}")
        print(f"Features: {len(feature_names)}")
        print(f"Classes: {len(class_names)}")
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit PCA
        pca = PCA()
        X_transformed = pca.fit_transform(X_scaled)
        
        # Plot explained variance
        pca.plot_explained_variance(max_components=min(10, X.shape[1]))
        
        # 2D visualization
        pca_2d = PCA(n_components=2)
        pca_2d.fit(X_scaled)
        pca_2d.plot_components_2d(X_scaled, y, feature_names, class_names)
        
        # Compare different numbers of components
        print("Analyzing optimal number of components...")
        compare_n_components(X_scaled, y, max_components=min(15, X.shape[1]))
        
        # Demonstrate dimensionality reduction
        demonstrate_dimensionality_reduction(X_scaled, y, feature_names, class_names)
        
        print("-" * 60)
    
    print("\nPCA Demonstration Complete!")

if __name__ == "__main__":
    main()