import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
import graphviz

class TreeNode:
    """
    Node class for decision tree.
    
    Attributes:
        feature (int): Feature index used for splitting
        threshold (float): Threshold value for splitting
        left (TreeNode): Left child node
        right (TreeNode): Right child node
        value (Union[int, float]): Prediction value for leaf nodes
        samples (int): Number of samples in this node
        impurity (float): Impurity measure of this node
    """
    
    def __init__(self, feature: Optional[int] = None, threshold: Optional[float] = None,
                 left: Optional['TreeNode'] = None, right: Optional['TreeNode'] = None,
                 value: Optional[Union[int, float]] = None, samples: int = 0, impurity: float = 0.0):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.samples = samples
        self.impurity = impurity
    
    def is_leaf(self) -> bool:
        """Check if node is a leaf."""
        return self.value is not None

class DecisionTree:
    """
    Decision Tree Classifier implementation.
    
    Attributes:
        max_depth (int): Maximum depth of the tree
        min_samples_split (int): Minimum samples required to split
        min_samples_leaf (int): Minimum samples required in a leaf
        criterion (str): Splitting criterion ('gini' or 'entropy')
        root (TreeNode): Root node of the tree
        feature_names (List[str]): Names of features
        class_names (List[str]): Names of classes
        feature_importances_ (np.ndarray): Feature importance scores
    """
    
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, criterion: str = 'gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.root = None
        self.feature_names = None
        self.class_names = None
        self.feature_importances_ = None
        self.n_features = 0
        self.n_classes = 0
    
    def _gini_impurity(self, y: np.ndarray) -> float:
        """
        Calculate Gini impurity.
        
        Args:
            y (np.ndarray): Target values
            
        Returns:
            float: Gini impurity
        """
        if len(y) == 0:
            return 0
        
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)
    
    def _entropy(self, y: np.ndarray) -> float:
        """
        Calculate entropy.
        
        Args:
            y (np.ndarray): Target values
            
        Returns:
            float: Entropy
        """
        if len(y) == 0:
            return 0
        
        proportions = np.bincount(y) / len(y)
        proportions = proportions[proportions > 0]  # Remove zeros
        return -np.sum(proportions * np.log2(proportions))
    
    def _impurity(self, y: np.ndarray) -> float:
        """Calculate impurity based on criterion."""
        if self.criterion == 'gini':
            return self._gini_impurity(y)
        elif self.criterion == 'entropy':
            return self._entropy(y)
        else:
            raise ValueError("Criterion must be 'gini' or 'entropy'")
    
    def _information_gain(self, X: np.ndarray, y: np.ndarray, 
                         feature: int, threshold: float) -> float:
        """
        Calculate information gain for a split.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target values
            feature (int): Feature index
            threshold (float): Split threshold
            
        Returns:
            float: Information gain
        """
        parent_impurity = self._impurity(y)
        
        # Split data
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        left_y = y[left_mask]
        right_y = y[right_mask]
        
        if len(left_y) == 0 or len(right_y) == 0:
            return 0
        
        # Calculate weighted impurity of children
        n = len(y)
        left_weight = len(left_y) / n
        right_weight = len(right_y) / n
        
        child_impurity = (left_weight * self._impurity(left_y) + 
                         right_weight * self._impurity(right_y))
        
        return parent_impurity - child_impurity
    
    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float, float]:
        """
        Find the best split for the data.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target values
            
        Returns:
            Tuple[int, float, float]: Best feature, threshold, and information gain
        """
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature in range(n_features):
            # Get unique values for thresholds
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> TreeNode:
        """
        Recursively build the decision tree.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target values
            depth (int): Current depth
            
        Returns:
            TreeNode: Root node of the subtree
        """
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        # Calculate node statistics
        most_common_class = Counter(y).most_common(1)[0][0]
        impurity = self._impurity(y)
        
        # Create leaf node if stopping criteria are met
        if (depth >= self.max_depth or 
            n_classes == 1 or 
            n_samples < self.min_samples_split):
            return TreeNode(value=most_common_class, samples=n_samples, impurity=impurity)
        
        # Find best split
        best_feature, best_threshold, best_gain = self._best_split(X, y)
        
        if best_feature is None or best_gain <= 0:
            return TreeNode(value=most_common_class, samples=n_samples, impurity=impurity)
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Check minimum samples in leaf
        if (np.sum(left_mask) < self.min_samples_leaf or 
            np.sum(right_mask) < self.min_samples_leaf):
            return TreeNode(value=most_common_class, samples=n_samples, impurity=impurity)
        
        # Recursively build left and right subtrees
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return TreeNode(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            samples=n_samples,
            impurity=impurity
        )
    
    def _calculate_feature_importances(self, node: TreeNode, 
                                     total_samples: int) -> Dict[int, float]:
        """
        Calculate feature importances recursively.
        
        Args:
            node (TreeNode): Current node
            total_samples (int): Total number of samples
            
        Returns:
            Dict[int, float]: Feature importances
        """
        if node.is_leaf():
            return {}
        
        importances = {}
        
        # Calculate importance for current split
        left_samples = node.left.samples
        right_samples = node.right.samples
        
        importance = (node.samples / total_samples) * (
            node.impurity - 
            (left_samples / node.samples) * node.left.impurity -
            (right_samples / node.samples) * node.right.impurity
        )
        
        importances[node.feature] = importance
        
        # Recursively calculate for children
        left_importances = self._calculate_feature_importances(node.left, total_samples)
        right_importances = self._calculate_feature_importances(node.right, total_samples)
        
        # Merge importances
        for feature, imp in left_importances.items():
            importances[feature] = importances.get(feature, 0) + imp
        
        for feature, imp in right_importances.items():
            importances[feature] = importances.get(feature, 0) + imp
        
        return importances
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTree':
        """
        Train the decision tree.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training targets
            
        Returns:
            DecisionTree: Self for method chaining
        """
        self.n_features = X.shape[1]
        self.n_classes = len(np.unique(y))
        
        # Build the tree
        self.root = self._build_tree(X, y)
        
        # Calculate feature importances
        importances_dict = self._calculate_feature_importances(self.root, len(y))
        self.feature_importances_ = np.zeros(self.n_features)
        
        for feature, importance in importances_dict.items():
            self.feature_importances_[feature] = importance
        
        # Normalize feature importances
        if self.feature_importances_.sum() > 0:
            self.feature_importances_ = self.feature_importances_ / self.feature_importances_.sum()
        
        return self
    
    def _predict_sample(self, x: np.ndarray, node: TreeNode) -> int:
        """
        Predict a single sample.
        
        Args:
            x (np.ndarray): Single sample
            node (TreeNode): Current node
            
        Returns:
            int: Predicted class
        """
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predictions
        """
        if self.root is None:
            raise ValueError("Tree must be trained before making predictions")
        
        predictions = []
        for x in X:
            pred = self._predict_sample(x, self.root)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score.
        
        Args:
            X (np.ndarray): Test features
            y (np.ndarray): True labels
            
        Returns:
            float: Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def _get_tree_depth(self, node: TreeNode) -> int:
        """Get the depth of the tree."""
        if node.is_leaf():
            return 1
        
        left_depth = self._get_tree_depth(node.left)
        right_depth = self._get_tree_depth(node.right)
        
        return 1 + max(left_depth, right_depth)
    
    def _count_nodes(self, node: TreeNode) -> int:
        """Count total number of nodes in the tree."""
        if node.is_leaf():
            return 1
        
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)
    
    def get_tree_info(self) -> Dict[str, int]:
        """Get information about the tree structure."""
        if self.root is None:
            return {"depth": 0, "nodes": 0, "leaves": 0}
        
        depth = self._get_tree_depth(self.root)
        total_nodes = self._count_nodes(self.root)
        leaves = self._count_leaves(self.root)
        
        return {
            "depth": depth,
            "nodes": total_nodes,
            "leaves": leaves
        }
    
    def _count_leaves(self, node: TreeNode) -> int:
        """Count number of leaf nodes."""
        if node.is_leaf():
            return 1
        
        return self._count_leaves(node.left) + self._count_leaves(node.right)
    
    def plot_feature_importances(self) -> None:
        """Plot feature importances."""
        if self.feature_importances_ is None:
            print("Model must be trained first")
            return
        
        feature_names = (self.feature_names if self.feature_names is not None 
                        else [f'Feature {i}' for i in range(len(self.feature_importances_))])
        
        plt.figure(figsize=(10, 6))
        indices = np.argsort(self.feature_importances_)[::-1]
        
        plt.bar(range(len(self.feature_importances_)), 
                self.feature_importances_[indices])
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Feature Importances')
        plt.xticks(range(len(self.feature_importances_)), 
                  [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()
    
    def visualize_tree(self, max_depth_show: int = 3) -> str:
        """
        Create a text representation of the tree.
        
        Args:
            max_depth_show (int): Maximum depth to show
            
        Returns:
            str: Text representation of the tree
        """
        if self.root is None:
            return "Tree not built yet"
        
        def _build_tree_string(node: TreeNode, depth: int = 0, prefix: str = "") -> str:
            if depth > max_depth_show:
                return prefix + "...\n"
            
            if node.is_leaf():
                class_name = (self.class_names[node.value] if self.class_names 
                             else f"Class {node.value}")
                return f"{prefix}└─ {class_name} (samples: {node.samples})\n"
            
            feature_name = (self.feature_names[node.feature] if self.feature_names 
                           else f"Feature {node.feature}")
            
            result = f"{prefix}├─ {feature_name} <= {node.threshold:.3f} (samples: {node.samples})\n"
            
            # Left child (True branch)
            result += _build_tree_string(node.left, depth + 1, prefix + "│  ")
            
            # Right child (False branch)
            result += f"{prefix}├─ {feature_name} > {node.threshold:.3f} (samples: {node.right.samples})\n"
            result += _build_tree_string(node.right, depth + 1, prefix + "   ")
            
            return result
        
        return _build_tree_string(self.root)

def create_sample_data():
    """Create sample datasets for demonstration."""
    # Dataset 1: Iris dataset
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    
    # Dataset 2: Synthetic dataset
    X_synthetic, y_synthetic = make_classification(
        n_samples=1000, n_features=4, n_informative=3, n_redundant=1,
        n_clusters_per_class=1, random_state=42
    )
    
    return [(X_iris, y_iris, iris.feature_names, iris.target_names),
            (X_synthetic, y_synthetic, 
             [f'Feature_{i}' for i in range(X_synthetic.shape[1])], 
             [f'Class_{i}' for i in range(len(np.unique(y_synthetic)))])]

def compare_criteria(X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
    """Compare different splitting criteria."""
    criteria = ['gini', 'entropy']
    results = {}
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    plt.figure(figsize=(15, 5))
    
    for i, criterion in enumerate(criteria, 1):
        # Train tree
        tree = DecisionTree(max_depth=5, criterion=criterion)
        tree.feature_names = feature_names
        tree.fit(X_train, y_train)
        
        # Evaluate
        train_score = tree.score(X_train, y_train)
        test_score = tree.score(X_test, y_test)
        tree_info = tree.get_tree_info()
        
        results[criterion] = {
            'train_score': train_score,
            'test_score': test_score,
            'tree_info': tree_info
        }
        
        # Plot feature importances
        plt.subplot(1, 3, i)
        if tree.feature_importances_ is not None:
            feature_names_plot = (feature_names if feature_names 
                                else [f'Feature {j}' for j in range(len(tree.feature_importances_))])
            indices = np.argsort(tree.feature_importances_)[::-1]
            
            plt.bar(range(len(tree.feature_importances_)), 
                    tree.feature_importances_[indices])
            plt.title(f'{criterion.title()} Criterion\nTest Acc: {test_score:.3f}')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.xticks(range(len(tree.feature_importances_)), 
                      [feature_names_plot[j] for j in indices], rotation=45)
    
    # Comparison plot
    plt.subplot(1, 3, 3)
    criteria_names = list(results.keys())
    train_scores = [results[c]['train_score'] for c in criteria_names]
    test_scores = [results[c]['test_score'] for c in criteria_names]
    
    x_pos = np.arange(len(criteria_names))
    width = 0.35
    
    plt.bar(x_pos - width/2, train_scores, width, label='Train', alpha=0.8)
    plt.bar(x_pos + width/2, test_scores, width, label='Test', alpha=0.8)
    
    plt.xlabel('Criterion')
    plt.ylabel('Accuracy')
    plt.title('Criterion Comparison')
    plt.xticks(x_pos, criteria_names)
    plt.legend()
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    return results

def study_overfitting(X: np.ndarray, y: np.ndarray):
    """Study overfitting with different max_depth values."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    max_depths = range(1, 16)
    train_scores = []
    test_scores = []
    
    for depth in max_depths:
        tree = DecisionTree(max_depth=depth, criterion='gini')
        tree.fit(X_train, y_train)
        
        train_score = tree.score(X_train, y_train)
        test_score = tree.score(X_test, y_test)
        
        train_scores.append(train_score)
        test_scores.append(test_score)
    
    plt.figure(figsize=(10, 6))
    plt.plot(max_depths, train_scores, 'o-', label='Training Accuracy', linewidth=2)
    plt.plot(max_depths, test_scores, 's-', label='Test Accuracy', linewidth=2)
    
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Training vs Test Accuracy by Tree Depth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Find optimal depth
    optimal_depth = max_depths[np.argmax(test_scores)]
    print(f"Optimal depth based on test accuracy: {optimal_depth}")
    
    return optimal_depth

def main():
    """Demonstrate decision tree implementation."""
    print("Decision Tree Implementation Demo")
    print("=" * 40)
    
    # Create sample data
    datasets = create_sample_data()
    
    for i, (X, y, feature_names, class_names) in enumerate(datasets, 1):
        dataset_name = "Iris Dataset" if i == 1 else "Synthetic Dataset"
        print(f"\n{dataset_name}")
        print(f"Data shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train decision tree
        tree = DecisionTree(max_depth=5, min_samples_split=5, criterion='gini')
        tree.feature_names = feature_names
        tree.class_names = class_names
        tree.fit(X_train, y_train)
        
        # Evaluate
        train_accuracy = tree.score(X_train, y_train)
        test_accuracy = tree.score(X_test, y_test)
        tree_info = tree.get_tree_info()
        
        print(f"Training Accuracy: {train_accuracy:.3f}")
        print(f"Test Accuracy: {test_accuracy:.3f}")
        print(f"Tree Depth: {tree_info['depth']}")
        print(f"Number of Nodes: {tree_info['nodes']}")
        print(f"Number of Leaves: {tree_info['leaves']}")
        
        # Show tree structure (first few levels)
        print("\nTree Structure (first 3 levels):")
        print(tree.visualize_tree(max_depth_show=3))
        
        # Plot feature importances
        tree.plot_feature_importances()
        
        # Compare criteria
        print("Comparing splitting criteria...")
        criteria_results = compare_criteria(X, y, feature_names)
        
        for criterion, results in criteria_results.items():
            print(f"{criterion.title()}: Train={results['train_score']:.3f}, "
                  f"Test={results['test_score']:.3f}, "
                  f"Nodes={results['tree_info']['nodes']}")
        
        # Study overfitting
        print("Studying overfitting effects...")
        optimal_depth = study_overfitting(X, y)
        
        # Train final model with optimal depth
        print(f"\nTraining final model with optimal depth ({optimal_depth})...")
        final_tree = DecisionTree(max_depth=optimal_depth, criterion='gini')
        final_tree.feature_names = feature_names
        final_tree.class_names = class_names
        final_tree.fit(X_train, y_train)
        
        final_train_acc = final_tree.score(X_train, y_train)
        final_test_acc = final_tree.score(X_test, y_test)
        
        print(f"Final Model - Train: {final_train_acc:.3f}, Test: {final_test_acc:.3f}")
        
        print("-" * 40)

if __name__ == "__main__":
    main()