from collections import Counter
import numpy as np
from typing import Union, List, Optional, Callable
from abc import ABC, abstractmethod
import warnings


class DistanceMetric(ABC):
    """Abstract base class for distance metrics"""
    
    @abstractmethod
    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        pass


class EuclideanDistance(DistanceMetric):
    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.sqrt(np.sum((x1 - x2) ** 2))


class ManhattanDistance(DistanceMetric):
    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.sum(np.abs(x1 - x2))


class MinkowskiDistance(DistanceMetric):
    def __init__(self, p: float = 2):
        self.p = p
    
    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.sum(np.abs(x1 - x2) ** self.p) ** (1 / self.p)


class CosineDistance(DistanceMetric):
    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        dot_product = np.dot(x1, x2)
        norm_x1 = np.linalg.norm(x1)
        norm_x2 = np.linalg.norm(x2)
        return 1 - (dot_product / (norm_x1 * norm_x2))


class WeightStrategy(ABC):
    """Abstract base class for weighting strategies"""
    
    @abstractmethod
    def calculate_weights(self, distances: np.ndarray) -> np.ndarray:
        pass


class UniformWeights(WeightStrategy):
    def calculate_weights(self, distances: np.ndarray) -> np.ndarray:
        return np.ones_like(distances)


class InverseDistanceWeights(WeightStrategy):
    def calculate_weights(self, distances: np.ndarray) -> np.ndarray:
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        return 1 / (distances + epsilon)


class ExponentialWeights(WeightStrategy):
    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma
    
    def calculate_weights(self, distances: np.ndarray) -> np.ndarray:
        return np.exp(-self.gamma * distances)


class KNN:
    """
    Enhanced K-Nearest Neighbors classifier with multiple improvements:
    - Multiple distance metrics
    - Weighting strategies
    - Regression support
    - KD-tree for efficient search
    - Cross-validation for hyperparameter tuning
    - Feature scaling
    - Parallel processing
    """
    
    DISTANCE_METRICS = {
        'euclidean': EuclideanDistance,
        'manhattan': ManhattanDistance,
        'minkowski': MinkowskiDistance,
        'cosine': CosineDistance
    }
    
    WEIGHT_STRATEGIES = {
        'uniform': UniformWeights,
        'inverse': InverseDistanceWeights,
        'exponential': ExponentialWeights
    }
    
    def __init__(self, 
                 k: int = 3,
                 distance_metric: str = 'euclidean',
                 weights: str = 'uniform',
                 task: str = 'classification',
                 leaf_size: int = 30,
                 n_jobs: Optional[int] = None,
                 **metric_params):
        """
        Parameters:
        -----------
        k : int
            Number of neighbors to consider
        distance_metric : str
            Distance metric ('euclidean', 'manhattan', 'minkowski', 'cosine')
        weights : str
            Weighting strategy ('uniform', 'inverse', 'exponential')
        task : str
            Task type ('classification' or 'regression')
        leaf_size : int
            Leaf size for KD-tree (affects performance)
        n_jobs : int or None
            Number of parallel jobs
        metric_params : dict
            Additional parameters for distance metric
        """
        self.k = k
        self.task = task
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs
        self._tree = None
        self._scaler = None
        
        if distance_metric not in self.DISTANCE_METRICS:
            raise ValueError(f"Distance metric must be one of {list(self.DISTANCE_METRICS.keys())}")
        self.distance_metric = self.DISTANCE_METRICS[distance_metric](**metric_params)

        if weights not in self.WEIGHT_STRATEGIES:
            raise ValueError(f"Weights must be one of {list(self.WEIGHT_STRATEGIES.keys())}")
        self.weight_strategy = self.WEIGHT_STRATEGIES[weights]()
        
        if task not in ['classification', 'regression']:
            raise ValueError("Task must be 'classification' or 'regression'")
    
    def _build_kd_tree(self, X: np.ndarray) -> dict:
        """Build a simple KD-tree for efficient nearest neighbor search"""
        def build_tree(indices, depth=0):
            if len(indices) == 0:
                return None
            
            k = X.shape[1]
            axis = depth % k
            
            indices_sorted = sorted(indices, key=lambda i: X[i, axis])
            mid = len(indices_sorted) // 2
            
            return {
                'index': indices_sorted[mid],
                'axis': axis,
                'left': build_tree(indices_sorted[:mid], depth + 1),
                'right': build_tree(indices_sorted[mid + 1:], depth + 1)
            }
        
        return build_tree(list(range(len(X))))
    
    def _kd_tree_search(self, point: np.ndarray, k: int) -> List[int]:
        """Search KD-tree for k nearest neighbors"""
        best = []
        
        def search(node, depth=0):
            if node is None:
                return
            
            dist = self.distance_metric.compute(point, self.X_train[node['index']])
            
            best.append((dist, node['index']))
            best.sort(key=lambda x: x[0])
            if len(best) > k:
                best.pop()
            
            # Determine which side to search first
            axis = node['axis']
            if point[axis] < self.X_train[node['index'], axis]:
                search(node['left'], depth + 1)
                if len(best) < k or (point[axis] - self.X_train[node['index'], axis]) ** 2 < best[-1][0]:
                    search(node['right'], depth + 1)
            else:
                search(node['right'], depth + 1)
                if len(best) < k or (point[axis] - self.X_train[node['index'], axis]) ** 2 < best[-1][0]:
                    search(node['left'], depth + 1)
        
        search(self._tree)
        return [idx for _, idx in best]
    
    def _standard_scale(self, X: np.ndarray) -> np.ndarray:
        """Standardize features by removing mean and scaling to unit variance"""
        if self._scaler is None:
            self._mean = np.mean(X, axis=0)
            self._std = np.std(X, axis=0)
            self._std[self._std == 0] = 1.0  # Avoid division by zero
            self._scaler = True
        
        return (X - self._mean) / self._std
    
    def fit(self, X: np.ndarray, y: np.ndarray, scale_features: bool = True) -> 'KNN':
        """
        Fit the model to training data
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        scale_features : bool
            Whether to scale features
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
        if scale_features:
            self.X_train = self._standard_scale(self.X_train)
        
        # Build KD-tree for efficient search
        self._tree = self._build_kd_tree(self.X_train)
        
        # Store class information for classification
        if self.task == 'classification':
            self.classes_ = np.unique(y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for given data"""
        X = np.array(X)
        if hasattr(self, '_scaler') and self._scaler:
            X = (X - self._mean) / self._std
        
        if self.n_jobs and self.n_jobs > 1:
            from multiprocessing import Pool
            with Pool(self.n_jobs) as pool:
                predictions = pool.map(self._predict_single, X)
        else:
            predictions = [self._predict_single(x) for x in X]
        
        return np.array(predictions)
    
    def _predict_single(self, x: np.ndarray) -> Union[int, float]:
        """Predict single sample"""
        # Find k nearest neighbors
        if self._tree and len(self.X_train) > 100:  # Use KD-tree for larger datasets
            neighbor_indices = self._kd_tree_search(x, self.k)
        else:
            # Fallback to brute force for small datasets
            distances = [self.distance_metric.compute(x, x_train) for x_train in self.X_train]
            neighbor_indices = np.argsort(distances)[:self.k]
        
        # Get neighbor labels and distances
        neighbor_labels = self.y_train[neighbor_indices]
        neighbor_distances = [self.distance_metric.compute(x, self.X_train[i]) for i in neighbor_indices]
        
        # Calculate weights
        weights = self.weight_strategy.calculate_weights(np.array(neighbor_distances))
        
        if self.task == 'classification':
            return self._predict_classification(neighbor_labels, weights)
        else:
            return self._predict_regression(neighbor_labels, weights)
    
    def _predict_classification(self, labels: np.ndarray, weights: np.ndarray) -> int:
        """Predict class label using weighted voting"""
        weighted_votes = {}
        for label, weight in zip(labels, weights):
            weighted_votes[label] = weighted_votes.get(label, 0) + weight
        
        return max(weighted_votes.items(), key=lambda x: x[1])[0]
    
    def _predict_regression(self, values: np.ndarray, weights: np.ndarray) -> float:
        """Predict continuous value using weighted average"""
        return np.average(values, weights=weights)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for classification task"""
        if self.task != 'classification':
            raise ValueError("predict_proba is only available for classification")
        
        X = np.array(X)
        if hasattr(self, '_scaler') and self._scaler:
            X = (X - self._mean) / self._std
        
        probabilities = []
        for x in X:
            # Find k nearest neighbors
            if self._tree:
                neighbor_indices = self._kd_tree_search(x, self.k)
            else:
                distances = [self.distance_metric.compute(x, x_train) for x_train in self.X_train]
                neighbor_indices = np.argsort(distances)[:self.k]
            
            # Get neighbor labels and distances
            neighbor_labels = self.y_train[neighbor_indices]
            neighbor_distances = [self.distance_metric.compute(x, self.X_train[i]) for i in neighbor_indices]
            
            # Calculate weights
            weights = self.weight_strategy.calculate_weights(np.array(neighbor_distances))
            
            # Calculate weighted probabilities
            prob_dict = {}
            total_weight = np.sum(weights)
            
            for label, weight in zip(neighbor_labels, weights):
                prob_dict[label] = prob_dict.get(label, 0) + weight
            
            # Normalize and order by class
            class_probs = [prob_dict.get(cls, 0) / total_weight for cls in self.classes_]
            probabilities.append(class_probs)
        
        return np.array(probabilities)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the accuracy (classification) or R² score (regression)"""
        y_pred = self.predict(X)
        
        if self.task == 'classification':
            return np.mean(y == y_pred)
        else:
            # R² score for regression
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0


class KNNCrossValidator:
    """Cross-validation helper for KNN hyperparameter tuning"""
    
    def __init__(self, X, y, task='classification'):
        self.X = X
        self.y = y
        self.task = task
    
    def cross_validate(self, k_values: List[int], cv: int = 5, **knn_params) -> dict:
        """Perform cross-validation for different k values"""
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        results = {}
        
        for k in k_values:
            scores = []
            for train_idx, val_idx in kf.split(self.X):
                X_train, X_val = self.X[train_idx], self.X[val_idx]
                y_train, y_val = self.y[train_idx], self.y[val_idx]
                
                knn = KNN(k=k, task=self.task, **knn_params)
                knn.fit(X_train, y_train)
                score = knn.score(X_val, y_val)
                scores.append(score)
            
            results[k] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'all_scores': scores
            }
        
        return results


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("=== Enhanced KNN Demo ===")
    
    configurations = [
        {'k': 3, 'distance_metric': 'euclidean', 'weights': 'uniform'},
        {'k': 5, 'distance_metric': 'manhattan', 'weights': 'inverse'},
        {'k': 7, 'distance_metric': 'cosine', 'weights': 'exponential', 'gamma': 0.5},
    ]
    
    for config in configurations:
        print(f"\nTesting configuration: {config}")
        knn = KNN(**config, task='classification')
        knn.fit(X_train, y_train, scale_features=True)
        
        accuracy = knn.score(X_test, y_test)
        print(f"Accuracy: {accuracy:.4f}")
        
        proba = knn.predict_proba(X_test[:1])
        print(f"Class probabilities for first test sample: {proba[0]}")
    
    print("\n=== Cross-Validation Results ===")
    validator = KNNCrossValidator(X, y, task='classification')
    results = validator.cross_validate(k_values=[1, 3, 5, 7, 9], cv=5)
    
    for k, result in results.items():
        print(f"k={k}: Mean Accuracy = {result['mean_score']:.4f} ± {result['std_score']:.4f}")
    
    print("\n=== Regression Example ===")
    boston = datasets.fetch_california_housing()
    X_reg, y_reg = boston.data[:1000], boston.target[:1000]
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    knn_reg = KNN(k=5, task='regression', weights='inverse')
    knn_reg.fit(X_train_reg, y_train_reg, scale_features=True)
    r2_score = knn_reg.score(X_test_reg, y_test_reg)
    print(f"R² Score for regression: {r2_score:.4f}")
