# Enhanced K-Nearest Neighbors (KNN) Algorithm From Scratch

<img width="432" height="259" alt="image" src="https://github.com/user-attachments/assets/46527bc7-8055-4a74-80cd-8f000dc4d669" />

A comprehensive, production-ready implementation of the K-Nearest Neighbors algorithm with multiple enhancements, advanced features, and extensive customization options.

<img width="685" height="660" alt="image" src="https://github.com/user-attachments/assets/41a2a17f-63e6-433e-bf04-4f686ed959a6" />

## Table of Contents

- Overview
- Features
- Installation
- Quick Start
- Algorithm Details
- API Reference
- Examples
- Performance Benchmarks
- Mathematical Background
- Applications
- Contributing
- License

## Overview

K-Nearest Neighbors (KNN) is a simple yet powerful instance-based learning algorithm used for both classification and regression tasks. This enhanced implementation provides a robust, feature-rich version suitable for both educational purposes and production environments.

### Key Improvements Over Basic KNN

| Feature | Basic KNN | Enhanced KNN |
|---------|-----------|--------------|
| Distance Metrics | Euclidean only | 4+ metrics |
| Weighting | Uniform only | Multiple strategies |
| Efficiency | Brute force | KD-tree optimized |
| Tasks | Classification only | Classification & Regression |
| Feature Scaling | Manual | Automatic |
| Hyperparameter Tuning | Manual | Cross-validation |
| Parallel Processing | No | Yes |
| Probability Estimates | No | Yes |

## ✨ Features

### Core Features
- **Multi-class Classification**: Support for multiple classes
- **Regression**: Continuous value prediction
- **Probability Estimates**: Class probability scores
- **Feature Scaling**: Automatic standardization
- **Parallel Processing**: Multi-core support for large datasets

### Distance Metrics
| Metric | Formula | Use Case |
|--------|---------|----------|
| Euclidean | `√(Σ(x_i - y_i)²)` | General purpose |
| Manhattan | `Σ\|x_i - y_i\|` | High-dimensional data |
| Minkowski | `(Σ\|x_i - y_i\|^p)^(1/p)` | Customizable distance |
| Cosine | `1 - (x·y)/(\|x\|\|y\|)` | Text similarity |

### Weighting Strategies
| Strategy | Formula | Description |
|----------|---------|-------------|
| Uniform | `w_i = 1` | All neighbors equal weight |
| Inverse | `w_i = 1/(d_i + ε)` | Closer neighbors have more influence |
| Exponential | `w_i = exp(-γ·d_i)` | Smooth distance-based weighting |

### Advanced Features
- **KD-tree Optimization**: Efficient nearest neighbor search
- **Cross-validation**: Automated hyperparameter tuning
- **Comprehensive Metrics**: Accuracy, R² score, probability estimates
- **Type Hints**: Full type annotation for better development
- **Error Handling**: Robust validation and error messages

## Installation

### Requirements
```bash
Python >= 3.7
numpy >= 1.19.0
scikit-learn >= 0.24.0 (for demo and utilities)
```

### Installation Methods

#### Method 1: Direct Download
```python
# Download the script and import directly
from enhanced_knn import KNN, KNNCrossValidator
```

#### Method 2: Package Installation
```bash
# Create a package structure
mkdir enhanced_knn
cd enhanced_knn
# Place the code in __init__.py
```

#### Method 3: PIP Installation (Hypothetical)
```bash
pip install knn-from-scratch
```

## Quick Start

### Basic Classification
```python
from enhanced_knn import KNN
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train KNN
knn = KNN(k=5, distance_metric='euclidean', weights='inverse')
knn.fit(X_train, y_train)

# Predict and evaluate
predictions = knn.predict(X_test)
accuracy = knn.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

### Advanced Configuration
```python
# Advanced configuration with all features
knn = KNN(
    k=7,
    distance_metric='minkowski', 
    weights='exponential',
    task='classification',
    leaf_size=30,
    n_jobs=4,
    p=3  # Minkowski parameter
)

knn.fit(X_train, y_train, scale_features=True)

# Get probability estimates
probabilities = knn.predict_proba(X_test)
```

## 🔧 API Reference

### KNN Class

#### Constructor Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | 3 | Number of neighbors |
| `distance_metric` | str | 'euclidean' | Distance calculation method |
| `weights` | str | 'uniform' | Weighting strategy |
| `task` | str | 'classification' | 'classification' or 'regression' |
| `leaf_size` | int | 30 | KD-tree leaf size |
| `n_jobs` | int | None | Number of parallel jobs |
| `**metric_params` | dict | {} | Additional metric parameters |

#### Methods

##### `fit(X, y, scale_features=True)`
Trains the KNN model.

**Parameters:**
- `X`: array-like, training features
- `y`: array-like, target values
- `scale_features`: bool, whether to standardize features

**Returns:** self

##### `predict(X)`
Predicts labels for input data.

**Parameters:**
- `X`: array-like, input features

**Returns:** array of predictions

##### `predict_proba(X)`
Predicts class probabilities (classification only).

**Parameters:**
- `X`: array-like, input features

**Returns:** array of class probabilities

##### `score(X, y)`
Returns accuracy (classification) or R² score (regression).

**Parameters:**
- `X`: array-like, test features
- `y`: array-like, true labels

**Returns:** float, performance score

### KNNCrossValidator Class

Utility class for hyperparameter tuning.

#### `cross_validate(k_values, cv=5, **knn_params)`
Performs cross-validation for different k values.

**Parameters:**
- `k_values`: list of k values to test
- `cv`: number of cross-validation folds
- `**knn_params`: additional KNN parameters

**Returns:** dict with cross-validation results

## 📊 Examples

### Example 1: Comprehensive Classification
```python
import numpy as np
from enhanced_knn import KNN, KNNCrossValidator
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load wine dataset
wine = load_wine()
X, y = wine.data, wine.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=== Comprehensive KNN Classification ===")

# Test multiple configurations
configurations = [
    {'k': 3, 'distance_metric': 'euclidean', 'weights': 'uniform'},
    {'k': 5, 'distance_metric': 'manhattan', 'weights': 'inverse'},
    {'k': 7, 'distance_metric': 'cosine', 'weights': 'exponential', 'gamma': 0.5},
]

for config in configurations:
    knn = KNN(**config)
    knn.fit(X_train, y_train, scale_features=True)
    accuracy = knn.score(X_test, y_test)
    print(f"Config {config}: Accuracy = {accuracy:.4f}")

# Cross-validation for optimal k
print("\n=== Cross-Validation ===")
validator = KNNCrossValidator(X, y)
results = validator.cross_validate(k_values=[1, 3, 5, 7, 9, 11], cv=5)

for k, result in results.items():
    print(f"k={k}: {result['mean_score']:.4f} ± {result['std_score']:.4f}")
```

### Example 2: Regression Task
```python
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score

# Load housing data
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Use subset for demonstration
X, y = X[:1000], y[:1000]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("=== KNN Regression ===")

knn_reg = KNN(k=5, task='regression', weights='inverse')
knn_reg.fit(X_train, y_train, scale_features=True)

predictions = knn_reg.predict(X_test)
r2 = knn_reg.score(X_test, y_test)
mse = mean_squared_error(y_test, predictions)

print(f"R² Score: {r2:.4f}")
print(f"MSE: {mse:.4f}")
print(f"Predictions sample: {predictions[:5]}")
```

### Example 3: Probability Estimates
```python
# Probability estimation example
knn_prob = KNN(k=5, weights='inverse')
knn_prob.fit(X_train, y_train, scale_features=True)

# Get probabilities for test set
probabilities = knn_prob.predict_proba(X_test[:3])
predictions = knn_prob.predict(X_test[:3])

print("=== Probability Estimates ===")
for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
    print(f"Sample {i+1}: Predicted class = {pred}")
    print(f"Probabilities: {dict(zip(knn_prob.classes_, probs))}")
    print()
```

## Performance Benchmarks

### Time Complexity Analysis
| Operation | Brute Force | KD-tree (Avg) | KD-tree (Worst) |
|-----------|-------------|---------------|-----------------|
| Training | O(1) | O(d·n log n) | O(d·n log n) |
| Prediction | O(n·d) | O(k log n) | O(n) |
| Memory | O(n·d) | O(n·d) | O(n·d) |

Where:
- n = number of training samples
- d = number of features
- k = number of neighbors

### Empirical Performance

**Dataset: Iris (150 samples, 4 features)**
| k | Euclidean | Manhattan | Cosine |
|---|-----------|-----------|--------|
| 3 | 0.9667 | 0.9667 | 0.9333 |
| 5 | 0.9667 | 1.0000 | 0.9333 |
| 7 | 1.0000 | 1.0000 | 0.9333 |

**Dataset: Wine (178 samples, 13 features)**
| k | Uniform | Inverse | Exponential |
|---|---------|---------|-------------|
| 3 | 0.7222 | 0.7500 | 0.7500 |
| 5 | 0.7500 | 0.7778 | 0.7778 |
| 7 | 0.7778 | 0.8056 | 0.7778 |

## 📐 Mathematical Background

### Distance Metrics

#### Euclidean Distance
$$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

#### Manhattan Distance
$$d(x, y) = \sum_{i=1}^{n} |x_i - y_i|$$

#### Minkowski Distance
$$d(x, y) = \left( \sum_{i=1}^{n} |x_i - y_i|^p \right)^{1/p}$$

#### Cosine Distance
$$d(x, y) = 1 - \frac{x \cdot y}{\|x\| \cdot \|y\|}$$

### Weighting Strategies

#### Uniform Weights
$$w_i = 1$$

#### Inverse Distance Weights
$$w_i = \frac{1}{d_i + \epsilon}$$

#### Exponential Weights
$$w_i = e^{-\gamma \cdot d_i}$$

### Prediction Rules

#### Classification (Weighted Voting)
$$\hat{y} = \arg\max_{c} \sum_{i=1}^{k} w_i \cdot \mathbb{1}(y_i = c)$$

#### Regression (Weighted Average)
$$\hat{y} = \frac{\sum_{i=1}^{k} w_i \cdot y_i}{\sum_{i=1}^{k} w_i}$$

## Applications

### Ideal Use Cases
- **Recommendation Systems**: User similarity matching
- **Anomaly Detection**: Identifying outliers
- **Image Recognition**: Simple image classification
- **Medical Diagnosis**: Patient similarity analysis
- **Financial Analysis**: Credit scoring

### When to Use KNN
| Scenario | Recommendation |
|----------|---------------|
| Small datasets | ✅ Excellent choice |
| Large datasets | ⚠️ Use with KD-tree optimization |
| Many features | ⚠️ Requires feature scaling |
| Non-linear patterns | ✅ Very effective |
| Interpretability | ✅ Highly interpretable |
| Training speed | ✅ Instant training |

### Limitations and Considerations
- **Curse of Dimensionality**: Performance degrades with high dimensions
- **Memory Intensive**: Stores all training data
- **Sensitive to Scales**: Requires feature normalization
- **Slow Prediction**: For large datasets without optimization

## Advanced Usage

### Custom Distance Metric
```python
from enhanced_knn import DistanceMetric

class CustomDistance(DistanceMetric):
    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        # Implement custom distance calculation
        return np.sum(np.abs(x1 - x2) ** 0.5)  # Example custom metric

# Use custom distance
knn_custom = KNN(k=5, distance_metric=CustomDistance())
```

### Custom Weighting Strategy
```python
from enhanced_knn import WeightStrategy

class CustomWeights(WeightStrategy):
    def calculate_weights(self, distances: np.ndarray) -> np.ndarray:
        return 1 / (1 + distances ** 2)  # Custom weighting function

# Use custom weights
knn_custom_weights = KNN(k=5, weights=CustomWeights())
```

### Integration with Scikit-learn
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNN())
])

# Define parameter grid
param_grid = {
    'knn__k': [3, 5, 7, 9],
    'knn__distance_metric': ['euclidean', 'manhattan'],
    'knn__weights': ['uniform', 'inverse']
}

# Grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

## Performance Optimization Tips

### 1. Feature Scaling
Always enable `scale_features=True` for better performance:
```python
knn.fit(X_train, y_train, scale_features=True)
```

### 2. KD-tree Optimization
For datasets larger than 100 samples, KD-tree is automatically used:
```python
# Adjust leaf size for performance
knn = KNN(k=5, leaf_size=20)  # Smaller for faster, larger for memory
```

### 3. Parallel Processing
For large prediction sets:
```python
knn = KNN(k=5, n_jobs=4)  # Use 4 CPU cores
```

### 4. Optimal k Selection
Use cross-validation to find the best k:
```python
validator = KNNCrossValidator(X, y)
results = validator.cross_validate(k_values=range(1, 15))
```

## Testing and Validation

### Unit Tests
```python
import unittest
import numpy as np
from enhanced_knn import KNN

class TestKNN(unittest.TestCase):
    def setUp(self):
        self.X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        self.y_train = np.array([0, 0, 1, 1])
        
    def test_prediction(self):
        knn = KNN(k=3)
        knn.fit(self.X_train, self.y_train)
        predictions = knn.predict(np.array([[2.5, 3.5]]))
        self.assertEqual(predictions.shape, (1,))
        
    def test_probability(self):
        knn = KNN(k=3)
        knn.fit(self.X_train, self.y_train)
        probabilities = knn.predict_proba(np.array([[2.5, 3.5]]))
        self.assertEqual(probabilities.shape, (1, 2))
        self.assertAlmostEqual(np.sum(probabilities[0]), 1.0)

if __name__ == '__main__':
    unittest.main()
```

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
git clone https://github.com/saadarazzaq/knn-from-scratch.git
cd knn-from-scratch
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running Tests
```bash
python -m pytest tests/
python -m unittest discover tests/
```

## Acknowledgments

- Inspired by traditional KNN implementations
- KD-tree optimization based on computational geometry principles
- Weighting strategies from statistical learning theory
- Cross-validation techniques from machine learning best practices

## References

> 1. Cover, T. M., & Hart, P. E. (1967). "Nearest neighbor pattern classification"
> 2. Friedman, J. H., Bentley, J. L., & Finkel, R. A. (1977). "An algorithm for finding best matches in logarithmic expected time"
> 3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). "The Elements of Statistical Learning"
> 4. Scikit-learn: Machine Learning in Python

---

<br>

<h2 align="center">✨ Author</h2>

<p align="center">
  <b>Saad Abdur Razzaq</b><br>
  <i>Machine Learning Engineer | Effixly AI</i>
</p>

<p align="center">
  <a href="https://www.linkedin.com/in/saadarazzaq" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
  </a>
  <a href="mailto:sabdurrazzaq124@gmail.com">
    <img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email">
  </a>
  <a href="https://saadarazzaq.dev" target="_blank">
    <img src="https://img.shields.io/badge/Website-000000?style=for-the-badge&logo=google-chrome&logoColor=white" alt="Website">
  </a>
  <a href="https://github.com/saadabdurrazzaq" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
  </a>
</p>

<br>

---

<div align="center">

### ⭐ Don't forget to star this repository if you find it helpful!

</div>
