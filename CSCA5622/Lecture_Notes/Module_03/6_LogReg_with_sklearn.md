# Logistic Regression with scikit-learn - Comprehensive Reference Notes
**CSCA5622 - Module 03: Classification**

---

## 📚 Overview

**Logistic Regression** (also called **logit** or **MaxEnt** classifier) is a fundamental classification algorithm that models the probability of a binary or multiclass outcome using a logistic (sigmoid) function. Despite its name, it's used for **classification**, not regression.

This document covers:
- Mathematical foundations
- scikit-learn implementation details
- All parameters explained
- Solver options and when to use them
- Regularization (L1, L2, Elastic Net)
- Binary and multiclass classification
- Practical examples and code
- Best practices

**Source:** Based on scikit-learn official documentation (v1.7.2+)

---

## 1. What Is Logistic Regression?

### 🔍 Definition

From sklearn documentation:
> "Logistic Regression (aka logit, MaxEnt) classifier implements regularized logistic regression using various solvers. It can handle both dense and sparse input."

### 📐 Mathematical Foundation

**For binary classification:**

The model predicts probability using the logistic/sigmoid function:

```
P(y=1|X) = 1 / (1 + e^(-(β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ)))
```

**Equivalently, the log-odds (logit):**
```
log(P(y=1|X) / P(y=0|X)) = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ
```

This transforms the linear combination to probabilities in range [0, 1].

### 🎯 Key Features

**Regularization by default:**
- Regularization is applied by default in sklearn
- Helps prevent overfitting
- Can use L1, L2, or Elastic Net penalties

**Multiple solvers available:**
- Different optimization algorithms for different scenarios
- Trade-offs between speed, accuracy, and features supported

**Handles multiclass:**
- Can solve binary or multiclass problems
- Uses one-vs-rest (OvR) or multinomial approaches

---

## 2. The LogisticRegression Class

### 📋 Basic Usage

```python
from sklearn.linear_model import LogisticRegression

# Create model
clf = LogisticRegression()

# Fit model
clf.fit(X_train, y_train)

# Predict classes
y_pred = clf.predict(X_test)

# Predict probabilities
y_proba = clf.predict_proba(X_test)

# Score (accuracy)
score = clf.score(X_test, y_test)
```

### 🎛️ Complete Parameter List

```python
LogisticRegression(
    penalty='l2',              # Regularization type
    dual=False,                # Dual or primal formulation
    tol=1e-4,                  # Stopping tolerance
    C=1.0,                     # Inverse regularization strength
    fit_intercept=True,        # Add intercept/bias term
    intercept_scaling=1,       # Scaling for intercept
    class_weight=None,         # Class weights
    random_state=None,         # Random seed
    solver='lbfgs',            # Optimization algorithm
    max_iter=100,              # Max iterations
    multi_class='auto',        # Multiclass strategy (deprecated)
    verbose=0,                 # Verbosity level
    warm_start=False,          # Reuse previous solution
    n_jobs=None,               # Parallel processing
    l1_ratio=None              # Elastic Net mixing
)
```

---

## 3. Core Parameters Explained

### 🔧 penalty: Regularization Type

**Purpose:** Specify the norm of the penalty to prevent overfitting

**Options:**
- **`'l2'`** (default): Ridge regularization (squared coefficients)
- **`'l1'`**: Lasso regularization (absolute coefficients)
- **`'elasticnet'`**: Combination of L1 and L2
- **`None`**: No regularization

**When to use:**
- **L2**: Default choice, works well generally
- **L1**: For feature selection (drives coefficients to exactly zero)
- **Elastic Net**: When you want both regularization and feature selection
- **None**: When you have very few features or want no regularization

**Example:**
```python
# L2 regularization (default)
clf_l2 = LogisticRegression(penalty='l2')

# L1 regularization for feature selection
clf_l1 = LogisticRegression(penalty='l1', solver='liblinear')

# Elastic Net (requires 'saga' solver)
clf_en = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5)

# No regularization
clf_none = LogisticRegression(penalty=None)
```

⚠️ **Warning:** Not all solvers support all penalties (see solver section below)

---

### 🎚️ C: Regularization Strength

**Purpose:** Inverse of regularization strength (smaller = stronger regularization)

**Type:** Positive float (default: 1.0)

**How it works:**
```
C = 1 / λ

Where λ is the regularization parameter
```

**Guidelines:**
- **Large C** (e.g., 100): Weak regularization, model fits training data closely
- **Small C** (e.g., 0.01): Strong regularization, simpler model
- **C = 1.0**: Default, balanced regularization

**Example:**
```python
# Strong regularization (simpler model)
clf_strong = LogisticRegression(C=0.01)

# Weak regularization (more complex model)
clf_weak = LogisticRegression(C=100)

# Tune C with GridSearchCV
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
print(f"Best C: {grid.best_params_['C']}")
```

**Relationship to bias-variance:**
- Small C → High bias, low variance (underfitting risk)
- Large C → Low bias, high variance (overfitting risk)

---

### ⚙️ solver: Optimization Algorithm

**Purpose:** Choose the algorithm to use for optimization

**Options:**
- **`'lbfgs'`** (default): Limited-memory BFGS
- **`'liblinear'`**: Coordinate descent
- **`'newton-cg'`**: Newton's method with conjugate gradient
- **`'newton-cholesky'`**: Newton's method with Cholesky factorization  
- **`'sag'`**: Stochastic Average Gradient
- **`'saga'`**: SAGA (supports all penalties)

### 📊 Solver Comparison Table

| Solver | Penalty Support | Multiclass | Best For | Notes |
|--------|----------------|------------|----------|-------|
| **lbfgs** | L2, None | Yes | Default choice, medium datasets | Good all-around |
| **liblinear** | L1, L2 | No (binary only) | Small datasets, L1 penalty | Fast for small data |
| **newton-cg** | L2, None | Yes | Medium datasets | Similar to lbfgs |
| **newton-cholesky** | L2, None | Yes | n_samples >> n_features | High memory usage |
| **sag** | L2, None | Yes | Large datasets | Fast, needs scaling |
| **saga** | L1, L2, Elastic Net, None | Yes | Any penalty, large data | Most flexible |

### 🎯 Choosing a Solver

**Decision flowchart:**

```
Do you need L1 or Elastic Net regularization?
├─ Yes → Use 'saga' (or 'liblinear' for L1 binary)
└─ No
   └─ Is your dataset large (>10,000 samples)?
      ├─ Yes → Use 'sag' or 'saga'
      └─ No
         └─ Is n_samples >> n_features (many samples, few features)?
            ├─ Yes → Use 'newton-cholesky'
            └─ No → Use 'lbfgs' (default)
```

**Examples:**
```python
# Default: good for most cases
clf_default = LogisticRegression(solver='lbfgs')

# Small dataset with L1 regularization
clf_small = LogisticRegression(solver='liblinear', penalty='l1')

# Large dataset
clf_large = LogisticRegression(solver='sag', max_iter=1000)

# Need Elastic Net
clf_elastic = LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.5)
```

⚠️ **Important:** SAG and SAGA require feature scaling for fast convergence!

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
clf = LogisticRegression(solver='saga').fit(X_scaled, y)
```

---

### 🔢 max_iter: Maximum Iterations

**Purpose:** Maximum number of iterations for solver to converge

**Default:** 100

**When to increase:**
- Model fails to converge (warning message appears)
- Using SAG/SAGA on large datasets
- Strong regularization (small C)

**Example:**
```python
# Increase iterations if convergence warning appears
clf = LogisticRegression(max_iter=1000)

# Check if converged
clf.fit(X_train, y_train)
print(f"Iterations used: {clf.n_iter_}")  # Check convergence
```

---

### 🎭 class_weight: Handling Imbalanced Data

**Purpose:** Adjust weights for imbalanced classes

**Options:**
- **`None`** (default): All classes have weight 1
- **`'balanced'`**: Automatically adjust weights inversely proportional to class frequencies
- **Dict:** Manual weights `{class_label: weight}`

**Automatic balancing formula:**
```
weight_for_class_i = n_samples / (n_classes × count_of_class_i)
```

**Example:**
```python
# Automatic balancing for imbalanced data
clf_balanced = LogisticRegression(class_weight='balanced')

# Manual weights
clf_manual = LogisticRegression(class_weight={0: 1, 1: 10})  # Class 1 has 10× weight

# Example: imbalanced dataset
# Class 0: 900 samples, Class 1: 100 samples
# Balanced weights: Class 0 = 1000/(2×900) ≈ 0.56
#                   Class 1 = 1000/(2×100) = 5.0
```

**When to use:**
- Imbalanced datasets (one class much more frequent)
- When false negatives/positives have different costs
- Medical diagnosis, fraud detection, etc.

---

### 🔀 l1_ratio: Elastic Net Mixing

**Purpose:** Balance between L1 and L2 penalties in Elastic Net

**Range:** 0 ≤ l1_ratio ≤ 1

**Only used when:** `penalty='elasticnet'`

**Interpretation:**
- **l1_ratio = 0**: Pure L2 (Ridge)
- **l1_ratio = 1**: Pure L1 (Lasso)
- **0 < l1_ratio < 1**: Combination

**Formula:**
```
penalty = l1_ratio × L1 + (1 - l1_ratio) × L2
```

**Example:**
```python
# 50% L1, 50% L2
clf_elastic = LogisticRegression(
    penalty='elasticnet',
    solver='saga',
    l1_ratio=0.5,
    C=1.0
)

# Mostly L1 (for feature selection)
clf_mostly_l1 = LogisticRegression(
    penalty='elasticnet',
    solver='saga',
    l1_ratio=0.9
)
```

---

### 🎲 random_state: Reproducibility

**Purpose:** Seed for random number generator

**Type:** Integer or None

**When it matters:**
- Solver uses randomness ('sag', 'saga', 'liblinear')
- Feature selection during fitting
- Data shuffling

**Example:**
```python
# Reproducible results
clf = LogisticRegression(random_state=42)
```

---

### 🔥 warm_start: Incremental Learning

**Purpose:** Reuse previous solution as initialization

**Default:** False

**When useful:**
- Training on batches of data
- Hyperparameter tuning (increasing max_iter)
- Online learning scenarios

**Example:**
```python
clf = LogisticRegression(warm_start=True, max_iter=100)

# First fit
clf.fit(X_batch1, y_batch1)

# Continue training from previous solution
clf.fit(X_batch2, y_batch2)
```

---

### 🔧 Other Parameters

#### fit_intercept (default: True)
Add bias/intercept term to decision function

```python
clf = LogisticRegression(fit_intercept=True)  # Usually keep True
```

#### dual (default: False)
- Dual vs primal formulation
- Only for L2 with liblinear
- Use `dual=False` when n_samples > n_features

#### tol (default: 1e-4)
Tolerance for stopping criteria (smaller = more precise)

#### verbose (default: 0)
Set to positive number for progress messages

#### n_jobs (default: None)
Number of CPU cores for parallel processing (when multi_class='ovr')

---

## 4. Model Attributes (After Fitting)

### 📊 Learned Parameters

After calling `.fit()`, these attributes become available:

#### coef_: Feature Coefficients

**Shape:** `(n_classes, n_features)` or `(1, n_features)` for binary

**Meaning:** Weights for each feature in decision function

```python
clf.fit(X_train, y_train)

print("Coefficients shape:", clf.coef_.shape)
print("Coefficients:", clf.coef_)

# For binary classification
# Positive coef → increases probability of class 1
# Negative coef → decreases probability of class 1
```

#### intercept_: Intercept Term

**Shape:** `(n_classes,)` or `(1,)` for binary

**Meaning:** Bias term added to decision function

```python
print("Intercept:", clf.intercept_)
```

#### classes_: Class Labels

**Type:** Array of class labels known to classifier

```python
print("Classes:", clf.classes_)  # e.g., [0, 1] or ['cat', 'dog']
```

#### n_features_in_: Number of Features

```python
print("Features used:", clf.n_features_in_)
```

#### feature_names_in_: Feature Names

Available if X was DataFrame with string column names

```python
print("Feature names:", clf.feature_names_in_)
```

#### n_iter_: Iterations Used

**Type:** Array showing iterations for convergence

```python
print("Iterations:", clf.n_iter_)
```

**Interpretation:**
- If equals max_iter → may not have converged (check for warning)
- Otherwise → successfully converged

---

## 5. Binary Classification

### 📊 Binary Classification Example

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Generate binary classification data
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and fit model
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

# Evaluation
print("Accuracy:", clf.score(X_test, y_test))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Examine coefficients
print("\nTop 5 most important features (by abs coefficient):")
feature_importance = np.abs(clf.coef_[0])
top_indices = feature_importance.argsort()[-5:][::-1]
for idx in top_indices:
    print(f"Feature {idx}: {clf.coef_[0][idx]:.4f}")
```

### 🎯 Interpreting Coefficients in Binary Classification

**For binary classification with classes [0, 1]:**

```python
# Decision function
decision = clf.intercept_ + np.dot(X, clf.coef_.T)

# Probability of class 1
prob_class_1 = 1 / (1 + np.exp(-decision))

# Probability of class 0
prob_class_0 = 1 - prob_class_1
```

**Coefficient interpretation:**
- **Positive coefficient:** Increases log-odds of class 1
- **Negative coefficient:** Decreases log-odds of class 1
- **Magnitude:** Strength of effect

**Example:**
```
coef_ = [0.5, -0.3, 1.2]

Feature 0: +0.5 → Each unit increase multiplies odds by e^0.5 ≈ 1.65
Feature 1: -0.3 → Each unit increase multiplies odds by e^-0.3 ≈ 0.74
Feature 2: +1.2 → Each unit increase multiplies odds by e^1.2 ≈ 3.32
```

---

## 6. Multiclass Classification

### 🎯 Strategies for Multiclass

**Two main approaches:**

#### 1. One-vs-Rest (OvR)
- Train N binary classifiers (one per class)
- Each classifier: "this class" vs "all others"
- Prediction: class with highest probability

#### 2. Multinomial
- Single model minimizing multinomial loss
- Models full probability distribution
- Generally better for multiclass

### 📊 Multiclass Example

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load iris dataset (3 classes)
iris = load_iris()
X, y = iris.data, iris.target

# Scale features (important for SAG/SAGA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Multinomial logistic regression
clf_multi = LogisticRegression(
    multi_class='multinomial',  # Note: deprecated, will be default
    solver='lbfgs',
    random_state=42
)
clf_multi.fit(X_train, y_train)

# Predictions
y_pred = clf_multi.predict(X_test)
y_proba = clf_multi.predict_proba(X_test)

print("Accuracy:", clf_multi.score(X_test, y_test))
print("\nPredicted probabilities for first 3 samples:")
print(y_proba[:3])
print("\nActual classes:", y_test[:3])
print("Predicted classes:", y_pred[:3])

# Coefficients shape for multiclass
print("\nCoefficients shape:", clf_multi.coef_.shape)  # (3, 4) for 3 classes, 4 features
print("Intercepts shape:", clf_multi.intercept_.shape)  # (3,)
```

### 🔍 Multiclass Coefficient Interpretation

```python
# Each row = coefficients for one class
print("Class 0 (setosa) coefficients:", clf_multi.coef_[0])
print("Class 1 (versicolor) coefficients:", clf_multi.coef_[1])
print("Class 2 (virginica) coefficients:", clf_multi.coef_[2])

# Feature importance per class
import pandas as pd

coef_df = pd.DataFrame(
    clf_multi.coef_,
    columns=iris.feature_names,
    index=[f"Class {i}" for i in range(3)]
)
print("\nCoefficients by class:")
print(coef_df)
```

---

## 7. Complete Practical Examples

### Example 1: Binary Classification with Model Tuning

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                           n_classes=2, random_state=42)

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(random_state=42))
])

# Parameter grid
param_grid = {
    'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'clf__penalty': ['l1', 'l2'],
    'clf__solver': ['liblinear', 'saga']
}

# Grid search
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X, y)

print("Best parameters:", grid.best_params_)
print("Best cross-validation score:", grid.best_score_)

# Use best model
best_clf = grid.best_estimator_
```

### Example 2: Imbalanced Data Handling

```python
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Create imbalanced dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    weights=[0.9, 0.1],  # 90% class 0, 10% class 1
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Class distribution in training:", np.bincount(y_train))

# Approach 1: class_weight='balanced'
clf_balanced = LogisticRegression(class_weight='balanced', random_state=42)
clf_balanced.fit(X_train, y_train)

# Approach 2: Manual weights (favor minority class)
clf_manual = LogisticRegression(class_weight={0: 1, 1: 9}, random_state=42)
clf_manual.fit(X_train, y_train)

# Approach 3: SMOTE oversampling
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
clf_smote = LogisticRegression(random_state=42)
clf_smote.fit(X_train_sm, y_train_sm)

# Compare approaches
for name, clf in [('Balanced', clf_balanced), ('Manual', clf_manual), ('SMOTE', clf_smote)]:
    y_pred = clf.predict(X_test)
    print(f"\n{name} approach:")
    print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))
```

### Example 3: Feature Selection with L1 Regularization

```python
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Generate data with many features
X, y = make_classification(
    n_samples=1000,
    n_features=100,
    n_informative=10,
    n_redundant=90,
    random_state=42
)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# L1 regularization for feature selection
clf_l1 = LogisticRegression(penalty='l1', solver='saga', C=0.1, max_iter=10000, random_state=42)
clf_l1.fit(X_scaled, y)

# Identify non-zero coefficients
non_zero_coef = np.abs(clf_l1.coef_[0]) > 0
n_selected = non_zero_coef.sum()

print(f"Features selected: {n_selected} out of {X.shape[1]}")
print(f"Selected feature indices: {np.where(non_zero_coef)[0]}")

# Compare with L2
clf_l2 = LogisticRegression(penalty='l2', C=0.1, random_state=42)
clf_l2.fit(X_scaled, y)

print(f"\nL1 - Non-zero coefficients: {n_selected}")
print(f"L2 - Non-zero coefficients: {(np.abs(clf_l2.coef_[0]) > 0.001).sum()}")
```

---

## 8. Best Practices

### ✅ Recommendations

**1. Always scale features when using SAG/SAGA:**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**2. Use cross-validation for hyperparameter tuning:**
```python
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
```

**3. Check for convergence warnings:**
```python
import warnings
warnings.filterwarnings('always')
clf.fit(X, y)
# If warning appears, increase max_iter
```

**4. Handle imbalanced data appropriately:**
```python
clf = LogisticRegression(class_weight='balanced')
```

**5. Use regularization to prevent overfitting:**
```python
# Don't use penalty=None unless you have good reason
clf = LogisticRegression(C=1.0)  # Regularization on by default
```

### ⚠️ Common Pitfalls

**1. Forgetting to scale with SAG/SAGA**
```python
# ❌ Bad - slow convergence
clf = LogisticRegression(solver='saga')
clf.fit(X, y)  # X not scaled!

# ✓ Good
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
clf.fit(X_scaled, y)
```

**2. Using wrong solver for penalty**
```python
# ❌ Bad - incompatible
clf = LogisticRegression(penalty='l1', solver='lbfgs')  # Error!

# ✓ Good
clf = LogisticRegression(penalty='l1', solver='liblinear')
```

**3. Not handling class imbalance**
```python
# ❌ Bad for 90-10 split
clf = LogisticRegression()

# ✓ Good
clf = LogisticRegression(class_weight='balanced')
```

**4. Using default max_iter for large datasets**
```python
# ❌ May not converge
clf = LogisticRegression(solver='sag')  # max_iter=100

# ✓ Better
clf = LogisticRegression(solver='sag', max_iter=1000)
```

---

## 9. Comparison with Other Classifiers

### 📊 When to Use Logistic Regression

**Advantages:**
- ✓ Fast training and prediction
- ✓ Probabilistic predictions
- ✓ Interpretable coefficients
- ✓ Works well with linearly separable data
- ✓ Low memory footprint
- ✓ Built-in regularization

**Disadvantages:**
- ✗ Assumes linear decision boundary
- ✗ May underfit complex patterns
- ✗ Sensitive to outliers
- ✗ Requires feature engineering for non-linear relationships

### 🔄 vs. Other Algorithms

| Algorithm | When Better | When Worse |
|-----------|-------------|------------|
| **Decision Trees** | Non-linear patterns, feature interactions | Interpretability, overfitting risk |
| **Random Forest** | Complex non-linear data, robustness | Speed, interpretability |
| **SVM** | High-dimensional spaces, kernel tricks | Large datasets, speed |
| **Naive Bayes** | Very fast predictions needed | Feature independence assumption |
| **Neural Networks** | Very complex patterns, lots of data | Interpretability, small datasets |

---

## 10. Summary

### 🎯 Key Takeaways

1. **Logistic Regression** is a fundamental classification algorithm using the logistic function
2. **Regularization** is applied by default (controlled by C parameter)
3. **Multiple solvers** available - choose based on data size and penalty type
4. **Handles multiclass** with OvR or multinomial strategies
5. **Feature scaling** important for SAG/SAGA solvers
6. **class_weight='balanced'** useful for imbalanced data
7. **L1 regularization** enables automatic feature selection
8. **Interpretable** - coefficients show feature importance

### 📋 Quick Reference

**Default setup (good starting point):**
```python
clf = LogisticRegression(random_state=42)
```

**For large datasets:**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
clf = LogisticRegression(solver='saga', max_iter=1000)
clf.fit(X_scaled, y)
```

**For feature selection:**
```python
clf = LogisticRegression(penalty='l1', solver='saga', C=0.1)
```

**For imbalanced data:**
```python
clf = LogisticRegression(class_weight='balanced')
```

**With hyperparameter tuning:**
```python
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
grid = GridSearchCV(LogisticRegression(solver='saga'), param_grid, cv=5)
grid.fit(X, y)
```

---

**End of Logistic Regression Notes - Module 03**
