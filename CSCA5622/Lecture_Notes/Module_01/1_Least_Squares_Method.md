# Least Squares Method - Detailed Lecture Notes
**CSCA5622 - Module 01**

---

## 📚 Overview

This document provides a comprehensive breakdown of the Least Squares Method, covering fundamental concepts essential for understanding linear regression in machine learning. Topics include:

- The effect of scaling on regression coefficients
- The normal equation for multivariate regression  
- Handling singular matrices with the pseudo-inverse
- Error metrics for evaluating regression performance

All concepts are explained with mathematical derivations, intuitive explanations, and practical Python examples.

---

## 1. Scaling Effects on Regression Coefficients

### 🔍 Core Concept

**Scaling** refers to changing the units or magnitude of your variables. This transformation directly affects regression coefficients, making it crucial to understand for proper model interpretation.

### 💡 Real-World Examples

#### Example 1: Housing Prices
- **Original**: `x` = house size in square feet, `y` = price in dollars
- **Scaled**: `x'` = house size in square meters (÷ 10.764), `y'` = price in thousands of dollars (÷ 1000)

#### Example 2: Temperature Prediction  
- **Original**: `x` = altitude in meters, `y` = temperature in Celsius
- **Scaled**: `x'` = altitude in kilometers (÷ 1000), `y'` = temperature in Fahrenheit (× 9/5 + 32)

### 🧮 Mathematical Relationships

Given scale factors:
- `r`: scale factor for x (e.g., r = 100 to convert meters → centimeters)
- `s`: scale factor for y (e.g., s = 0.000001 to convert dollars → millions)

**Transformed variables:**
- `x'` = r × x  
- `y'` = s × y

**Updated coefficients:**
```
β₁' = (s/r) × β₁
β₀' = s × (ȳ - β₁ × x̄)
```

### 📊 Practical Python Example

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Original data: house size (sq ft) vs price ($)
data = pd.DataFrame({
    'size_sqft': [1000, 1500, 2000, 2500, 3000],
    'price_dollars': [200000, 275000, 350000, 425000, 500000]
})

# Original regression
X_orig = data[['size_sqft']]
y_orig = data['price_dollars']
model_orig = LinearRegression().fit(X_orig, y_orig)

print(f"Original: β₁ = {model_orig.coef_[0]:.2f} $/sqft")
print(f"Original: β₀ = {model_orig.intercept_:.2f} $")

# Scale: sqft → sq meters (÷10.764), dollars → thousands (÷1000)
r = 1/10.764  # x scale factor  
s = 1/1000    # y scale factor

X_scaled = data[['size_sqft']] * r  # Convert to sq meters
y_scaled = data['price_dollars'] * s  # Convert to thousands

model_scaled = LinearRegression().fit(X_scaled, y_scaled)

print(f"Scaled: β₁ = {model_scaled.coef_[0]:.2f} k$/sq_m")
print(f"Scaled: β₀ = {model_scaled.intercept_:.2f} k$")

# Verify relationship: β₁' = (s/r) × β₁
expected_slope = (s/r) * model_orig.coef_[0]
print(f"Expected slope: {expected_slope:.2f}")
print(f"Actual slope: {model_scaled.coef_[0]:.2f}")
```

### 🧠 Key Insights

1. **Slope sensitivity**: Smaller input units → larger slope coefficients
2. **Intercept scaling**: Only depends on y-scaling, not x-scaling  
3. **Interpretation**: Always check units when comparing coefficients across models

---

## 2. The Normal Equation - Mathematical Foundation

### 🔍 Goal & Setup

Find the optimal coefficient vector **β̂** that minimizes the sum of squared residuals for the linear model:

**y = Xβ + ε**

Where:
- `y`: target vector (n × 1)
- `X`: design matrix (n × p), includes intercept column
- `β`: coefficient vector (p × 1)  
- `ε`: error vector (n × 1)

### 🧮 Step-by-Step Derivation

#### Step 1: Define the Objective Function
Minimize the **Residual Sum of Squares (RSS)**:
```
RSS(β) = ||y - Xβ||²
```

#### Step 2: Matrix Expansion
```
RSS(β) = (y - Xβ)ᵀ(y - Xβ)
       = yᵀy - yᵀXβ - βᵀXᵀy + βᵀXᵀXβ
       = yᵀy - 2βᵀXᵀy + βᵀXᵀXβ
```

*(Note: yᵀXβ = βᵀXᵀy since both are scalars)*

#### Step 3: Take the Gradient
```
∇RSS(β) = d/dβ [yᵀy - 2βᵀXᵀy + βᵀXᵀXβ]
        = 0 - 2Xᵀy + 2XᵀXβ
        = -2Xᵀy + 2XᵀXβ
```

#### Step 4: Set Gradient to Zero
```
-2Xᵀy + 2XᵀXβ = 0
XᵀXβ = Xᵀy
```

#### Step 5: Solve for β
```
β̂ = (XᵀX)⁻¹Xᵀy
```

### 🧠 Geometric Intuition

- **XᵀX**: Captures correlations between predictors
- **Xᵀy**: Captures how predictors relate to the target
- **Inverse operation**: "Untangles" correlations to isolate individual effects

### 🛠 Complete Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

def normal_equation_regression(X, y):
    """
    Implement linear regression using the normal equation.
    
    Parameters:
    X: Feature matrix (n_samples, n_features)
    y: Target vector (n_samples,)
    
    Returns:
    beta: Coefficient vector
    """
    # Add intercept column (column of ones)
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
    
    # Normal equation: β = (XᵀX)⁻¹Xᵀy
    XtX = X_with_intercept.T @ X_with_intercept
    Xty = X_with_intercept.T @ y
    beta = np.linalg.inv(XtX) @ Xty
    
    return beta

# Example usage
np.random.seed(42)
X = np.random.randn(100, 2)  # 2 features
true_beta = np.array([3, 1.5, -2])  # [intercept, coef1, coef2]
y = np.column_stack([np.ones(100), X]) @ true_beta + 0.1 * np.random.randn(100)

# Fit using normal equation
estimated_beta = normal_equation_regression(X, y)
print("True coefficients:", true_beta)
print("Estimated coefficients:", estimated_beta)

# Compare with sklearn
from sklearn.linear_model import LinearRegression
sklearn_model = LinearRegression().fit(X, y)
sklearn_beta = np.array([sklearn_model.intercept_] + list(sklearn_model.coef_))
print("Sklearn coefficients:", sklearn_beta)
```

---

## 3. Singular Matrices & The Pseudo-Inverse Solution

### ❓ What is a Singular Matrix?

A matrix is **singular** (non-invertible) when:
- Its determinant equals zero
- Its columns are linearly dependent  
- It doesn't have full rank

### 🚨 When Does This Happen in Regression?

#### Common Scenarios:

1. **Perfect Multicollinearity**:
   ```python
   # Feature 2 is exactly 2x Feature 1
   X = np.array([[1, 2], [2, 4], [3, 6], [4, 8]])
   ```

2. **More Features than Samples**:
   ```python
   # 5 features, only 3 samples → underdetermined system
   X = np.random.randn(3, 5)
   ```

3. **Duplicate Features**:
   ```python
   # Accidentally included the same feature twice
   X = np.array([[1, 1], [2, 2], [3, 3]])
   ```

### ⚠️ Problems with Singular Matrices

```python
# This will fail!
try:
    XtX = X.T @ X
    beta = np.linalg.inv(XtX) @ X.T @ y  # Error: Singular matrix
except np.linalg.LinAlgError:
    print("Cannot invert singular matrix!")
```

**Why it fails:**
- No unique solution exists
- Infinite possible coefficient combinations yield the same fit
- Model cannot determine individual feature importance

### 🛠 Solution: Moore-Penrose Pseudo-Inverse

The **pseudo-inverse** X⁺ provides the solution with minimum norm when multiple solutions exist.

**Mathematical definition:**
```
β̂ = X⁺y
```

Where X⁺ is computed via **Singular Value Decomposition (SVD)**:
```
X = UΣVᵀ
X⁺ = VΣ⁺Uᵀ
```

### 📊 Practical Implementation & Comparison

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Create problematic data (perfect multicollinearity)
n_samples = 100
X1 = np.random.randn(n_samples)
X2 = 2 * X1  # Perfect correlation!
X = np.column_stack([np.ones(n_samples), X1, X2])
y = 3 + 2*X1 + np.random.randn(n_samples) * 0.1

print("Matrix rank:", np.linalg.matrix_rank(X))
print("Matrix shape:", X.shape)
print("Is singular?", np.linalg.matrix_rank(X) < X.shape[1])

# Method 1: Normal equation (will fail)
try:
    beta_normal = np.linalg.inv(X.T @ X) @ X.T @ y
    print("Normal equation succeeded")
except np.linalg.LinAlgError:
    print("Normal equation failed (singular matrix)")

# Method 2: Pseudo-inverse (will work)
beta_pinv = np.linalg.pinv(X) @ y
print("Pseudo-inverse coefficients:", beta_pinv)

# Method 3: Sklearn (uses SVD internally)
sklearn_model = LinearRegression().fit(X[:, 1:], y)  # Exclude manual intercept
sklearn_beta = [sklearn_model.intercept_] + list(sklearn_model.coef_)
print("Sklearn coefficients:", sklearn_beta)

# Verify predictions are the same
y_pred_pinv = X @ beta_pinv
y_pred_sklearn = sklearn_model.predict(X[:, 1:])
print("Predictions match:", np.allclose(y_pred_pinv, y_pred_sklearn))
```

### 🧠 Key Properties of Pseudo-Inverse

1. **Always exists**: Even for non-square or singular matrices
2. **Minimum norm solution**: Among all solutions, picks the one with smallest ||β||
3. **Reduces to regular inverse**: When X is invertible, X⁺ = X⁻¹
4. **Used by default**: Most ML libraries use SVD-based methods

---

## 4. Error Metrics for Regression Evaluation

### 📏 Understanding Residuals

**Residuals** are the building blocks of all error metrics:
```
eᵢ = yᵢ - ŷᵢ   (actual - predicted)
```

### 📉 Mean Absolute Error (MAE)

**Formula:**
```
MAE = (1/n) Σ |yᵢ - ŷᵢ|
```

**Properties:**
- ✅ Same units as target variable
- ✅ Robust to outliers  
- ✅ Intuitive interpretation
- ❌ Not differentiable at zero

**When to use:** When you want to penalize all errors equally

### 🔲 Mean Squared Error (MSE)

**Formula:**
```  
MSE = (1/n) Σ (yᵢ - ŷᵢ)²
```

**Properties:**
- ✅ Differentiable (useful for optimization)
- ✅ Emphasizes large errors
- ❌ Units are squared (harder to interpret)
- ❌ Sensitive to outliers

**When to use:** When large errors are particularly problematic

### 🧮 Root Mean Squared Error (RMSE)

**Formula:**
```
RMSE = √MSE = √[(1/n) Σ (yᵢ - ŷᵢ)²]
```

**Properties:**
- ✅ Same units as target (like MAE)
- ✅ Penalizes large errors (like MSE)  
- ✅ Most popular regression metric
- ❌ Still sensitive to outliers

### 📊 Mean Absolute Percentage Error (MAPE)

**Formula:**
```
MAPE = (1/n) Σ |yᵢ - ŷᵢ|/|yᵢ| × 100%
```

**Properties:**
- ✅ Unit-free (percentage)
- ✅ Easy to interpret
- ❌ Undefined when yᵢ = 0
- ❌ Asymmetric (over-prediction vs under-prediction)

### 🎯 Practical Comparison Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Generate sample data with outliers
np.random.seed(42)
X = np.linspace(0, 10, 50).reshape(-1, 1)
y_true = 2 + 3*X.flatten() + np.random.normal(0, 1, 50)

# Add some outliers
outlier_indices = [10, 25, 40]
y_true[outlier_indices] += np.array([15, -12, 18])

# Fit model
model = LinearRegression().fit(X, y_true)
y_pred = model.predict(X)

# Calculate all metrics
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f"MAE:  {mae:.3f}")
print(f"MSE:  {mse:.3f}")  
print(f"RMSE: {rmse:.3f}")
print(f"MAPE: {mape:.3f}%")

# Visualize residuals
residuals = y_true - y_pred
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(X, y_true, alpha=0.7, label='Actual')
plt.plot(X, y_pred, 'r-', label='Predicted')
plt.legend()
plt.title('Predictions vs Actual')

plt.subplot(1, 2, 2)
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.tight_layout()
plt.show()

# Show impact of outliers
print("\n--- Impact of Outliers ---")
mask = np.ones(len(y_true), dtype=bool)
mask[outlier_indices] = False

mae_no_outliers = mean_absolute_error(y_true[mask], y_pred[mask])
mse_no_outliers = mean_squared_error(y_true[mask], y_pred[mask])
rmse_no_outliers = np.sqrt(mse_no_outliers)

print(f"MAE without outliers:  {mae_no_outliers:.3f} (vs {mae:.3f})")
print(f"RMSE without outliers: {rmse_no_outliers:.3f} (vs {rmse:.3f})")
```

---

## 5. Summary & Key Takeaways

| **Concept** | **Key Points** | **Practical Impact** |
|-------------|----------------|---------------------|
| **Scaling** | Changes coefficient values but not model fit | Critical for interpretation and feature comparison |
| **Normal Equation** | Closed-form solution: β̂ = (XᵀX)⁻¹Xᵀy | Fast for small datasets, foundations of linear regression |
| **Singular Matrices** | Occur with multicollinearity or p > n | Use pseudo-inverse; indicates data quality issues |
| **Error Metrics** | MAE (robust), MSE (differentiable), RMSE (interpretable) | Choose based on problem requirements and outlier sensitivity |

### 🧠 Decision Framework

**When to use Normal Equation:**
- Small datasets (n < 10,000)
- Need exact solution
- Learning/debugging purposes

**When to use Pseudo-Inverse:**  
- Multicollinearity present
- More features than samples
- Robust implementation needed

**Error Metric Selection:**
- **MAE**: Outliers present, equal error weighting desired
- **RMSE**: Standard choice, large errors are costly  
- **MAPE**: Need percentage-based interpretation

### 🚀 Next Steps

1. Practice implementing these concepts on real datasets
2. Explore regularization (Ridge/Lasso) for handling multicollinearity
3. Learn gradient descent as an alternative to normal equation
4. Study advanced error metrics (R², adjusted R²)

---

**📚 Further Reading:**
- Elements of Statistical Learning (Chapter 3)
- Pattern Recognition and Machine Learning (Bishop, Chapter 3)
- Scikit-learn documentation on Linear Models
