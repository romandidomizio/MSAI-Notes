# Least Squared Method - Lecture 2

## Overview
This lecture covers how linear regression finds the optimal coefficients using the **Least Squares Method**. We'll explore error metrics, the mathematical optimization process, scaling effects, and extension to multivariate regression.

---

## Table of Contents
1. [Finding the Coefficients](#finding-the-coefficients)
2. [Residuals Visualization](#residuals-visualization)
3. [Error Metrics](#error-metrics)
4. [Least Squares Optimization](#least-squares-optimization)
5. [Error Surface Visualization](#error-surface-visualization)
6. [Closed-Form Solution](#closed-form-solution)
7. [Practice Problem: Variable Scaling](#practice-problem-variable-scaling)
8. [Multivariate Least Squares](#multivariate-least-squares)
9. [Matrix Invertibility and Pseudo-Inverse](#matrix-invertibility-and-pseudo-inverse)

---

## Finding the Coefficients

> **"So again, this beta 0 beta 1 are coefficients and this is my model and the difference between the target value and the predicted value is called the residual."**

### Model Formulation

$$\hat{y} = \beta_0 + \beta_1 x$$

Where:
- $\beta_0$ = intercept (coefficient)
- $\beta_1$ = slope (coefficient)
- $\hat{y}$ = predicted value

### Residual Definition

$$\text{Residual: } \epsilon_i = y_i - \hat{y}_i$$

---

## Residuals Visualization

> **"So here is the plot that plots the residuals. So this is a residue that has positive value and these are the residues that have negative value."**

**Positive residuals**: Data points **above** the regression line (actual > predicted)  
**Negative residuals**: Data points **below** the regression line (actual < predicted)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate sample data
np.random.seed(42)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
y = np.array([2.5, 3.8, 4.2, 6.1, 7.3, 7.8, 9.5, 10.2])

# Fit model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X, y, s=100, alpha=0.6, color='blue', label='Actual data', zorder=3)
plt.plot(X, y_pred, 'r-', linewidth=2, label='Regression line', zorder=2)

# Plot residuals
for i in range(len(X)):
    color = 'green' if y[i] > y_pred[i] else 'orange'
    plt.plot([X[i], X[i]], [y[i], y_pred[i]], color=color, 
             linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
    # Annotate residual value
    mid_y = (y[i] + y_pred[i]) / 2
    residual_val = y[i] - y_pred[i]
    plt.text(X[i][0] + 0.1, mid_y, f'{residual_val:.2f}', fontsize=9)

plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Residuals: Positive (green) and Negative (orange)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Error Metrics

> **"So when you say how good my model is, that means how small is the error overall? So we need to find an error measure that accounts to all these residuals from all the points."**

### 1. Simple Sum (Not Useful)

$$\sum_{i=1}^{n} \epsilon_i = \sum_{i=1}^{n} (y_i - \hat{y}_i)$$

> **"However, it's going to be 0 all the times if the regression line was fit, so this is not very useful."**

**Problem**: Positive and negative residuals cancel out, always summing to approximately zero.

---

### 2. Mean Absolute Error (MAE)

> **"So, we're going to define another error measure that measures the distance instead of just summing all of these residuals."**

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**Characteristics**:
- Uses absolute values to prevent cancellation
- Measures average absolute distance from regression line
- Same units as target variable
- **Limitation**: Can be arbitrarily large depending on scale of Y

---

### 3. Mean Squared Error (MSE)

> **"Another way we can do it is we can maybe square each residuals and then sum them up. And also we can divide by N and this gives mean squared error."**

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Characteristics**:
- Squares residuals to prevent cancellation
- Penalizes large errors more heavily than MAE
- **Limitation**: Units are squared (e.g., dollars²)

> **"So these two are very popular error measuring regression tasks."**

---

### 4. Mean Absolute Percentage Error (MAPE)

> **"So, we talked about MAE but MAE can be arbitrary large depending on how large Ys are. Therefore, we can define percent absolute error instead."**

$$\text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100\%$$

**Advantages**:
- Scale-independent (percentage-based)
- Easy to interpret
- Useful for comparing models across different datasets

---

### 5. Root Mean Squared Error (RMSE)

> **"We talked about mean squared error, but mean squared error unit is different from Y's unit. So in case we want to compare in the same unit, we can take a square root, then it becomes a root mean square error, which is also good metric in regression."**

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

**Advantages**:
- Same units as target variable (fixes MSE's unit problem)
- Still penalizes large errors
- Commonly used in practice

---

### Python Example: Computing All Error Metrics

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Sample data
y_true = np.array([100, 150, 200, 250, 300])
y_pred = np.array([110, 140, 210, 240, 295])

# Calculate all error metrics
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

# MAPE (manual calculation)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print("="*60)
print("ERROR METRICS COMPARISON")
print("="*60)
print(f"Mean Absolute Error (MAE):           {mae:.4f}")
print(f"Mean Squared Error (MSE):            {mse:.4f}")
print(f"Root Mean Squared Error (RMSE):      {rmse:.4f}")
print(f"Mean Absolute Percentage Error:      {mape:.2f}%")
print("="*60)

# Show individual errors
print("\nIndividual Residuals:")
for i in range(len(y_true)):
    residual = y_true[i] - y_pred[i]
    abs_error = abs(residual)
    sq_error = residual**2
    pct_error = abs(residual / y_true[i]) * 100
    print(f"Point {i+1}: True={y_true[i]:>3}, Pred={y_pred[i]:>3}, "
          f"Residual={residual:>4}, |ε|={abs_error:>2}, "
          f"ε²={sq_error:>3}, %Error={pct_error:>5.2f}%")
```

---

## Least Squares Optimization

> **"All right, so let's talk about how the optimization in linear regression work. There could be various methods but this method called least squares method is the most popular and almost all Python package that solves a linear regression uses this method."**

### The Optimization Process

**Reminder of supervised learning workflow**:
1. Model takes **features** (X)
2. Has **internal parameters** (β₀, β₁)
3. Linear regression has **no hyperparameters**
4. Makes **predictions** (ŷ)
5. **Optimization** finds parameter values to make predictions as accurate as possible

### Least Squares Method

> **"And for this squared method, the linear regression mostly use takes the feature and target value and find a solution for the parameters. And the name suggests least squares because it uses a squared error."**

**Objective**: Minimize Mean Squared Error (MSE)

$$\min_{\beta_0, \beta_1} \text{MSE} = \min_{\beta_0, \beta_1} \frac{1}{n} \sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_i)^2$$

---

## Error Surface Visualization

> **"So we're going to use MSE and let's have a look what the error surface of MSE look like."**

### 3D Error Surface Description

> **"So in MSE, the error in the coefficient space where this axis is one of the coefficient and the other axis is the other coefficient and this axis represent the error, the size of the error. Then this takes a kind of bowl shape like this."**

**Structure**:
- **X-axis**: Coefficient A (e.g., β₀ or β₁)
- **Y-axis**: Coefficient B (e.g., β₁ or β₀)
- **Z-axis**: Error (MSE value)
- **Shape**: Bowl-shaped (convex)

### Different Views

> **"So if you look at from the top, the contour will look like an ellipsoid like this. And then if you look from the side, then it will look like parabola."**

**Top view (contours)**: Ellipsoid/elliptical contours  
**Side view**: Parabola shape

### The Minimum

> **"So it has some minimum value at the bottom of this bowl, so we would like to find a solution in one of the values for A and B at the bottom of this MSE surface."**

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate sample data
np.random.seed(42)
X_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([2, 4, 5, 4, 5])

# Create grid of coefficient values
beta0_range = np.linspace(-2, 6, 100)
beta1_range = np.linspace(-1, 3, 100)
Beta0, Beta1 = np.meshgrid(beta0_range, beta1_range)

# Calculate MSE for each combination of coefficients
MSE = np.zeros_like(Beta0)
for i in range(Beta0.shape[0]):
    for j in range(Beta0.shape[1]):
        y_pred = Beta0[i, j] + Beta1[i, j] * X_data
        MSE[i, j] = np.mean((y_data - y_pred)**2)

# Find optimal coefficients
min_idx = np.unravel_index(np.argmin(MSE), MSE.shape)
optimal_beta0 = Beta0[min_idx]
optimal_beta1 = Beta1[min_idx]
min_mse = MSE[min_idx]

# Create visualizations
fig = plt.figure(figsize=(16, 5))

# 3D surface plot
ax1 = fi.add_subplot(131, projection='3d')
surface = ax1.plot_surface(Beta0, Beta1, MSE, cmap='viridis', alpha=0.8)
ax1.scatter([optimal_beta0], [optimal_beta1], [min_mse], 
            color='red', s=200, marker='*', label='Minimum')
ax1.set_xlabel('β₀ (Intercept)', fontsize=10)
ax1.set_ylabel('β₁ (Slope)', fontsize=10)
ax1.set_zlabel('MSE', fontsize=10)
ax1.set_title('3D MSE Surface (Bowl Shape)', fontsize=12, fontweight='bold')
ax1.legend()

# Contour plot (top view)
ax2 = fig.add_subplot(132)
contour = ax2.contour(Beta0, Beta1, MSE, levels=20, cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8)
ax2.scatter([optimal_beta0], [optimal_beta1], 
            color='red', s=200, marker='*', label='Minimum', zorder=5)
ax2.set_xlabel('β₀ (Intercept)', fontsize=10)
ax2.set_ylabel('β₁ (Slope)', fontsize=10)
ax2.set_title('Contour Plot (Top View - Ellipsoid)', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Side view (parabola)
ax3 = fig.add_subplot(133)
# Fix beta1 at optimal value and plot MSE vs beta0
mse_slice = MSE[min_idx[0], :]
ax3.plot(beta1_range, mse_slice, 'b-', linewidth=2)
ax3.scatter([optimal_beta1], [min_mse], 
            color='red', s=200, marker='*', label='Minimum')
ax3.set_xlabel('β₁ (Slope)', fontsize=10)
ax3.set_ylabel('MSE', fontsize=10)
ax3.set_title('Side View (Parabola Shape)', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Optimal coefficients: β₀ = {optimal_beta0:.4f}, β₁ = {optimal_beta1:.4f}")
print(f"Minimum MSE: {min_mse:.4f}")
```

---

## Closed-Form Solution

> **"To find out the minimum value of the MSE, we'll take a derivative with respect to each of coefficient."**

### Mathematical Approach

$$\frac{\partial \text{MSE}}{\partial \beta_0} = 0$$

$$\frac{\partial \text{MSE}}{\partial \beta_1} = 0$$

> **"Why we do that? If you think about the parabola shape at the bottom, the slope or the gradient becomes 0, so we'll use that fact."**

**Key Insight**: At the minimum of a parabola, the derivative (slope) equals zero.

### The Solutions

> **"And if we do the algebra, we're going to get the solution without derivation, but you can look at the supplemental node that has all the derivation."**

#### Slope Formula

$$\beta_1 = \frac{\text{Cov}(X, Y)}{\text{Var}(X)}$$

> **"Important thing to remember is that the slope is proportional to the covariance of the variable X and Y and then inversely proportional to the variance of X."**

Where:
- $\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]$
- $\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]$

#### Intercept Formula

$$\beta_0 = \mathbb{E}[Y] - \beta_1 \mathbb{E}[X]$$

> **"And similarly, intercept has this relation. And if you look at carefully, this suggests that actually the regression line passes through the center which is mean of X and comma mean of Y."**

**Key Property**: The regression line **always passes through** the point $(\bar{x}, \bar{y})$

### Python Implementation

```python
import numpy as np

# Sample data
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Manual calculation using formulas
X_mean = np.mean(X)
y_mean = np.mean(y)

# Covariance and Variance
cov_xy = np.mean((X - X_mean) * (y - y_mean))
var_x = np.mean((X - X_mean)**2)

# Coefficients
beta_1 = cov_xy / var_x
beta_0 = y_mean - beta_1 * X_mean

print("="*60)
print("MANUAL CALCULATION OF COEFFICIENTS")
print("="*60)
print(f"Mean of X: {X_mean:.4f}")
print(f"Mean of Y: {y_mean:.4f}")
print(f"Cov(X, Y): {cov_xy:.4f}")
print(f"Var(X):    {var_x:.4f}")
print(f"\nβ₁ (slope) = Cov(X,Y) / Var(X) = {cov_xy:.4f} / {var_x:.4f} = {beta_1:.4f}")
print(f"β₀ (intercept) = ȳ - β₁·x̄ = {y_mean:.4f} - {beta_1:.4f}·{X_mean:.4f} = {beta_0:.4f}")
print(f"\nRegression line: ŷ = {beta_0:.4f} + {beta_1:.4f}x")
print(f"Line passes through point ({X_mean:.4f}, {y_mean:.4f}) ✓")
print("="*60)

# Verify with sklearn
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X.reshape(-1, 1), y)
print(f"\nVerification with sklearn:")
print(f"β₀ = {model.intercept_:.4f}")
print(f"β₁ = {model.coef_[0]:.4f}")
```

---

## Practice Problem: Variable Scaling

> **"What happens when we change the scale of the variables?"**

### Problem Setup

> **"So for example, we have a big value for living space square foot and a really big value for sales price. So we want to change the unit, for example, makes it million dollar as a unit and use a small number and maybe we can divide by 1,000 for the square foot living."**

**Original variables**:
- X: Living space (square feet) - large numbers
- Y: Sales price (dollars) - very large numbers

**Scaled variables**:
- X': X / 1000 (in thousands of square feet)
- Y': Y / 1,000,000 (in millions of dollars)

**Scaling factors**:
- r = 1/1000 = 10⁻³ for X
- s = 1/1,000,000 = 10⁻⁶ for Y

> **"So in that case, if we change this by 1 over 1000 of original value and this is 10 to the -6 of original value, what happens to my value for beta1 and beta0? You can think about it for a while."**

---

### Solution: Effect on β₁ (Slope)

> **"All right, we're back. So we're going to call it r and we're going to call this as s."**

Let:
- r = scaling factor for X
- s = scaling factor for Y

#### Step 1: Recall the formulas

$$\beta_1 = \frac{\text{Cov}(X, Y)}{\text{Var}(X)}$$

> **"And as you know, covariance is calculated by this formula x - x mean times y - y mean, an expectation of this value. And then similarly, the expectation of x - x mean squared."**

$$\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]$$

$$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]$$

#### Step 2: Apply scaling

When X → rX and Y → sY:

$$\text{Cov}(rX, sY) = \mathbb{E}[(rX - r\mathbb{E}[X])(sY - s\mathbb{E}[Y])]$$
$$= \mathbb{E}[r(X - \mathbb{E}[X]) \cdot s(Y - \mathbb{E}[Y])]$$
$$= rs \cdot \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]$$
$$= rs \cdot \text{Cov}(X, Y)$$

$$\text{Var}(rX) = \mathbb{E}[(rX - r\mathbb{E}[X])^2]$$
$$= \mathbb{E}[r^2(X - \mathbb{E}[X])^2]$$
$$= r^2 \cdot \text{Var}(X)$$

#### Step 3: Calculate new slope

> **"So, this part is scaled by r, and this part is scaled by s, and this part scaled by r squared."**

$$\beta_1' = \frac{\text{Cov}(rX, sY)}{\text{Var}(rX)} = \frac{rs \cdot \text{Cov}(X, Y)}{r^2 \cdot \text{Var}(X)}$$

$$= \frac{s}{r} \cdot \frac{\text{Cov}(X, Y)}{\text{Var}(X)} = \frac{s}{r} \cdot \beta_1$$

> **"So as a result, we're going to get s divided by r times original value of beta 1."**

#### Step 4: Plug in values

> **"So if we plug these numbers, we're going to get 10 to the -3 times original value for beta 1."**

$$\beta_1' = \frac{s}{r} \cdot \beta_1 = \frac{10^{-6}}{10^{-3}} \cdot \beta_1 = 10^{-3} \cdot \beta_1$$

**Result**: 
> **"So my slope get 1,000 times smaller, if I make my x variable 1,000 times smaller and make my y variable a million times smaller."**

$$\boxed{\beta_1' = 0.001 \cdot \beta_1}$$

---

### Solution: Effect on β₀ (Intercept)

> **"What happens to beta 0, my intercept?"**

#### Original formula:

$$\beta_0 = \mathbb{E}[Y] - \beta_1 \mathbb{E}[X]$$

#### After scaling:

$$\beta_0' = \mathbb{E}[sY] - \beta_1' \mathbb{E}[rX]$$

> **"So this is going to be s times original value of E[Y]."**

$$= s \cdot \mathbb{E}[Y] - \beta_1' \cdot r \cdot \mathbb{E}[X]$$

> **"And this, we already calculated it's going to be s over r times the original value, which I'm going to just say."**

$$= s \cdot \mathbb{E}[Y] - \left(\frac{s}{r}\beta_1\right) \cdot r \cdot \mathbb{E}[X]$$

> **"And then this quantity becomes r times the E[X], so this cancels out."**

$$= s \cdot \mathbb{E}[Y] - s \cdot \beta_1 \cdot \mathbb{E}[X]$$

$$= s(\mathbb{E}[Y] - \beta_1 \mathbb{E}[X])$$

$$= s \cdot \beta_0$$

> **"And then we're going to get s times original value beta 0."**

**Result**:
> **"So my intercept doesn't change when I scale the X, however, it's going to change when I scale the Y, and it only depends on the scaling of the Y."**

$$\boxed{\beta_0' = s \cdot \beta_0 = 10^{-6} \cdot \beta_0}$$

---

### Summary of Scaling Effects

| Coefficient | Original | After Scaling | Factor |
|------------|----------|---------------|---------|
| **Slope (β₁)** | β₁ | s/r · β₁ | 10⁻⁶/10⁻³ = 10⁻³ |
| **Intercept (β₀)** | β₀ | s · β₀ | 10⁻⁶ |

**Key Takeaways**:
- β₁ depends on **both** X and Y scaling (ratio s/r)
- β₀ depends **only** on Y scaling (factor s)
- β₀ is **independent** of X scaling

### Python Verification

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Original data (large numbers)
X_original = np.array([1000, 1500, 2000, 2500, 3000]).reshape(-1, 1)
y_original = np.array([200000, 350000, 400000, 550000, 650000])

# Fit original model
model_original = LinearRegression()
model_original.fit(X_original, y_original)
beta_0_orig = model_original.intercept_
beta_1_orig = model_original.coef_[0]

# Scaled data
r = 1/1000  # X scaling (to thousands)
s = 1/1_000_000  # Y scaling (to millions)

X_scaled = X_original * r
y_scaled = y_original * s

# Fit scaled model
model_scaled = LinearRegression()
model_scaled.fit(X_scaled, y_scaled)
beta_0_scaled = model_scaled.intercept_
beta_1_scaled = model_scaled.coef_[0]

# Theoretical predictions
beta_1_theory = (s/r) * beta_1_orig
beta_0_theory = s * beta_0_orig

print("="*70)
print("SCALING VERIFICATION")
print("="*70)
print(f"Original Model: ŷ = {beta_0_orig:.2f} + {beta_1_orig:.2f}·x")
print(f"Scaled Model:   ŷ = {beta_0_scaled:.6f} + {beta_1_scaled:.6f}·x")
print("\n" + "-"*70)
print("SLOPE (β₁) ANALYSIS:")
print("-"*70)
print(f"Original β₁:              {beta_1_orig:.4f}")
print(f"Scaled β₁ (actual):       {beta_1_scaled:.6f}")
print(f"Theoretical β₁ (s/r·β₁):  {beta_1_theory:.6f}")
print(f"Scaling factor (s/r):     {s/r:.6f} = 10⁻³")
print(f"Match: {np.isclose(beta_1_scaled, beta_1_theory)}")

print("\n" + "-"*70)
print("INTERCEPT (β₀) ANALYSIS:")
print("-"*70)
print(f"Original β₀:              {beta_0_orig:.2f}")
print(f"Scaled β₀ (actual):       {beta_0_scaled:.6f}")
print(f"Theoretical β₀ (s·β₀):    {beta_0_theory:.6f}")
print(f"Scaling factor (s):       {s:.6f} = 10⁻⁶")
print(f"Match: {np.isclose(beta_0_scaled, beta_0_theory)}")
print("="*70)
```

---

## Multivariate Least Squares

> **"So let's talk about how we generalize the least squares method to multivariate case."**

### Multiple Features

> **"So when we have P number of features, this is a feature matrix."**

When we have **p features**, our data is organized as:

**Feature matrix X** (n × p):
```
        X₁   X₂   ...  Xₚ
    ┌                    ┐
    │  x₁₁  x₁₂  ... x₁ₚ │  ← sample 1
    │  x₂₁  x₂₂  ... x₂ₚ │  ← sample 2
    │   ⋮    ⋮    ⋱   ⋮  │
    │  xₙ₁  xₙ₂  ... xₙₚ │  ← sample n
    └                    ┘
```

### Design Matrix

> **"And we add a column that has a ones, so that it can take care of the intercept term. So together with this, this total metrics is called design metrics."**

**Design matrix** (n × (p+1)):
```
        1    X₁   X₂   ...  Xₚ
    ┌                         ┐
    │  1   x₁₁  x₁₂  ... x₁ₚ │
    │  1   x₂₁  x₂₂  ... x₂ₚ │
    │  ⋮    ⋮    ⋮    ⋱   ⋮  │
    │  1   xₙ₁  xₙ₂  ... xₙₚ │
    └                         ┘
```

> **"And this index 1 to n, is for the sample index and this 0 to p is for the feature index including the intercept."**

**Coefficient vector β**:
$$\beta = \begin{bmatrix} \beta_0 \\ \beta_1 \\ \beta_2 \\ \vdots \\ \beta_p \end{bmatrix}$$

---

### Matrix Formulation

> **"So, MSE in matrix form is going to look like this, y - x beta, and these are all matrices."**

$$\text{MSE} = \frac{1}{n} \|y - X\beta\|_2^2$$

> **"And then two norm of the matrices, that is actually the y - x beta transpose and y - x beta."**

$$= \frac{1}{n}(y - X\beta)^T(y - X\beta)$$

---

### Normal Equations

> **"So when we take a derivative with respect to beta, then we're going to get this equation."**

Taking the derivative and setting to zero:

$$\frac{\partial \text{MSE}}{\partial \beta} = 0$$

> **"And if you further simplify it will look like this. And this is called a normal equation."**

$$X^T X \beta = X^T y$$

**This is called the Normal Equation.**

---

### Closed-Form Solution

> **"And then solving this equation for beta, it gives a solution like this."**

$$\boxed{\beta = (X^T X)^{-1} X^T y}$$

This is the **closed-form solution** for multivariate linear regression.

---

## Matrix Invertibility and Pseudo-Inverse

> **"So it involves an inverse of these metrics inside. And sometimes it can be a problem if the rank of this matrix X T and X are not equal to N."**

### When Is $(X^TX)$ Non-Invertible?

> **"And when does it happen? It happens when there are two or more variables or the features are linearly correlated."**

**Problem**: Multicollinearity - when features are linearly dependent

### Example of Linear Dependence

> **"So for example, if my X1 values were 1, 2, 3 and some of the other feature, let's say X5 was linearly dependent on X1, so for example, two times of this, something like that, then these two features are redundant."**

```
X₁: [1, 2, 3, 4, 5]
X₅: [2, 4, 6, 8, 10]  ← X₅ = 2·X₁ (linearly dependent!)
```

> **"Therefore, this metrics becomes a non-invertible."**

When $X_5 = 2 \cdot X_1$, the matrix $X^TX$ becomes **singular** (non-invertible).

---

### Consequences

> **"And then there is a problem when we try to get the solution beta. It actually doesn't mean that we don't have solution, it means that we have a solution that are not unique."**

**Key Point**: 
- Solutions **exist**
- Solutions are **not unique**
- Hard to determine which solution to use

---

### The Solution: Pseudo-Inverse

> **"So we're going to have a hard time to determine unique solution. But anyway, almost all Python packages that solves the ordinary least squares, OLS, has some mechanism to find the inverse metrics of this called pseudo inverse."**

**Pseudo-inverse** (also called Moore-Penrose Inverse):
$$X^+ = (X^TX)^{-1}X^T$$

When $X^TX$ is singular, the pseudo-inverse finds a solution.

> **"And sometimes this is called Moore-Penrose, Inverse."**

> **"So with this, we don't have to worry about non-inverted matrices."**

---

### Python Example: Pseudo-Inverse

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Create data with linear dependence
X1 = np.array([1, 2, 3, 4, 5])
X2 = np.array([2, 3, 4, 5, 6])
X3 = 2 * X1  # Linearly dependent on X1!

X = np.column_stack([X1, X2, X3])
y = np.array([2, 4, 5, 4, 5])

print("="*70)
print("MULTICOLLINEARITY EXAMPLE")
print("="*70)
print("Feature matrix X:")
print(X)
print(f"\nNote: X3 = 2·X1 (linearly dependent)")

# Check if XTX is invertible
XTX = X.T @ X
print(f"\nRank of X: {np.linalg.matrix_rank(X)}")
print(f"Number of features: {X.shape[1]}")
print(f"Determinant of X^T X: {np.linalg.det(XTX):.10f}")
print(f"Matrix is singular (non-invertible): {np.linalg.det(XTX) < 1e-10}")

# Try to compute pseudo-inverse
X_pinv = np.linalg.pinv(X.T @ X) @ X.T
beta_manual = X_pinv @ y

print(f"\nUsing pseudo-inverse:")
print(f"Coefficients: {beta_manual}")

# sklearn handles this automatically
model = LinearRegression()
model.fit(X, y)
print(f"\nsklearn (uses pseudo-inverse internally):")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

print("\n" + "="*70)
print("sklearn automatically handles multicollinearity using pseudo-inverse!")
print("="*70)
```

---

## Summary

**Key Concepts Covered**:

✅ **Residuals**: Difference between actual and predicted values (positive and negative)

✅ **Error Metrics**:
- MAE: Mean Absolute Error
- MSE: Mean Squared Error (most common for optimization)
- MAPE: Mean Absolute Percentage Error
- RMSE: Root Mean Squared Error

✅ **Least Squares Method**: Minimizes MSE to find optimal coefficients

✅ **Error Surface**: Bowl-shaped (convex), with minimum at optimal coefficients

✅ **Closed-Form Solutions**:
- $\beta_1 = \frac{\text{Cov}(X,Y)}{\text{Var}(X)}$
- $\beta_0 = \bar{y} - \beta_1\bar{x}$
- Regression line passes through $(\bar{x}, \bar{y})$

✅ **Variable Scaling Effects**:
- β₁ scales by s/r (both X and Y)
- β₀ scales by s only (Y only)

✅ **Multivariate Case**:
- Normal equations: $X^TX\beta = X^Ty$
- Solution: $\beta = (X^TX)^{-1}X^Ty$

✅ **Pseudo-Inverse**: Handles multicollinearity when $X^TX$ is singular

---

## Key Formulas Reference

| Concept | Formula |
|---------|---------|
| **Residual** | $\epsilon_i = y_i - \hat{y}_i$ |
| **MAE** | $\frac{1}{n}\sum_{i=1}^{n}\|y_i - \hat{y}_i\|$ |
| **MSE** | $\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$ |
| **RMSE** | $\sqrt{\text{MSE}}$ |
| **MAPE** | $\frac{1}{n}\sum_{i=1}^{n}\left\|\frac{y_i - \hat{y}_i}{y_i}\right\| \times 100\%$ |
| **Slope** | $\beta_1 = \frac{\text{Cov}(X,Y)}{\text{Var}(X)}$ |
| **Intercept** | $\beta_0 = \bar{y} - \beta_1\bar{x}$ |
| **Normal Equation** | $X^TX\beta = X^Ty$ |
| **OLS Solution** | $\beta = (X^TX)^{-1}X^Ty$ |

---

**End of Lecture 2 Notes**
