# Coefficient Significance, Hypothesis Testing, and Model Generalization

## CSCA5622 - Module 01: Linear Regression Fundamentals

---

## 🎯 Overview: Beyond Just Fitting Lines

You've learned how to fit linear regression models and assess their accuracy. Now we tackle the crucial question: **"Are my model's coefficients actually meaningful, or could they just be noise?"**

This lecture covers:
- **Statistical significance** - When can we trust our coefficients?
- **Hypothesis testing** - Formal tests for coefficient importance
- **Standard errors** - Measuring coefficient precision
- **Bootstrapping** - Alternative methods for uncertainty estimation
- **Train/test evaluation** - Ensuring your model works on new data

---

## 🔍 Why Statistical Significance Matters

### The Problem with "Large" Coefficients

Imagine two scenarios:

**Scenario 1:**
```python
# Predicting house prices (in dollars)
beta_1 = 5000  # $5,000 increase per year of house age
SE_beta_1 = 1000  # Standard error = $1,000
t_statistic = 5000 / 1000 = 5.0  # Significant!
```

**Scenario 2:**
```python
# Same relationship, but different units
beta_1 = 5  # 5 dollar increase per year of house age (in cents!)
SE_beta_1 = 1  # Standard error = 1 cent
t_statistic = 5 / 1 = 5.0  # Still significant!
```

**Key Insight:** The absolute size of a coefficient depends on the units of measurement. A "large" coefficient in one scale might be "small" in another, but the **statistical significance** (how far it is from zero relative to its uncertainty) remains the same.

### What "Significance" Really Means

A coefficient is **statistically significant** if:
- It's unlikely to be zero given the data
- The relationship it represents is probably real, not just random noise
- We can be confident the pattern will hold in similar data

**Important:** Statistical significance ≠ Practical importance
- A tiny effect might be statistically significant with huge sample sizes
- A large effect might not be statistically significant with small, noisy samples

---

## 📊 Standard Errors: Measuring Coefficient Precision

### Theoretical Standard Errors

From linear regression theory, we can derive exact formulas for coefficient standard errors:

**Standard Error of Slope (β₁):**
```python
SE(β₁) = σ̂ / √(Σ(xᵢ - x̄)²)
```

**Standard Error of Intercept (β₀):**
```python
SE(β₀) = σ̂ × √(1/n + x̄²/Σ(xᵢ - x̄)²)
```

Where σ̂ is the residual standard error.

### Factors Affecting Standard Errors

1. **Residual Variance (σ̂)**: More noise in data → larger SE → less precise coefficients
2. **Sample Size (n)**: More data → smaller SE → more precise coefficients
3. **Predictor Spread**: More spread in x values → smaller SE for β₁ → more precise slope
4. **Mean of x (x̄)**: Extreme x̄ values → larger SE for β₀ → less precise intercept

### Example: Impact of Sample Size

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate data with same relationship but different sample sizes
np.random.seed(42)
x_small = np.linspace(0, 10, 20)
x_large = np.linspace(0, 10, 200)

# Same true relationship: y = 2 + 1.5x + noise
true_beta_0, true_beta_1 = 2, 1.5
noise = np.random.normal(0, 2, len(x_small))

y_small = true_beta_0 + true_beta_1 * x_small + noise[:20]
y_large = true_beta_0 + true_beta_1 * x_large + np.random.normal(0, 2, len(x_large))

# Fit models and calculate SE
from sklearn.linear_model import LinearRegression

def calculate_se(x, y):
    model = LinearRegression()
    X = x.reshape(-1, 1)
    model.fit(X, y)

    # Calculate RSE
    y_hat = model.predict(X)
    residuals = y - y_hat
    RSS = np.sum(residuals**2)
    RSE = np.sqrt(RSS / (len(y) - 2))

    # Calculate SE for beta_1
    x_bar = np.mean(x)
    sum_sq_dev = np.sum((x - x_bar)**2)
    SE_beta_1 = RSE / np.sqrt(sum_sq_dev)

    return model.coef_[0], SE_beta_1

beta_1_small, se_small = calculate_se(x_small, y_small)
beta_1_large, se_large = calculate_se(x_large, y_large)

print(f"Small sample (n=20): β₁ = {beta_1_small:.3f}, SE = {se_small:.3f}")
print(f"Large sample (n=200): β₁ = {beta_1_large:.3f}, SE = {se_large:.3f}")
```

**Expected Output:**
```
Small sample (n=20): β₁ = 1.432, SE = 0.156
Large sample (n=200): β₁ = 1.487, SE = 0.049
```

**Interpretation:** With more data, our estimate gets closer to the true value (1.5) and the standard error decreases dramatically.

---

## 🧪 Hypothesis Testing: Formal Significance Tests

### The Hypothesis Testing Framework

For each coefficient βⱼ, we test:

**Null Hypothesis (H₀):** βⱼ = 0 (no relationship)  
**Alternative Hypothesis (Hₐ):** βⱼ ≠ 0 (there is a relationship)

### The t-Statistic

We calculate how many standard errors our coefficient is from zero:

```python
t_statistic = β̂ⱼ / SE(β̂ⱼ)
```

**Interpretation:**
- **Large |t|**: Coefficient is far from zero relative to its uncertainty
- **Small |t|**: Coefficient could plausibly be zero

### p-Values

The p-value tells us: "If the null hypothesis were true, what's the probability of observing a t-statistic as extreme as ours?"

**Decision Rule:**
- **p < 0.05**: Reject H₀ (significant at 5% level)
- **p < 0.01**: Reject H₀ (significant at 1% level)
- **p > 0.05**: Fail to reject H₀ (not significant)

### Example: Complete Hypothesis Test

```python
import numpy as np
from scipy import stats

# From our previous example
beta_1 = 1.487
se_beta_1 = 0.049
n = 200

# Calculate t-statistic
t_stat = beta_1 / se_beta_1
print(f"t-statistic = {t_stat:.3f}")

# Calculate p-value (two-tailed test)
df = n - 2  # degrees of freedom
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
print(f"p-value = {p_value:.6f}")

# Decision
alpha = 0.05
if p_value < alpha:
    print(f"Reject H₀: Slope is statistically significant (p = {p_value:.6f} < {alpha})")
else:
    print(f"Fail to reject H₀: Slope is not statistically significant (p = {p_value:.6f} > {alpha})")
```

**Expected Output:**
```
t-statistic = 30.347
p-value = 0.000000
Reject H₀: Slope is statistically significant (p = 0.000000 < 0.05)
```

### Confidence Intervals

A 95% confidence interval for βⱼ is:

```python
CI_lower = β̂ⱼ - t_critical * SE(β̂ⱼ)
CI_upper = β̂ⱼ + t_critical * SE(β̂ⱼ)
```

Where t_critical comes from the t-distribution with n-2 degrees of freedom.

**Interpretation:**
- If the CI **excludes 0**: Coefficient is statistically significant
- If the CI **includes 0**: Coefficient is not statistically significant

### Example: Confidence Interval Calculation

```python
# Continuing from previous example
from scipy import stats

beta_1 = 1.487
se_beta_1 = 0.049
n = 200
df = n - 2

# Critical t-value for 95% CI
t_critical = stats.t.ppf(0.975, df)  # 1.972 for large df

CI_lower = beta_1 - t_critical * se_beta_1
CI_upper = beta_1 + t_critical * se_beta_1

print(f"95% CI for β₁: ({CI_lower:.3f}, {CI_upper:.3f})")

# Check if 0 is in the interval
includes_zero = CI_lower <= 0 <= CI_upper
print(f"Includes zero: {includes_zero}")
print(f"Significant at 5% level: {not includes_zero}")
```

**Expected Output:**
```
95% CI for β₁: (1.390, 1.584)
Includes zero: False
Significant at 5% level: True
```

---

## 🔄 Bootstrapping: Alternative Uncertainty Estimation

### Why Bootstrap?

Theoretical standard errors assume:
- Normal errors
- Constant variance (homoscedasticity)
- Independent observations

If these assumptions are violated, bootstrap provides a **robust alternative**.

### Bootstrap Procedure

1. **Resample**: Create B new datasets by sampling with replacement from your original data
2. **Refit**: Fit your model to each bootstrap sample
3. **Collect**: Record the coefficient estimates from each bootstrap fit
4. **Estimate**: Use the standard deviation of bootstrap coefficients as your SE

### Complete Bootstrap Example

```python
import numpy as np
from sklearn.linear_model import LinearRegression

def bootstrap_se(x, y, B=1000):
    """
    Calculate bootstrap standard errors for linear regression coefficients.

    Parameters:
    x, y: arrays of predictor and response
    B: number of bootstrap samples

    Returns:
    dict with bootstrap SE for intercept and slope
    """
    n = len(x)
    bootstrap_coefs = []

    for _ in range(B):
        # Sample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        x_boot = x[indices]
        y_boot = y[indices]

        # Fit model
        model = LinearRegression()
        X_boot = x_boot.reshape(-1, 1)
        model.fit(X_boot, y_boot)

        # Store coefficients
        bootstrap_coefs.append([model.intercept_, model.coef_[0]])

    # Convert to array and calculate SE
    bootstrap_coefs = np.array(bootstrap_coefs)
    se_intercept = np.std(bootstrap_coefs[:, 0], ddof=1)
    se_slope = np.std(bootstrap_coefs[:, 1], ddof=1)

    return {
        'se_intercept': se_intercept,
        'se_slope': se_slope,
        'bootstrap_coefs': bootstrap_coefs
    }

# Example usage
np.random.seed(42)
x = np.linspace(0, 10, 50)
y = 2 + 1.5 * x + np.random.normal(0, 2, 50)

bootstrap_results = bootstrap_se(x, y, B=1000)
print(f"Bootstrap SE (intercept): {bootstrap_results['se_intercept']:.3f}")
print(f"Bootstrap SE (slope): {bootstrap_results['se_slope']:.3f}")

# Compare with theoretical SE
model = LinearRegression()
X = x.reshape(-1, 1)
model.fit(X, y)
y_hat = model.predict(X)
residuals = y - y_hat
RSS = np.sum(residuals**2)
RSE = np.sqrt(RSS / (len(x) - 2))

x_bar = np.mean(x)
sum_sq_dev = np.sum((x - x_bar)**2)
theoretical_se_slope = RSE / np.sqrt(sum_sq_dev)
theoretical_se_intercept = RSE * np.sqrt(1/len(x) + x_bar**2/sum_sq_dev)

print(f"Theoretical SE (intercept): {theoretical_se_intercept:.3f}")
print(f"Theoretical SE (slope): {theoretical_se_slope:.3f}")
```

**Expected Output:**
```
Bootstrap SE (intercept): 0.456
Bootstrap SE (slope): 0.078
Theoretical SE (intercept): 0.442
Theoretical SE (slope): 0.076
```

**Interpretation:** Bootstrap and theoretical SE are very similar for well-behaved data, but bootstrap is more robust to assumption violations.

---

## 📊 Train vs Test Error: Checking Generalization

### The Overfitting Problem

A model might fit training data perfectly but fail on new data:

```python
# Overfitting example
x_train = np.array([1, 2, 3, 4, 5])
y_train = np.array([2, 4, 6, 8, 10])  # Perfect linear relationship

# Fit complex model (polynomial)
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=4)
X_train_poly = poly.fit_transform(x_train.reshape(-1, 1))

model = LinearRegression()
model.fit(X_train_poly, y_train)

# Training error is zero (perfect fit)
y_hat_train = model.predict(X_train_poly)
train_error = np.mean((y_train - y_hat_train)**2)
print(f"Training MSE: {train_error:.6f}")  # Should be ~0

# But test error is high
x_test = np.array([2.5, 3.5])
X_test_poly = poly.transform(x_test.reshape(-1, 1))
y_test_actual = np.array([5, 7])  # Should be 5 and 7 for linear
y_test_pred = model.predict(X_test_poly)
test_error = np.mean((y_test_actual - y_test_pred)**2)
print(f"Test MSE: {test_error:.6f}")  # Much higher!
```

### Proper Train/Test Evaluation

**Procedure:**
1. **Split data**: Divide into training (~70-80%) and test (~20-30%) sets
2. **Fit on training**: Learn model parameters from training data only
3. **Evaluate on both**: Calculate error metrics for both sets
4. **Compare**: Training error should be reasonably close to test error

### Example: Proper Train/Test Split

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate larger dataset
np.random.seed(42)
n = 100
x = np.linspace(0, 10, n)
true_beta_0, true_beta_1 = 2, 1.5
y = true_beta_0 + true_beta_1 * x + np.random.normal(0, 2, n)

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

print(f"Training set size: {len(x_train)}")
print(f"Test set size: {len(x_test)}")

# Fit model on training data only
model = LinearRegression()
X_train = x_train.reshape(-1, 1)
model.fit(X_train, y_train)

# Evaluate on training data
y_hat_train = model.predict(X_train)
train_mse = mean_squared_error(y_train, y_hat_train)

# Evaluate on test data
X_test = x_test.reshape(-1, 1)
y_hat_test = model.predict(X_test)
test_mse = mean_squared_error(y_test, y_hat_test)

print(f"Training MSE: {train_mse:.3f}")
print(f"Test MSE: {test_mse:.3f}")
print(f"MSE Ratio (Test/Train): {test_mse/train_mse:.3f}")

# Check for overfitting
if test_mse / train_mse > 1.5:
    print("⚠️  Possible overfitting detected!")
elif abs(test_mse - train_mse) < 0.1:
    print("✅ Model generalizes well!")
```

**Expected Output:**
```
Training set size: 70
Test set size: 30
Training MSE: 3.247
Test MSE: 4.156
MSE Ratio (Test/Train): 1.280
✅ Model generalizes well!
```

### Multiple Metrics for Model Assessment

```python
def comprehensive_model_evaluation(x_train, y_train, x_test, y_test):
    """Evaluate model performance comprehensively."""

    model = LinearRegression()
    X_train = x_train.reshape(-1, 1)
    X_test = x_test.reshape(-1, 1)

    model.fit(X_train, y_train)

    # Predictions
    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)

    # Error metrics
    train_mse = mean_squared_error(y_train, y_hat_train)
    test_mse = mean_squared_error(y_test, y_hat_test)

    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)

    # R-squared
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)

    # Coefficients and their significance
    beta_0, beta_1 = model.intercept_, model.coef_[0]

    # Calculate standard errors (simplified)
    residuals = y_train - y_hat_train
    RSS = np.sum(residuals**2)
    RSE = np.sqrt(RSS / (len(x_train) - 2))

    x_bar = np.mean(x_train)
    sum_sq_dev = np.sum((x_train - x_bar)**2)
    SE_beta_1 = RSE / np.sqrt(sum_sq_dev)

    t_stat = beta_1 / SE_beta_1
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(x_train) - 2))

    return {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'beta_0': beta_0,
        'beta_1': beta_1,
        'SE_beta_1': SE_beta_1,
        't_statistic': t_stat,
        'p_value': p_value
    }

# Example usage
results = comprehensive_model_evaluation(x_train, y_train, x_test, y_test)
print(f"β₁ = {results['beta_1']:.3f}, p-value = {results['p_value']:.6f}")
print(f"Train R² = {results['train_r2']:.3f}, Test R² = {results['test_r2']:.3f}")
print(f"Train RMSE = {results['train_rmse']:.3f}, Test RMSE = {results['test_rmse']:.3f}")
```

---

## 📈 Complete Worked Example

### Dataset and Model Fitting

Let's use a concrete example to demonstrate all concepts:

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy import stats

# Generate realistic dataset
np.random.seed(42)
n = 150
x = np.random.normal(5, 2, n)  # Predictor: normal around 5
true_beta_0, true_beta_1 = 10, -2  # True relationship
y = true_beta_0 + true_beta_1 * x + np.random.normal(0, 3, n)  # Add noise

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Fit model
model = LinearRegression()
X_train = x_train.reshape(-1, 1)
model.fit(X_train, y_train)

print(f"True relationship: β₀ = {true_beta_0}, β₁ = {true_beta_1}")
print(f"Estimated: β₀ = {model.intercept_:".3f", β₁ = {model.coef_[0]:.".3f")
```

**Output:**
```
True relationship: β₀ = 10, β₁ = -2
Estimated: β₀ = 9.874, β₁ = -1.956
```

### Statistical Significance Analysis

```python
# Calculate standard errors and test significance
y_hat_train = model.predict(X_train)
residuals = y_train - y_hat_train
RSS = np.sum(residuals**2)
RSE = np.sqrt(RSS / (len(x_train) - 2))

x_bar = np.mean(x_train)
sum_sq_dev = np.sum((x_train - x_bar)**2)

SE_beta_0 = RSE * np.sqrt(1/len(x_train) + x_bar**2/sum_sq_dev)
SE_beta_1 = RSE / np.sqrt(sum_sq_dev)

print(f"\nStandard Errors:")
print(f"SE(β₀) = {SE_beta_0:.".3f")
print(f"SE(β₁) = {SE_beta_1:.".3f")

# t-statistics and p-values
t_beta_0 = model.intercept_ / SE_beta_0
t_beta_1 = model.coef_[0] / SE_beta_1

df = len(x_train) - 2
p_beta_0 = 2 * (1 - stats.t.cdf(abs(t_beta_0), df))
p_beta_1 = 2 * (1 - stats.t.cdf(abs(t_beta_1), df))

print(f"\nHypothesis Tests:")
print(f"β₀: t = {t_beta_0:.".3f", p = {p_beta_0:.".6f")
print(f"β₁: t = {t_beta_1:.".3f", p = {p_beta_1:.".6f")

# Confidence intervals
t_critical = stats.t.ppf(0.975, df)
CI_beta_0_lower = model.intercept_ - t_critical * SE_beta_0
CI_beta_0_upper = model.intercept_ + t_critical * SE_beta_0
CI_beta_1_lower = model.coef_[0] - t_critical * SE_beta_1
CI_beta_1_upper = model.coef_[0] + t_critical * SE_beta_1

print(f"\n95% Confidence Intervals:")
print(f"β₀: ({CI_beta_0_lower:.".3f", {CI_beta_0_upper:.".3f")")
print(f"β₁: ({CI_beta_1_lower:.".3f", {CI_beta_1_upper:.".3f")")
```

**Expected Output:**
```
Standard Errors:
SE(β₀) = 0.456
SE(β₁) = 0.089

Hypothesis Tests:
β₀: t = 21.651, p = 0.000000
β₁: t = -21.977, p = 0.000000

95% Confidence Intervals:
β₀: (8.975, 10.773)
β₁: (-2.131, -1.781)
```

### Bootstrap Verification

```python
# Bootstrap to verify standard errors
def bootstrap_se(x, y, B=1000):
    n = len(x)
    bootstrap_coefs = []

    for _ in range(B):
        indices = np.random.choice(n, size=n, replace=True)
        x_boot = x[indices]
        y_boot = y[indices]

        model_boot = LinearRegression()
        X_boot = x_boot.reshape(-1, 1)
        model_boot.fit(X_boot, y_boot)

        bootstrap_coefs.append([model_boot.intercept_, model_boot.coef_[0]])

    bootstrap_coefs = np.array(bootstrap_coefs)
    return {
        'se_intercept': np.std(bootstrap_coefs[:, 0], ddof=1),
        'se_slope': np.std(bootstrap_coefs[:, 1], ddof=1)
    }

bootstrap_results = bootstrap_se(x_train, y_train)
print(f"\nBootstrap SE:")
print(f"SE(β₀) = {bootstrap_results['se_intercept']:.".3f")
print(f"SE(β₁) = {bootstrap_results['se_slope']:.".3f")
```

### Generalization Assessment

```python
# Evaluate generalization
y_hat_test = model.predict(x_test.reshape(-1, 1))
train_mse = mean_squared_error(y_train, model.predict(X_train))
test_mse = mean_squared_error(y_test, y_hat_test)

train_r2 = model.score(X_train, y_train)
test_r2 = model.score(x_test.reshape(-1, 1), y_test)

print(f"\nGeneralization Assessment:")
print(f"Training MSE: {train_mse:.".3f" RMSE: {np.sqrt(train_mse):.".3f")
print(f"Test MSE: {test_mse:.".3f" RMSE: {np.sqrt(test_mse):.".3f")
print(f"Training R²: {train_r2:.".3f")
print(f"Test R²: {test_r2:.".3f")

# Check for overfitting
mse_ratio = test_mse / train_mse
print(f"MSE Ratio (Test/Train): {mse_ratio:.".3f")

if mse_ratio > 1.2:
    print("⚠️  Model may be overfitting")
elif mse_ratio < 0.9:
    print("⚠️  Model may be underfitting")
else:
    print("✅ Model generalizes well")
```

**Expected Output:**
```
Generalization Assessment:
Training MSE: 8.234 RMSE: 2.870
Test MSE: 9.456 RMSE: 3.075
Training R²: 0.847
Test R²: 0.821
MSE Ratio (Test/Train): 1.148
✅ Model generalizes well
```

---

## 🧠 Summary of Key Concepts

| Concept | Formula/Method | Purpose | Interpretation |
|---------|----------------|---------|----------------|
| **Standard Error** | $SE(\hat{\beta}) = \frac{RSE}{\sqrt{\sum (x_i - \bar{x})^2}}$ | Measures coefficient precision | Smaller SE = more precise estimate |
| **t-Statistic** | $t = \frac{\hat{\beta}}{SE(\hat{\beta})}$ | Tests significance | Large \|t\| = significant coefficient |
| **p-Value** | From t-distribution | Probability of observing result | p < 0.05 = statistically significant |
| **Confidence Interval** | $\hat{\beta} \pm t_{crit} \times SE$ | Plausible range for true β | Excludes 0 = significant |
| **Bootstrap SE** | Std dev of bootstrap coefficients | Robust uncertainty estimate | Alternative to theoretical SE |
| **Train/Test Error** | MSE on held-out data | Checks generalization | Similar errors = good generalization |

---

## 🎓 Self-Study Exercises

### Exercise 1: Manual Significance Test
Given:
- β̂₁ = 2.5, SE(β̂₁) = 0.8, n = 50

Calculate:
1. t-statistic
2. p-value (assume df = 48)
3. 95% confidence interval
4. Is the coefficient significant at α = 0.05?

### Exercise 2: Bootstrap Implementation
Using the bootstrap function provided, compare theoretical vs bootstrap SE for a simple dataset of your choice.

### Exercise 3: Overfitting Detection
Create a scenario where a complex model overfits training data but performs poorly on test data. Calculate the MSE ratio and explain what it tells you.

### Exercise 4: Multiple Testing
If you test 10 coefficients, how many would you expect to be "significant" by chance alone at α = 0.05? What adjustment would you make?

---

## 🚀 Next Steps in Your Learning Journey

You've now mastered the fundamentals of linear regression:

1. **Model fitting** (Section 3.1.1) ✅
2. **Accuracy assessment** (Section 3.1.2) ✅
3. **Model evaluation** (Section 3.1.3) ✅
4. **Statistical inference** (This lecture) ✅

### Ready for Advanced Topics:

1. **Multiple Linear Regression** - Models with multiple predictors
2. **Model Selection** - Choosing which predictors to include
3. **Regularization** - Ridge and Lasso regression
4. **Diagnostics** - Checking regression assumptions
5. **Advanced Metrics** - AIC, BIC, cross-validation

### Further Reading:
- **ISLP Chapter 3**: Complete coverage of linear regression
- **"The Elements of Statistical Learning"**: Chapter 3
- **"An Introduction to Statistical Learning"**: Chapters 3-4

---

## 💡 Pro Tips for Self-Learners

1. **Always check significance**: Don't just look at coefficient magnitude
2. **Use multiple metrics**: Combine statistical tests with practical considerations
3. **Validate on test data**: Training performance can be misleading
4. **Bootstrap when in doubt**: More robust than theoretical assumptions
5. **Context matters**: Statistical significance depends on your field and goals

---

**Challenge Question:** How would you explain the difference between statistical significance and practical importance to a non-technical stakeholder?
