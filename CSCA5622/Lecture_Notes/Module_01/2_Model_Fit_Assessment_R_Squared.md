# Model Fit Assessment - R² and Adjusted R²

## CSCA5622 - Module 01: Linear Regression Fundamentals

---

## 🎯 Overview: Measuring How Well Your Model Fits

When we build a linear regression model like:

$$
\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i
$$

We need to answer a crucial question: **"How well does this model explain the variation in my data?"**

This is where **R-Squared (R²)** and **Adjusted R-Squared** come in. These metrics help us understand:

- What proportion of the variation in our outcome variable is explained by our model
- How our model compares to a simple baseline (predicting the mean)
- Whether adding more variables actually improves our model

---

## 🔍 Why We Need Model Fit Metrics

### The Problem with RSS Alone

Recall that **RSS (Residual Sum of Squares)** measures the total error in our model:

$$
RSS = \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

While RSS tells us "how much error" our model has, it has limitations:

1. **Scale-dependent**: A model with RSS = 100 might be great for predicting house prices (in thousands) but terrible for predicting height (in inches)
2. **No baseline comparison**: It doesn't tell us if our model is better than simply predicting the average
3. **Sample size sensitive**: Larger datasets naturally have larger RSS values

### The Solution: Standardized Metrics

R² and Adjusted R² solve these problems by:
- **Standardizing** the error measure (0 to 1 scale)
- **Comparing against a baseline** (the null model)
- **Providing intuitive interpretation** ("X% of variation is explained")

---

## 📊 The Null Model: Predicting the Mean

### What is the Null Model?

The **null model** is the simplest possible model - it just predicts the average value of y for every observation:

$$
\hat{y}_i = \bar{y} \quad \text{(for all i)}
$$

This represents the "no relationship" baseline.

### Total Sum of Squares (TSS)

To measure how much variation exists in our data, we calculate:

$$
TSS = \sum_{i=1}^n (y_i - \bar{y})^2
$$

**Interpretation:**
- TSS represents the **total variation** in the outcome variable
- It's the "information content" we're trying to explain
- Larger TSS means more variation to potentially explain

**Example Calculation:**
```python
import numpy as np

y = np.array([3, 5, 4, 7, 2, 8, 6])
y_bar = np.mean(y)  # 5.0

# Calculate TSS
TSS = np.sum((y - y_bar)**2)
print(f"TSS = {TSS:.2f}")  # TSS = 26.00

# This means the values vary by a total of 26 squared units from their mean
```

---

## 🥇 R-Squared (R²): Proportion of Variance Explained

### The Formula

$$
R^2 = 1 - \frac{RSS}{TSS}
$$

**Interpretation:**
- **R² = 1**: Perfect fit - model explains 100% of variation
- **R² = 0**: Model explains 0% of variation - no better than predicting the mean
- **R² = 0.75**: Model explains 75% of variation - pretty good fit!

### Intuitive Understanding

R² tells us: **"What fraction of the total variation in y is explained by our model?"**

Breaking it down:
- **TSS**: Total "information" in the data
- **RSS**: Information left unexplained by our model
- **TSS - RSS**: Information explained by our model
- **R² = (TSS - RSS) / TSS**: Proportion of total information explained

### Example Walkthrough

Let's use concrete numbers:

```python
import numpy as np

# Sample data
x = np.array([1, 2, 3, 4, 5, 6, 7])
y = np.array([2, 4, 3, 6, 5, 8, 7])

# Calculate TSS (null model)
y_bar = np.mean(y)  # 5.0
TSS = np.sum((y - y_bar)**2)  # 26.0

# Simple linear regression model
# β₁ = cov(x,y) / var(x) = 4.0 / 4.666 ≈ 0.857
# β₀ = ȳ - β₁x̄ = 5.0 - 0.857*4.0 ≈ 1.571
y_hat = 1.571 + 0.857 * x

# Calculate RSS
RSS = np.sum((y - y_hat)**2)  # 8.857

# Calculate R²
R_squared = 1 - (RSS / TSS)
print(f"R² = {R_squared:.3f}")  # R² = 0.659
```

**What this means:**
- Our model explains about 66% of the variation in y
- 34% of the variation remains unexplained (could be noise, missing variables, etc.)

### Visual Interpretation

Imagine plotting your data:
- **TSS** represents the total "spread" of points around the mean
- **RSS** represents the remaining "spread" around your regression line
- **R²** is the proportion of that spread your model "captures"

---

## 📉 When R-Squared Can Be Negative

### The Problem

R² can actually be **negative** if your model performs worse than the null model!

**When does this happen?**
- When RSS > TSS
- This means your model's predictions are so bad that they're worse than just predicting the mean

### Example of Negative R²

```python
# Very bad model that performs worse than mean
y_bad = np.array([5, 5, 5, 5, 5, 5, 5])  # Constant predictions
RSS_bad = np.sum((y - y_bad)**2)  # This would be > TSS

R_squared_bad = 1 - (RSS_bad / TSS)
print(f"Bad model R² = {R_squared_bad:.3f}")  # Could be negative!
```

**Interpretation:** A negative R² means "you're better off ignoring your model and just predicting the average."

### When This Happens in Practice

Negative R² typically occurs when:
- You fit a model to inappropriate data
- You have severe overfitting
- The relationship between variables is very weak

**Important:** If you get a negative R², it's a red flag that something is wrong with your model or data.

---

## 🧮 Adjusted R-Squared: The Overfitting Protector

### The Problem with Regular R²

As you add more predictors to your model, R² **always increases**, even if those predictors are completely useless!

**Example:**
```python
# Original model with 1 predictor: R² = 0.65
# Add a random predictor: R² = 0.67 (went up!)
# Add another random predictor: R² = 0.69 (went up again!)
```

This is misleading because random predictors shouldn't improve your model.

### The Solution: Adjusted R-Squared

Adjusted R² penalizes models with too many predictors:

$$
\text{Adjusted } R^2 = 1 - \frac{RSS / (n - p - 1)}{TSS / (n - 1)}
$$

Where:
- **n**: Number of observations
- **p**: Number of predictors (excluding intercept)

**Key differences:**
- **Regular R²**: $1 - \frac{RSS}{TSS}$
- **Adjusted R²**: $1 - \frac{RSS/(n-p-1)}{TSS/(n-1)}$

### Why This Adjustment Works

The adjusted formula:
1. **Divides RSS by (n-p-1)**: Penalizes models with more parameters
2. **Divides TSS by (n-1)**: Uses proper degrees of freedom
3. **Result**: Adjusted R² only increases if new predictors truly improve the model

### Example: Regular vs Adjusted R²

```python
# Model 1: 1 predictor, R² = 0.65
# Model 2: 2 predictors, R² = 0.67 (but one is useless)
# Model 3: 3 predictors, R² = 0.69 (two are useless)

# Adjusted R² calculation:
n = 100  # sample size
p1, p2, p3 = 1, 2, 3  # number of predictors

adj_R2_1 = 1 - (RSS1/(n-p1-1)) / (TSS/(n-1))
adj_R2_2 = 1 - (RSS2/(n-p2-1)) / (TSS/(n-1))
adj_R2_3 = 1 - (RSS3/(n-p3-1)) / (TSS/(n-1))

# Result: adj_R2_1 > adj_R2_2 > adj_R2_3
# Adjusted R² decreases as we add useless predictors!
```

---

## ⚠️ The Intercept-Free Model Trap

### The Problem

Sometimes you might consider fitting a model **without an intercept**:

$$
\hat{y}_i = \beta_1 x_i \quad (\text{no } \beta_0)
$$

This creates a problem for R² calculation.

### Why It's Problematic

When you remove the intercept:
- The null model becomes $\hat{y}_i = 0$ instead of $\hat{y}_i = \bar{y}$
- TSS becomes $\sum (y_i - 0)^2 = \sum y_i^2$
- This artificially inflates TSS and makes R² look better

**Example:**
```python
# Model with intercept: R² = 0.65
# Model without intercept: R² = 0.82 (inflated!)

# But this is misleading because the baseline is different
```

### The Solution

**Always use the same baseline for comparison:**

1. **Include intercept** in your models (recommended)
2. **Compare models with same null model** (both with or both without intercept)
3. **Use adjusted R²** when comparing models with different numbers of predictors

### When to Use Intercept-Free Models

Only consider removing the intercept when:
- You have strong theoretical justification
- The relationship must pass through the origin (0,0)
- You're doing specialized modeling (like ratios or certain scientific applications)

---

## 📊 Complete Worked Example

### Dataset

Let's use a concrete example with real calculations:

| x | y  |
|---|----|
| 1 | 2  |
| 2 | 4  |
| 3 | 3  |
| 4 | 6  |
| 5 | 5  |

### Step-by-Step Calculation

**Step 1: Calculate means**
```python
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 3, 6, 5])

x_bar = np.mean(x)  # 3.0
y_bar = np.mean(y)  # 4.0
```

**Step 2: Fit linear regression**
```python
# Calculate slope and intercept
numerator = np.sum((x - x_bar) * (y - y_bar))  # 6.0
denominator = np.sum((x - x_bar)**2)          # 10.0
beta_1 = numerator / denominator              # 0.6
beta_0 = y_bar - beta_1 * x_bar              # 4.0 - 0.6*3.0 = 2.2

y_hat = beta_0 + beta_1 * x
# y_hat = [2.8, 3.4, 4.0, 4.6, 5.2]
```

**Step 3: Calculate TSS**
```python
TSS = np.sum((y - y_bar)**2)
# (2-4)² + (4-4)² + (3-4)² + (6-4)² + (5-4)² = 4 + 0 + 1 + 4 + 1 = 10.0
```

**Step 4: Calculate RSS**
```python
residuals = y - y_hat
# [2-2.8, 4-3.4, 3-4.0, 6-4.6, 5-5.2] = [-0.8, 0.6, -1.0, 1.4, -0.2]
RSS = np.sum(residuals**2)  # 0.64 + 0.36 + 1.0 + 1.96 + 0.04 = 4.0
```

**Step 5: Calculate R²**
```python
R_squared = 1 - (RSS / TSS)  # 1 - (4.0 / 10.0) = 0.6
```

**Step 6: Calculate Adjusted R²**
```python
n = 5  # observations
p = 1  # predictors (excluding intercept)
adj_R_squared = 1 - (RSS / (n - p - 1)) / (TSS / (n - 1))
# 1 - (4.0 / 3) / (10.0 / 4) = 1 - (1.333) / (2.5) = 1 - 0.533 = 0.467
```

### Interpretation

- **R² = 0.6**: Our model explains 60% of the variation in y
- **Adjusted R² = 0.467**: After accounting for having 1 predictor, we explain about 47% of variation
- **Difference**: The adjustment penalizes us for using a predictor, showing that some of the "explained" variation might be due to chance

---

## 💻 Python Implementation

### Complete Code for R² Calculation

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def calculate_r_squared(x, y):
    """
    Calculate R² and Adjusted R² for a simple linear regression.

    Parameters:
    x: array-like, predictor variable
    y: array-like, response variable

    Returns:
    dict with R², Adjusted R², RSS, TSS, and model coefficients
    """

    # Convert to numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Calculate means
    x_bar = np.mean(x)
    y_bar = np.mean(y)

    # Fit linear regression
    model = LinearRegression()
    X = x.reshape(-1, 1)  # sklearn needs 2D array
    model.fit(X, y)

    # Get predictions
    y_hat = model.predict(X)

    # Calculate metrics
    TSS = np.sum((y - y_bar)**2)
    RSS = np.sum((y - y_hat)**2)

    R_squared = 1 - (RSS / TSS)

    # Adjusted R²
    n = len(y)
    p = 1  # number of predictors (excluding intercept)
    adj_R_squared = 1 - (RSS / (n - p - 1)) / (TSS / (n - 1))

    return {
        'R_squared': R_squared,
        'Adjusted_R_squared': adj_R_squared,
        'RSS': RSS,
        'TSS': TSS,
        'beta_0': model.intercept_,
        'beta_1': model.coef_[0]
    }

# Example usage
x = [1, 2, 3, 4, 5]
y = [2, 4, 3, 6, 5]

results = calculate_r_squared(x, y)
print(f"R² = {results['R_squared']:.3f}")
print(f"Adjusted R² = {results['Adjusted_R_squared']:.3f}")
print(f"β₀ = {results['beta_0']:.3f}, β₁ = {results['beta_1']:.3f}")
```

**Expected Output:**
```
R² = 0.600
Adjusted R² = 0.467
β₀ = 2.200, β₁ = 0.600
```

---

## 🧠 Key Takeaways Summary

| Metric | Formula | Purpose | When to Use |
|--------|---------|---------|-------------|
| **TSS** | $\sum (y_i - \bar{y})^2$ | Total variation in data | Baseline for comparison |
| **RSS** | $\sum (y_i - \hat{y}_i)^2$ | Unexplained variation | Lower = better fit |
| **R²** | $1 - \frac{RSS}{TSS}$ | Proportion explained | Model comparison, goodness of fit |
| **Adj. R²** | $1 - \frac{RSS/(n-p-1)}{TSS/(n-1)}$ | Adjusted for predictors | Model selection, overfitting protection |

### Decision Guidelines

**Choose R² when:**
- Comparing models with the same number of predictors
- You want a simple, interpretable metric
- Sample size is large

**Choose Adjusted R² when:**
- Comparing models with different numbers of predictors
- You want to guard against overfitting
- Sample size is small relative to number of predictors

**Be cautious when:**
- R² is very close to 0 or 1 (might indicate problems)
- Comparing models with and without intercepts
- Sample size is very small (R² becomes unreliable)

---

## 🎓 Self-Study Exercises

### Exercise 1: Manual Calculation
Given the dataset:
- x = [2, 4, 6, 8]
- y = [3, 7, 5, 9]

Calculate by hand:
1. TSS and RSS
2. R²
3. Adjusted R² (n=4, p=1)

### Exercise 2: Model Comparison
You have two models for the same data:
- Model A: R² = 0.85, 2 predictors
- Model B: R² = 0.82, 1 predictor

Which model is better? Why?

### Exercise 3: The Negative R² Case
Create a scenario where R² would be negative. Explain why this happens.

---

## 🚀 Next Steps in Your Learning Journey

Now that you understand R² and Adjusted R², you're ready to:

1. **Practice with real datasets** - Try calculating these metrics on actual data
2. **Learn about multiple regression** - How R² works with multiple predictors
3. **Explore model selection** - Using Adjusted R² for choosing the best model
4. **Study prediction intervals** - Going beyond point estimates to prediction uncertainty

### Further Reading

- **ISLP Chapter 3**: More details on model assessment
- **"The Elements of Statistical Learning"**: Chapter 7 covers these topics in depth
- **Scikit-learn documentation**: `r2_score` and related functions

---

## 💡 Pro Tips for Self-Learners

1. **Always visualize**: Plot your data and regression line alongside the mean to see what R² represents
2. **Check assumptions**: R² assumes your model includes an intercept and uses the mean as baseline
3. **Context matters**: A "good" R² depends on your field (physics: 0.95+, social science: 0.3+)
4. **Don't over-interpret**: R² measures fit, not causation or model correctness
5. **Combine metrics**: Use R² alongside residual plots, hypothesis tests, and domain knowledge

---

**Ready for a challenge?** Try implementing R² calculation for multiple linear regression, or explore how R² behaves with polynomial models!
