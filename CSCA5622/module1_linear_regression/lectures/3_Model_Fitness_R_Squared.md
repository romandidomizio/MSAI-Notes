# Model Fitness and R-squared - Lecture 3

## Overview
This lecture explains how to measure how well a linear regression model fits the data using **R-squared** and **Adjusted R-squared**. We'll derive R-squared from first principles, understand its range, and explore common pitfalls when interpreting R-squared values.

---

## Table of Contents
1. [Introduction to Model Fit Metrics](#introduction-to-model-fit-metrics)
2. [Deriving R-squared](#deriving-r-squared)
3. [Understanding RSS and TSS](#understanding-rss-and-tss)
4. [R-squared Value Ranges](#r-squared-value-ranges)
5. [Practice Problem: Models With and Without Intercept](#practice-problem-models-with-and-without-intercept)
6. [The Uncentered R-squared Pitfall](#the-uncentered-r-squared-pitfall)

---

## Introduction to Model Fit Metrics

> **"Let's talk about how well my model fits. We're going to look at the numbers R-squared value and Adj-R-squared. These are metric for how well the model fits."**

### R-squared vs. Adjusted R-squared

**R-squared (R²)**: Measures the proportion of variance in the target variable explained by the model

**Adjusted R-squared (Adj-R²)**: Same as R-squared but penalizes for the number of features

> **"Adj-R-squared is actually same as R-squared except that it also takes number of features into account."**

### When Are They the Same?

> **"However, when the number of samples are much larger than number of features in the model, these two numbers are essentially the same."**

**Rule of thumb**: When n >> p (many more samples than features), R² ≈ Adj-R²

**Formulas**:
- $R^2 = 1 - \frac{RSS}{TSS}$
- $\text{Adj-}R^2 = 1 - \frac{RSS/(n-p-1)}{TSS/(n-1)}$

Where:
- n = number of samples
- p = number of features

---

## Deriving R-squared

> **"Let's derive R-squared as a measure of model fit."**

### When Does a Model Have Good Fit?

> **"When do we know that model has a good fit? From the least squares method that we used to determine our coefficient values, we know that model has a good fit when we have a squared error is minimized."**

**Key principle**: Good fit = minimized squared error

### Residual Sum of Squares (RSS)

> **"Again, we can use MAC or RSS. RSS is the residual sum of squares is nothing but same as MSE. We wrote the averaging vector."**

**Definition**:
$$RSS = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Relationship to MSE**:
$$MSE = \frac{1}{n} RSS$$

> **"We define this quantity and we know that if this quantity is minimized, we know the model has a good fit."**

### The Problem with RSS Alone

> **"However, there is a little bit of problem with this metric. One is that this value can be arbitrarily large depending on our unit of the target variable."**

**Problems with RSS**:
1. **Scale-dependent**: Changes with units (dollars vs. thousands of dollars)
2. **Not comparable**: Different datasets have different RSS values

> **"Also if you have a different set of data, this quantity will be different."**

**Question**: How do we make RSS comparable across datasets?

> **"We want to normalize by something similar error measure that has same union. What would it be a good way to do that?"**

---

## Understanding RSS and TSS

### The Benchmark Model

> **"We can define a benchmark model, say y equals y min and then we can compare how good is my error from my model y equals Beta zero plus Beta one x compared to the editor of my benchmark model, which is y equals y min."**

**Null/Benchmark Model**: $\hat{y} = \bar{y}$ (predict the mean for every input)

**Idea**: Compare our model's error to the error of simply predicting the mean

### Total Sum of Squares (TSS)

> **"We're going to define another quantity called the TSS, total sum of squares that actually quantifies the error between my null model and my training data points."**

**Definition**:
$$TSS = \sum_{i=1}^{n} (y_i - \bar{y})^2$$

**Interpretation**: 
- Total variance in the data
- Error of the benchmark model (predicting $\bar{y}$ for everything)

### Creating a Dimensionless Ratio

> **"With that, we can define a dimensionless quantity by dividing RSS by TSS."**

$$\frac{RSS}{TSS} = \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

> **"This is the quantity essentially telling that what's the ratio of the error from my model to the error from the null model or benchmark model."**

**Interpretation**:
- **RSS/TSS = 0**: Perfect fit (our model has zero error)
- **RSS/TSS = 1**: Our model is no better than predicting the mean
- **RSS/TSS > 1**: Our model is worse than predicting the mean!

### Flipping the Sign

> **"This can be a good quantity that measures how well my model fits compared to my null model. But we also won our continuity or R-squared value to be higher when my model fits better."**

**Problem**: RSS/TSS decreases when model improves (counterintuitive)

> **"If you see RSS goes down when the model fits better so we're going to flip the sign. We're going to subtract this quantity from one."**

### R-squared Definition

> **"Then actually it becomes the definition of R-squared."**

$$\boxed{R^2 = 1 - \frac{RSS}{TSS} = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}}$$

**Interpretation**:
- **R² = 1**: Perfect fit
- **R² = 0**: Model is no better than predicting the mean
- **R² < 0**: Model is worse than predicting the mean

### Python Example: Computing R-squared

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Generate sample data
np.random.seed(42)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([2.3, 3.8, 5.2, 6.9, 8.1, 9.5, 11.2, 12.8, 14.1, 15.7])

# Fit model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Manual R-squared calculation
y_mean = np.mean(y)
RSS = np.sum((y - y_pred)**2)
TSS = np.sum((y - y_mean)**2)
R2_manual = 1 - (RSS / TSS)

# Using sklearn
R2_sklearn = r2_score(y, y_pred)
R2_model = model.score(X, y)

print("="*60)
print("R-SQUARED CALCULATION")
print("="*60)
print(f"Mean of y (ȳ): {y_mean:.4f}")
print(f"\nRSS (Residual Sum of Squares): {RSS:.4f}")
print(f"TSS (Total Sum of Squares):    {TSS:.4f}")
print(f"\nRSS/TSS ratio: {RSS/TSS:.4f}")
print(f"1 - (RSS/TSS): {R2_manual:.4f}")
print(f"\nR² (manual calculation): {R2_manual:.4f}")
print(f"R² (sklearn r2_score):   {R2_sklearn:.4f}")
print(f"R² (model.score):        {R2_model:.4f}")
print("="*60)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Model fit with residuals
ax1 = axes[0]
ax1.scatter(X, y, s=100, alpha=0.6, label='Actual data', zorder=3)
ax1.plot(X, y_pred, 'r-', linewidth=2, label=f'Model: R² = {R2_manual:.4f}', zorder=2)
ax1.axhline(y=y_mean, color='green', linestyle='--', linewidth=2, label=f'Baseline (ȳ = {y_mean:.2f})', zorder=1)

# Draw residuals to model
for i in range(len(X)):
    ax1.plot([X[i], X[i]], [y[i], y_pred[i]], 'orange', linestyle='--', linewidth=1, alpha=0.7)

ax1.set_xlabel('X', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Model Fit (RSS = residuals to red line)', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: Baseline model with deviations
ax2 = axes[1]
ax2.scatter(X, y, s=100, alpha=0.6, label='Actual data', zorder=3)
ax2.axhline(y=y_mean, color='green', linestyle='--', linewidth=2, label=f'Baseline (ȳ = {y_mean:.2f})', zorder=2)

# Draw deviations from mean
for i in range(len(X)):
    ax2.plot([X[i], X[i]], [y[i], y_mean], 'blue', linestyle='--', linewidth=1, alpha=0.7)

ax2.set_xlabel('X', fontsize=12)
ax2.set_ylabel('y', fontsize=12)
ax2.set_title('Baseline Model (TSS = deviations from mean)', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nInterpretation: {R2_manual*100:.2f}% of variance in y is explained by X")
```

---

## R-squared Value Ranges

> **"Let's take a moment and think about what values R-square can take. We're going to think about two extreme cases."**

### Case 1: Perfect Fit (R² = 1)

> **"One extreme case is that when my RSS is 0, that means my model fits perfectly all of the data points, which we'll never happen in practice."**

**Scenario**: All data points lie exactly on the regression line

> **"I think that my model is so good that all the data points are on my models line. Then this term goes to 0 and my R-squared value will go 1. That's one extreme."**

$$RSS = 0 \implies R^2 = 1 - \frac{0}{TSS} = 1$$

**In practice**: Perfect R² = 1.0 almost never happens with real data

### Can R² > 1?

> **"Can R-square go larger than 1? R-squared value cannot be larger than 1 because RSS cannot be negative."**

**Answer**: **NO**, R² cannot exceed 1

**Reason**: RSS is a sum of squares, so RSS ≥ 0 always

$$R^2 = 1 - \frac{RSS}{TSS} \leq 1 \quad \text{(since } RSS \geq 0\text{)}$$

### Case 2: No Better Than Baseline (R² = 0)

> **"Another extreme case is that by model is actually just as good as my null model, y equals y mean."**

**Scenario**: Our regression line performs no better than predicting $\bar{y}$

> **"In that case, my RSS value will be same as TSS so this goes to one. Then my R-squared will go to 0."**

$$RSS = TSS \implies R^2 = 1 - \frac{TSS}{TSS} = 1 - 1 = 0$$

**Interpretation**: The features have zero predictive power

### Can R² < 0?

> **"Can R-squared value go negative? Yes, it can."**

**Answer**: **YES**, R² can be negative!

> **"In practice, if you use a package to fit your regression line, you will almost never happen."**

**When does it happen?**

> **"But in case your model is this bad, like this, the slope is totally wrong. Then you might have an RSS that's larger than TSS than this R-squared value can go negative."**

**Scenario**: Model fits worse than the baseline (e.g., slope has wrong sign)

$$RSS > TSS \implies R^2 = 1 - \frac{RSS}{TSS} < 0$$

### Simple vs. Complex Models

> **"For simple linear regression, this may not happen. But as you might see later in the more complex models, sometimes the model can fit worse than the baseline so remember the R-squared can go negative as well."**

**Summary of R² Ranges**:

| Range | Interpretation | Likelihood |
|-------|---------------|------------|
| **R² = 1** | Perfect fit | Very rare (real data) |
| **0.7 < R² < 1** | Strong fit | Common for good models |
| **0.3 < R² < 0.7** | Moderate fit | Acceptable |
| **0 < R² < 0.3** | Weak fit | Questionable utility |
| **R² = 0** | No better than mean | Features useless |
| **R² < 0** | Worse than mean | Model is harmful |

### Python Example: Different R² Values

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

np.random.seed(42)
X = np.linspace(0, 10, 50).reshape(-1, 1)

# Create datasets with different R² values
y_perfect = 2 * X.flatten() + 1  # R² = 1.0
y_strong = 2 * X.flatten() + 1 + np.random.normal(0, 1, 50)  # R² ≈ 0.95
y_weak = 2 * X.flatten() + 1 + np.random.normal(0, 8, 50)  # R² ≈ 0.30
y_none = np.random.normal(5, 5, 50)  # R² ≈ 0.0

datasets = [
    (y_perfect, 'Perfect Fit'),
    (y_strong, 'Strong Fit'),
    (y_weak, 'Weak Fit'),
    (y_none, 'No Fit')
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (y, title) in enumerate(datasets):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    ax = axes[idx]
    ax.scatter(X, y, alpha=0.6, s=50)
    ax.plot(X, y_pred, 'r-', linewidth=2, label=f'Model')
    ax.axhline(y=np.mean(y), color='green', linestyle='--', linewidth=2, label='Baseline (ȳ)')
    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_title(f'{title}: R² = {r2:.4f}', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Practice Problem: Models With and Without Intercept

> **"We saw the R-square value could be a good measure of how my model fits. However, you have to be careful when you interpret the value from your summary table."**

### The Scenario

> **"Let's take an example where we might want a model that takes the form of ax and there's no intercept."**

**Two models to compare**:
1. **Model with intercept**: $\hat{y} = \beta_0 + \beta_1 x$ (standard)
2. **Model without intercept**: $\hat{y} = \beta_1 x$ (through origin)

### Why Remove the Intercept?

> **"Why would we want to do that? Let's have a look at the intercept value. It's a negative value."**

**Issue with intercept**:

> **"That means my sales price will go negative when my living space is 0. That doesn't make a lot of sense so maybe instead of having this uninterpretable intercept, maybe we want to have a model that has no intercept."**

**Example**: House price vs. living space
- With intercept: When living space = 0, price = $\beta_0$ (could be negative!)
- Without intercept: When living space = 0, price = 0 (makes sense)

> **"Then yeah, that sounds good. My sales price of house should be 0 when the living space is 0."**

### Fitting the No-Intercept Model

> **"Let's take a fit and look at the summary table. We have a square feet living coefficient, which is similar to the previous value, which is good."**

**Observation**: The slope coefficient is similar in both models

### The Surprising R² Increase

> **"But then we suddenly see R-squared value has gone up. What does that mean?"**

**Observation**: R² for no-intercept model > R² for intercept model

> **"Does it mean our new model, y equals ax is better than our old model, ax plus b? Well, not necessarily."**

**⚠️ WARNING**: Higher R² doesn't always mean better model!

---

## The Uncentered R-squared Pitfall

### The Clue in the Output

> **"If you look at carefully, you're going to see uncentered next to the R-squared. What does that mean?"**

**Key observation**: The summary table shows "R-squared (uncentered)"

### Different Null Models

> **"It turns out that this R-squared value is calculated such that RSS of our new model, this guy and then divide by TSS of the new null model, which is not y equals y min."**

**Standard R² (centered)**:
$$R^2_{\text{centered}} = 1 - \frac{RSS}{TSS}, \quad \text{where } TSS = \sum(y_i - \bar{y})^2$$

**Null model**: $\hat{y} = \bar{y}$ (predict the mean)

**Uncentered R² (for no-intercept model)**:
$$R^2_{\text{uncentered}} = 1 - \frac{RSS}{TSS_{\text{uncent}}}, \quad \text{where } TSS_{\text{uncent}} = \sum y_i^2$$

> **"But now our new null model is y equals 0."**

**Null model**: $\hat{y} = 0$ (predict zero)

### Why Uncentered R² Is Higher

> **"This goes to here. Then the total sum of squares from y equals will be way higher."**

$$TSS_{\text{uncentered}} = \sum_{i=1}^{n} y_i^2 \quad \text{(usually much larger)}$$

$$TSS_{\text{centered}} = \sum_{i=1}^{n} (y_i - \bar{y})^2 \quad \text{(smaller)}$$

**Result**: Larger denominator → smaller fraction → larger R²

> **"Therefore, the R-squared value can be much larger than the previous one."**

**⚠️ CRITICAL**: You **cannot compare** centered and uncentered R²!

### The Correct Comparison

> **"If you want to compare apple to apple, how my new model is doing in terms of the error, you can just directly calculate RSS for our new model."**

**Correct approach**: Compare RSS values directly

> **"Let's say y equals ax. Then compare with the previous model. Notice as y equals ax plus b, then you're going to see this RSS is larger than this one."**

**Comparison**:
- $RSS_{\text{no intercept}} > RSS_{\text{with intercept}}$

**Conclusion**: Model with intercept actually fits better!

> **"But the value that gives here in the summary table is a little bit deceptive."**

**Key Lesson**: Don't trust R² values when comparing models with and without intercepts!

### Python Example: The Intercept Problem

```python
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Generate house price data
np.random.seed(42)
sqft_living = np.array([800, 1200, 1500, 1800, 2000, 2500, 3000, 3500, 4000])
price = 50 + 0.15 * sqft_living + np.random.normal(0, 20, len(sqft_living))

print("="*70)
print("COMPARING MODELS WITH AND WITHOUT INTERCEPT")
print("="*70)

# Model 1: WITH intercept (y = β₀ + β₁x)
X_with_intercept = sm.add_constant(sqft_living)
model_with = sm.OLS(price, X_with_intercept).fit()

y_pred_with = model_with.predict(X_with_intercept)
RSS_with = np.sum((price - y_pred_with)**2)

# Model 2: WITHOUT intercept (y = β₁x)
X_no_intercept = sqft_living.reshape(-1, 1)
model_without = sm.OLS(price, X_no_intercept).fit()

y_pred_without = model_without.predict(X_no_intercept)
RSS_without = np.sum((price - y_pred_without)**2)

# Manual R² calculations
y_mean = np.mean(price)
TSS_centered = np.sum((price - y_mean)**2)
TSS_uncentered = np.sum(price**2)

R2_with_manual = 1 - (RSS_with / TSS_centered)
R2_without_centered = 1 - (RSS_without / TSS_centered)  # Fair comparison
R2_without_uncentered = 1 - (RSS_without / TSS_uncentered)  # Reported value

print("\nMODEL 1 (WITH INTERCEPT): ŷ = β₀ + β₁·x")
print("-"*70)
print(f"  β₀ (intercept): {model_with.params[0]:.4f}")
print(f"  β₁ (slope):     {model_with.params[1]:.4f}")
print(f"  RSS:            {RSS_with:.4f}")
print(f"  R² (centered):  {model_with.rsquared:.4f}")

if model_with.params[0] < 0:
    print(f"  ⚠️  Negative intercept: price = {model_with.params[0]:.2f} when sqft = 0")

print("\nMODEL 2 (WITHOUT INTERCEPT): ŷ = β₁·x")
print("-"*70)
print(f"  β₁ (slope):              {model_without.params[0]:.4f}")
print(f"  RSS:                     {RSS_without:.4f}")
print(f"  R² (uncentered):         {model_without.rsquared:.4f}  ← Reported")
print(f"  R² (centered, manual):   {R2_without_centered:.4f}  ← Fair comparison")

print("\n" + "="*70)
print("COMPARISON ANALYSIS")
print("="*70)
print(f"RSS (with intercept):    {RSS_with:.4f}")
print(f"RSS (without intercept): {RSS_without:.4f}")
print(f"Difference:              {RSS_without - RSS_with:.4f}")

if RSS_without > RSS_with:
    print("\n✓ Model WITH intercept fits better (lower RSS)")
else:
    print("\n✓ Model WITHOUT intercept fits better (lower RSS)")

print(f"\nR² comparison (MISLEADING if using uncentered):")
print(f"  Model with intercept:    R² = {model_with.rsquared:.4f}")
print(f"  Model without (uncent):  R² = {model_without.rsquared:.4f}  ← HIGHER but WRONG comparison!")
print(f"  Model without (cent):    R² = {R2_without_centered:.4f}  ← Correct for comparison")

print(f"\nWhy uncentered R² is misleading:")
print(f"  TSS (centered):   {TSS_centered:.4f}  ← Used for model with intercept")
print(f"  TSS (uncentered): {TSS_uncentered:.4f}  ← Used for model without intercept")
print(f"  Ratio: {TSS_uncentered/TSS_centered:.2f}x larger!")

print("\n" + "="*70)
print("⚠️  LESSON: Never compare R² between models with/without intercept!")
print("    Use RSS instead for fair comparison.")
print("="*70)

# Visualization
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Model comparison
ax1 = axes[0]
ax1.scatter(sqft_living, price, s=100, alpha=0.6, label='Data', zorder=3)
ax1.plot(sqft_living, y_pred_with, 'r-', linewidth=2, 
         label=f'With intercept (R²={R2_with_manual:.3f})', zorder=2)
ax1.plot(sqft_living, y_pred_without, 'b--', linewidth=2,
         label=f'Without intercept (R²={R2_without_centered:.3f})', zorder=2)
ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
ax1.set_xlabel('Square Feet Living', fontsize=12)
ax1.set_ylabel('Price ($1000s)', fontsize=12)
ax1.set_title('Model Comparison (Fair R² comparison)', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, max(sqft_living)*1.1)

# Right: R² values
ax2 = axes[1]
models = ['With\nIntercept', 'Without\n(Uncentered)', 'Without\n(Centered)']
r2_values = [model_with.rsquared, model_without.rsquared, R2_without_centered]
colors = ['green', 'red', 'orange']

bars = ax2.bar(models, r2_values, color=colors, alpha=0.7, edgecolor='black')
for i, (bar, val) in enumerate(zip(bars, r2_values)):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.4f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.set_ylabel('R² Value', fontsize=12)
ax2.set_title('R² Comparison: The Deceptive Increase', fontsize=13, fontweight='bold')
ax2.set_ylim(0, max(r2_values) * 1.15)
ax2.grid(True, axis='y', alpha=0.3)
ax2.axhline(y=model_with.rsquared, color='green', linestyle='--', alpha=0.5, 
            label='True baseline')
ax2 legend()

plt.tight_layout()
plt.show()
```

---

## Summary

**Key Concepts Covered**:

✅ **R-squared**: Proportion of variance explained, $R^2 = 1 - \frac{RSS}{TSS}$

✅ **Adjusted R-squared**: Penalizes for number of features, similar to R² when n >> p

✅ **RSS**: Residual Sum of Squares = $\sum(y_i - \hat{y}_i)^2$

✅ **TSS**: Total Sum of Squares = $\sum(y_i - \bar{y})^2$

✅ **R² Ranges**:
- R² = 1: Perfect fit
- R² = 0: No better than mean
- R² < 0: Worse than mean (possible!)

✅ **Models Without Intercept**:
- Can make intuitive sense (e.g., price = 0 when size = 0)
- Use "uncentered" R² (not comparable to standard R²)
- Compare using RSS, not R²!

✅ **Critical Warning**: Never compare R² between models with and without intercepts

---

## Key Formulas

| Concept | Formula |
|---------|---------|
| **R-squared** | $R^2 = 1 - \frac{RSS}{TSS}$ |
| **RSS** | $\sum_{i=1}^{n} (y_i - \hat{y}_i)^2$ |
| **TSS (centered)** | $\sum_{i=1}^{n} (y_i - \bar{y})^2$ |
| **TSS (uncentered)** | $\sum_{i=1}^{n} y_i^2$ |
| **Adjusted R²** | $1 - \frac{RSS/(n-p-1)}{TSS/(n-1)}$ |

---

## Important Takeaways

🎯 **R² measures goodness of fit** by comparing model error (RSS) to baseline error (TSS)

🎯 **R² is dimensionless** and comparable across datasets (unlike RSS)

🎯 **R² can be negative** when the model is worse than predicting the mean

🎯 **Uncentered R² is deceptive** - always check if comparing models with/without intercepts

🎯 **Use RSS for fair comparison** when models have different structures

---

**End of Lecture 3 Notes**
