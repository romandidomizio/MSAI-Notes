# Simple Linear Regression - Lecture 1

## Overview
This lecture introduces **linear regression**, one of the simplest yet most fundamental supervised learning models. We'll explore how the model works, how it's optimized, important quantities for evaluating performance, and how to assess statistical significance.

---

## Table of Contents
1. [Supervised Learning Review](#supervised-learning-review)
2. [What is Linear Regression?](#what-is-linear-regression)
3. [Linear Relationships](#linear-relationships)
4. [Multiple Features and Linear Combination](#multiple-features-and-linear-combination)
5. [Real-World Example: House Price Prediction](#real-world-example-house-price-prediction)
6. [Correlation Matrix](#correlation-matrix)
7. [Simple (Univariate) Linear Regression](#simple-univariate-linear-regression)
8. [Model Training and Implementation](#model-training-and-implementation)
9. [Key Questions to Address](#key-questions-to-address)

---

## Supervised Learning Review

> **Supervised learning needs training data that feeds to the model.**

The supervised learning workflow consists of several key components:

1. **Training Data**: The dataset used to train the model
2. **Model**: Has internal parameters (some models don't have parameters at all)
3. **Hyperparameters**: User-defined settings that need to be tweaked
4. **Prediction**: The model's output value
5. **Target**: The actual value we're trying to predict

### The Goal of Supervised Learning

> **"Our goal is to tweak this parameter by optimization so that the model makes a prediction that's close to the target as much as possible."**

**Initial State**: If the parameters for the parametric model are not optimized, the prediction value will be far away from the target.

**Optimization Process**: Through training, we adjust parameters to minimize the difference between predictions and actual target values.

### Python Example: Supervised Learning Concept

```python
import numpy as np
import matplotlib.pyplot as plt

# Example: Initial vs. Optimized Model
np.random.seed(42)
X = np.linspace(0, 10, 50)
y_true = 2.5 * X + 1.0 + np.random.normal(0, 2, 50)  # True relationship with noise

# Poor initial parameters
initial_slope = 0.5
initial_intercept = 5
y_initial = initial_slope * X + initial_intercept

# Optimized parameters (after training)
optimized_slope = 2.5
optimized_intercept = 1.0
y_optimized = optimized_slope * X + optimized_intercept

# Visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X, y_true, alpha=0.6, label='Actual Data')
plt.plot(X, y_initial, 'r-', linewidth=2, label='Initial Model (Poor Fit)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Before Optimization')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(X, y_true, alpha=0.6, label='Actual Data')
plt.plot(X, y_optimized, 'g-', linewidth=2, label='Optimized Model (Good Fit)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('After Optimization')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## What is Linear Regression?

> **"It is one of the simplest kind of supervised learning model."**

### Key Characteristics

1. **Regression Task**: Predicts a **real-value number** (continuous output)
2. **Has Parameters**: Contains internal parameters often called **coefficients**
3. **No Hyperparameters**: The user doesn't need to figure out design parameters in advance or during training
4. **Linear Relationship Assumption**: ⚠️ **CRITICAL** - Linear regression assumes a **linear relationship** between features and the target variable

### What Makes It "Linear"?

The model assumes that the relationship between input features (X) and the target variable (Y) can be represented as a straight line (in 1D) or a hyperplane (in higher dimensions).

---

## Linear Relationships

> **"It means the feature, let's say we have only one feature for now, has a linear relationship to the target variable."**

### Example 1: House Size vs. House Price

**Feature**: House size (square feet)  
**Target**: House price

**Relationship**: When the house size gets larger, the house price gets larger.

```python
# Example: House Size vs Price
import numpy as np
import matplotlib.pyplot as plt

# Simulated data
house_sizes = np.array([800, 1000, 1200, 1500, 1800, 2000, 2200, 2500, 3000, 3500])
house_prices = np.array([150, 180, 210, 250, 290, 320, 350, 400, 470, 550])  # in thousands

plt.figure(figsize=(8, 6))
plt.scatter(house_sizes, house_prices, s=100, alpha=0.6, color='blue')
plt.xlabel('House Size (sq ft)', fontsize=12)
plt.ylabel('House Price ($1000s)', fontsize=12)
plt.title('Positive Linear Relationship: House Size vs Price', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.show()
```

**Observation**: Positive slope - as X increases, Y increases.

---

### Example 2: Years of Experience vs. Salary

**Feature**: Years of experience  
**Target**: Salary

**Relationship**: When years of experience goes up, salary goes up.

```python
# Example: Experience vs Salary
years_exp = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10])
salary = np.array([40, 43, 48, 52, 58, 63, 68, 75, 82, 95])  # in thousands

plt.figure(figsize=(8, 6))
plt.scatter(years_exp, salary, s=100, alpha=0.6, color='green')
plt.xlabel('Years of Experience', fontsize=12)
plt.ylabel('Salary ($1000s)', fontsize=12)
plt.title('Positive Linear Relationship: Experience vs Salary', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.show()
```

---

### Example 3: Age vs. Survival Rate

**Feature**: Age  
**Target**: Survival rate from disease (e.g., cancer)

**Relationship**: As age goes up, survival rate goes down.

> **"It doesn't have to be positive slope all the time."**

```python
# Example: Age vs Survival Rate
age = np.array([20, 30, 40, 50, 60, 70, 80, 90])
survival_rate = np.array([95, 92, 88, 82, 75, 65, 50, 35])  # percentage

plt.figure(figsize=(8, 6))
plt.scatter(age, survival_rate, s=100, alpha=0.6, color='red')
plt.xlabel('Age', fontsize=12)
plt.ylabel('Survival Rate (%)', fontsize=12)
plt.title('Negative Linear Relationship: Age vs Survival Rate', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.show()
```

**Observation**: Negative slope - as X increases, Y decreases.

---

### Key Takeaway

> **"These examples show some kind of linear relationship of the feature to the target variable."**

Linear regression is appropriate when:
- The data shows a trend that can be approximated by a straight line
- The relationship can be positive (upward slope) or negative (downward slope)
- The relationship is consistent across the range of values

---

## Multiple Features and Linear Combination

> **"When we have multiple features, linear model also have some linear combination shape."**

### Mathematical Formulation

When we have **p features** (X₁, X₂, ..., Xₚ), the linear model takes the form:

$$\hat{y} = a_0 + a_1 X_1 + a_2 X_2 + \ldots + a_p X_p$$

Where:
- $\hat{y}$ = predicted value
- $a_0$ = **intercept** (free parameter, bias term)
- $a_1, a_2, \ldots, a_p$ = **coefficients** for each feature
- $X_1, X_2, \ldots, X_p$ = feature values

### What is a "Linear Combination"?

A **linear combination** means each feature is:
1. Multiplied by its own coefficient
2. All terms are added together
3. An intercept term can be added

**Important**: The model is linear in the **coefficients**, not necessarily in the features themselves.

### Python Example: Multiple Linear Regression

```python
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

# Example: Predicting house price with multiple features
# Features: size (sq ft), bedrooms, age (years)
data = {
    'size': [1500, 1800, 2000, 2200, 2500, 1200, 1400, 3000, 2800, 1600],
    'bedrooms': [3, 3, 4, 4, 4, 2, 2, 5, 4, 3],
    'age': [10, 5, 15, 8, 2, 20, 25, 1, 3, 12],
    'price': [250, 300, 280, 350, 420, 180, 170, 500, 480, 260]  # in $1000s
}

df = pd.DataFrame(data)

# Prepare features and target
X = df[['size', 'bedrooms', 'age']].values
y = df['price'].values

# Train linear regression model
model = LinearRegression()
model.fit(X, y)

# Display coefficients
print("Multiple Linear Regression Results")
print("=" * 50)
print(f"Intercept (a₀): {model.intercept_:.4f}")
print(f"Coefficient for Size (a₁): {model.coef_[0]:.4f}")
print(f"Coefficient for Bedrooms (a₂): {model.coef_[1]:.4f}")
print(f"Coefficient for Age (a₃): {model.coef_[2]:.4f}")
print("\nModel Equation:")
print(f"Price = {model.intercept_:.2f} + {model.coef_[0]:.4f}*Size + {model.coef_[1]:.4f}*Bedrooms + {model.coef_[2]:.4f}*Age")

# Make a prediction
new_house = np.array([[2000, 3, 10]])
predicted_price = model.predict(new_house)
print(f"\nPredicted price for a 2000 sq ft, 3-bedroom, 10-year-old house: ${predicted_price[0]:.2f}k")
```

---

## Real-World Example: House Price Prediction

> **"Let's take an example. This data is coming from Kaggle website."**

### About Kaggle

**Kaggle** is:
- A repository for machine learning datasets
- A platform to build and train ML models
- A host for ML competitions where competitors compare model performance
- "Super fun, so you should try" (direct quote from lecture)

### The Dataset

**Source**: Kaggle - House Sales Price in Washington State

**Target Variable**: `price` (what we want to predict)

**Features**: All other columns describe characteristics of the house, such as:
- Number of bedrooms
- Number of bathrooms
- Square footage (living space)
- Square footage (lot)
- Number of floors
- Waterfront (yes/no)
- View rating
- Condition
- Grade
- Square footage above ground
- Square footage basement
- Year built
- Year renovated
- Zipcode
- Latitude
- Longitude
- ...and more (21 features total)

### Feature Selection Challenge

> **"Because we want to build a simple regression model like this, we want to find out which feature could be a good predictor to predict the house sales price."**

#### Approaches to Feature Selection:

**1. Domain Knowledge**
Think about what matters when buying a house:
- Number of bedrooms might be important → more bedrooms = more expensive?
- Size of the house matters
- Location of the house matters most

**2. Quantitative Evidence: Correlation Matrix**

> **"However, to quantify and have some evidence that which features is most important or likely to important to predict the price, we can have a look at the correlation matrix."**

---

## Correlation Matrix

### What is a Correlation Matrix?

> **"Correlation matrix gives correlation values between the features."**

A correlation matrix is a table showing correlation coefficients between variables:
- **Diagonal elements**: Correlation of a feature with itself = 1.0 (always)
- **Off-diagonal elements**: Correlation between different features

### Interpreting Correlation Values

- **+1.0**: Perfect positive correlation
- **0.0**: No linear correlation
- **-1.0**: Perfect negative correlation

**Correlation Range**: -1 ≤ r ≤ +1

### Example Findings from the Housing Data

> **"You can figure out the square foot living, which is a house size, is most correlated to the price."**

From the first row (correlations with price):
- `sqft_living` (house size) has the **highest correlation** with price
- `grade` (quality of house) also has comparable correlation with price
- Other features like `view`, `floors`, etc. have varying correlations

### Python Example: Computing Correlation Matrix

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Simulated housing data (simplified version)
np.random.seed(42)
n_samples = 100

data = {
    'price': np.random.normal(300, 100, n_samples),
    'sqft_living': np.random.normal(2000, 500, n_samples),
    'bedrooms': np.random.randint(1, 6, n_samples),
    'bathrooms': np.random.randint(1, 4, n_samples),
    'floors': np.random.randint(1, 4, n_samples),
    'grade': np.random.randint(5, 12, n_samples),
    'view': np.random.randint(0, 5, n_samples),
    'condition': np.random.randint(1, 6, n_samples),
}

# Add correlations manually for demonstration
df = pd.DataFrame(data)
# Make sqft_living highly correlated with price
df['sqft_living'] = df['price'] * 5 + np.random.normal(0, 100, n_samples)
# Make grade somewhat correlated
df['grade'] = df['price'] * 0.02 + np.random.normal(6, 2, n_samples)

# Compute correlation matrix
corr_matrix = df.corr()

# Display correlation matrix
print("Correlation Matrix:")
print(corr_matrix)

# Visualize with heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Show correlations with price specifically
print("\nCorrelations with Price (sorted by absolute value):")
price_corr = corr_matrix['price'].drop('price').sort_values(ascending=False)
print(price_corr)
```

---

### ⚠️ **IMPORTANT WARNING**: Correlation vs. Feature Importance

> **"You should be careful when you select multiple features based on correlation matrix because the order of correlation, that means a high correlation or absolute value of a correlation to lower ones. These orders are not directly related to how important the features are."**

#### The Problem: Multicollinearity

**Example from lecture**:
- `sqft_living` has high correlation with `price`
- `grade` has comparable correlation with `price`
- **BUT**: `sqft_living` and `grade` are highly correlated with **each other**!

**Implication**: 
> **"When I add this feature to my model on top of square foot living, that doesn't add so much value because this is pretty similar to this one."**

#### Better Feature Selection

In this case, a different feature like `floors` or `view` might add **more value** than `grade`, even if they have lower individual correlation with `price`, because they provide **new information** not already captured by `sqft_living`.

#### The Lecture's Guidance

> **"In that case, some other variables such as floors or something like that, or maybe view would add better value to predict the price than this one that has a high correlation to the price. You have to be little bit careful."**

> **"We're going to go through a method that actually helps to select the features in right order, but to select just one feature correlation matrix gives a good information."**

**Bottom Line**: For **simple linear regression** (one feature), correlation matrix is very useful. For multiple features, more sophisticated methods are needed.

### Python Example: Demonstrating Multicollinearity

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Simulate data where two features are highly correlated
np.random.seed(123)
n = 100

# Create base feature
sqft_living = np.random.normal(2000, 500, n)

# Create target with relationship to sqft_living
price = 100 + 0.15 * sqft_living + np.random.normal(0, 20, n)

# Create grade highly correlated with sqft_living
grade = 5 + 0.002 * sqft_living + np.random.normal(0, 0.5, n)

# Create view - less correlated with sqft_living
view = np.random.randint(0, 5, n) + 0.0005 * sqft_living + np.random.normal(0, 1, n)

df = pd.DataFrame({
    'price': price,
    'sqft_living': sqft_living,
    'grade': grade,
    'view': view
})

# Check correlations
print("Correlation Matrix:")
print(df.corr())

# Model 1: Only sqft_living
model1 = LinearRegression()
model1.fit(df[['sqft_living']], df['price'])
r2_1 = model1.score(df[['sqft_living']], df['price'])

# Model 2: sqft_living + grade (highly correlated features)
model2 = LinearRegression()
model2.fit(df[['sqft_living', 'grade']], df['price'])
r2_2 = model2.score(df[['sqft_living', 'grade']], df['price'])

# Model 3: sqft_living + view (less correlated features)
model3 = LinearRegression()
model3.fit(df[['sqft_living', 'view']], df['price'])
r2_3 = model3.score(df[['sqft_living', 'view']], df['price'])

print(f"\nR² Score Comparison:")
print(f"Model 1 (sqft_living only): {r2_1:.4f}")
print(f"Model 2 (sqft_living + grade): {r2_2:.4f}  [Improvement: {r2_2 - r2_1:.4f}]")
print(f"Model 3 (sqft_living + view): {r2_3:.4f}  [Improvement: {r2_3 - r2_1:.4f}]")

print("\nNotice: Even though 'grade' might have higher individual correlation with price,")
print("'view' might provide more improvement because it's less correlated with sqft_living.")
```

---

## Simple (Univariate) Linear Regression

> **"Let's begin by that. Let's talk about univariate linear regression. Univariate means the variable is only one."**

### Terminology

- **Univariate Linear Regression** = Simple Linear Regression
- "Univariate" means there is **only one feature/variable**

### Mathematical Form

The simple linear regression model takes this form:

$$\hat{y} = \beta_0 + \beta_1 x + \epsilon$$

Where:
- $\hat{y}$ = predicted value
- $\beta_0$ = **intercept** (y-intercept, where the line crosses the y-axis)
- $\beta_1$ = **slope** (how much y changes for a unit change in x)
- $x$ = feature value
- $\epsilon$ = **residual** (error term)

### Residuals

> **"Then it has residuals that measures the difference between the target value and the prediction value by our model."**

**Definition**: Residual = Actual Value - Predicted Value

$$\epsilon_i = y_i - \hat{y}_i$$

For each data point *i*:
- $y_i$ = actual target value
- $\hat{y}_i$ = predicted value by the model
- $\epsilon_i$ = residual (error)

> **"This residual is important to measure the error and this is for each data point."**

### Visual Example

Imagine we have data points and a regression line:

```
        y
        |     • (data point)
        |    /|
        |   / | ← residual (vertical distance)
        |  /  •
        | /   |
        |/____|_________ x
      intercept
       (β₀)
```

- The **intercept** (β₀) is where the line crosses the y-axis
- The **slope** (β₁) determines the steepness of the line
- Each **residual** is the vertical distance from a data point to the regression line

> **"Each discrepancy of the data points to the regression line is called the residuals."**

### The Optimization Goal

> **"Our goal is to minimize the overall residuals of my model and make my model to produce or predict the value that's as close as possible to the target variable."**

**Objective**: Find the values of β₀ and β₁ that minimize the total error (residuals) across all data points.

**Method** (to be covered in detail later): **Ordinary Least Squares (OLS)** - minimizes the sum of squared residuals.

### Python Example: Visualizing Residuals

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate sample data
np.random.seed(42)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = 2.5 * X.flatten() + 3 + np.random.normal(0, 2, 10)

# Fit linear regression
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Extract coefficients
beta_0 = model.intercept_
beta_1 = model.coef_[0]

# Visualization
plt.figure(figsize=(12, 6))

# Left plot: Regression line with residuals
plt.subplot(1, 2, 1)
plt.scatter(X, y, s=100, alpha=0.6, color='blue', label='Actual Data', zorder=3)
plt.plot(X, y_pred, 'r-', linewidth=2, label=f'Regression Line: ŷ = {beta_0:.2f} + {beta_1:.2f}x', zorder=2)

# Draw residuals as vertical lines
for i in range(len(X)):
    plt.plot([X[i], X[i]], [y[i], y_pred[i]], 'g--', linewidth=1.5, alpha=0.7, zorder=1)
    
plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Simple Linear Regression with Residuals', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Right plot: Residuals distribution
plt.subplot(1, 2, 2)
residuals = y - y_pred
plt.scatter(X, residuals, s=100, alpha=0.6, color='green')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Residual Line')
plt.xlabel('X', fontsize=12)
plt.ylabel('Residuals (ε)', fontsize=12)
plt.title('Residual Plot', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print coefficients and residuals
print(f"Intercept (β₀): {beta_0:.4f}")
print(f"Slope (β₁): {beta_1:.4f}")
print(f"\nResiduals for each data point:")
for i in range(len(X)):
    print(f"  Point {i+1}: y={y[i]:.2f}, ŷ={y_pred[i]:.2f}, ε={residuals[i]:.2f}")
print(f"\nSum of Squared Residuals (SSR): {np.sum(residuals**2):.4f}")
```

---

## Model Training and Implementation

> **"This can be done using a single line using statsmodel OLS package, or there are other packages such as sklearn linear model."**

### Using `statsmodels` OLS

**OLS** = Ordinary Least Squares

> **"However, this is useful because it generates some summary table like this."**

#### Advantages of statsmodels:
- Provides a detailed **summary table** with statistical information
- Includes hypothesis tests for coefficients
- Shows R², adjusted R², F-statistic, and more
- Better for statistical analysis and inference

### Python Example: statsmodels OLS

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Create sample data (using the sqft_living as predictor for price)
np.random.seed(42)
n = 50

data = pd.DataFrame({
    'sqft_living': np.random.normal(2000, 500, n),
})
data['price'] = 100 + 0.15 * data['sqft_living'] + np.random.normal(0, 20, n)

# Fit OLS model using statsmodels
# Method 1: Using formula API (similar to R)
model = ols('price ~ sqft_living', data=data).fit()

# Display summary
print(model.summary())

# Method 2: Using arrays with added constant
X = sm.add_constant(data['sqft_living'])  # Adds intercept term
y = data['price']
model2 = sm.OLS(y, X).fit()

print("\n" + "="*70)
print("Coefficient Summary:")
print("="*70)
print(model2.summary())
```

### Understanding the Summary Table

> **"This summary table has a lot of information including the most interesting part, or rather my coefficient values."**

The summary table includes:

1. **Model Statistics**:
   - R-squared
   - Adjusted R-squared
   - F-statistic
   - Log-Likelihood
   - AIC, BIC

2. **Coefficient Table** (most important):
   - **coef**: Coefficient values (β₀ and β₁)
   - **std err**: Standard error of coefficient estimates
   - **t**: t-statistic for hypothesis testing
   - **P>|t|**: p-value (statistical significance)
   - **[0.025 0.975]**: 95% confidence interval

> **"With this coefficient value, I can determine what's my slope and my intercept is for my simple linear regression model."**

### Using `sklearn` LinearRegression

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data
X = np.array([[1500], [1800], [2000], [2200], [2500]])
y = np.array([250, 300, 320, 350, 400])

# Create and fit model
model = LinearRegression()
model.fit(X, y)

# Get coefficients
print(f"Intercept (β₀): {model.intercept_}")
print(f"Slope (β₁): {model.coef_[0]}")

# Make predictions
X_new = np.array([[2100]])
y_pred = model.predict(X_new)
print(f"Predicted price for 2100 sqft house: ${y_pred[0]:.2f}k")
```

**Note**: sklearn is simpler but doesn't provide the detailed statistical summary that statsmodels offers.

---

## Key Questions to Address

> **"Beside of coefficient values, we can ask some other questions that are important to linear regression."**

The lecture introduces four fundamental questions that will be explored in subsequent lectures:

### 1. How do we determine the coefficients?

> **"We'll begin by how do we determine the coefficients? In other words, how does the model training works under the hood of this package?"**

**What we'll learn**:
- The mathematical optimization process
- Ordinary Least Squares (OLS) method
- How to minimize the sum of squared residuals
- Gradient descent (if applicable)

**Why it matters**: Understanding the underlying math helps you:
- Debug model issues
- Understand when linear regression is appropriate
- Interpret results correctly

---

### 2. How well does my model fit?

> **"We'll also discuss how well my model fits. From the summary table values, what gives an idea of how my model fits?"**

**What we'll learn**:
- **R-squared (R²)**: Proportion of variance explained
- **Adjusted R²**: R² adjusted for number of features
- **Residual plots**: Visual assessment of fit quality
- **Mean Squared Error (MSE)**: Average squared residual

**Why it matters**: 
- Tells you if your model captures the underlying pattern
- Helps compare different models
- Identifies if you need a more complex model

---

### 3. How statistically significant are my coefficients?

> **"Then we'll also talk about how statistically significant my coefficients are. That means how robust our estimation for the coefficient is."**

**What we'll learn**:
- **p-values**: Probability that coefficient is zero
- **t-statistics**: How many standard errors away from zero
- **Confidence intervals**: Range of plausible coefficient values
- **Hypothesis testing**: H₀: β = 0 vs. H₁: β ≠ 0

**Why it matters**:
- Determines if a feature truly matters
- Identifies which features to keep or remove
- Provides confidence in your model's conclusions

**Example interpretation**:
- p-value < 0.05: Feature is statistically significant (reject H₀)
- p-value > 0.05: Cannot conclude feature matters (fail to reject H₀)

---

### 4. How well does my model generalize?

> **"We're going to also talk about how well my model predicts on unseen data. That means, how well does it generalize, which is very important in machine learning."**

**What we'll learn**:
- **Train/Test split**: Separating data for validation
- **Cross-validation**: Robust evaluation technique
- **Overfitting**: When model memorizes training data
- **Underfitting**: When model is too simple

**Why it matters**:
- A model that works only on training data is useless
- Real-world performance depends on generalization
- Helps you build models that work in production

**Best practice**: Always evaluate on held-out test data that the model has never seen during training.

---

## Summary

In this lecture, we covered:

✅ **Supervised Learning Fundamentals**: How models learn from data through parameter optimization

✅ **Linear Regression Definition**: A simple supervised learning model that predicts continuous values

✅ **Key Assumptions**: Linear relationship between features and target

✅ **Linear Relationships**: Examples with positive slopes (house size → price) and negative slopes (age → survival rate)

✅ **Multiple Features**: Linear combination form with multiple coefficients

✅ **Real-World Example**: Kaggle house price dataset from Washington state

✅ **Correlation Analysis**: Using correlation matrix for feature selection (with important caveats about multicollinearity)

✅ **Simple Linear Regression**: Mathematical form with intercept (β₀) and slope (β₁)

✅ **Residuals**: The difference between actual and predicted values, which we aim to minimize

✅ **Implementation**: Using statsmodels OLS or sklearn LinearRegression

✅ **Four Key Questions**:
1. How are coefficients determined?
2. How well does the model fit?
3. How statistically significant are coefficients?
4. How well does the model generalize?

---

## Next Steps

In the following lectures for Module 1, we will explore:
- **Lecture 2**: Coefficient estimation using Ordinary Least Squares (OLS) method
- **Lecture 3**: Model evaluation metrics and goodness-of-fit measures
- **Lecture 4**: Statistical inference and model validation

---

## Additional Resources

### Python Libraries Used:
```python
import numpy as np                          # Numerical computing
import pandas as pd                         # Data manipulation
import matplotlib.pyplot as plt             # Visualization
import seaborn as sns                       # Statistical visualization
from sklearn.linear_model import LinearRegression  # ML model
import statsmodels.api as sm                # Statistical models
from statsmodels.formula.api import ols     # OLS regression
```

### Key Terminology Glossary:

- **Supervised Learning**: Learning from labeled training data
- **Regression**: Predicting continuous numerical values
- **Coefficients**: Parameters of the linear model (β₀, β₁, ...)
- **Intercept**: Where regression line crosses y-axis (β₀)
- **Slope**: Rate of change of y with respect to x (β₁)
- **Residual**: Difference between actual and predicted values (ε)
- **OLS**: Ordinary Least Squares - method to find best coefficients
- **Correlation**: Measure of linear relationship between variables (-1 to +1)
- **R²**: Proportion of variance in y explained by x (0 to 1)
- **p-value**: Probability of observing data if null hypothesis is true
- **Multicollinearity**: When features are highly correlated with each other
- **Generalization**: Model's ability to perform well on unseen data

---

## Practice Exercise

**Exercise**: Use the following data to build a simple linear regression model predicting salary based on years of experience.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Data
data = pd.DataFrame({
    'years_experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'salary': [42, 46, 51, 55, 61, 66, 71, 77, 83, 90]  # in thousands
})

# TODO: Complete the following steps

# Step 1: Visualize the data
plt.figure(figsize=(8, 6))
plt.scatter(data['years_experience'], data['salary'], s=100, alpha=0.6)
plt.xlabel('Years of Experience')
plt.ylabel('Salary ($1000s)')
plt.title('Experience vs Salary')
plt.grid(True, alpha=0.3)
plt.show()

# Step 2: Fit the model using sklearn
X = data[['years_experience']].values
y = data['salary'].values

model = LinearRegression()
model.fit(X, y)

print("=" * 60)
print("SKLEARN RESULTS")
print("=" * 60)
print(f"Intercept (β₀): {model.intercept_:.4f}")
print(f"Slope (β₁): {model.coef_[0]:.4f}")
print(f"\nModel Equation: Salary = {model.intercept_:.2f} + {model.coef_[0]:.2f} * Years")

# Step 3: Fit the model using statsmodels for detailed statistics
X_sm = sm.add_constant(data['years_experience'])
model_sm = sm.OLS(data['salary'], X_sm).fit()

print("\n" + "=" * 60)
print("STATSMODELS RESULTS (Detailed Summary)")
print("=" * 60)
print(model_sm.summary())

# Step 4: Make predictions
y_pred = model.predict(X)

# Step 5: Visualize the regression line
plt.figure(figsize=(10, 6))
plt.scatter(data['years_experience'], data['salary'], s=100, alpha=0.6, label='Actual Data')
plt.plot(data['years_experience'], y_pred, 'r-', linewidth=2, 
         label=f'Regression Line: ŷ = {model.intercept_:.2f} + {model.coef_[0]:.2f}x')
plt.xlabel('Years of Experience', fontsize=12)
plt.ylabel('Salary ($1000s)', fontsize=12)
plt.title('Simple Linear Regression: Experience vs Salary', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Step 6: Calculate and visualize residuals
residuals = y - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(data['years_experience'], residuals, s=100, alpha=0.6, color='green')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Years of Experience', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Residual Plot', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.show()

print("\n" + "=" * 60)
print("RESIDUAL ANALYSIS")
print("=" * 60)
for i in range(len(data)):
    print(f"Point {i+1}: Years={data['years_experience'].iloc[i]}, "
          f"Actual Salary={y[i]:.2f}, Predicted={y_pred[i]:.2f}, "
          f"Residual={residuals[i]:.2f}")

print(f"\nSum of Squared Residuals: {np.sum(residuals**2):.4f}")
print(f"Mean Squared Error: {np.mean(residuals**2):.4f}")
print(f"Root Mean Squared Error: {np.sqrt(np.mean(residuals**2)):.4f}")
print(f"R² Score: {model.score(X, y):.4f}")

# Step 7: Make a prediction for a new data point
new_experience = np.array([[12]])
predicted_salary = model.predict(new_experience)
print(f"\n" + "=" * 60)
print(f"Predicted salary for 12 years of experience: ${predicted_salary[0]:.2f}k")
print("=" * 60)
```

### Expected Output Analysis:

**Questions to answer**:
1. What is the intercept? What does it represent in this context?
2. What is the slope? How do you interpret it?
3. For every additional year of experience, how much does salary increase?
4. Is the relationship statistically significant? (Check p-value from statsmodels)
5. What percentage of variance in salary is explained by years of experience? (Check R²)
6. What would you expect the salary to be for someone with 15 years of experience?
7. Looking at the residual plot, do residuals appear randomly distributed around zero?

---

**End of Lecture 1 Notes**
