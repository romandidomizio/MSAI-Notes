# Chapter 3 - Linear Regression

## ISLP (Introduction to Statistical Learning with Python)

---

## Section 3.1 - Simple Linear Regression

simple linear regression models the relationship between a single predictor variable X\nand a response variable Y. The goal is to model Y as simply an approximate linear function of X.

## Model Equation:
\\X Y = \beta_0 + \beta_1 X \\plus \\epsilon\\\n

- `Y` : response variable
- `X`: predictor variable
- `\beta_0`: intercept -- value of Y when X = 0
- `\beta_1`: slope -- how much Y changes with one unit increase in X
- `epsilon`: error term -- captures random variation in Y not explained by X alone

Example: Suppose we are trying to predict salary (Y`) from years of experience (X). We use a model:

    salary = 30,000 + 5,000 * years_experience

This means each added year of experience adds $5,000 to the predicted salary. The "linearity" assumption lets us make predictions with a line, and the "error term" explains why the prediction isn't exact.

---

## Section 3.1.1 - Estimating the Coefficients

### 📘 Concept Introduction

In simple linear regression, we want to fit a straight line that best describes the relationship between `X` (predictor) and `Y` (response). The model is:

[
\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i
]

Our goal is to estimate the coefficients (\hat{\beta}_0) (intercept) and (\hat{\beta}_1) (slope).

---

### 🧠 Mathematical Foundation

We estimate the line using **Least Squares**, by minimizing the **Residual Sum of Squares (RSS)**:

[
RSS = \sum_{i=1}^{n} (y_i - \hat{y}*i)^2 = \sum*{i=1}^{n} (y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i)^2
]

To minimize this, we take the **partial derivatives** of RSS with respect to each parameter, set them to zero, and solve.

1. **Take derivative of RSS w.r.t.** (\hat{\beta}_0):
   [
   \frac{\partial RSS}{\partial \hat{\beta}_0} = -2 \sum (y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i)
   ]

2. **Take derivative w.r.t.** (\hat{\beta}_1):
   [
   \frac{\partial RSS}{\partial \hat{\beta}_1} = -2 \sum x_i (y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i)
   ]

Set both equal to 0, solve the system, and derive the formulas:

[
\hat{\beta}_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}
]

[
\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}
]

* Numerator = **Covariance** of X and Y
* Denominator = **Variance** of X

---

### 💡 Intuitive Explanation

* The **slope** (\hat{\beta}_1) tells us how much Y changes on average for a one-unit increase in X.
* The **intercept** (\hat{\beta}_0) is the predicted value of Y when X = 0.
* We minimize RSS because it penalizes large prediction errors and gives us the "best fitting line".

---

### 🧮 Practical Example

Let's say:

```python
X = [1, 2, 3, 4, 5]
Y = [1, 3, 3, 5, 7]
```

You'd calculate:

1. Means:
   (\bar{x} = 3), (\bar{y} = 3.8)

2. Numerator (covariance):
   (\sum (x_i - \bar{x})(y_i - \bar{y}) = 10)

3. Denominator (variance):
   (\sum (x_i - \bar{x})^2 = 10)

4. Final coefficients:
   (\hat{\beta}_1 = 1),
   (\hat{\beta}_0 = 0.8)

So the best-fit line is:

[
\hat{y} = 0.8 + 1 \cdot x
]

---

### 🧪 Python Code Implementation (Manual)

```python
import pandas as pd
import matplotlib.pyplot as plt

data = {'x': [1, 2, 3, 4, 5], 'y': [1, 3, 3, 5, 7]}
df = pd.DataFrame(data)

x_mean = df['x'].mean()
y_mean = df['y'].mean()

numerator = sum((df['x'] - x_mean) * (df['y'] - y_mean))
denominator = sum((df['x'] - x_mean)**2)

beta_1 = numerator / denominator
beta_0 = y_mean - beta_1 * x_mean

df['y_hat'] = beta_0 + beta_1 * df['x']

plt.scatter(df['x'], df['y'], label='Data')
plt.plot(df['x'], df['y_hat'], color='red', label='Regression Line')
plt.legend()
plt.title('Simple Linear Regression')
plt.show()
```

---

### 🤖 Python Code Implementation (Scikit-Learn)

```python
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array(df['x']).reshape(-1, 1)  # must be 2D
y = np.array(df['y'])

model = LinearRegression()
model.fit(X, y)

print("Slope (beta_1):", model.coef_[0])
print("Intercept (beta_0):", model.intercept_)

df['y_hat_sklearn'] = model.predict(X)

plt.scatter(df['x'], df['y'], label='Data')
plt.plot(df['x'], df['y_hat_sklearn'], color='green', label='Sklearn Line')
plt.legend()
plt.title('Sklearn Regression')
plt.show()
```

---

### ✅ Comprehension Recap & Key Answers

1. **Why minimize RSS?**
   It gives the best-fitting line by penalizing large squared errors.

2. **Why derivatives?**
   They tell us how RSS changes as we tweak coefficients. Setting derivatives to 0 finds the minimum (bottom of the loss curve).

3. **Why is variance in denominator squared?**
   It's not. The denominator is the **sum of squared deviations**, which defines variance.

4. **Why is x 2D in sklearn?**
   scikit-learn expects X to be a matrix (even if just one feature), hence shape `(n_samples, 1)`.

---

## Section 3.1.2 - Assessing the Accuracy of the Coefficient Estimates
*[Content to be added]*

## Section 3.1.3 - Assessing the Accuracy of the Model
*[Content to be added]*

---

## Section 3.2 - Multiple Linear Regression

### 3.2.1 - Estimating the Regression Coefficients
*[Content to be added]*

### 3.2.2 - Some Important Questions
*[Content to be added]*

---

## Section 3.3 - Other Considerations in the Regression Model

### 3.3.1 - Qualitative Predictors
*[Content to be added]*

### 3.3.2 - Extensions of the Linear Model
*[Content to be added]*

### 3.3.3 - Potential Problems
*[Content to be added]*

---

## Section 3.4 - The Marketing Plan
*[Content to be added]*

---

## Section 3.5 - Comparison of Linear Regression with K-Nearest Neighbors
*[Content to be added]*

---

## Section 3.6 - Lab: Linear Regression

### 3.6.1 - Importing packages
*[Content to be added]*

### 3.6.2 - Simple Linear Regression
*[Content to be added]*

### 3.6.3 - Multiple Linear Regression
*[Content to be added]*

### 3.6.4 - Multivariate Goodness of Fit
*[Content to be added]*

### 3.6.5 - Interaction Terms
*[Content to be added]*

### 3.6.6 - Non-linear Transformations of the Predictors
*[Content to be added]*

### 3.6.7 - Qualitative Predictors
*[Content to be added]*

---

## Section 3.7 - Exercises
*[Content to be added]*

---

## Notes
*[Add your notes here]*
