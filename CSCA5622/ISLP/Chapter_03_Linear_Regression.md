# Chapter 3 - Linear Regression

## ISLP (Introduction to Statistical Learning with Python)

---

## Section 3.1 - Simple Linear Regression

Simple linear regression models the relationship between a single predictor variable $X$
and a response variable $Y$. The goal is to model $Y$ as simply an approximate linear function of $X$.

## Model Equation:

$$
Y = \beta_0 + \beta_1 X + \epsilon
$$

* **$Y$** : response variable
* **$X$** : predictor variable
* **$\beta_0$** : intercept (value of $Y$ when $X = 0$)
* **$\beta_1$** : slope (how much $Y$ changes with one unit increase in $X$)
* **$\epsilon$** : error term (captures random variation in $Y$ not explained by $X$ alone)

Example: Suppose we are trying to predict salary ($Y$) from years of experience ($X$). We use a model:

$$
\text{salary} = 30{,}000 + 5{,}000 \times \text{years_experience}
$$

This means each added year of experience adds $5,000 to the predicted salary.
The "linearity" assumption lets us make predictions with a line, and the "error term" explains why the prediction isn't exact.

---

## Section 3.1.1 - Estimating the Coefficients

### 📘 Objective:

Estimate the best-fitting straight line $(\hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x)$ for a set of data points using the **least squares** method.

---

### 🔧 Key Concepts

#### 1. **What is RSS?**

* **Residual**: $(e_i = y_i - \hat{y}_i)$ is the error between the actual and predicted value.
* **Residual Sum of Squares (RSS):**
  $$
  RSS = \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \sum_{i=1}^n \left(y_i - (\hat{\beta}_0 + \hat{\beta}_1 x_i)\right)^2
  $$
* RSS is what we **minimize** to find the best line. This means finding the parameters $(\hat{\beta}_0)$ and $(\hat{\beta}_1)$ that make the total squared error as small as possible.

---

#### 2. **Why Use Derivatives?**

* A **derivative** tells us how a function changes — it's like asking: "if I nudge $(\beta_0)$ or $(\beta_1)$, how does RSS respond?"
* We **take the derivative of RSS** with respect to $(\beta_0)$ and $(\beta_1)$, then **set those derivatives equal to 0** to find the **minimum point** of the RSS "valley".
* This gives us a **system of equations** whose solution yields the best estimates.

---

#### 3. **Final Closed-Form Coefficients**

* **Slope**:
  $$
  \hat\beta_1 = \frac{ \sum (x_i - \bar x)(y_i - \bar y) }{ \sum (x_i - \bar x)^2 }
  $$
  This is the **covariance of x and y** over the **variance of x**.
* **Intercept**:
  $$
  \hat\beta_0 = \bar y - \hat\beta_1 \bar x
  $$
  This ensures the line always passes through the mean point $((\bar x, \bar y))$.

---

### 🧠 Important Clarifications

* **Why is variance squared?**
  It's not. The denominator is already the squared deviation from the mean — the formula itself is the definition of variance, not its square.

* **Why use the mean?**
  It centers the data, helping the line balance positive and negative residuals.

* **Why set the derivative = 0?**
  That's how we find **minimum RSS** — when the slope of the RSS function is zero, we're at the bottom of the "valley".

* **What is $(n)$?**
  It's the number of data points. It comes up when summing over all samples (e.g., in Equation A: $(\sum y_i = n \beta_0 + \beta_1 \sum x_i)$).

* **How does β₁ relate to β₀?**
  You need β₁ to compute β₀, because $(\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x})$

---

### ✍️ Manual Python Implementation (No Libraries)

```python
import pandas as pd
import matplotlib.pyplot as plt

# Sample data
data = {
    "x": [1, 2, 3, 4, 5],
    "y": [2, 4, 5, 4, 5]
}
df = pd.DataFrame(data)

# Calculate means
x = df["x"]
y = df["y"]
x_mean = x.mean()
y_mean = y.mean()

# Calculate slope (beta_1)
numerator = sum((x - x_mean) * (y - y_mean))
denominator = sum((x - x_mean)**2)
beta_1 = numerator / denominator

# Calculate intercept (beta_0)
beta_0 = y_mean - beta_1 * x_mean

# Predict values
df["y_hat"] = beta_0 + beta_1 * df["x"]

# Plot
plt.scatter(df["x"], df["y"], color="blue", label="Actual")
plt.plot(df["x"], df["y_hat"], color="red", label="Regression Line")
plt.legend()
plt.title("Manual Linear Regression")
plt.grid(True)
plt.show()
```

---

### 🤖 `scikit-learn` Implementation

```python
from sklearn.linear_model import LinearRegression

# Step 1: Prepare inputs
X = df[["x"]]  # 2D array (n samples x 1 feature)
y = df["y"]    # 1D array (n samples)

# Step 2: Create and train the model
model = LinearRegression()
model.fit(X, y)

# Step 3: Get coefficients
sk_beta_1 = model.coef_[0]
sk_beta_0 = model.intercept_

# Step 4: Predict
df["y_hat_sklearn"] = model.predict(X)
```

---

### 🧠 Explanation of Each Line in Scikit-learn

* `df[["x"]]` keeps `X` as a 2D array → required because `sklearn` expects matrix-shaped input
* `model.fit(X, y)` runs the least squares solution under the hood
* `model.coef_` gives β₁ (the slope)
* `model.intercept_` gives β₀ (the intercept)
* `model.predict(X)` uses $(\hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x)$

---

### 🧠 Your Questions Answered

**Q: Why does X need to be 2D?**
A: Because `scikit-learn` is built for multi-feature datasets. Even with one feature, it expects the shape $(n_samples, n_features) = (5, 1)$

**Q: Why does y not need to be 2D?**
A: Because the target is always a single value per row — a 1D vector $(shape = (n,))$ is fine.

**Q: What does model.fit do?**
A: Internally computes β₀ and β₁ by minimizing RSS and stores them in the model.

**Q: What do model.coef_ and model.intercept_ do?**
A: They store and return the fitted slope (β₁) and intercept (β₀) respectively.

---

### ✅ Summary of Learned Skills

You now know how to:

* Define and interpret RSS
* Derive $\beta_0$ and $\beta_1$ using calculus (least squares)
* Implement linear regression from scratch in Python
* Use `scikit-learn` for regression
* Plot and interpret regression lines
* Explain regression theory and code in your own words

---

## Section 3.1.2 - Assessing the Accuracy of the Coefficient Estimates

---

#### 🔍 Why This Section Matters

When we fit a linear regression model using least squares, we obtain estimated values for the intercept (\hat{\beta}_0) and slope (\hat{\beta}_1). These values are **estimates**, not exact truths — they change with different samples due to natural data variation. In this section, we learn how to **quantify our uncertainty** in these estimates using **standard errors**, **confidence intervals**, and **hypothesis tests**.

---

### 🔢 Key Concepts and Assumptions

#### ✅ Assumptions:

To assess accuracy reliably, we assume:

* **Independence of Errors**: Observations don’t influence each other.
* **Constant Variance (Homoscedasticity)**: All errors have the same variance, (\sigma^2).
* **Normality (Optional)**: Errors are normally distributed.

---

### 📐 Definitions

#### ✅ Residual Standard Error (RSE):

Estimates the variance of residuals:
[
\hat\sigma^2 = \frac{RSS}{n - 2}, \quad \hat\sigma = \sqrt{\hat\sigma^2}
]

---

### 📊 Variance and Standard Errors of Coefficients

Using least squares, we derive the variance and standard error of the estimated coefficients:

#### Variance and Standard Error of (\hat\beta_1):

[
\mathrm{Var}(\hat\beta_1) = \frac{\sigma^2}{\sum (x_i - \bar{x})^2}, \quad SE(\hat\beta_1) = \sqrt{\mathrm{Var}(\hat\beta_1)}
]

#### Variance and Standard Error of (\hat\beta_0):

[
\mathrm{Var}(\hat\beta_0) = \sigma^2 \left( \frac{1}{n} + \frac{\bar{x}^2}{\sum (x_i - \bar{x})^2} \right), \quad SE(\hat\beta_0) = \sqrt{\mathrm{Var}(\hat\beta_0)}
]

* Interpretation:

  * More spread in (x) → smaller SE((\hat\beta_1))
  * Larger (n) → smaller SE((\hat\beta_0))

---

### 🧪 Hypothesis Testing

#### Null Hypothesis:

[
H_0: \beta_j = 0
]

#### t-Statistic:

[
t = \frac{ \hat\beta_j - 0 }{ SE(\hat\beta_j) }
]

Compare this value to a t-distribution with (n - 2) degrees of freedom. If (t) is large in magnitude, we reject the null (i.e., the predictor is significant).

---

### 🟦 Confidence Intervals

For a 95% CI:
[
\hat\beta_j \pm t_{\alpha/2, df=n-2} \times SE(\hat\beta_j)
]

Approximate:
[
\hat\beta_j \pm 2 \cdot SE(\hat\beta_j)
]

---

### 🧮 Fully Worked Example (Manual Calculation)

Given dataset:
[
(x, y) = (1, 2), (2, 3), (3, 5)
]

Using standard regression formulas:

* (\hat\beta_1 = 1.5)
* (\hat\beta_0 = 0.333)
* (\hat y_i = \hat\beta_0 + \hat\beta_1 x_i)
* (RSS = \sum (y_i - \hat y_i)^2)
* (\hat\sigma^2 = \frac{RSS}{1}) (since (n=3))
* Plug into formulas above to compute SEs and t-values

---

### 🧪 Python Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Dataset
x = np.array([1, 2, 3])
y = np.array([2, 3, 5])

# Means
x_mean = np.mean(x)
y_mean = np.mean(y)

# Coefficients
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean)**2)
beta_1 = numerator / denominator
beta_0 = y_mean - beta_1 * x_mean

# Predictions
y_hat = beta_0 + beta_1 * x

# Residuals and RSS
residuals = y - y_hat
RSS = np.sum(residuals**2)

# Variance estimate
n = len(x)
sigma_squared_hat = RSS / (n - 2)

# Standard Errors
SE_beta_1 = np.sqrt(sigma_squared_hat / np.sum((x - x_mean)**2))
SE_beta_0 = np.sqrt(sigma_squared_hat * (1/n + (x_mean**2 / np.sum((x - x_mean)**2))))

# Confidence Intervals
CI_beta_1 = (beta_1 - 2 * SE_beta_1, beta_1 + 2 * SE_beta_1)
CI_beta_0 = (beta_0 - 2 * SE_beta_0, beta_0 + 2 * SE_beta_0)

# t-values
t_beta_1 = beta_1 / SE_beta_1
t_beta_0 = beta_0 / SE_beta_0

# Display Results
print("Beta_1:", beta_1, "SE:", SE_beta_1, "95% CI:", CI_beta_1)
print("Beta_0:", beta_0, "SE:", SE_beta_0, "95% CI:", CI_beta_0)
```

---

### 🔑 Summary of Outputs

| Coefficient | Estimate | SE    | t-value | 95% CI        |
| ----------- | -------- | ----- | ------- | ------------- |
| (\beta_0)   | 0.333    | 0.624 | 0.53    | (-0.91, 1.58) |
| (\beta_1)   | 1.5      | 0.289 | 5.20    | (0.92, 2.08)  |

---

### 🔁 Final Notes

* **High t-value** (like for (\beta_1)) suggests strong evidence that the slope isn’t zero.
* **Wider CI** → more uncertainty; **narrow CI** → more precise estimate.
* Always estimate **error variance** using residuals to plug into SE formulas.

---

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
