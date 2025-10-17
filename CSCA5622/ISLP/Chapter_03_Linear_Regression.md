# Chapter 3 - Linear Regression

## ISLP (Introduction to Statistical Learning with Python)

---

## Section 3.1 - Simple Linear Regression

Simple linear regression models the relationship between a single predictor variable $X$ and a response variable $Y$. The goal is to model $Y$ as simply an approximate linear function of $X$.

### Model Equation

$$
Y = \beta_0 + \beta_1 X + \epsilon
$$

**Where:**
- **$Y$**: Response variable
- **$X$**: Predictor variable
- **$\beta_0$**: Intercept (value of $Y$ when $X = 0$)
- **$\beta_1$**: Slope (how much $Y$ changes with one unit increase in $X$)
- **$\epsilon$**: Error term (captures random variation in $Y$ not explained by $X$ alone)

**Example:** Suppose we are trying to predict salary ($Y$) from years of experience ($X$). We use a model:

$$
\text{salary} = 30{,}000 + 5{,}000 \times \text{years_experience}
$$

This means each added year of experience adds $5,000 to the predicted salary. The "linearity" assumption lets us make predictions with a line, and the "error term" explains why the prediction isn't exact.

---

## Section 3.1.1 - Estimating the Coefficients

### 📘 Objective

Estimate the best-fitting straight line $(\hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x)$ for a set of data points using the **least squares** method.

---

### 🔧 Key Concepts

#### 1. What is RSS?

* **Residual**: $(e_i = y_i - \hat{y}_i)$ is the error between the actual and predicted value.
* **Residual Sum of Squares (RSS)**:

  $$
  RSS = \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \sum_{i=1}^n \left(y_i - (\hat{\beta}_0 + \hat{\beta}_1 x_i)\right)^2
  $$

* RSS is what we **minimize** to find the best line. This means finding the parameters $(\hat{\beta}_0)$ and $(\hat{\beta}_1)$ that make the total squared error as small as possible.

---

#### 2. Why Use Derivatives?

* A **derivative** tells us how a function changes — it's like asking: "if I nudge $(\beta_0)$ or $(\beta_1)$, how does RSS respond?"
* We **take the derivative of RSS** with respect to $(\beta_0)$ and $(\beta_1)$, then **set those derivatives equal to 0** to find the **minimum point** of the RSS "valley".
* This gives us a **system of equations** whose solution yields the best estimates.

---

#### 3. Final Closed-Form Coefficients

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

* **Why is variance squared?** It's not. The denominator is already the squared deviation from the mean — the formula itself is the definition of variance, not its square.

* **Why use the mean?** It centers the data, helping the line balance positive and negative residuals.

* **Why set the derivative = 0?** That's how we find **minimum RSS** — when the slope of the RSS function is zero, we're at the bottom of the "valley".

* **What is $(n)$?** It's the number of data points. It comes up when summing over all samples (e.g., in Equation A: $(\sum y_i = n \beta_0 + \beta_1 \sum x_i)$).

* **How does β₁ relate to β₀?** You need β₁ to compute β₀, because $(\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x})$

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

### 🤖 Scikit-learn Implementation

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

### 🔍 Overview

Once we fit a linear regression model and obtain the estimated coefficients $(\hat{\beta}_0)$ and $(\hat{\beta}_1)$, we want to evaluate how accurate those estimates are. This is essential for determining how much confidence we can place in our model's predictions and whether the relationship between the predictor $(X)$ and the response $(Y)$ is statistically significant.

We achieve this by calculating:
- Standard errors of the coefficient estimates
- Confidence intervals for the estimates
- t-statistics for hypothesis testing

This process builds on the least squares method introduced in Section 3.1.1.

---

### ✅ Key Assumptions

To assess the accuracy, we assume the following:

**Linearity:** The relationship between $(X)$ and $(Y)$ is linear.  
**Independence:** Observations are independent of each other.  
**Homoscedasticity:** The error terms $(\varepsilon_i)$ have constant variance $(\sigma^2)$.  
**Normality (for small samples):** The errors are normally distributed (not required for large samples due to CLT).

---

### 🔢 Variance and Standard Errors

**Residual Sum of Squares (RSS):**

$$
RSS = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

We use this to estimate $(\sigma^2)$:

$$
\hat{\sigma}^2 = \frac{RSS}{n - 2}
$$

**Then, we compute the variances and standard errors for the coefficients:**

**Variance of $(\hat{\beta}_1)$:**

$$
\text{Var}(\hat{\beta}_1) = \frac{\sigma^2}{\sum (x_i - \bar{x})^2}, \quad SE(\hat{\beta}_1) = \sqrt{\text{Var}(\hat{\beta}_1)}
$$

**Variance of $(\hat{\beta}_0)$:**

$$
\text{Var}(\hat{\beta}_0) = \sigma^2 \left( \frac{1}{n} + \frac{\bar{x}^2}{\sum (x_i - \bar{x})^2} \right), \quad SE(\hat{\beta}_0) = \sqrt{\text{Var}(\hat{\beta}_0)}
$$

---

### 📊 Worked Example: Manual Calculation

**Given the dataset:**  
$X = [1, 2, 3]$  
$Y = [2, 3, 5]$

**Step-by-Step:**

1. **Compute means:**
   $$
   \bar{x} = 2, \quad \bar{y} = 3.33
   $$

2. **Compute slope and intercept:**
   $$
   \hat{\beta}_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2} = \frac{3}{2} = 1.5
   $$
   $$
   \hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x} = 3.33 - (1.5)(2) = 0.33
   $$

3. **Predicted values:**
   $$
   \hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i \Rightarrow [1.83, 3.33, 4.83]
   $$

4. **Compute RSS:**
   $$
   RSS = \sum (y_i - \hat{y}_i)^2 = (2-1.83)^2 + (3-3.33)^2 + (5-4.83)^2 = 0.0667 + 0.1089 + 0.0289 = 0.2045
   $$

5. **Estimate $(\hat{\sigma}^2)$:**
   $$
   \hat{\sigma}^2 = \frac{RSS}{n-2} = 0.2045 / 1 = 0.2045, \quad \hat{\sigma} = \sqrt{0.2045} \approx 0.452
   $$

6. **Standard errors:**
   $$
   SE(\hat{\beta}_1) = \sqrt{\frac{0.2045}{2}} = 0.319
   $$
   $$
   SE(\hat{\beta}_0) = \sqrt{0.2045 \left( \frac{1}{3} + \frac{2^2}{2} \right)} = \sqrt{0.2045 \times 2.833} \approx 0.758
   $$

---

### 🎓 Python Implementation (With Interpretation)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.array([1, 2, 3])
y = np.array([2, 3, 5])

x_mean = np.mean(x)
y_mean = np.mean(y)

# Coefficients
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean)**2)
beta_1 = numerator / denominator
beta_0 = y_mean - beta_1 * x_mean

y_hat = beta_0 + beta_1 * x

# RSS
residuals = y - y_hat
RSS = np.sum(residuals**2)
n = len(x)
sigma_squared_hat = RSS / (n - 2)

# Standard Errors
SE_beta_1 = np.sqrt(sigma_squared_hat / np.sum((x - x_mean)**2))
SE_beta_0 = np.sqrt(sigma_squared_hat * (1/n + (x_mean**2 / np.sum((x - x_mean)**2))))

# t-values
t_beta_1 = beta_1 / SE_beta_1
t_beta_0 = beta_0 / SE_beta_0

# Confidence Intervals (95%)
CI_beta_1 = (beta_1 - 2 * SE_beta_1, beta_1 + 2 * SE_beta_1)
CI_beta_0 = (beta_0 - 2 * SE_beta_0, beta_0 + 2 * SE_beta_0)

print("Beta_1:", beta_1, "SE:", SE_beta_1, "CI:", CI_beta_1)
print("Beta_0:", beta_0, "SE:", SE_beta_0, "CI:", CI_beta_0)
```

---

### 🔑 Interpretation

- **Standard Error (SE)** tells us how much our estimate would vary across different datasets.
- **t-value** helps test the hypothesis $(H_0: \beta_j = 0)$.
- **Confidence Intervals** provide a range within which the true coefficient likely lies.
- **A small SE and large t-value** suggest strong evidence that the coefficient is significant.

---

### 🔄 Summary Table

| Coefficient | Estimate | SE    | t-value | 95% CI        |
|-------------|----------|-------|---------|---------------|
| $\beta_0$   | 0.333    | 0.758 | 0.439   | (-1.18, 1.85) |
| $\beta_1$   | 1.5      | 0.319 | 4.70    | (0.862, 2.138)|

---

## Section 3.1.3 - Assessing the Accuracy of the Model

### 🎯 Objective

Once we estimate a linear regression model:

$$
\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i
$$

We want to **evaluate how accurate this model is**. That means:

* How far are actual values from predicted values?
* How precise are our slope and intercept estimates?
* Can we trust these values statistically?

---

### 🔍 Key Concepts in This Section

* **Residuals**: $( e_i = y_i - \hat{y}_i )$
* **Residual Sum of Squares (RSS)**: Measures total error
* **Residual Standard Error (RSE)**: Measures average error
* **Standard Errors (SE)**: Measures precision of $(\hat{\beta}_0)$ and $(\hat{\beta}_1)$
* **Confidence Intervals**: Range of plausible values for coefficients
* **Hypothesis Testing**: Testing whether coefficients are significant

---

### 📘 Step-by-Step Breakdown

#### 1. Residuals

Each prediction has some error. That error is called the **residual**:

$$
e_i = y_i - \hat{y}_i
$$

**Interpretation:**
- **Positive residual**: Model underpredicts the actual value
- **Negative residual**: Model overpredicts the actual value
- **Zero residual**: Perfect prediction (rare in practice)

**Example:** If actual salary is $60,000 but model predicts $55,000, the residual is +$5,000.

---

#### 2. Residual Sum of Squares (RSS)

This is the sum of squared residuals:

$$
RSS = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} e_i^2
$$

**Why square the residuals?**
- Squaring makes all errors positive
- Prevents positive and negative errors from canceling out
- Penalizes large errors more heavily (mathematical convenience for optimization)

**Interpretation:**
- **Lower RSS**: Better model fit
- **Higher RSS**: More unexplained variation in the data
- **Units**: Squared units of the response variable

**Example Calculation:**
```python
# For our dataset: y = [2, 3, 5], ŷ = [1.83, 3.33, 4.83]
residuals = [2-1.83, 3-3.33, 5-4.83] = [0.17, -0.33, 0.17]
RSS = (0.17)² + (-0.33)² + (0.17)² = 0.0289 + 0.1089 + 0.0289 = 0.1667
```

---

#### 3. Residual Standard Error (RSE)

RSS grows with number of data points, so we divide by degrees of freedom to get a standard error:

$$
RSE = \sqrt{\frac{RSS}{n - 2}}
$$

**Why $(n - 2)$ degrees of freedom?**
- We have $n$ data points
- We estimate 2 parameters $(\beta_0, \beta_1)$
- Degrees of freedom = $n - 2$

**Interpretation:**
- This gives us the **average prediction error**
- In units of the response variable $(y)$
- Roughly represents the typical distance between actual and predicted values

**Example:**
```python
RSS = 0.1667  # From above
n = 3         # Number of data points
df = n - 2    # Degrees of freedom
RSE = sqrt(RSS / df) = sqrt(0.1667 / 1) = sqrt(0.1667) ≈ 0.408
```

**Rule of thumb:** RSE represents the typical size of a residual. In our example, the model typically misses by about $408 when predicting salary.

---

#### 4. Standard Errors of Coefficients

These tell us how much $(\hat{\beta}_0)$ and $(\hat{\beta}_1)$ would vary if we collected new data:

**Standard Error of Slope:**

$$
SE(\hat{\beta}_1) = \frac{RSE}{\sqrt{\sum (x_i - \bar{x})^2}}
$$

**Standard Error of Intercept:**

$$
SE(\hat{\beta}_0) = RSE \cdot \sqrt{ \frac{1}{n} + \frac{\bar{x}^2}{\sum (x_i - \bar{x})^2} }
$$

**Interpretation:**
- **Small SE**: Precise estimate, little variation expected in new samples
- **Large SE**: Imprecise estimate, would vary a lot with new data
- **SE depends on**: Sample size $(n)$, spread of $x$ values, and RSE

**Example Calculation:**
```python
# From our dataset
sum_squared_deviations = sum((x - x_bar)²) = (1-2)² + (2-2)² + (3-2)² = 1 + 0 + 1 = 2
RSE = 0.408

SE_beta_1 = RSE / sqrt(sum_squared_deviations) = 0.408 / sqrt(2) ≈ 0.408 / 1.414 ≈ 0.289
SE_beta_0 = RSE * sqrt(1/3 + (2)²/2) = 0.408 * sqrt(0.333 + 4/2) = 0.408 * sqrt(0.333 + 2) = 0.408 * sqrt(2.333) ≈ 0.408 * 1.527 ≈ 0.623
```

---

#### 5. Confidence Intervals

For 95% confidence:

$$
\hat{\beta}_j \pm 2 \cdot SE(\hat{\beta}_j)
$$

**Meaning:**
- There's a 95% chance the true coefficient lies in this interval (assuming normal errors)
- **Wider interval**: More uncertainty about the true value
- **Narrower interval**: More confidence in our estimate

**Example:**
```python
beta_1 = 1.5, SE_beta_1 = 0.289
CI_beta_1 = (1.5 - 2*0.289, 1.5 + 2*0.289) = (1.5 - 0.578, 1.5 + 0.578) = (0.922, 2.078)

beta_0 = 0.33, SE_beta_0 = 0.623
CI_beta_0 = (0.33 - 2*0.623, 0.33 + 2*0.623) = (0.33 - 1.246, 0.33 + 1.246) = (-0.916, 1.576)
```

**Interpretation:** We're 95% confident that:
- The true slope is between 0.922 and 2.078 (doesn't include 0, so significant)
- The true intercept is between -0.916 and 1.576 (includes 0, so not significant)

---

#### 6. Hypothesis Testing (t-test)

We ask: "Does a variable really affect the outcome? Is $\beta_j \neq 0$? Or could it just be noise?"

**Null Hypothesis:**
$$
H_0: \beta_j = 0
$$

**Alternative Hypothesis:**
$$
H_a: \beta_j \neq 0
$$

**t-Statistic:**

$$
t = \frac{\hat{\beta}_j}{SE(\hat{\beta}_j)}
$$

**Decision Rule:**
- Compare $t$ to a t-distribution with $(n - 2)$ degrees of freedom
- Large absolute $t$-value (typically $> 2$) → reject $H_0$ (significant)
- Small absolute $t$-value → fail to reject $H_0$ (not significant)

**Example:**
```python
t_beta_1 = beta_1 / SE_beta_1 = 1.5 / 0.289 ≈ 5.19
t_beta_0 = beta_0 / SE_beta_0 = 0.33 / 0.623 ≈ 0.53

# t_beta_1 = 5.19 > 2, so slope is significant (p < 0.05)
# t_beta_0 = 0.53 < 2, so intercept is not significant (p > 0.05)
```

**Interpretation:**
- **Slope significant**: Years of experience has a meaningful relationship with salary
- **Intercept not significant**: When experience = 0, salary could plausibly be 0 (or any value in our CI)

---

### 📊 Complete Worked Example (Manual Calculation)

**Given dataset:**

| x | y  |
|---|----|
| 1 | 2  |
| 2 | 3  |
| 3 | 5  |

**Step 1: Calculate means**
$$
\bar{x} = \frac{1+2+3}{3} = 2, \quad \bar{y} = \frac{2+3+5}{3} = \frac{10}{3} \approx 3.333
$$

**Step 2: Calculate slope**
$$
\hat{\beta}_1 = \frac{ \sum (x_i - \bar{x})(y_i - \bar{y}) }{ \sum (x_i - \bar{x})^2 } = \frac{ (1-2)(2-3.333) + (2-2)(3-3.333) + (3-2)(5-3.333) }{ (1-2)^2 + (2-2)^2 + (3-2)^2 } = \frac{ (-1)(-1.333) + 0 + (1)(1.667) }{ 1 + 0 + 1 } = \frac{1.333 + 1.667}{2} = \frac{3}{2} = 1.5
$$

**Step 3: Calculate intercept**
$$
\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x} = 3.333 - 1.5 \times 2 = 3.333 - 3 = 0.333
$$

**Step 4: Generate predictions**
$$
\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i = 0.333 + 1.5 x_i
$$
- For $x=1$: $\hat{y} = 0.333 + 1.5 \times 1 = 1.833$
- For $x=2$: $\hat{y} = 0.333 + 1.5 \times 2 = 3.333$
- For $x=3$: $\hat{y} = 0.333 + 1.5 \times 3 = 4.833$

**Step 5: Calculate residuals**
- $e_1 = 2 - 1.833 = 0.167$
- $e_2 = 3 - 3.333 = -0.333$
- $e_3 = 5 - 4.833 = 0.167$

**Step 6: Calculate RSS**
$$
RSS = (0.167)^2 + (-0.333)^2 + (0.167)^2 = 0.0279 + 0.1109 + 0.0279 = 0.1667
$$

**Step 7: Calculate RSE**
$$
RSE = \sqrt{\frac{RSS}{n-2}} = \sqrt{\frac{0.1667}{1}} = \sqrt{0.1667} \approx 0.408
$$

**Step 8: Calculate standard errors**
**For slope:**
$$
SE(\hat{\beta}_1) = \frac{RSE}{\sqrt{\sum (x_i - \bar{x})^2}} = \frac{0.408}{\sqrt{2}} \approx \frac{0.408}{1.414} \approx 0.289
$$

**For intercept:**
$$
SE(\hat{\beta}_0) = RSE \cdot \sqrt{ \frac{1}{n} + \frac{\bar{x}^2}{\sum (x_i - \bar{x})^2} } = 0.408 \cdot \sqrt{ \frac{1}{3} + \frac{4}{2} } = 0.408 \cdot \sqrt{0.333 + 2} = 0.408 \cdot \sqrt{2.333} \approx 0.408 \cdot 1.527 \approx 0.623
$$

**Step 9: Calculate t-statistics**
$$
t_{\beta_1} = \frac{\hat{\beta}_1}{SE(\hat{\beta}_1)} = \frac{1.5}{0.289} \approx 5.19
$$
$$
t_{\beta_0} = \frac{\hat{\beta}_0}{SE(\hat{\beta}_0)} = \frac{0.333}{0.623} \approx 0.534
$$

**Step 10: Calculate 95% confidence intervals**
**For slope:**
$$
CI_{\beta_1} = 1.5 \pm 2 \times 0.289 = (1.5 - 0.578, 1.5 + 0.578) = (0.922, 2.078)
$$

**For intercept:**
$$
CI_{\beta_0} = 0.333 \pm 2 \times 0.623 = (0.333 - 1.246, 0.333 + 1.246) = (-0.913, 1.579)
$$

---

### 💻 Complete Python Implementation

```python
import numpy as np

# Dataset
x = np.array([1, 2, 3])
y = np.array([2, 3, 5])

# Step 1: Calculate means
x_bar = np.mean(x)
y_bar = np.mean(y)

# Step 2: Estimate coefficients
numerator = np.sum((x - x_bar) * (y - y_bar))
denominator = np.sum((x - x_bar)**2)
beta_1 = numerator / denominator
beta_0 = y_bar - beta_1 * x_bar

# Step 3: Generate predictions
y_hat = beta_0 + beta_1 * x

# Step 4: Calculate residuals and RSS
residuals = y - y_hat
RSS = np.sum(residuals**2)

# Step 5: Calculate RSE
n = len(x)
RSE = np.sqrt(RSS / (n - 2))

# Step 6: Calculate standard errors
SE_beta_1 = RSE / np.sqrt(np.sum((x - x_bar)**2))
SE_beta_0 = RSE * np.sqrt(1/n + (x_bar**2 / np.sum((x - x_bar)**2)))

# Step 7: Calculate t-statistics
t_beta_1 = beta_1 / SE_beta_1
t_beta_0 = beta_0 / SE_beta_0

# Step 8: Calculate 95% confidence intervals
CI_beta_1_lower = beta_1 - 2 * SE_beta_1
CI_beta_1_upper = beta_1 + 2 * SE_beta_1
CI_beta_0_lower = beta_0 - 2 * SE_beta_0
CI_beta_0_upper = beta_0 + 2 * SE_beta_0

# Display results
print(f"β₁ = {beta_1:.3f}, SE = {SE_beta_1:.3f}, t = {t_beta_1:.3f}, 95% CI = ({CI_beta_1_lower:.3f}, {CI_beta_1_upper:.3f})")
print(f"β₀ = {beta_0:.3f}, SE = {SE_beta_0:.3f}, t = {t_beta_0:.3f}, 95% CI = ({CI_beta_0_lower:.3f}, {CI_beta_0_upper:.3f})")
print(f"RSS = {RSS:.4f}, RSE = {RSE:.3f}")
```

**Expected Output:**
```
β₁ = 1.500, SE = 0.289, t = 5.190, 95% CI = (0.922, 2.078)
β₀ = 0.333, SE = 0.623, t = 0.534, 95% CI = (-0.913, 1.579)
RSS = 0.1667, RSE = 0.408
```

---

### 🧠 Key Takeaways

| Term | Purpose | Interpretation |
|------|---------|----------------|
| **Residuals** | Individual errors | $e_i = y_i - \hat{y}_i$ |
| **RSS** | Total squared error | $\sum e_i^2$ - lower is better |
| **RSE** | Average prediction error | $\sqrt{RSS/(n-2)}$ - in units of y |
| **SE** | Precision of estimates | How much $\hat{\beta}$ varies |
| **t-value** | Test significance | $t = \hat{\beta}/SE$ - large = significant |
| **CI** | Plausible range | $\hat{\beta} \pm 2 \times SE$ |

**Statistical Significance:**
- **Slope significant** (t = 5.19 > 2): Years of experience meaningfully affects salary
- **Intercept not significant** (t = 0.53 < 2): When experience = 0, salary could plausibly be 0

**Model Quality:**
- RSE ≈ 0.408 means the model typically misses by about $408 when predicting salary
- This represents the irreducible error - even a perfect model couldn't do better than this

---

### 🎓 Learning Check

**Questions to test understanding:**

1. **What does a residual of -5 mean?**
2. **Why do we square residuals in RSS?**
3. **Why divide by (n-2) for RSE?**
4. **What does a large t-value tell us?**
5. **When would you trust a confidence interval?**

**Answers:**
1. The model overpredicted by 5 units
2. To make all errors positive and penalize large errors more
3. Because we estimate 2 parameters, leaving n-2 degrees of freedom
4. Strong evidence against the null hypothesis (coefficient ≠ 0)
5. When the interval is narrow and doesn't include problematic values

---

### 🚀 Next Steps

Now that you understand model accuracy assessment, you're ready to:
- Evaluate regression models in practice
- Compare different models using these metrics
- Move on to multiple linear regression (Section 3.2)
- Learn about model selection and validation techniques

---

## Section 3.2 – Multiple Linear Regression

When we have more than one predictor variable, the simple linear regression model is too restrictive. Multiple linear regression lets us model the response \(Y\) as a linear function of several predictors \(X_1, X_2, \dots, X_p\).

### 3.2.1 Estimating the Regression Coefficients

#### Model Definition

We assume the model:

\[
Y_i = \beta_0 + \beta_1 X_{i1} + \beta_2 X_{i2} + \cdots + \beta_p X_{ip} + \varepsilon_i
\]

- Here \(\varepsilon_i\) are error terms assumed to satisfy \(E[\varepsilon_i] = 0\), \(\mathrm{Var}(\varepsilon_i) = \sigma^2\), independence, etc.
- Each \(\beta_j\) is interpreted as “the expected change in \(Y\) when \(X_j\) increases by 1 unit, *holding all other predictors constant*.”

We choose \(\hat\beta_0, \hat\beta_1, \dots, \hat\beta_p\) to minimize the **Residual Sum of Squares (RSS)**:

\[
RSS = \sum_{i=1}^n \bigl( y_i - \hat\beta_0 - \hat\beta_1 x_{i1} - \cdots - \hat\beta_p x_{ip} \bigr)^2
\]

#### Matrix Form & Normal Equations

Let:

- \(X\) be the \(n \times (p + 1)\) **design matrix**, with first column all 1s (for the intercept) and subsequent columns the predictor values.
- \(\boldsymbol{\beta} = [\beta_0, \beta_1, \dots, \beta_p]^T\)
- \(\mathbf{y} = [y_1, \dots, y_n]^T\)

Then the least squares solution is:

\[
\hat{\boldsymbol{\beta}} = (X^\top X)^{-1} X^\top \mathbf{y}
\]

provided \(X^\top X\) is invertible (i.e. predictors are not perfectly collinear).

This generalizes the “normal equations” from simple regression to multiple predictors.

### 3.2.2 Some Important Questions

When applying multiple regression, there are several key statistical questions we must answer:

#### Question 1: Is there a relationship between the response and predictors?

This is a test of the null hypothesis:

\[
H_0: \beta_1 = \beta_2 = \cdots = \beta_p = 0
\]

versus the alternative that *at least one* \(\beta_j \neq 0\).

We use the **F-statistic**:

\[
F = \frac{ (TSS - RSS) / p }{ RSS / (n - p - 1) }
\]

where

\[
TSS = \sum_{i=1}^n (y_i - \bar{y})^2
\]

If \(F\) is large, we reject \(H_0\), concluding that at least one predictor is useful.

#### Question 2: Are all predictors useful?

After rejecting the global null, we typically examine each coefficient \(\beta_j\) individually using a **t-test**:

\[
t_j = \frac{\hat{\beta}_j}{ SE(\hat{\beta}_j) }
\]

We compute \(SE(\hat{\beta}_j)\) from the variance-covariance matrix:

\[
\mathrm{Var}(\hat{\boldsymbol{\beta}}) = \sigma^2 (X^\top X)^{-1}
\]

Estimate \(\sigma^2\) by:

\[
\hat{\sigma}^2 = \frac{RSS}{n - p - 1}
\]

Then:

\[
SE(\hat{\beta}_j) = \sqrt{ \hat{\sigma}^2 \, [(X^\top X)^{-1}]_{jj} }
\]

We compare \(t_j\) to a \(t\)-distribution with \(n - p - 1\) degrees of freedom.

#### Question 3: How well does the model fit?

Use the **coefficient of determination**:

\[
R^2 = 1 - \frac{RSS}{TSS}
\]

- \(R^2\) measures the proportion of variance in \(y\) explained by predictors.
- In simple regression, \(R^2 = r^2\) (the square of the correlation), but in multiple regression this direct equivalence breaks down. :contentReference[oaicite:0]{index=0}  
- Because adding more predictors always reduces \(RSS\), \(R^2\) will never decrease when you add predictors—even if they are useless. That’s why we also consider **Adjusted \(R^2\)**:

\[
\text{Adjusted } R^2 = 1 - \frac{RSS/(n - p - 1)}{TSS/(n - 1)}
\]

Adjusted \(R^2\) imposes a penalty for adding predictors that do not sufficiently improve model fit.

#### Question 4: How well can we predict new observations?

Given a new predictor vector \(\mathbf{x}^* = [1, x_1^*, \dots, x_p^*]^\top\), the predicted response is:

\[
\hat{y}^* = \mathbf{x}^{*\top} \hat{\boldsymbol{\beta}} = \hat\beta_0 + \hat\beta_1 x_1^* + \dots + \hat\beta_p x_p^*
\]

We can form:

- A **confidence interval** for the mean response at \(\mathbf{x}^*\)
- A **prediction interval** for a future observed \(y^*\), which is wider because it includes irreducible error

### 3.2.3 Properties & Inference

Under standard assumptions (linearity, independence, homoscedasticity, no perfect multicollinearity):

- \(\hat{\boldsymbol{\beta}}\) is **unbiased**: \(E[\hat{\beta}_j] = \beta_j\)
- The variance-covariance matrix is \(\sigma^2 (X^\top X)^{-1}\)
- Standard errors and hypothesis tests follow as above

However, some complications arise:

- **Multicollinearity**: predictors highly correlated with each other inflate variances of \(\hat{\beta}_j\), making individual coefficients unstable.
- **High-leverage and influential points**: some observations may disproportionately affect model fit.
- **Nonlinearity, heteroscedasticity, autocorrelation, outliers**: these violate model assumptions and distort inference. (See text’s “Potential Problems” section) :contentReference[oaicite:1]{index=1}

### 3.2.4 Diagnostics, Model Selection, and Practical Concerns

- **Residual plots** (residuals vs. fitted values) help detect patterns, non-constant variance, or non-linearity.
- **Variance inflation factor (VIF)** is a diagnostic for multicollinearity.
- **Model selection strategies** (in later chapters) include *forward selection*, *backward elimination*, and *mixed stepwise methods*.
- **Overfitting risk**: as you add more variables, the model may start fitting noise rather than signal. Use cross-validation and validation sets.
- Be cautious interpreting \(R^2\) alone; also inspect Adjusted \(R^2\), standard errors, and significance tests.

### 📎 Python / Code Guidance (Scaffold, not full solution)

```python
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import numpy as np

# Suppose df is your DataFrame with predictors x1, x2, …, xp and target y
X = df[['x1', 'x2', 'x3']]    # (n × p) predictors
X_with_const = sm.add_constant(X)  # adds intercept column
y = df['y']

# Using statsmodels for inference
model = sm.OLS(y, X_with_const).fit()
print(model.summary())  # provides coefficients, SEs, t-stats, p-values, R^2, F-stat

# Using sklearn for prediction (no inference)
lr = LinearRegression()
lr.fit(X, y)
coefs = lr.coef_
intercept = lr.intercept_

y_hat = lr.predict(X)
residuals = y - y_hat
RSS = np.sum(residuals**2)
n, p = X.shape
sigma2_hat = RSS / (n - p - 1)
````

You will want to understand:

* What `model.summary()` reports (coefficients, SE, (t)-stats, p-values, (R^2), (F)-statistic)
* The difference between sklearn (for prediction) and statsmodels (for statistical inference)
* How to compute residuals, RSS, degrees of freedom, and (\hat\sigma^2)

---

### ✅ Summary

* Multiple regression extends simple regression to multiple predictors with conditional (ceteris paribus) interpretation of coefficients.
* You estimate coefficients by minimizing RSS using the normal equation ( \hat\beta = (X^\top X)^{-1} X^\top y ).
* Global model significance is tested via an (F)-test; individual coefficients via (t)-tests.
* (R^2) measures model explanatory power, but **Adjusted (R^2)** accounts for model complexity.
* Diagnostics are critical for reliability—watch out for multicollinearity, outliers, nonlinearity, heteroscedasticity.

---

## Section 3.3 – Other Considerations in the Regression Model

When applying linear regression in practice, there are several real‑world issues and extensions we must consider. This section explores:

- Qualitative (categorical) predictors  
- Extensions to relax standard linearity/additivity assumptions  
- Potential problems and diagnostic concerns  

---

### 3.3.1 Qualitative Predictors

Not all predictors are numeric. Some are categorical (e.g. gender, region, color). To include these in a linear model, we use **indicator (dummy) variables**.

#### Two-Level Qualitative Variables

If a predictor \(Z\) has two levels (e.g. Male / Female), we encode:

\[
D_i = \begin{cases}
1 & \text{if } Z_i = \text{Female} \\
0 & \text{if } Z_i = \text{Male}
\end{cases}
\]

Then include \(D_i\) as a predictor:

\[
Y_i = \beta_0 + \beta_1 D_i + \varepsilon_i
\]

Interpretation:

- The intercept \(\beta_0\) is the mean outcome for the baseline category (Male, when \(D = 0\))
- \(\beta_1\) is the difference between Female and Male

If you instead coded 0 = Female and 1 = Male, the numeric values of \(\beta_0, \beta_1\) would change, but **predicted values** and **interpretations relative to baseline** remain consistent.

#### Qualitative Predictors with More Than Two Levels

If \(Z\) has \(K\) categories (e.g. 4 regions: North, South, East, West), you need \(K - 1\) dummy variables:

\[
D_{i1}, D_{i2}, \dots, D_{i, K-1}
\]

One category is left as the reference (baseline). For example, if baseline is “North”:

- \(D_{i1} = 1\) if “South”, 0 otherwise  
- \(D_{i2} = 1\) if “East”, 0 otherwise  
- \(D_{i3} = 1\) if “West”, 0 otherwise  

Then:

\[
Y_i = \beta_0 + \beta_1 D_{i1} + \beta_2 D_{i2} + \beta_3 D_{i3} + \varepsilon_i
\]

Interpretation:

- \(\beta_1\) = difference (South vs North); \(\beta_2\) = (East vs North); etc.
- Always omit one dummy to avoid **perfect multicollinearity** (dummy trap).

---

### 3.3.2 Extensions of the Linear Model

Linear models as originally formulated assume two restrictive assumptions:

1. **Additivity**: the effect of each predictor is independent (no interactions)  
2. **Linearity**: each predictor has a linear effect on \(Y\), i.e. one unit change in \(X\) changes \(Y\) by a constant amount

To relax these, we can incorporate:

#### Interaction Terms (Removing Additive Assumption)

Two variables may **interact** so that the effect of one depends on the level of another.

For two predictors \(X_1\) and \(X_2\):

\[
Y_i = \beta_0 + \beta_1 X_{i1} + \beta_2 X_{i2} + \beta_3 (X_{i1} \cdot X_{i2}) + \varepsilon_i
\]

Here \(\beta_3\) captures the **interaction effect**.

- The marginal effect of \(X_1\) on \(Y\) becomes \(\beta_1 + \beta_3 X_2\). It depends on \(X_2\).  
- By the **hierarchical principle**, if you include the interaction term, you should include the main (lower-order) terms \(X_1\) and \(X_2\) even if they are not statistically significant on their own.

You can also interact **qualitative and quantitative** variables:

\[
Y_i = \beta_0 + \beta_1 X + \beta_2 D + \beta_3 (X \cdot D) + \varepsilon_i
\]

Interpretation: effect of \(X\) may differ by group defined by \(D\).

#### Polynomial Terms & Nonlinear Transformations

To relax the linearity assumption, include nonlinear terms of predictors (while keeping linearity in parameters). For example:

\[
Y_i = \beta_0 + \beta_1 X_i + \beta_2 X_i^2 + \beta_3 X_i^3 + \varepsilon_i
\]

This is still a linear model in \(\beta\)s, but captures curvature.

Alternatively, you can transform predictors (e.g. \(\log X\), \(\sqrt X\)) or use basis expansions (splines, piecewise polynomials) in advanced topics.

#### Summary of Extensions

- **Interactions**: allow one predictor’s effect to depend on another  
- **Polynomial / nonlinear terms**: allow curvature  
- **Combining categorical & continuous**: via interaction of dummy × numeric  
- Always check interpretable meaning: marginal effects become context‑dependent

---

### 3.3.3 Potential Problems and Diagnostics

Even with extensions, linear models may suffer from issues. This section describes common problems and how to detect/fix them.

| Problem | Description | Diagnostics / Remedies |
|---|---|---|
| **Nonlinearity** | True relationship is not linear | Residual plots vs fitted values; include polynomial or transform variables |
| **Heteroscedasticity** | Residual variance increases or decreases with predictors | Residual vs fitted plot shows “funnel” shape; use weighted least squares or transform response |
| **Correlated Errors (Autocorrelation)** | Residuals not independent, e.g. time series data | Durbin–Watson test, autocorrelation plots; use time-series models (ARIMA) |
| **Outliers** | Observations with large error | Examine studentized residuals, Cook’s distance; consider removing or modeling separately |
| **High Leverage / Influential Points** | Observations far in predictor space that have undue influence | Leverage values, influence measures; inspect and possibly drop or model separately |
| **Multicollinearity** | Predictors highly correlated with each other | Variance Inflation Factor (VIF), condition indices; drop or combine variables, use regularization (ridge / lasso) |

A few additional notes:

- **Residual plots**: In multiple regression, residuals are often plotted against fitted values \(\hat{y}_i\) rather than a single \(X\) because there are many predictors.
- **Transformations**: e.g. \(\log Y\) or \(\sqrt Y\) are common when variance of errors grows with level of \(Y\).
- **Outlier impact**: Just because an outlier is extreme does not mean it should be removed — it may reflect an interesting regime or anomaly.
- **Multicollinearity’s effect**: It inflates standard errors, making coefficients less reliable even if \(R^2\) is high.

---

### 📎 Example Sketch with Extensions (Code Guidance)

```python
import pandas as pd
import statsmodels.api as sm
import numpy as np

# Suppose df with numerical predictors x1, x2, and categorical D (0/1)
X = df[['x1', 'x2']]
D = df['D']
X['interaction'] = X['x1'] * X['x2']         # numeric × numeric
X['x1_D'] = X['x1'] * D                       # numeric × dummy interaction
X['x2_D'] = X['x2'] * D                       # numeric × dummy interaction
X['x1_sq'] = X['x1']**2                       # polynomial term

Xc = sm.add_constant(X)
model = sm.OLS(df['y'], Xc).fit()
print(model.summary())
````

In this setup:

* `interaction` captures (x_1 \times x_2)
* `x1_D` and `x2_D` capture how slopes differ by group
* `x1_sq` adds quadratic term to model curvature

The `model.summary()` output will include coefficients, standard errors, p-values, (R^2), F-statistic, etc.

---

### ✅ Summary & Study Tips (for 3.3.2 Focus)

* Extensions allow flexibility beyond additive and linear assumptions.
* **Interactions** provide conditional effects among predictors.
* **Polynomials and transforms** let you model curvature while retaining interpretability.
* Always check diagnostics to validate assumptions.
* Don’t over-interpret coefficients when interactions or nonlinear terms are included — interpret marginal effects.

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
