# Coefficient Significance and Test Error

**Lecture**: Module 1, Lecture 4  
**Course**: CSCA5622  
**Topic**: Statistical Significance of Coefficients, Hypothesis Testing, Confidence Intervals, and Train/Test Split

---

## 1. Introduction to Coefficient Significance

The core question addressing in this lecture is: **When can we say a coefficient value is "significant"?**

### The Concept of Significance
In the context of linear regression, "significance" is a statistical term with a specific meaning:
*   **Significant**: The relationship captured by the coefficient is real; the feature associated with this coefficient actually has a non-zero impact on the target variable. We are confident that if we gathered more data, this relationship would persist.
*   **Not Significant**: The non-zero value of the coefficient we obtained is likely just due to random chance or noise in the specific sample of data we used. The true value of the coefficient in the population is likely **zero**. If the true coefficient is zero, then the feature has no effect on the target.

### The Magnitude Trap: Absolute Values are Misleading
A common misconception is that a "large" coefficient number automatically implies significance, and a "small" number implies insignificance. This is **incorrect**.

*   **Scenario**: 
    *   Coefficient $\beta_1 = -4000$ (Large negative number)
    *   Coefficient $\beta_2 = 280$ (Smaller positive number)
    
*   **Intuitive (but wrong) thought**: 
    *   "$\beta_1$ is massive, so it must be very significant. It's far from zero."
    *   "$\beta_2$ is smaller, maybe less significant."

*   **Reality Check**:
    *   The absolute magnitude depends entirely on the **units** of the input features and the target variable.
    *   **Example**:
        *   Suppose we are predicting **House Price**.
        *   If the target is measured in **Dollars**, a coefficient might be 100,000.
        *   If the target is measured in **Millions of Dollars**, that same coefficient becomes 0.1.
        *   The *relationship* hasn't changed, only the *scale*.
    *   Therefore, the distance from zero ($|\hat{\beta} - 0|$) is meaningless without a "measuring stick" to judge scale.

> **Slide Visualization**:
> Imagine a slide showing a table of coefficients with different magnitudes.
> *   `Coef A`: -4123.5
> *   `Coef B`: 281.2
> *   The lecturer emphasizes that without context (Standard Error), these numbers tell us nothing about reliability. The slide likely has a big red "X" or question mark over the idea of judging by size alone.

---

## 2. Standard Error of Coefficients

To judge if a coefficient is significant, we need that "measuring stick". We need to know the **uncertainty** or **spread** of our estimate. This is the **Standard Error (SE)**.

### Visualizing the Distribution of Coefficients
When we fit a model, we get *one* estimate for $\beta_1$ (let's call it $\hat{\beta}_1$). But if we had sampled a different set of training data from the same population, we would have gotten a slightly different $\hat{\beta}_1$.

*   Imagine the estimated coefficient $\hat{\beta}$ is a random variable drawn from a theoretical distribution of all possible coefficient estimates.
*   **Center**: The mean of this distribution is the true population parameter $\beta$ (if our estimator is unbiased).
*   **Spread (Variance)**: This tells us how "wobbly" our estimate is.
    *   **Wide Spread (High SE)**: The distribution is flat and wide. Our specific estimate $\hat{\beta}_1$ could be far from the true mean. Even if we see a value of 5, the "true" value could easily be 0.
    *   **Narrow Spread (Low SE)**: The distribution is sharp and peaked. Our estimate is likely very close to the true value. If we see 5, and the spread is 0.1, we are very confident the true value is not 0.

### Methods to Calculate Standard Error

We can derive the Standard Error in two main ways: using a theoretical model or using computational resampling.

#### A. Model-Based Method (Theoretical)
This approach relies on statistical theory and assumptions about the underlying data generation process.

1.  **Covariance Matrix**:
    *   In matrix notation, the variance-covariance matrix of the coefficients is given by:
        $$ \text{Var}(\hat{\beta}) = (X^T X)^{-1} \sigma^2 $$
    *   Here, $\sigma^2$ is the variance of the error terms (residuals).
    *   The diagonal elements of this matrix give us the variances of each coefficient ($\text{Var}(\hat{\beta}_0), \text{Var}(\hat{\beta}_1)$).
    *   The square root of these variances is the **Standard Error**.

2.  **Key Dependencies**:
    *   **Residual Variance ($\sigma^2$)**: The Standard Error is proportional to the noise in the data. 
        *   More noise in data $\rightarrow$ Higher SE $\rightarrow$ Less significance.
    *   **Data Spread ($X$)**: The Standard Error is inversely proportional to the spread of the feature $X$.
        *   More spread in $X$ $\rightarrow$ Lower SE $\rightarrow$ More significance.
        *   *Analogy*: It's easier to balance a ruler on your finger if it's long (wide spread of mass) than if it's short. Similarly, it's easier to fix a regression line if the X points are far apart.

**Critical Assumption: Homoscedasticity**
*   **Definition**: The term means "same scatter". It assumes that the variance of the error terms ($\epsilon$) is constant for all values of $X$.
*   **Violation: Heteroscedasticity**:
    *   This means "different scatter". The error variance changes as $X$ changes.
    *   **Slide Visualization**:
        *   A scatter plot showing a **"Cone Shape"**.
        *   On the left (low X), points are tightly clustered around the line.
        *   On the right (high X), points fan out widely.
        *   This violates the assumption used in the simple model-based formula.
    *   **Consequence**: If we have heteroscedasticity, the standard model-based Standard Errors will be wrong (usually too small), leading us to claim significance when we shouldn't.

#### B. Bootstrapping (Resampling Method)
If we don't trust the assumptions (like if we see that cone shape), we can use **Bootstrapping**. This uses the computer's power to build the distribution empirically.

**The Bootstrapping Algorithm (Step-by-Step)**:

1.  **Original Sample**: Start with your original dataset of size $N$.
2.  **Resample**: Create a "bootstrap sample" by randomly drawing $N$ observations from the original dataset **with replacement**.
    *   *With Replacement* means the same data point can be picked multiple times, and some might not be picked at all.
3.  **Fit**: Train your linear regression model on this bootstrap sample.
4.  **Record**: Save the coefficient values (e.g., slope and intercept) from this fit.
5.  **Iterate**: Repeat steps 2-4 many times (e.g., $B=1000$ times).
6.  **Calculate Stats**: You now have a list of 1000 slope estimates.
    *   **Bootstrap Mean**: The average of these 1000 slopes.
    *   **Bootstrap Standard Error**: The standard deviation of these 1000 slopes.

This gives us a robust measure of standard error without relying on the homoscedasticity assumption.

---

### Python Example: Bootstrapping from Scratch

This script generates synthetic data, introduces heteroscedasticity (noise increases with X), and then uses bootstrapping to estimate the coefficient uncertainty.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def generate_data(n=100):
    """
    Generates data with heteroscedasticity (cone shape).
    """
    np.random.seed(42)
    X = 2 * np.random.rand(n, 1)
    # Noise increases as X increases (X * randn)
    noise = (0.5 + X) * np.random.randn(n, 1) 
    y = 4 + 3 * X + noise
    return X, y

def bootstrap_coefficients(X, y, n_iterations=1000):
    """
    Performs bootstrapping to estimate coefficient distributions.
    """
    n_size = len(X)
    slopes = []
    intercepts = []
    
    for i in range(n_iterations):
        # 1. Resample indices with replacement
        indices = np.random.randint(0, n_size, n_size)
        X_sample = X[indices]
        y_sample = y[indices]
        
        # 2. Fit model
        model = LinearRegression()
        model.fit(X_sample, y_sample)
        
        # 3. Store coefficients
        slopes.append(model.coef_[0][0])
        intercepts.append(model.intercept_[0])
        
    return np.array(slopes), np.array(intercepts)

# Main Execution
X, y = generate_data(100)
slopes, intercepts = bootstrap_coefficients(X, y)

# Analysis
slope_mean = np.mean(slopes)
slope_se = np.std(slopes)
intercept_mean = np.mean(intercepts)
intercept_se = np.std(intercepts)

print("--- Bootstrap Results ---")
print(f"Slope Mean:       {slope_mean:.4f}")
print(f"Slope Std Error:  {slope_se:.4f}")
print(f"Intercept Mean:   {intercept_mean:.4f}")
print(f"Intercept SE:     {intercept_se:.4f}")

# Visualization of Slope Distribution
plt.figure(figsize=(10, 6))
plt.hist(slopes, bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)

# Plot Mean
plt.axvline(slope_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {slope_mean:.2f}')

# Plot +/- 2 SE lines (Approx 95% CI)
plt.axvline(slope_mean - 2*slope_se, color='green', linestyle=':', linewidth=2, label='-2 SE')
plt.axvline(slope_mean + 2*slope_se, color='green', linestyle=':', linewidth=2, label='+2 SE')

plt.title('Bootstrap Distribution of Slope Coefficient')
plt.xlabel('Estimated Slope Value')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**Interpretation of the Graph**:
*   The histogram shows the probability distribution of the slope.
*   The width of the histogram visually represents the Standard Error.
*   If the value **0** falls far outside the histogram (to the left or right), the coefficient is significant.

---

## 3. Hypothesis Testing for Coefficients

Once we have the Standard Error (SE), we formalize the significance check using Hypothesis Testing.

### The Hypotheses Framework
We set up two competing claims about the world:
1.  **Null Hypothesis ($H_0$)**: 
    *   "The coefficient is actually zero."
    *   $\beta_1 = 0$
    *   Meaning: The feature $X$ has no effect on $Y$. Any non-zero slope we see is just noise.
2.  **Alternative Hypothesis ($H_1$ or $H_a$)**: 
    *   "The coefficient is not zero."
    *   $\beta_1 \neq 0$
    *   Meaning: The feature $X$ has a real effect on $Y$.

### The Test Statistic (t-score)
We calculate a score that tells us: *How many standard errors away from zero is our estimate?*

$$ t = \frac{\text{Observed Value} - \text{Hypothesized Value}}{\text{Standard Error}} $$

Since our hypothesized value for $H_0$ is 0:

$$ t = \frac{\hat{\beta} - 0}{SE(\hat{\beta})} = \frac{\hat{\beta}}{SE(\hat{\beta})} $$

*   **Large |t|**: The estimate is far from zero relative to the error. Evidence against $H_0$.
*   **Small |t|**: The estimate is close to zero relative to the error. Consistent with $H_0$.

> **Distribution Note**: This statistic follows a **Student's t-distribution**.
*   The t-distribution looks like a Normal distribution but with "fatter tails".
*   Fatter tails reflect the extra uncertainty when we estimate variance from small samples.
*   As sample size $N$ increases ($N > 30$), the t-distribution converges to the **Normal (Z) distribution**.

### P-Value and Critical Regions

How large does $t$ have to be to reject the null hypothesis?

1.  **Significance Level ($\alpha$)**: 
    *   The threshold for probability we are willing to accept for making a "Type I Error" (rejecting $H_0$ when it's actually true).
    *   Standard convention: $\alpha = 0.05$ (5%).
    *   This means we demand 95% confidence.

2.  **Rejection Regions**:
    *   For a 2-tailed test (checking if $\beta \neq 0$), we split $\alpha$ in half.
    *   2.5% on the far left tail, 2.5% on the far right tail.
    *   **Critical Values**: For a standard normal distribution, these boundaries are at **-1.96** and **+1.96**.
    *   If our $t$-score falls outside these bounds (i.e., $|t| > 1.96$), we reject $H_0$.

3.  **The P-Value**:
    *   Instead of just checking the bounds, we calculate the exact probability.
    *   **Definition**: The probability of observing a $t$-statistic as extreme (or more extreme) than the one we calculated, assuming $H_0$ is true.
    *   **Visual**: The area under the tails of the distribution starting from our $t$-score.

### Decision Rule Summary
*   **If P-value < $\alpha$ (0.05)**:
    *   The observation is extremely unlikely under the Null Hypothesis.
    *   **Action**: Reject $H_0$.
    *   **Conclusion**: The coefficient is **Statistically Significant**.
*   **If P-value $\ge$ $\alpha$ (0.05)**:
    *   The observation is plausibly consistent with the Null Hypothesis.
    *   **Action**: Fail to Reject $H_0$.
    *   **Conclusion**: The coefficient is **Not Significant**.

> **Slide Visualization**:
> *   The slide shows a standard bell curve centered at 0.
> *   Two vertical lines at -1.96 and +1.96 mark the "Critical Values".
> *   The tails beyond these lines are shaded Red ("Rejection Region").
> *   A specific t-score is plotted.
> *   The area beyond that t-score is shaded Green ("p-value").
> *   **Visual Logic**: If the Green area is smaller than the Red area (Total 0.05), the result is significant.

### Lecture Example Calculation
*   **Intercept**:
    *   T-score: $-10$
    *   Analysis: This is 10 standard deviations below the mean. Extremely rare.
    *   Conclusion: Significant.
*   **Slope**:
    *   T-score: $145$
    *   Analysis: This is 145 standard deviations above the mean. Practically impossible under $H_0$.
    *   Conclusion: Significant.
*   **P-values**: The transcript mentions p-values are "almost 0".

---

### Python Example: Statsmodels for Hypothesis Testing

While `scikit-learn` is great for prediction, `statsmodels` is the go-to library for this kind of statistical inference (hypothesis testing, p-values).

```python
import numpy as np
import statsmodels.api as sm

# Generate some linear data
np.random.seed(10)
X = np.random.rand(100, 1) * 10
y = 3 + 2.5 * X + np.random.randn(100, 1) * 2  # True: Intercept=3, Slope=2.5

# CRITICAL STEP: Add a constant term for the intercept
# Statsmodels does not add an intercept by default (unlike sklearn)
X_with_const = sm.add_constant(X)

# Fit the OLS (Ordinary Least Squares) model
model = sm.OLS(y, X_with_const)
results = model.fit()

# Print the full statistical summary
print(results.summary())
```

**Analyzing the Output Table**:
Look for the middle table in the output:
```text
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          3.1234      0.345      9.053      0.000       2.439       3.808
x1             2.4876      0.058     42.890      0.000       2.372       2.603
==============================================================================
```
*   **coef**: The estimated coefficients ($\hat{\beta}$).
*   **std err**: The Standard Error ($SE$).
*   **t**: The t-score. ($t = \text{coef} / \text{std err}$).
    *   Example: $2.4876 / 0.058 \approx 42.89$.
*   **P>|t|**: The p-value.
    *   `0.000` means $< 0.001$. Very significant.
*   **[0.025 0.975]**: The 95% Confidence Interval.

---

## 4. Confidence Intervals (CI)

Hypothesis testing gives a Yes/No answer. Confidence Intervals give a **Range** of plausible values.

### A. Coefficient Confidence Interval (95%)
Instead of saying "The slope is 2.5", we say "We are 95% confident the true slope is between 2.37 and 2.60".

**Formula**:
$$ CI_{95\%} = \hat{\beta} \pm (1.96 \times SE(\hat{\beta})) $$
*(Note: 1.96 is for Normal distribution; for small samples, use the critical t-value).*

**Connection to Significance**:
*   If the 95% CI **does not include 0** (e.g., [2.3, 2.6]), the coefficient is statistically significant at $\alpha=0.05$.
*   If the 95% CI **includes 0** (e.g., [-0.5, 0.5]), the coefficient is not significant. We cannot rule out zero.

### B. Regression Line Confidence Interval
This creates a "band" around the fitted line.
*   It represents the uncertainty of the **mean response** for a given X.
*   "If we repeated the experiment many times, the average regression line would fall in this band 95% of the time."
*   **Shape**: It is narrowest at the mean of X ($\bar{x}$) and gets wider as we move away from the center (hourglass shape).
*   **Reason**: The regression line pivots around the center point $(\bar{x}, \bar{y})$, so leverage is higher at the ends.

### C. Prediction Interval (95%)
This creates a wider band around the line.
*   It represents the range where a **single new data point** is likely to fall.
*   It accounts for two sources of error:
    1.  Uncertainty in the model (the line could be wrong).
    2.  Inherent noise in the data (the $\epsilon$ term).
*   **Formula Logic**:
    $$ \text{Variance}_{pred} = \text{Variance}_{model} + \text{Variance}_{data} $$
*   Since it adds the data variance, the Prediction Interval is always **wider** than the Confidence Interval of the line.

> **Slide Visualization**:
> *   A scatter plot with the regression line in the middle.
> *   **Inner Shaded Region (Orange)**: The 95% Confidence Interval for the line. Tighter.
> *   **Outer Shaded Region (Blue)**: The 95% Prediction Interval for data points. Wider.
> *   The lecturer points out that this blue region is useful for detecting **outliers**. If a point falls outside the blue region, it is statistically very unlikely (an anomaly).

---

## 5. Training vs Test Error

Finally, we move from interpreting coefficients to evaluating **predictive performance**.

### The Problem with Training Error
If we calculate the Mean Squared Error (MSE) on the same data we used to train the model, we get an optimistic bias.
*   The model has "seen" these answers.
*   It might have memorized noise (Overfitting).
*   A complex model might have 0 Training Error but be useless for new data.

### The Solution: Train/Test Split
We physically divide the dataset into two mutually exclusive sets.

1.  **Training Set (e.g., 80%)**:
    *   Used to run gradient descent or the normal equation.
    *   Used to find $\hat{\beta}$ values.
2.  **Test Set (e.g., 20%)**:
    *   **Held Out**: The model is NOT allowed to see this during training.
    *   Used ONLY to calculate error metrics at the end.
    *   Simulates "real world" performance on unseen data.

### The Workflow & Mathematics
1.  **Fit**:
    $$ \hat{\beta} = \text{fit}(X_{train}, y_{train}) $$
2.  **Predict (Train)**:
    $$ \hat{y}_{train} = X_{train} \cdot \hat{\beta} $$
3.  **Predict (Test)**:
    $$ \hat{y}_{test} = X_{test} \cdot \hat{\beta} $$
    *   *Note: We use the SAME $\hat{\beta}$ derived from training.*
4.  **Evaluate**:
    $$ MSE_{train} = \frac{1}{N_{train}} \sum (y_{train} - \hat{y}_{train})^2 $$
    $$ MSE_{test} = \frac{1}{N_{test}} \sum (y_{test} - \hat{y}_{test})^2 $$

### Interpretation of Results
*   **Scenario A: $MSE_{train} \approx MSE_{test}$ (and both low)**
    *   Good fit. The model generalizes well.
*   **Scenario B: $MSE_{train} \ll MSE_{test}$**
    *   **Overfitting**. The model is "Over-Parameterized".
    *   It fits the noise in the training set but fails on general patterns.
    *   This discrepancy is the primary signal for overfitting.
*   **Scenario C: $MSE_{train}$ and $MSE_{test}$ are both high**
    *   **Underfitting**. The model is too simple (e.g., fitting a line to a curve).

---

### Python Example: Train/Test Split Workflow

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Generate Data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 2. Split Data
# test_size=0.2 means 20% of data goes to test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Data Size: {X_train.shape[0]}")
print(f"Test Data Size:     {X_test.shape[0]}")

# 3. Train Model (Fit on Train ONLY)
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Predict
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)  # Predict on unseen data

# 5. Evaluate
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print("-" * 30)
print(f"Training MSE: {mse_train:.4f}")
print(f"Test MSE:     {mse_test:.4f}")
print(f"Test R^2:     {r2_test:.4f}")
print("-" * 30)

# Logic check
if mse_test > mse_train * 1.5:
    print("Warning: Test error significantly higher than Train error. Possible Overfitting.")
else:
    print("Model seems to generalize well.")
```

---

## 6. Conceptual Practice & Self-Assessment

Based on the lecture concepts, here are practice scenarios to test your understanding.

### Practice Scenario 1: Interpreting Significance
**Given**:
*   A linear regression model predicts `Salary` based on `Years_Experience`.
*   Coefficient for `Years_Experience`: $5000$.
*   Standard Error for this coefficient: $6000$.
*   $\alpha = 0.05$.

**Tasks**:
1.  Calculate the t-score.
2.  Determine if the coefficient is significant.

**Solution**:
1.  **Calculate t-score**:
    $$ t = \frac{\text{Coef}}{SE} = \frac{5000}{6000} \approx 0.833 $$
2.  **Conclusion**:
    *   The critical value for $\alpha=0.05$ (large N) is approx $1.96$.
    *   Our $|t| = 0.833$.
    *   $0.833 < 1.96$.
    *   The t-score is **not** in the rejection region.
    *   **Result**: Fail to reject Null Hypothesis. The relationship is **not significant**. The large coefficient ($5000$) is unreliable because the error ($6000$) is even larger.

### Practice Scenario 2: Sample Size Effects
**Question**: If we keep the same coefficient estimate ($\hat{\beta}$) and the same data variance, but we **increase the sample size N** significantly, what happens to the t-score and significance?

**Answer**:
*   Standard Error formula involves dividing by $\sqrt{N}$ (roughly).
*   As $N$ increases, Standard Error **decreases**.
*   Since $t = \hat{\beta} / SE$, as SE decreases, the t-score **increases**.
*   **Result**: With more data, we become more confident. A coefficient that was borderline insignificant might become significant simply by collecting more data, because we are shrinking the uncertainty spread.

### Practice Scenario 3: Test Error Logic
**Given**:
*   Model A: Train MSE = 10, Test MSE = 12.
*   Model B: Train MSE = 5, Test MSE = 50.

**Question**: Which model is better? Why?

**Answer**:
*   **Model A** is better.
    *   The gap between Train (10) and Test (12) is small, implying good generalization.
*   **Model B** is suffering from severe **overfitting**.
    *   It memorized the training set (Train MSE 5 is very low), but fails completely on new data (Test MSE 50).
    *   We always care about **Test Error** (performance on unseen data) over Training Error.

---

## 7. Summary of Module 1 (Simple Linear Regression)

This lecture concludes the module on Simple Linear Regression. Here is the roadmap of what has been covered:

1.  **The Model**: 
    *   $y = \beta_0 + \beta_1 x + \epsilon$.
    *   Assumption of linearity.
2.  **Fitting (Least Squares)**: 
    *   Objective: Minimize Residual Sum of Squares (RSS).
    *   Derivatives set to 0 to find optimal $\beta$.
3.  **Evaluation (Goodness of Fit)**: 
    *   $R^2$: Percentage of variance explained.
    *   RMSE: Average error in target units.
4.  **Inference (Significance)**: 
    *   Standard Errors, t-scores, p-values.
    *   Confidence Intervals.
    *   Understanding that a line is just an *estimate* with uncertainty.
5.  **Validation**: 
    *   Train/Test split to ensure the model actually works on future data.

**Next Steps**: 
In the upcoming videos, we will move beyond Simple Linear Regression to **Multiple Linear Regression** (more features) and **Polynomial Regression** (higher-order complexity).

---

## 8. Glossary of Terms

*   **Coefficient**: The weight assigned to a feature. Represents the change in $Y$ for a 1-unit change in $X$.
*   **Standard Error (SE)**: A measure of the statistical accuracy of an estimate. The standard deviation of the theoretical distribution of the coefficient.
*   **Homoscedasticity**: The assumption that the variance of residuals is constant across all $X$.
*   **Heteroscedasticity**: The violation where residual variance changes (often cone-shaped).
*   **Bootstrapping**: A resampling technique to estimate standard errors by repeatedly sampling with replacement from the data.
*   **Null Hypothesis ($H_0$)**: The default assumption that there is no relationship ($\beta = 0$).
*   **t-score**: A standardized score indicating how many standard errors an estimate is from zero.
*   **p-value**: The probability of seeing the data if the Null Hypothesis were true. Low p-value = High significance.
*   **Confidence Interval (CI)**: A range of values that likely contains the true population parameter.
*   **Prediction Interval**: A wider range that likely contains a specific future data point.
*   **Train/Test Split**: Dividing data to separate model fitting (Train) from model evaluation (Test).
*   **Overfitting**: When a model fits training noise rather than the underlying pattern, characterized by low Train Error but high Test Error.
