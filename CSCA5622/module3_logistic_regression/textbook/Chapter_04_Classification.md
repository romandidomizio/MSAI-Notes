# Chapter 4 - Classification

## ISLP (Introduction to Statistical Learning with Python)

---

## Section 4.1 – An Overview of Classification

### What Is Classification?

* In regression (Chapters 1–3), the **response variable ( Y )** is **quantitative (numeric)**.

* But many real problems have **qualitative (categorical)** responses. For example:
   - Will a person default on a credit card? (Yes / No)
   - Is an email spam or not?
   - What is a person’s disease class?

* Predicting a categorical (qualitative) response is called **classification**.

* Many classification methods first estimate **probabilities** for each class, and then decide a class label (for example by thresholding).

  * So classification methods often behave like regression in the probability domain.

### Example from the Book: Default Data

* The authors use the **Default** dataset (10,000 observations). (ISLR/ISLP) ([Bookdown][1])

  * Predictors: `balance` (monthly credit—how much the person owes), `income`
  * Response: `default` (Yes / No)
* They illustrate plotting the points (balance vs income), coloring by default status to visualize separation. ([Bookdown][1])
* Then they use boxplots of `balance` and `income` by default status to show how predictors differ by class. ([Bookdown][1])

### Key Points in the Overview

* Classification is ubiquitous in practice (medicine, fraud detection, email spam, etc.).
* Because the response is categorical, we can’t directly apply ordinary regression.
* Instead, models estimate **( \Pr(Y = \text{class} \mid X) )**, then assign the most probable class.
* The chapter will cover several classification methods:
   - Logistic regression
   - Linear discriminant analysis (LDA)
   - Quadratic discriminant analysis (QDA)
   - Naive Bayes
   - (K)-nearest neighbors (KNN) ([Bookdown][1])

---

## Section 4.2 – Why Not Linear Regression?

This section argues why we cannot simply adapt standard linear regression to classification tasks. The main reasons are:

### Problem 1: Predictions Below 0 or Above 1

* Suppose we encode the binary response as (Y = 0) (No) or (Y = 1) (Yes).
* If we fit a linear regression model:

[
\hat{Y} = \hat{\beta}_0 + \hat{\beta}_1 X
]

* Some predicted values might be **less than 0** or **greater than 1**, which make no sense as probabilities. ([Bookdown][1])
* Such values cannot properly be interpreted as probabilities (which must lie in ([0,1])).

### Problem 2: Inappropriate for More Than Two Classes

* If the response has **more than two categories**, there's **no natural numerical encoding** to fit a linear regression.

  * E.g. three disease diagnoses (stroke, overdose, seizure): encoding them as 1, 2, 3 imposes artificial order and equal spacing assumptions.
  * Different arbitrary codings yield different regression fits. ([Cross Validated][2])
* Even with binary cases, the linear model is a crude approximation and lacks probabilistic interpretability.

### Problem 3: Interpretation as Probability Is Flawed

* If we interpret (\hat{Y}) from linear regression as a probability (e.g. (\hat{Y} = 0.2) means 20 %), the problems above spoil this interpretation.
* Also, the linear model does not account for **nonlinear relationship** between predictors and probability — probabilities often change nonlinearly with predictors.

### Summary of Why Linear Regression Fails for Classification

| Issue                                  | Description                                                                               |
| -------------------------------------- | ----------------------------------------------------------------------------------------- |
| Predictions out of ([0,1])             | Linear regression can give negative or >1 predictions                                     |
| No consistent encoding for multi-class | Arbitrary numeric codes impose false structure                                            |
| Poor probabilistic interpretation      | Doesn’t model the log-odds or nonlinear effects                                           |
| Lack of a proper loss function         | Classification tasks require loss functions suited to probabilities (like log-likelihood) |

Because of these issues, we adopt classification-specific approaches (e.g. logistic regression) that produce probability estimates constrained to ([0,1]), handle multi-class cases, and provide sound statistical interpretation.

---

Here’s an expanded, detailed set of notes for **Section 4.3: Logistic Regression**, including more examples and code sketches. You can integrate this into your notes under your classification chapter.

---

## Section 4.3 — Logistic Regression

### 4.3.1 The Logistic Model

#### Motivation

* For a binary response (Y \in {0,1}), we want:
  [
  p(x) = \Pr(Y = 1 \mid X = x)
  ]
* A naive linear model (p(x) = \beta_0 + \beta_1 x) can produce predictions outside ([0,1]), which makes no probabilistic sense.
* Instead, logistic regression uses a **link function** to map real values to ((0,1)).

#### The Logistic / Sigmoid Formulation

Define:

[
p(x) = \frac{e^{\beta_0 + \beta_1 x}}{1 + e^{\beta_0 + \beta_1 x}}
]

Equivalently:

* **Odds**:

  [
  \text{odds}(x) = \frac{p(x)}{1 - p(x)} = e^{\beta_0 + \beta_1 x}
  ]

* **Log-odds / Logit**:

  [
  \log \left( \frac{p(x)}{1 - p(x)} \right) = \beta_0 + \beta_1 x
  ]

Thus logistic regression models the log-odds as a linear function of predictors.

#### Interpretation

* (\beta_1) is the change in log-odds for a one-unit increase in (x).
* (\exp(\beta_1)) is the **odds ratio**: how much the odds multiply when (x) increases by 1.
* The relationship between (x) and the probability (p(x)) is **nonlinear** (S-shaped). At low or high values of (x), changes in (x) have smaller effect on (p); in the mid-range, changes in (x) shift (p) more sharply.

#### Example (Credit Default, from Default Dataset)

Let’s use the ISLR/ISLP **Default** dataset (10,000 observations). Key variables:

* `default`: “Yes” / “No” (binary response)
* `balance`: numeric predictor
* `income`: numeric predictor
* `student`: categorical predictor (Yes / No)

One can model:

[
\log\left( \frac{p(\text{default} = \text{Yes} \mid \text{balance}, \text{income}, \text{student})}{1 - p(\cdot)} \right)
= \beta_0 + \beta_1 \cdot \text{balance} + \beta_2 \cdot \text{income} + \beta_3 \cdot \mathbf{1}[\text{student} = \text{Yes}]
]

Then:

[
\hat{p} = \frac{e^{\beta_0 + \beta_1 , \text{balance} + \beta_2 , \text{income} + \beta_3 , \text{studentYes}}}{1 + e^{\beta_0 + \beta_1 , \text{balance} + \beta_2 , \text{income} + \beta_3 , \text{studentYes}}}
]

Interpretation:

* (\beta_1): how the log-odds of default change per unit increase in `balance`, holding other variables fixed.
* (\exp(\beta_1)): factor change in odds per unit balance change.
* A positive (\beta_1) means higher balance → higher probability of default.

---

### 4.3.2 Estimating the Regression Coefficients

#### Likelihood & Log-Likelihood

Because (Y) is binary, the likelihood for a single observation ((x_i, y_i)) is:

[
P(Y_i = y_i \mid x_i) = p(x_i)^{y_i} , [1 - p(x_i)]^{1 - y_i}
]

Overall **likelihood**:

[
L(\beta) = \prod_{i=1}^n p(x_i)^{y_i} [1 - p(x_i)]^{1 - y_i}
]

We typically maximize the **log-likelihood**:

[
\ell(\beta) = \sum_{i=1}^n \Bigl( y_i \ln p(x_i) + (1 - y_i)\ln(1 - p(x_i)) \Bigr)
]

We choose (\hat\beta) to maximize (\ell(\beta)).

#### No Closed-Form — Iterative Algorithms

Unlike linear regression, logistic regression **does not** have a closed-form solution. Instead, we use numerical optimization (e.g. Newton-Raphson, iteratively reweighted least squares) to find (\hat\beta).

The algorithm roughly:

1. Initialize (\beta) (often zeros)
2. Compute current predicted probabilities (p(x_i))
3. Compute the gradient vector and Hessian matrix of (\ell(\beta))
4. Update (\beta \leftarrow \beta + H^{-1} , \nabla \ell)
5. Repeat until convergence (changes in (\beta) small)

#### Inference: Standard Errors, Tests, Confidence Intervals

After fitting:

* We compute the **variance–covariance matrix** of (\hat\beta). Often software outputs standard errors of coefficients.

* A **Wald statistic** or **z-statistic** is:

  [
  z_j = \frac{ \hat\beta_j }{ SE(\hat\beta_j) }
  ]

  This is analogous to the (t)-statistic in linear regression for testing ( H_0: \beta_j = 0).

* The **likelihood ratio test** (comparing full model vs reduced model) is also used to test groups of predictors or model significance.

#### Example & Code Sketch (Python with scikit-learn / statsmodels)

Here’s a sketch (not final assignment code) to help you implement logistic regression in Python:

```python
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import numpy as np

# Suppose df is your DataFrame with predictors and binary target 'default'
# Encode target as 0 / 1 (e.g. default = 'Yes' → 1, 'No' → 0)
df['default01'] = (df['default'] == 'Yes').astype(int)

X = df[['balance', 'income', 'studentYes']]  # numeric predictors plus dummy variable
y = df['default01']

# Add intercept for statsmodels
X_with_const = sm.add_constant(X)

# Fit logistic regression with statsmodels for inference
model = sm.Logit(y, X_with_const).fit()
print(model.summary())  # shows coefficients, SEs, z-values, p-values etc.

# Using sklearn for prediction (no built-in inference)
lr = LogisticRegression()
lr.fit(X, y)
coefs = lr.coef_  # slopes
intercept = lr.intercept_

# Predict probabilities and classes
y_prob = lr.predict_proba(X)[:, 1]   # probability of class 1
y_pred = (y_prob >= 0.5).astype(int)  # threshold at 0.5

# Evaluate accuracy
accuracy = np.mean(y_pred == y)
```

**Notes to Understand Each Line:**

* We create a dummy variable `studentYes` (0/1) from the categorical `student`.
* `sm.Logit(...)` fits the logistic model, analogous to `sm.OLS` in linear regression.
* `model.summary()` provides detailed output: coefficient estimates, standard errors, z-statistics, p-values, and likelihood-based metrics.
* `sklearn.LogisticRegression()` is convenient for prediction but does not by default provide standard errors or p-values.
* `predict_proba()` gives class probabilities; thresholding yields class predictions.

#### More Concrete Example

Let’s assume from `sm.Logit` summary we get:

| Predictor  | Coefficient | Std. Error | z-value | p-value |
| ---------- | ----------- | ---------- | ------- | ------- |
| const      | –11.0       | 0.50       | –22.0   | < 2e–16 |
| studentYes | –0.65       | 0.236      | –2.74   | 0.006   |
| balance    | 0.00574     | 0.000232   | 24.74   | < 2e–16 |
| income     | 0.00000303  | 0.0000082  | 0.370   | 0.711   |

Interpretation:

* `balance` has a strong positive effect on default odds — each $1 increase in balance multiplies the odds of default by ( e^{0.00574} \approx 1.00575 ).
* `studentYes` is negative: being a student reduces the log-odds of default (holding balance & income constant).
* `income` has a non-significant coefficient (large p-value ~ 0.711), suggesting it may not contribute predictive power given balance and student status.

One can compute predicted probabilities:

[
\hat{p_i} = \frac{e^{\beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \beta_3 x_{i3}}}{1 + e^{\beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \beta_3 x_{i3}}}
]

Then classify: predict default (class = 1) if (\hat{p_i} \ge 0.5).

You also can compute the **confusion matrix**, **accuracy**, **precision/recall**, or **ROC / AUC** to evaluate classifier performance.

---

### 4.3.3 - Making Predictions
*[Content to be added]*

### 4.3.4 - Multiple Logistic Regression
*[Content to be added]*

### 4.3.5 - Multinomial Logistic Regression
*[Content to be added]*

---

## Section 4.4 - Generative Models for Classification

### 4.4.1 - Linear Discriminant Analysis for p = 1
*[Content to be added]*

### 4.4.2 - Linear Discriminant Analysis for p > 1
*[Content to be added]*

### 4.4.3 - Quadratic Discriminant Analysis
*[Content to be added]*

### 4.4.4 - Naive Bayes
*[Content to be added]*

---

## Section 4.5 - A Comparison of Classification Methods

### 4.5.1 - An Analytical Comparison
*[Content to be added]*

### 4.5.2 - An Empirical Comparison
*[Content to be added]*

---

## Section 4.6 - Generalized Linear Models

### 4.6.1 - Linear Regression on the Bikeshare Data
*[Content to be added]*

### 4.6.2 - Poisson Regression on the Bikeshare Data
*[Content to be added]*

### 4.6.3 - Generalized Linear Models in Greater Generality
*[Content to be added]*

---

## Section 4.7 - Lab: Logistic Regression, LDA, QDA, and KNN

### 4.7.1 - The Stock Market Data
*[Content to be added]*

### 4.7.2 - Logistic Regression
*[Content to be added]*

### 4.7.3 - Linear Discriminant Analysis
*[Content to be added]*

### 4.7.4 - Quadratic Discriminant Analysis
*[Content to be added]*

### 4.7.5 - Naive Bayes
*[Content to be added]*

### 4.7.6 - K-Nearest Neighbors
*[Content to be added]*

### 4.7.7 - Linear and Poisson Regression on the Bikeshare Data
*[Content to be added]*

---

## Section 4.8 - Exercises
*[Content to be added]*

---

## Notes
*[Add your notes here]*
