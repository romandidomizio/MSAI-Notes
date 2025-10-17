# Chapter 6 - Linear Model Selection and Regularization

## ISLP (Introduction to Statistical Learning with Python)

---

## Section 6.1 – Subset Selection

Subset selection is a class of methods for fitting linear models by selecting a *subset* of the predictors instead of using all of them. The goal is to improve interpretability and reduce overfitting by discarding irrelevant predictors.

Three common subset selection strategies:

* Best subset selection
* Stepwise selection (forward, backward, or hybrid)
* Methods for choosing the optimal model among fitted subsets

---

### 6.1.1 Best Subset Selection

**Definition / Approach**

* Suppose we have (p) candidate predictors (features).

* **Best subset selection** fits *all possible* regression models that use any subset of these (p) predictors.

* That means:

  * Fit all ( \binom{p}{1} ) models with exactly 1 predictor
  * All ( \binom{p}{2} ) models with exactly 2 predictors
  * …
  * All ( \binom{p}{p} = 1 ) model with all predictors

* For each subset size (d = 0,1,2,\dots,p) (0 meaning no predictors, just intercept), select the *best* model of size (d) (by a criterion such as smallest RSS or largest (R^2)).

* This yields *(p + 1)* candidate models: one best of each size.

* Then we choose among those (p+1) models (of different sizes) using a model selection criterion (e.g. cross-validation, (C_p), BIC, adjusted (R^2)).

**Advantages & Disadvantages**

| Pros                                                      | Cons                                                                       |
| --------------------------------------------------------- | -------------------------------------------------------------------------- |
| Finds the best possible subset (for the given criterion)  | Computationally infeasible for large (p), since there are (2^p) subsets    |
| Guarantees that the best model of each size is considered | High variance / overfitting risk when (p) is large (many candidate models) |

**Computational Burden**

* The total number of models is (2^p), which grows exponentially.
* Even for moderate (p) (e.g. 20), (2^p = 1,048,576) models — often too many.
* As a result, best subset selection is feasible only when (p) is small.

**Example Sketch**

If (p = 3), predictors (X_1, X_2, X_3):

* Fit:

  * size-1 models: ({X_1}, {X_2}, {X_3})
  * size-2 models: ({X_1, X_2}, {X_1, X_3}, {X_2, X_3})
  * size-3 model: ({X_1, X_2, X_3})
* Choose best model of each size (e.g. by lowest RSS).
* Then compare these 4 models (size 0 through 3) by e.g. cross-validated error or penalized criterion to pick the final one.

---

### 6.1.2 Stepwise Selection

Because best subset selection becomes infeasible for large (p), stepwise methods provide more efficient heuristics:

Two main variants:

* **Forward Stepwise Selection**
* **Backward Stepwise Selection**
* (Also hybrid / bidirectional stepwise)

#### Forward Stepwise Selection

* Start with the **null model** (no predictors).
* At each step, consider adding **one** predictor not yet in the model.
* Among all possible additions, choose the one that yields the greatest improvement (e.g. reduces RSS most, or increases (R^2) most).
* Continue adding until all predictors are in, or until an improvement threshold is reached.
* This greedy approach avoids fitting all (2^p) models; instead it fits only:

  [
  1 + (p) + (p-1) + (p-2) + \cdots + 1 = \frac{p(p+1)}{2} + 1
  ]

  models in total. (This is far less than (2^p).)

**Limitation:** Because it's greedy, it may miss the *globally best* model. For instance, it may commit to a predictor early that precludes a better subset later.

#### Backward Stepwise Selection

* Start with the **full model** (all (p) predictors).
* At each step, consider **removing** one predictor.
* Remove the one whose omission leads to the smallest increase in RSS (or minimal harm) — i.e., worst predictor by some criterion.
* Continue removing until no predictors remain (or based on stopping rule).
* Also fits about (\frac{p(p+1)}{2} + 1) models.

**Limitation:** Requires that (n > p) (so the full model is estimable). Also greedy, so not guaranteed optimal.

#### Hybrid / Bidirectional Stepwise

* At each step you can both add and (possibly) remove predictors.
* This allows more flexibility: after adding a variable, you can remove an earlier one if it no longer helps.
* This attempts to mitigate some of the greedy path dependence of pure forward/backward.

---

### 6.1.3 Choosing the Optimal Model

After generating candidate models (via best subset or stepwise methods), we need a principled way to choose which one is best in terms of predictive performance (not just training fit).

#### Problem with Training Metrics

* As we increase model size (more predictors), **RSS always decreases** and **(R^2) always increases** on training data — even if the added predictors are noise.
* Thus, **training error** metrics cannot reliably select among nested models of different sizes.

Hence, we need methods that penalize complexity or estimate test error:

#### Approaches to Select Optimal Model

1. **Cross-Validation / Validation Set Approach**

   * Use held-out or cross-validated error to estimate test error for each candidate model size. Pick the model with lowest estimated test error.

2. **Penalty-based Criteria (Adjustment to Training Error)**

   * **Mallows’ (C_p)**
   * **Akaike Information Criterion (AIC)**
   * **Bayesian Information Criterion (BIC)**
   * **Adjusted (R^2)**

These penalize model complexity (number of predictors) to offset the optimistic bias of training error.

For example, for a model with (d) predictors (excluding intercept):

[
C_p = \frac{1}{n} \left( RSS + 2 d \hat{\sigma}^2 \right)
]

where (\hat{\sigma}^2) is an estimate of the error variance. Lower (C_p) is better.

Similarly, BIC uses a penalty term ( d \log(n) ) instead of (2d). Adjusted (R^2) adjusts (R^2) downward based on (n) and (d).

#### Example / Illustration (From Credit Data in ISLR)

* In the Credit data example in ISLR/ISLP, Table 6.1 shows which predictors are selected at sizes 1,2,3,4 under best subset and forward stepwise. For the first three models, both methods choose the same predictors; but at size 4 they diverge (best subset replaces “rating” with “cards”, while forward stepwise keeps “rating” since it was included earlier). ([INFO 523][1])
* This illustrates that greedy stepwise methods may fail to find the best subset.

---

## Section 6.2 – Shrinkage Methods

Shrinkage methods (also called regularization) are techniques used to improve prediction accuracy and interpretability by **adding a penalty** to the ordinary least squares objective. They “shrink” coefficient estimates toward zero (and possibly set some to exactly zero).

They combat **overfitting** when ( p ) (number of predictors) is large, or predictors are multicollinear, or sample size is limited.

The two principal shrinkage methods discussed are:

* **Ridge regression**
* **Lasso (least absolute shrinkage and selection operator)**

Then, we need a method for **selecting the tuning parameter** (penalty strength).

---

### 6.2.1 Ridge Regression

#### Objective / Formulation

Recall from ordinary least squares, we choose ( \beta = (\beta_0, \beta_1, \dots, \beta_p) ) to minimize

[
RSS = \sum_{i=1}^n \bigl( y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij} \bigr)^2
]

Ridge regression modifies this by adding a penalty term on the sum of squares of coefficients:

[
\hat\beta^{\text{ridge}} = \arg\min_{\beta_0, \beta} ; \Biggl{ \sum_{i=1}^n \bigl( y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij} \bigr)^2 + \lambda \sum_{j=1}^p \beta_j^2 \Biggr}
]

* Here (\lambda \ge 0) is a **tuning parameter** (also called penalty parameter).
* The penalty ( \lambda \sum_j \beta_j^2 ) is sometimes called the **shrinkage penalty**.
* The penalty is only applied to (\beta_1, \dots, \beta_p), **not** to the intercept (\beta_0).

#### Intuition & Behavior

* When (\lambda = 0), the penalty term vanishes, and ridge gives ordinary least squares (OLS) estimates.
* As (\lambda \to \infty), the penalty dominates and forces all (\beta_j) toward zero (making a very simple model).
* The coefficient estimates are **shrunk toward zero**, but not exactly zero (unless (\lambda) infinite).
* Ridge is especially helpful when predictors are collinear or when ( p ) is large, because it stabilizes estimates by controlling their variance.

Mathematically, for standardized predictors (centered (X) and mean-zero (y)), one can show the ridge estimate has the closed-form:

[
\hat\beta^{\text{ridge}} = (X^\top X + \lambda I)^{-1} X^\top y
]

where (I) is the identity matrix (size (p \times p)). This differs from the OLS formula ((X^\top X)^{-1} X^\top y).

#### Example & Coefficient Paths

* As you increase (\lambda), ridge coefficients move continuously toward 0.
* One often plots **coefficient paths**: coefficients as a function of (\lambda). This shows which variables shrink faster.
* In practice, one typically **standardizes** predictors so they are on comparable scale (zero mean and unit variance), so the penalty treats them fairly.

#### Pros & Limitations

* **Pros**:

  * Reduces variance and helps prevent overfitting.
  * Keeps all predictors in the model (no variable selection) — which can be good if you believe many predictors have small effects.
* **Limitations**:

  * Does *not* set any coefficient exactly to zero; you lose interpretability in terms of selecting predictors.
  * The choice of (\lambda) is critical.

---

### 6.2.2 The Lasso

#### Objective / Formulation

The Lasso modifies the OLS objective by adding an (L_1) penalty (sum of absolute values of coefficients):

[
\hat\beta^{\text{lasso}} = \arg\min_{\beta_0, \beta} ; \Biggl{ \sum_{i=1}^n ( y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij} )^2 + \lambda \sum_{j=1}^p |\beta_j| \Biggr}
]

Again, (\lambda \ge 0) tunes the strength of the penalty.

Alternatively, one can frame ridge and lasso in **constrained form**:

* Ridge: minimize (RSS) subject to (\sum_{j} \beta_j^2 \le s)
* Lasso: minimize (RSS) subject to (\sum_{j} |\beta_j| \le s)

These are equivalent reparameterizations (for each (\lambda) there is some (s)).

#### Key Differences vs Ridge & Behavior

* The (L_1) penalty tends to **shrink some coefficients exactly to zero** for sufficiently large (\lambda). This means the lasso performs **variable selection** plus shrinkage.
* Geometric intuition: in the constrained form, the lasso feasible region ((\ell_1)-ball) has “corners” at axes. The RSS contours often touch these corners, forcing some coefficients to zero.
* In contrast, the ridge constraint region ((\ell_2)-ball or sphere) is smooth and symmetric; it tends to shrink all coefficients but rarely exactly to zero.

Thus:

* **Lasso** yields sparse models (some coefficients = 0), which helps interpretability.
* **Ridge** shrinks all coefficients a bit, retaining all variables.

#### Example (Special Case)

In a very simplified scenario (orthonormal predictors):

* For ridge: (\hat\beta_j^{\text{ridge}} = \hat\beta_j^{\text{OLS}} / (1 + \lambda))
* For lasso: (\hat\beta_j^{\text{lasso}}) is given by a **soft-thresholding** rule:

[
\hat\beta_j^{\text{lasso}} = \begin{cases}
\hat\beta_j^{\text{OLS}} - \frac{\lambda}{2} & \text{if } \hat\beta_j^{\text{OLS}} > \frac{\lambda}{2} \
\hat\beta_j^{\text{OLS}} + \frac{\lambda}{2} & \text{if } \hat\beta_j^{\text{OLS}} < -\frac{\lambda}{2} \
0 & \text{if } |\hat\beta_j^{\text{OLS}}| \le \frac{\lambda}{2}
\end{cases}
]

This shows lasso shrinks small coefficients to 0, while ridge never zeros them out.

#### Pros & Limitations

* **Pros**:

  * Variable selection (sparse model) → interpretability.
  * Good when you expect many predictors to have zero or negligible effects.
* **Limitations**:

  * When predictors are highly correlated, lasso may arbitrarily pick one and ignore others.
  * For large (\lambda), too aggressive shrinkage may remove useful predictors.
  * Solving optimization is more complex (non-differentiable at zero) — needs specialized algorithms (e.g. coordinate descent, LARS).

---

### 6.2.3 Selecting the Tuning Parameter

The performance of both ridge and lasso depends critically on choosing (\lambda) appropriately. The goal is to find a (\lambda) that balances **bias and variance**, leading to lowest test error.

#### Methods to Choose (\lambda)

* **Cross-validation** (e.g. (k)-fold CV): fit models across a grid of (\lambda) values, compute CV error for each, choose the (\lambda) with lowest CV error.
* **Validation set approach**: hold out a validation set, fit models on training for each (\lambda), choose (\lambda) that yields lowest error on validation.
* **Analytic / Information criteria**: in some cases, one might use AIC, BIC, or generalized cross-validation (GCV) to select (\lambda).

After selecting (\lambda), one typically refits the model (ridge or lasso) on the **entire dataset** using that chosen (\lambda).

#### Considerations & Trade-offs

* The grid of (\lambda) values should usually span a wide range (from very small to large) to see where shrinkage becomes too strong.
* Because CV splits introduce variability, one sometimes uses repeated CV or nested CV to ensure stability in (\lambda) selection.
* The “optimal” (\lambda) from CV might not be the best for interpretability — sometimes one picks a slightly larger (\lambda) (simpler model) if the error increase is small (~ one‑standard-error rule).
* The selected (\lambda) balances **bias** (too much shrinkage → underfitting) and **variance** (too little shrinkage → overfitting).

---

### Examples & Code Sketches

Below are sketches (not full assignment code) to help you understand how to implement ridge and lasso, and tune (\lambda) with cross-validation in Python (e.g. via `sklearn` / `glmnet`-style libraries).

```python
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler

# Assume X, y are your data arrays (n × p, and length n)
# Standardize X so that each feature has mean=0 and unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define a grid of lambda (called alpha in sklearn) values to search
alphas = np.logspace(-4, 4, 100)

# Ridge regression + CV
ridge = Ridge()
ridge_cv = GridSearchCV(ridge, param_grid={"alpha": alphas}, cv=5, scoring="neg_mean_squared_error")
ridge_cv.fit(X_scaled, y)
best_alpha_ridge = ridge_cv.best_params_["alpha"]
best_ridge_model = ridge_cv.best_estimator_

# Lasso regression + CV
lasso = Lasso(max_iter=10000)
lasso_cv = GridSearchCV(lasso, param_grid={"alpha": alphas}, cv=5, scoring="neg_mean_squared_error")
lasso_cv.fit(X_scaled, y)
best_alpha_lasso = lasso_cv.best_params_["alpha"]
best_lasso_model = lasso_cv.best_estimator_

# Examine coefficients
coef_ridge = best_ridge_model.coef_
coef_lasso = best_lasso_model.coef_

# Compare number of non-zero coefficients in lasso
num_nonzero = np.sum(coef_lasso != 0)

print("Best alpha for Ridge:", best_alpha_ridge)
print("Best alpha for Lasso:", best_alpha_lasso)
print("Number of non-zero coeffs in Lasso:", num_nonzero)
```

**What to pay attention to:**

* Standardizing (X) is very important: penalties depend on scale.
* `alpha` in sklearn is the penalty parameter (analogous to (\lambda)).
* `GridSearchCV` with cross-validation picks the best `alpha` by minimizing validation error.
* For lasso, you will often see many coefficients exactly zero at moderate `alpha` values, which is the variable selection effect.

You can also plot **coefficient paths** (coefficients vs. (\log \lambda)) to see how coefficients shrink and vanish as (\lambda) increases.

---

## Section 6.3 - Dimension Reduction Methods

### 6.3.1 - Principal Components Regression
*[Content to be added]*

### 6.3.2 - Partial Least Squares
*[Content to be added]*

---

## Section 6.4 - Considerations in High Dimensions

### 6.4.1 - High-Dimensional Data
*[Content to be added]*

### 6.4.2 - What Goes Wrong in High Dimensions?
*[Content to be added]*

### 6.4.3 - Regression in High Dimensions
*[Content to be added]*

### 6.4.4 - Interpreting Results in High Dimensions
*[Content to be added]*

---

## Section 6.5 - Lab: Linear Models and Regularization Methods

### 6.5.1 - Subset Selection Methods
*[Content to be added]*

### 6.5.2 - Ridge Regression and the Lasso
*[Content to be added]*

### 6.5.3 - PCR and PLS Regression
*[Content to be added]*

---

## Section 6.6 - Exercises
*[Content to be added]*

---

## Notes
*[Add your notes here]*
