# Chapter 5 - Resampling Methods

## ISLP (Introduction to Statistical Learning with Python)

---

## Section 5.1 – Cross-Validation

Cross‑validation is a **resampling technique** used to estimate a model’s test error (i.e. predictive performance on unseen data) by repeatedly splitting the available data into training and validation subsets. It helps assess generalization and choose model complexity without needing a separate external test set. ([Bookdown][1])

### Key Idea

* True test error is what we care about — how well the model predicts new data.
* We often do *not* have a large independent test set.
* Cross-validation allows us to *simulate* test error from the training data itself by holding out parts of it, training on the rest, and measuring how the model performs on held-out parts. ([mathstat.dal.ca][2])
* Averaging over multiple such splits yields a more stable estimate of test error than a single hold-out split.

---

### 5.1.1 The Validation Set Approach

**Method:**

1. Randomly split the dataset into two parts:

   * **Training set**
   * **Validation (or hold-out) set**
2. Fit the model (or several candidate models) on the training set.
3. Use the fitted model(s) to predict on the validation set.
4. Compute error metric (e.g. MSE for regression, misclassification rate for classification).
5. That error is an estimate of test error.

**Pros and Cons:**

* **Pros:** Simple; easy to implement.
* **Cons:**

  * The performance estimate depends heavily on how **one single split** was done — high variance.
  * Uses only part of data for training (so model is less powerful than when using full data).
  * Because of randomness in splitting, the error estimate can be noisy.

**Example sketch:**

Suppose you have (n = 100) observations. You split randomly into 70 training + 30 validation.

* Fit linear regression on 70 training samples.
* Predict on 30 validation samples; compute ( \text{MSE}_{\text{val}} = \frac{1}{30} \sum (y_i - \hat{y}_i)^2 ).
* Use that as approximation to test MSE.

If you repeat this split multiple times, you may get different validation errors because of randomness.

---

### 5.1.2 Leave-One-Out Cross-Validation (LOOCV)

LOOCV is a special case of cross-validation where **each** observation in turn is used as the validation set of size **1**, and the rest (n-1) observations are used to train the model.

**Procedure:**

* For (i = 1) to (n):

  * Hold out observation (i) as validation.
  * Fit the model on the remaining (n-1) observations.
  * Predict (\hat{y}_i) for the held-out observation.
  * Compute error (Err_i = (y_i - \hat{y}_i)^2) or classification error (I(y_i \neq \hat{y}_i)).
* The LOOCV estimate of test error is:

[
CV_{LOO} = \frac{1}{n} \sum_{i=1}^n Err_i
]

**Features / Pros & Cons:**

* **Pros:**

  * Uses almost all data for training each time, so minimal bias.
  * No randomness: the splits are deterministic (one per observation).
* **Cons:**

  * Computationally expensive: you must fit the model (n) times.
  * High variance: Because the training sets overlap heavily, the error estimates can fluctuate. ([Cross Validated][3])
  * For some methods, there are shortcuts to compute LOOCV without refitting fully each time (e.g. for linear models, leave-one-out formulas), but not in general.

**Example sketch:**

If (n = 100), do 100 fits, each omitting one sample, collect the squared error on the omitted sample, average them.

Because each model is only missing one sample, the models are quite similar, so variance among the predicted errors can be large.

---

### 5.1.3 (k)-Fold Cross-Validation

(k)-fold cross-validation is the most commonly used practical cross-validation method.

**Procedure:**

1. Split the data into (k) roughly equal-sized **folds** (partitions).
2. For each (j = 1 \dots k):

   * Treat fold (j) as validation set.
   * Fit model on the remaining (k-1) folds (the training folds).
   * Predict on fold (j) and compute validation error for that fold.
3. Compute the average validation error across all (k) folds:

[
CV_k = \frac{1}{k} \sum_{j=1}^k \text{Error}_j
]

4. Use (CV_k) as estimate of test error.

**Special case:** When (k = n), (k)-fold becomes LOOCV.

**Typical choices:** (k = 5) or (k = 10) are common in practice. ([Bijen Patel][4])

**Pros & Cons:**

* **Pros:**

  * More stable than a single validation set (less variance) because each fold is used as validation exactly once.
  * Less computational cost than LOOCV (only (k) fits, vs (n) fits).
* **Cons:**

  * Slight bias (because models are trained on slightly less data than full set). ([Stanford University][5])
  * The estimate depends on how the data is partitioned (fold assignments).

**Bias–Variance Trade-Off for (k)-Fold:**
As (k) increases, training sets are larger (less bias) but variance of estimates can increase (because folds smaller) — there is a trade-off.
When (k = n), it's LOOCV (very low bias, but high variance).
When (k) is small (e.g. 5), variance is lower but bias is somewhat higher. ([Stanford University][5])

**Example sketch / code logic:**

```python
from sklearn.model_selection import KFold
import numpy as np

X = ...  # predictors
y = ...  # responses
kf = KFold(n_splits=k, shuffle=True, random_state=seed)

errors = []
for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    err = mean_squared_error(y_val, y_pred)
    errors.append(err)

cv_error = np.mean(errors)
```

---

### 5.1.4 Bias–Variance Trade-Off for (k)-Fold Cross-Validation

This subsection discusses how the choice of (k) affects the bias and variance of the cross-validation estimate.

**Effect of (k):**

* **As (k) increases (toward (n)):**

  * Training sets become closer to full dataset → **bias decreases** (model is closer to one trained on full data).
  * But error estimates from each fold are based on small validation sets, leading to **higher variance** in those estimates.
* **As (k) decreases (e.g. to 5):**

  * More bias (models are trained on smaller fractions)
  * But each validation fold is larger and more stable → **variance of estimate is lower**.

Thus, there is a **bias–variance trade-off** in choosing (k).

LOOCV ((k = n)) yields nearly unbiased estimate but high variance. A moderate (k) (like 5 or 10) often gives a better trade-off in practice. ([mathstat.dal.ca][2])

**Practical recommendation:** Use (k = 5) or (k = 10) in standard settings.

---

### 5.1.5 Cross-Validation on Classification Problems

Cross-validation extends naturally to classification tasks, but we replace numeric error metrics (like MSE) by classification metrics such as **misclassification rate** (or other metrics: precision, recall, AUC, etc.). ([r4ds.github.io][6])

**Procedure:**

* Use the same schemes (validation set, LOOCV, (k)-fold).
* For each held-out fold or instance:

  * Predict class label (\hat{y}_i).
  * Compute error indicator ( Err_i = I(y_i \neq \hat{y}_i) ) (1 if misclassified, 0 if correct).
* Cross‑validation error is:

[
\frac{1}{n} \sum_{i=1}^n Err_i \quad \text{(for LOOCV)} \quad \text{or} \quad \frac{1}{k} \sum_{j=1}^k \text{error}_j \quad \text{for }k\text{-fold}
]

We often call this the **CV misclassification rate**. ([r4ds.github.io][6])

**Example sketch:**

```python
from sklearn.model_selection import cross_val_score

clf = LogisticRegression()
# scoring = 'accuracy' or 'neg_log_loss' or other metrics
scores = cross_val_score(clf, X, y, cv=k, scoring='accuracy')
cv_error = 1 - np.mean(scores)
```

Here `cv_error` is cross-validated misclassification error (on average).

**Note:** Because classification tasks often have class imbalance, one might prefer stratified folds (i.e. preserve class ratios in folds) rather than naive random splits.

---

## Section 5.2 - The Bootstrap
*[Content to be added]*

---

## Section 5.3 - Lab: Cross-Validation and the Bootstrap

### 5.3.1 - The Validation Set Approach
*[Content to be added]*

### 5.3.2 - Cross-Validation
*[Content to be added]*

### 5.3.3 - The Bootstrap
*[Content to be added]*

---

## Section 5.4 - Exercises
*[Content to be added]*

---

## Notes
*[Add your notes here]*
