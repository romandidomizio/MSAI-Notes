# Chapter 9 - Support Vector Machines

## ISLP (Introduction to Statistical Learning with Python)

---

## Section 9.1 — Maximal Margin Classifier

This section introduces the idealized classifier that separates two classes by the maximum margin — assuming perfect separability. It sets up the geometric and optimization framework that will be relaxed later (Support Vector Classifier) to handle non‑separable data.

---

### 9.1.1 What Is a Hyperplane?

**Definition & Geometry**

* In (p)-dimensional space, a **hyperplane** is an affine subspace of dimension (p-1).

  * In 2D: a hyperplane is a line.
  * In 3D: a hyperplane is a plane.
  * In higher dimensions, it generalizes the notion of a flat dividing boundary.
* A hyperplane separates space into two half-spaces.

Mathematically, a hyperplane can be written as:

[
\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p = 0
]

Or in vector form:

[
\beta_0 + \boldsymbol{\beta}^\top \mathbf{x} = 0
]

Here:

* (\boldsymbol{\beta} = (\beta_1, \dots, \beta_p)) is the normal vector to the hyperplane (orthogonal to the hyperplane).
* (\beta_0) is the intercept or offset term.
* For any point (\mathbf{x}), the sign of (\beta_0 + \boldsymbol{\beta}^\top \mathbf{x}) tells which side of the hyperplane (\mathbf{x}) lies on:

  * If (\beta_0 + \boldsymbol{\beta}^\top \mathbf{x} > 0): one side
  * If (< 0): the other side
  * If = 0: exactly on the hyperplane.

The **distance** from a point (\mathbf{x}) to the hyperplane is:

[
\frac{|\beta_0 + \boldsymbol{\beta}^\top \mathbf{x}|}{|\boldsymbol{\beta}|}
]

(where (|\boldsymbol{\beta}|) is the Euclidean norm). This distance formula arises from projecting onto the normal direction.

Thus a hyperplane defines a linear classifier dividing (\mathbb{R}^p) into two classes (if separable).

---

### 9.1.2 Classification Using a Separating Hyperplane

**Separability Assumption**

* The maximal margin classifier assumes the two classes are **linearly separable**: there exists at least one hyperplane that perfectly divides all class 1 points from class −1 points without error.

Given training data ({(\mathbf{x}_i, y_i)}) with (y_i \in {-1, +1}), a separating hyperplane satisfies:

[
y_i \bigl( \beta_0 + \boldsymbol{\beta}^\top \mathbf{x}_i \bigr) > 0 \quad \forall i
]

This means each training point lies on the correct side of the hyperplane.

There may be many such hyperplanes. We want to pick the *best* one, according to some optimality criterion: **margin maximization** (largest separation from the nearest points).

---

### 9.1.3 The Maximal Margin Classifier

**Margin Definition**

* The **margin** is the distance from the hyperplane to the nearest training point (on either side).
* Given a hyperplane defined by ((\beta_0, \boldsymbol{\beta})), the margin is:

[
M = \min_i \frac{ y_i (\beta_0 + \boldsymbol{\beta}^\top \mathbf{x}_i) }{|\boldsymbol{\beta}|}
]

We want to **maximize (M)**, the minimal distance to the hyperplane among all training points.

**Maximal Margin Classifier**

* This is the hyperplane that **maximizes** the margin (M), under the constraint of perfect separability.
* Geometrically, we imagine “sliding” a hyperplane between the two classes so that it is as far as possible from both classes but still does not misclassify any training point.

Key properties:

* The maximal margin hyperplane is uniquely determined by a subset of the points (those lying exactly on the margin boundaries).
* These critical points are termed **support vectors**. Moving non-support vectors slightly (within margin) does not change the hyperplane.

Thus the maximal margin classifier depends **only** on the support vectors, not on all training points. ([Bijen Patel][1])

In the ideal case, the maximal margin classifier gives the most robust separating hyperplane under perfect separability.

---

### 9.1.4 Construction of the Maximal Margin Classifier

This section describes how to formulate and solve for the maximal margin hyperplane with constraints.

**Optimization Formulation**

We want to maximize margin (M), subject to:

1. Normalization of the normal vector: (|\boldsymbol{\beta}| = 1) (or equivalently (\sum_j \beta_j^2 = 1))
2. All points lie outside the margin:

[
y_i (\beta_0 + \boldsymbol{\beta}^\top \mathbf{x}_i) \ge M \quad \forall i
]

Thus the optimization is:

[
\begin{aligned}
\text{maximize}_{\beta_0, \boldsymbol{\beta}} \quad & M \
\text{subject to} \quad & |\boldsymbol{\beta}| = 1, \
& y_i (\beta_0 + \boldsymbol{\beta}^\top \mathbf{x}_i) \ge M, \quad i = 1,\ldots,n.
\end{aligned}
]

In many derivations, one removes the explicit norm constraint and rewrites in a dual or equivalent form. But the core idea is: maximize margin while classifying all points correctly.

Support vector theory shows that in the optimal solution, many (\beta_i = 0) for non-support vectors; only support vectors remain active constraints.

Once solved, one obtains (\beta_0, \boldsymbol{\beta}). For classification of a new point (\mathbf{x}), compute (\text{sign}(\beta_0 + \boldsymbol{\beta}^\top \mathbf{x})).

Because the optimization is a convex quadratic programming problem, the solution is tractable in moderate dimension.

---

### 9.1.5 The Non-separable Case

In real data, classes are often **not perfectly separable** by a linear boundary. The maximal margin classifier (hard margin) fails: constraints may be unsatisfiable.

To handle this, we relax the constraints to allow some violations (slack). This leads to the **support vector classifier** (soft margin), introduced in Section 9.2, which allows misclassifications or margin violations.

Key ideas in non-separable case:

* Introduce **slack variables** (\xi_i \ge 0) for each point to allow some margin violation:

  [
  y_i (\beta_0 + \boldsymbol{\beta}^\top \mathbf{x}_i) \ge M - \xi_i
  ]

* Add a penalty for (\xi_i)'s in the objective, controlled by a tuning parameter (C) (or penalty parameter).

* The new objective trades off margin size and degree of misclassification or margin violation.

Because non‑separable data is typical, the soft margin / support vector classifier is the practical version of the maximal margin idea for most real-world datasets.

---

### Insights, Remarks & Intuitions

* The maximal margin classifier is elegant and geometrically intuitive, but strictly only applies under ideal separability.
* Its strength lies in the fact that only support vectors matter: many training points don’t influence the decision boundary.
* However, because it forces perfect separation, it is sensitive to outliers and noise.
* Soft margin (in SVC) relaxes perfection to gain robustness.

---

## Section 9.2 - Support Vector Classifiers

### 9.2.1 - Overview of the Support Vector Classifier

**The Problem with Perfect Separation**

* In practice, observations from two classes are **not necessarily separable** by a hyperplane.
* Even when a separating hyperplane exists, a classifier based on it can be undesirable:
  * **Overfitting**: Perfect classification of training data can lead to sensitivity to individual observations
  * **Small margins**: A single observation can dramatically shift the maximal margin hyperplane (see Figure 9.5)
  * **Poor generalization**: Extreme sensitivity suggests the model may overfit training data

**The Soft Margin Approach**

The **support vector classifier** (also called **soft margin classifier**) addresses these issues by:

* Allowing some observations to be on the **incorrect side of the margin**
* Even permitting some to be on the **incorrect side of the hyperplane**
* Trading off perfect separation for:
  * Greater **robustness** to individual observations
  * Better **classification of most** training observations

**Key Concept**: It may be worthwhile to **misclassify a few training observations** to do a better job classifying the remaining ones.

**Visual Understanding** (Figure 9.6):

* Most observations are on the correct side of the margin
* A small subset can be:
  * On the wrong side of the margin (but correct side of hyperplane)
  * On the wrong side of the hyperplane (misclassified)

**The Margin is "Soft"**

* The margin can be **violated** by some training observations
* This flexibility allows the classifier to handle non-separable data
* When no separating hyperplane exists, violations are inevitable

### 9.2.2 - Details of the Support Vector Classifier

**Optimization Problem**

The support vector classifier is the solution to:

[
\begin{aligned}
\text{maximize}_{\beta_0, \beta_1, \ldots, \beta_p, \epsilon_1, \ldots, \epsilon_n, M} \quad & M \\
\text{subject to} \quad & \sum_{j=1}^p \beta_j^2 = 1, \\
& y_i(\beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_p x_{ip}) \geq M(1 - \epsilon_i), \\
& \epsilon_i \geq 0, \quad \sum_{i=1}^n \epsilon_i \leq C
\end{aligned}
]

Where:
* **M**: width of the margin (we seek to maximize this)
* **ε_i**: **slack variables** allowing violations
* **C**: nonnegative **tuning parameter** (budget for violations)

**Understanding Slack Variables (ε_i)**

For the i-th observation:

* **ε_i = 0**: observation is on the **correct side of the margin**
* **ε_i > 0**: observation **violates the margin** (on wrong side)
* **ε_i > 1**: observation is on the **wrong side of the hyperplane** (misclassified)

**The Role of Tuning Parameter C**

* **C bounds** the sum of slack variables: Σε_i ≤ C
* C determines the **number and severity** of margin violations tolerated
* Think of C as a **budget** for margin violations:

  * **C = 0**: No violations allowed → maximal margin classifier (if separable)
  * **C > 0**: At most C observations can be on wrong side of hyperplane
  * **Larger C**: More tolerant of violations → **wider margin**
  * **Smaller C**: Less tolerant → **narrower margin**

**Bias-Variance Trade-off**

* **Small C**: 
  * Narrow margins, rarely violated
  * Highly fit to data
  * **Low bias, high variance**
  
* **Large C**:
  * Wide margin, more violations allowed
  * Less hard fit to data
  * **Higher bias, lower variance**

* **C is chosen via cross-validation** in practice

**Support Vectors**

**Critical Property**: Only observations that lie on the margin or violate it affect the hyperplane.

* Observations **strictly on correct side** of margin: Do NOT affect the classifier
* **Support vectors**: Observations that:
  * Lie directly on the margin, OR
  * Are on wrong side of margin for their class
* Only support vectors affect the support vector classifier
* Changing non-support vector positions doesn't change the classifier (if they stay on correct side)

**Impact on Bias-Variance**

* **Large C** → Wide margin → Many support vectors:
  * Low variance (many observations involved)
  * Potentially high bias
  
* **Small C** → Narrow margin → Few support vectors:
  * Low bias
  * High variance

**Robustness Property**

The support vector classifier is **robust** to observations far from the hyperplane:

* Decision rule based only on support vectors (small subset of training data)
* Observations far from boundary don't influence the classifier
* **Contrast with LDA**: Uses mean of ALL observations in each class and within-class covariance from all observations
* **Similar to logistic regression**: Also has low sensitivity to observations far from decision boundary
* (Section 9.5 shows SVC and logistic regression are closely related)

---

## Section 9.3 - Support Vector Machines

### 9.3.1 - Classification with Non-Linear Decision Boundaries

**The Problem**

* Support vector classifier is natural for two-class problems with **linear boundaries**
* In practice, we often face **non-linear class boundaries**
* Linear classifiers (including SVC) perform poorly on non-linear data (see Figure 9.8)

**Solution Approach: Enlarge the Feature Space**

**Analogy to Linear Regression** (Chapter 7):
* Linear regression suffers when relationship between predictors and outcome is non-linear
* Solution: Enlarge feature space using functions of predictors (quadratic, cubic terms, etc.)

**Applying to Support Vector Classifier**:

Instead of using p features:
[
X_1, X_2, \ldots, X_p
]

Fit SVC using **2p features** (include squared terms):
[
X_1, X_1^2, X_2, X_2^2, \ldots, X_p, X_p^2
]

**Modified Optimization Problem** (9.16):

[
\begin{aligned}
\text{maximize} \quad & M \\
\text{subject to} \quad & y_i \left( \beta_0 + \sum_{j=1}^p \beta_{j1} x_{ij} + \sum_{j=1}^p \beta_{j2} x_{ij}^2 \right) \geq M(1 - \epsilon_i), \\
& \sum_{i=1}^n \epsilon_i \leq C, \quad \epsilon_i \geq 0, \\
& \sum_{j=1}^p \sum_{k=1}^2 \beta_{jk}^2 = 1
\end{aligned}
]

**Why This Creates Non-linear Boundaries**

* In the **enlarged feature space**, the decision boundary is **linear**
* In the **original feature space**, the boundary has form q(x) = 0 where q is quadratic
* Solutions to quadratic equations are generally **non-linear**

**Extensions**

* Could add higher-order polynomial terms
* Could include interaction terms: X_j × X_j' for j ≠ j'
* Could use other functions beyond polynomials

**The Challenge**

* Many possible ways to enlarge feature space
* **Risk**: End up with huge number of features
* Computations become **unmanageable**

**Preview of Solution**

The **support vector machine** (next section) provides a way to enlarge the feature space using **kernels** that:
* Enables non-linear boundaries
* Leads to **efficient computations**

### 9.3.2 - The Support Vector Machine

**Definition**

The **support vector machine (SVM)** extends the support vector classifier by enlarging the feature space using **kernels**.

**Main Idea** (from 9.3.1):
* Enlarge feature space to accommodate non-linear boundaries
* Use **kernel approach** for efficient computation

**The Kernel Trick**

**Key Mathematical Insight**: The support vector classifier solution involves only **inner products** of observations (not the observations themselves).

**Inner Product Definition**:

For two r-vectors a and b:
[
\langle a, b \rangle = \sum_{i=1}^r a_i b_i
]

For observations x_i and x_i':
[
\langle x_i, x_{i'} \rangle = \sum_{j=1}^p x_{ij} x_{i'j}
]

**Representation as Inner Products**:

The linear support vector classifier can be represented as:
[
f(x) = \beta_0 + \sum_{i=1}^n \alpha_i \langle x, x_i \rangle
]

Where:
* n parameters α_i (one per training observation)
* Only need n(n-1)/2 inner products ⟨x_i, x_i'⟩ to estimate parameters

**Sparsity Property**:
* α_i is **nonzero only for support vectors**
* If observation is not a support vector, α_i = 0
* Let S = collection of support vector indices

Then:
[
f(x) = \beta_0 + \sum_{i \in S} \alpha_i \langle x, x_i \rangle
]

This involves far fewer terms than full summation!

**Kernels: Generalizing Inner Products**

**Key Idea**: Replace every inner product ⟨x_i, x_i'⟩ with a **kernel function** K(x_i, x_i').

**Kernel Definition**: A function quantifying **similarity** of two observations.

**Common Kernels**:

1. **Linear Kernel** (standard):
   [
   K(x_i, x_{i'}) = \sum_{j=1}^p x_{ij} x_{i'j}
   ]
   * This gives back the support vector classifier
   * Quantifies similarity using Pearson correlation

2. **Polynomial Kernel** (degree d):
   [
   K(x_i, x_{i'}) = \left(1 + \sum_{j=1}^p x_{ij} x_{i'j}\right)^d
   ]
   * d is a positive integer
   * d > 1 creates much more flexible decision boundary
   * Equivalent to fitting SVC in higher-dimensional space with polynomials of degree d
   * When d = 1, reduces to linear SVC
   * See Figure 9.9 (left) for example with d = 3

3. **Radial Kernel** (RBF/Gaussian):
   [
   K(x_i, x_{i'}) = \exp\left(-\gamma \sum_{j=1}^p (x_{ij} - x_{i'j})^2\right)
   ]
   * γ is a positive constant
   * See Figure 9.9 (right) for example

**SVM with Non-linear Kernel**

When using a non-linear kernel, the classifier becomes:
[
f(x) = \beta_0 + \sum_{i \in S} \alpha_i K(x, x_i)
]

**How the Radial Kernel Works**

* If test observation x* is **far** from training observation x_i:
  * Euclidean distance Σ(x*_j - x_ij)² is large
  * K(x*, x_i) ≈ 0 (exponentially small)
  * x_i plays virtually **no role** in f(x*)

* If x* is **near** x_i:
  * Distance is small
  * K(x*, x_i) ≈ 1
  * x_i **strongly influences** f(x*)

* **Very local behavior**: Only nearby training observations affect class label prediction

**Advantages of Kernels**

1. **Computational efficiency**:
   * Only need to compute K(x_i, x_i') for all n(n-1)/2 pairs
   * Don't need to explicitly work in enlarged feature space
   * Important when enlarged space is very large (computations would be intractable)

2. **Implicit feature spaces**:
   * Some kernels (e.g., radial) have **infinite-dimensional** implicit feature space
   * Could never compute in that space directly!
   * Kernel trick makes it tractable

### 9.3.3 - An Application to the Heart Disease Data

**Dataset & Setup**

* **Heart Disease Data** (from Chapter 8)
* **Goal**: Predict heart disease using 13 predictors (Age, Sex, Chol, etc.)
* **Data preparation**:
  * 297 subjects after removing 6 missing observations
  * Random split: 207 training, 90 test observations

**Methods Compared**

1. **LDA** (Linear Discriminant Analysis)
2. **Support Vector Classifier** (linear kernel, equivalent to SVM with polynomial kernel d = 1)
3. **SVM with Radial Kernel** (various γ values)

**Evaluation Metric: ROC Curves**

* All classifiers compute scores: f̂(X) = β̂₀ + β̂₁X₁ + β̂₂X₂ + ... + β̂ₚXₚ
* For cutoff t, classify as:
  * Heart disease if f̂(X) ≥ t
  * No heart disease if f̂(X) < t
* **ROC curve**: Plot false positive rate vs. true positive rate across range of t values
* **Optimal classifier**: Hugs top-left corner of ROC plot

**Training Set Results** (Figure 9.10)

**Left Panel - SVC vs. LDA**:
* Both perform well
* Support vector classifier appears **slightly superior** to LDA

**Right Panel - Radial Kernel SVMs**:
* Tested γ = 10⁻³, 10⁻², 10⁻¹
* As **γ increases** → fit becomes **more non-linear** → ROC curves improve
* γ = 10⁻¹ appears almost perfect (on training data)

**Test Set Results** (Figure 9.11) - The Reality Check

**Left Panel - SVC vs. LDA**:
* Support vector classifier has small advantage over LDA
* Differences not statistically significant

**Right Panel - Radial Kernel SVMs**:
* **γ = 10⁻¹** (best on training data) → **WORST on test data**!
  * Classic example of overfitting
  * More flexible method ≠ better test performance
  
* **γ = 10⁻²** and **γ = 10⁻³**:
  * Perform comparably to support vector classifier
  * All three outperform γ = 10⁻¹

**Key Lessons**

1. **Training error is misleading**:
   * Lower training error doesn't guarantee better test performance
   * Must evaluate on independent test data

2. **Flexibility trade-off**:
   * More flexible models (high γ) can overfit
   * Moderate flexibility often generalizes better

3. **Tuning is critical**:
   * γ parameter controls degree of non-linearity
   * Must be chosen carefully (via cross-validation)

4. **Linear vs. Non-linear**:
   * Linear methods (SVC, LDA) competitive on this dataset
   * Non-linear kernels don't always provide dramatic improvements

---

## Section 9.4 - SVMs with More than Two Classes

### 9.4.1 - One-Versus-One Classification
*[Content to be added]*

### 9.4.2 - One-Versus-All Classification
*[Content to be added]*

---

## Section 9.5 - Relationship to Logistic Regression
*[Content to be added]*

---

## Section 9.6 - Lab: Support Vector Machines

### 9.6.1 - Support Vector Classifier
*[Content to be added]*

### 9.6.2 - Support Vector Machine
*[Content to be added]*

### 9.6.3 - ROC Curves
*[Content to be added]*

### 9.6.4 - SVM with Multiple Classes
*[Content to be added]*

### 9.6.5 - Application to Gene Expression Data
*[Content to be added]*

---

## Section 9.7 - Exercises
*[Content to be added]*

---

## Notes
*[Add your notes here]*
