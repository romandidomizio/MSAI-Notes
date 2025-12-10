# Support Vector Machine: Soft Margin Classifier

**Lecture**: Module 6, Lecture 2  
**Course**: CSCA5622  
**Topic**: Soft Margin SVM, Slack Variables, C Parameter, Bias-Variance Trade-off

---

## Table of Contents
1. [Review: Hard Margin Classifier](#1-review-hard-margin-classifier)
2. [Mathematical Derivation: Distance to Hyperplane](#2-mathematical-derivation-distance-to-hyperplane)
3. [Soft Margin Classifier](#3-soft-margin-classifier)
4. [Slack Variables](#4-slack-variables)
5. [Margin Classification Zones](#5-margin-classification-zones)
6. [The C Parameter](#6-the-c-parameter)
7. [Bias-Variance Trade-off](#7-bias-variance-trade-off)
8. [Python Examples](#8-python-examples)
9. [Practice Problems](#9-practice-problems)

---

## 1. Review: Hard Margin Classifier

"Last time we talked about **maximum margin classifier**, another name, **hard merging classifier**, which has a **hyperplane** such that the **margins or the distance between the support and the hyperplane will be maximized**."

**Components**:
- **Hyperplane**: $w^Tx + b = 0$
- **Support Vectors**: Closest points
- **Margins**: Parallel boundaries

"The **goal is to make these margins as big as possible**. Having **bigger margin means that we have more safety or competence in terms of classification**."

---

## 2. Mathematical Derivation: Distance to Hyperplane

### Setup

"Let's **derive some math formula** that can be **useful for describing this optimization technique**."

**Goal**: Calculate perpendicular distance from point to hyperplane.

### Vectors

- $\vec{X}_A$: Vector to support point
- $\vec{X}_B$: Vector to arbitrary point on hyperplane
- $\vec{S} = \vec{X}_A - \vec{X}_B$: Difference vector

### Distance Formula

"We would like to **calculate the d**, which will be **S scalar value times the cosine Theta**, which is the **same as the S vector dot product the unit vector n**."

$$d = ||\vec{S}|| \cos(\theta) = \vec{S} \cdot \hat{n}$$

### Three-Dimensional Example

For 3D: $\vec{S} = [S_1, S_2, S_3]^T$ and $\hat{n} = [w_1, w_2, w_3]^T$

$$d = S_1 w_1 + S_2 w_2 + S_3 w_3$$

### Simplification

"Let's just simplify, **X_A is actually X**. Then we can do **X_1 W_1 + X_2 W_2 + X_3 W_3** and we can **call this guy, the rest, to be just some simple constant**, let's say **b**."

$$d = w^TX + b$$

### Handling Negative Distances

"To take care of that case, we're going to **assign a variable**, let's say **y for the point a is going to be plus one value** when it's **above the hyperplane**. It's **minus one** when it's **below the hyperplane**."

**Unified Constraint**:
$$y_i(w^Tx_i + b) \geq M$$

Normalized: $y_i(w^Tx_i + b) \geq 1$

---

## 3. Soft Margin Classifier

### Motivation

"When we have **in separate data**, what we need to do is that we need to just **relax the condition**. Instead of **having heart margin**, we **accept some errors by softening the margin**."

"This is called the **soft margin classifier**, or on other words, **support vector classifier**."

---

## 4. Slack Variables

### Introduction

"When they say we **relax the condition**, we introduce a **new variable called a slack variable**. **So this one**, which **helps to give some wiggle room for this M**."

**Soft Margin Constraint**:
$$y_i(w^Tx_i + b) \geq 1 - \xi_i$$

where $\xi_i \geq 0$ is the slack variable for point $i$.

### Constraints

1. $\xi_i \geq 0$ for all $i$
2. $\sum_{i=1}^{n} \xi_i \leq C$

### C Parameter

"**This c represent the budget for the error**. In other words, if **C is large**, then we can **tolerate more errors**. Also **c is a hyperparameter** so the **user gets to choose how much better budget we have**."

### Complete Formulation

$$\min_{w,b,\xi} \frac{1}{2}||w||^2 + C\sum_{i=1}^{n}\xi_i$$

Subject to:
- $y_i(w^Tx_i + b) \geq 1 - \xi_i$ for all $i$
- $\xi_i \geq 0$ for all $i$

---

## 5. Margin Classification Zones

### Zone Definitions

| Zone | Constraint | Slack $\xi_i$ | Status |
|------|------------|---------------|--------|
| **Safe Margin** | $y_i(w^Tx_i + b) > 1$ | $\xi_i = 0$ | Correct ✓ |
| **On Margin** | $y_i(w^Tx_i + b) = 1$ | $\xi_i = 0$ | Correct ✓ (Support) |
| **Wrong Side of Margin** | $0 < y_i(w^Tx_i + b) < 1$ | $0 < \xi_i < 1$ | Correct ✓ (Violates margin) |
| **On Hyperplane** | $y_i(w^Tx_i + b) = 0$ | $\xi_i = 1$ | Ambiguous |
| **Misclassified** | $y_i(w^Tx_i + b) < 0$ | $\xi_i > 1$ | Wrong ✗ |

---

## 6. The C Parameter

### Question 1: Maximum Misclassifications

"What is the **maximum number of supports in the wrong side of the hyperplane** when the **C is given**?"

**Answer**: Maximum of $\lfloor C \rfloor$ misclassified points.

"The **maximum number of errors can be C** if **all of the slack variables are equal to one**."

### Question 2: Effect on Margin

"**What happens to the margin when C decreases**? The **answer is** the **margin becomes narrower**."

- Small C → Narrow margin, less tolerance
- Large C → Wide margin, more tolerance

### Question 3: Bias-Variance

"**Small C means a tighter margin**. **Less tolerance to the error means that we will get a more accurate model**. That means **less bias**. **Bias decreases**, but we will have instead a **higher variance**."

| C Value | Margin | Bias | Variance |
|---------|--------|------|----------|
| Small | Narrow | Low | High (Overfit) |
| Large | Wide | High | Low (Underfit) |

---

## 7. Lecture Recap

"We talked about a **hard margin classifier**, which has a **hyperplane that separates the support to this closest point to the hyperplane** as much as possible."

"We need to **introduce a slack variable** that will **make this condition a little bit softer**, which **allows some of the data points can be wrong side of the hyperplane** or **wrong side of the margin**."

"We also talked about **C parameter**, which is a **hyperparameter that we set**, which value **acts as a budget for the total error**."

### Preview: Kernel Methods

"**So far**, we talked about **linearly separable data**. However, in **some cases**, there's **no way to separate this data with just one hyperplane**. In that case, we will have to **use some more general form of kernel**. **We'll talk about kernel method in the next video**."

---

## 8. Python Examples

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification

# Generate overlapping data
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                          n_informative=2, n_clusters_per_class=1,
                          class_sep=0.8, random_state=42)

# Train SVMs with different C values
C_values = [0.1, 1, 10, 100]
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, C in enumerate(C_values):
    svm = SVC(kernel='linear', C=C)
    svm.fit(X, y)
    
    ax = axes[idx]
    # Plot decision boundary
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100),
                         np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 100))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    
    # Plot data
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='coolwarm', edgecolors='k')
    ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
              s=200, linewidths=2, facecolors='none', edgecolors='lime')
    
    margin_width = 2 / np.linalg.norm(svm.coef_)
    ax.set_title(f'C = {C}, SVs: {len(svm.support_vectors_)}, Margin: {margin_width:.3f}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 9. Practice Problems

### Problem 1: Slack Calculation

Given hyperplane: $2x_1 + 3x_2 - 6 = 0$

Calculate slack for $(2, 2)$ with $y = +1$:

$$f(x) = 2(2) + 3(2) - 6 = 4$$
$$y \cdot f(x) = (+1) \times 4 = 4$$
$$\xi = \max(0, 1 - 4) = 0$$

**Answer**: $\xi = 0$ (Safe Margin zone)

### Problem 2: C Parameter

With $C = 10$ and 5 misclassified points ($\xi_i = 1.5$ each):

Used: $5 \times 1.5 = 7.5$
Remaining: $10 - 7.5 = 2.5$

Maximum additional violations ($\xi_i = 0.5$): $2.5 / 0.5 = 5$

**Answer**: 5 additional points can violate margin.

### Problem 3: Bias-Variance

Three SVMs:
- A: $C = 0.1$, margin = 0.8, 50 SVs
- B: $C = 1.0$, margin = 0.6, 30 SVs
- C: $C = 100$, margin = 0.3, 10 SVs

**Analysis**:
- SVM A: High bias, low variance (underfit)
- SVM B: Balanced (best for CV)
- SVM C: Low bias, high variance (overfit)

**Answer**: SVM B likely best cross-validation performance.

---

## 10. Key Takeaways

**1. Slack Variables**: $\xi_i = \max(0, 1 - y_i(w^Tx_i + b))$

**2. C Parameter**: Controls error budget ($\sum \xi_i \leq C$)
- Small C → Narrow margin, low bias, high variance
- Large C → Wide margin, high bias, low variance

**3. Five Zones**: Safe, On Margin, Wrong Side Margin, On Hyperplane, Misclassified

**4. Optimization**: $\min \frac{1}{2}||w||^2 + C\sum\xi_i$

**Next**: Kernel methods for non-linear boundaries

