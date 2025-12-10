# Support Vector Machine: Kernel Trick

**Lecture**: Module 6, Lecture 3  
**Course**: CSCA5622  
**Topic**: Kernel Methods, Polynomial Kernels, RBF Kernels, Non-linear Decision Boundaries

---

## Table of Contents
1. [Recap: Hard and Soft Margin Classifiers](#1-recap-hard-and-soft-margin-classifiers)
2. [SVM as Non-Parametric: The Paradox](#2-svm-as-non-parametric-the-paradox)
3. [SVC vs SVM: Implementation Differences](#3-svc-vs-svm-implementation-differences)
4. [Inner Product Formulation](#4-inner-product-formulation)
5. [The Kernel Trick](#5-the-kernel-trick)
6. [Polynomial Kernels](#6-polynomial-kernels)
7. [Radial Basis Function (RBF) Kernels](#7-radial-basis-function-rbf-kernels)
8. [Kernel Selection](#8-kernel-selection)
9. [Python Examples](#9-python-examples)
10. [Practice Problems](#10-practice-problems)

---

## 1. Recap: Hard and Soft Margin Classifiers

### Hard Margin Classifier

"**Just a brief recap**, we talked about **hard margin classifier**, which **goal is to maximize this margin**, which is the **distance between the hyperplane and the support vectors**."

> **Slide Visualization**: 
> - Linear hyperplane separating two classes
> - Support vectors circled on both sides
> - Margin boundaries (dashed lines) parallel to hyperplane
> - Arrow indicating margin width

**Goal**: Maximize margin = $\frac{2}{||w||}$

### Soft Margin Classifier

"And then in case we had **inseparable data like this**, we simply **added the select variable epsilon to all these data points** and then **this epsilon to specify how much they deviated from the margin**."

> **Slide Visualization**: 
> - Overlapping data with hyperplane
> - Some points violating margin (inside margin boundaries)
> - Slack variables $\xi$ shown with arrows indicating deviation

"**So this for red points** and then **this amount for different bullet points here**."

**Slack Variables**: $\xi_i \geq 0$ for each point $i$

"And **these slack variables need to satisfy two condition** such as **it has to be no negative value**. And then we also **define the C parameter**, which gives an **idea how much of error budget we have**."

**Constraints**:
1. $\xi_i \geq 0$ for all $i$
2. $\sum_{i=1}^{n} \xi_i \leq C$

### Motivation for Kernels

"Also, we mentioned that **sometimes the data can be not possible to use one hyperplane to separate the data**."

> **Slide Visualization**: 
> - XOR pattern or concentric circles
> - No single linear hyperplane can separate

"So in that case, we need to use some **special trick called the Kerner trick**, which will be **the subject of this video**."

---

## 2. SVM as Non-Parametric: The Paradox

### The Question

"**Before we go on what the kernels are**, let's think about this."

"So in **support vector classifier**, which is **another name for submerging classifier**, we mentioned that we had to **satisfy all these conditions**."

**SVC Hyperplane Formula**:
$$f(x) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_p x_p$$

Or in vector notation:
$$f(x) = \beta_0 + \beta^T x$$

"And **this part is the formula for the hyperplane**, let's call it **fx**. And then **this beta 0 and beta 1 and all the way to the beta p are the coefficients** for this equation."

"And the **optimizer will find the values for this coefficient**."

### The Paradox

"**Now, we can ask ourselves why do we call SVM as a non parametric method** when we **do see these parameters in the equation**?"

**Apparent Contradiction**:
- SVM has parameters: $\beta_0, \beta_1, ..., \beta_p$
- Yet classified as non-parametric method

### The Answer

"**That's very much related to the use of kernels**."

**Key Insight**: The number of effective "parameters" depends on the **number of support vectors**, which is **data-dependent**, not fixed beforehand. This makes SVM non-parametric.

---

## 3. SVC vs SVM: Implementation Differences

### Terminology Clarification

"And you might notice that **I use this term SVC**, which is a **support vector classifier versus SVM**, **support vector machines**."

"**It's not very important** but **support vector machine generally refers to some generalization of support vector classifier** where a **support vector classifier usually refers to the soft margin classifier**."

**Distinction**:
- **SVC (Support Vector Classifier)**: Usually soft margin with linear kernel
- **SVM (Support Vector Machine)**: General term, includes kernel methods

### Implementation Libraries

"In epsilon, they **use a different algorithm**. So **SPC uses a liblinear**. So it's very much **similar to the optimization algorithm that we use in logistic regression**."

**SVC Implementation**:
- Library: **liblinear**
- Similar to logistic regression optimization
- Efficient for linear problems

"**Whereas this SVM uses the SVM algorithm**, which is **specially made for SVM**. And **this algorithm uses the kernels**, all right?"

**SVM Implementation**:
- Library: **libSVM**
- Designed specifically for kernel methods
- More general but potentially slower

---

## 4. Inner Product Formulation

### Standard Hyperplane Formula

"So let's talk about **what the kernels are**. So this is again, **hard merging classifier** and this is **soft merging classifier**."

"And **this is the formula for the hyperplane**."

$$f(x) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_p x_p$$

### Alternative Formulation

"**We're going to introduce a different mass formula**, which is **equivalent to this formula f(x)**. However, we **skip the derivation and just show the result**."

"So **using the inner product**, it is known that **this formula f(x) can be rewritten to this formula**."

**Inner Product Formulation**:
$$f(x) = \alpha_0 + \sum_{i' \in SV} \alpha_{i'} y_{i'} \langle x_{i'}, x \rangle$$

where:
- $\alpha_{i'}$: Coefficients (dual variables)
- $y_{i'}$: Label of support vector $i'$
- $\langle x_{i'}, x \rangle$: Inner product between support vector $i'$ and test point $x$
- $SV$: Set of support vectors

"And **this is a dot product**. So if you have **xi prime**, then **this is that product between a point i' and a point i**."

### Linear Kernel Definition

"And **is that product represent a linear kernel**, oftentimes they will **call it as k kernel xi, xi'**."

**Linear Kernel**:
$$K(x_{i'}, x_i) = \langle x_{i'}, x_i \rangle = x_{i'}^T x_i$$

**Rewritten Formula**:
$$f(x) = \alpha_0 + \sum_{i' \in SV} \alpha_{i'} y_{i'} K(x_{i'}, x)$$

"**That product again is a linear kernel**. So **essentially this is same as this one**."

### Time Complexity Comparison

"However, when we **implement the algorithm**, it will have a **different time complexity**."

#### SVC (liblinear)

"So for example, the **SVC that use the liblinear library**, will have **time complexity of number of data point times number of features**."

**SVC Time Complexity**: $O(n \times p)$

where:
- $n$: Number of data points
- $p$: Number of features

#### SVM with Linear Kernel (libSVM)

"And if you **use the libSVM and solve for linear data**, then it's **going to take more time**. It's going to have **SVM with linear kernel**. It's going to take **n squared x p**."

**SVM Linear Kernel Time Complexity**: $O(n^2 \times p)$

### When Kernels Are Useful

"So **by using kernel**, it doesn't seem **it's useful for the linear data**."

For linear data: SVC is more efficient ($O(np)$ vs $O(n^2p)$)

"**However**, the **kernel method shines when it comes to complex data**."

---

## 5. The Kernel Trick

### Motivation: Non-Linear Data

"So let's have a look. When we have **this type of data that's not possible to separate by linear hyperplane**, what we want to do is this..."

> **Slide Visualization**: 
> - 1D data showing points: `- + + - -`
> - In one dimension, impossible to separate with single point

### 1D Example: Adding Dimension

"...so let's say a **simpler example**, we have a **data that's not linear separable**."

"**So in the one dimension**, The **hyperplane will be just a point**. So we need a **two hyperplane in order to separate perfectly**. However, **it's not possible**."

**Problem**: In 1D, need 2 decision points to separate `- + + - -`, but SVM uses only one hyperplane.

"So **the trick is**, we can **add one dimension here** and then now we can **separate this perfectly with this one hyperplane**."

> **Slide Visualization**: 
> - Original 1D: `x-axis` with points scattered
> - After adding dimension: 2D plot with parabolic transformation
> - Points now linearly separable with single line

**Key Idea**: Transform $x \to (x, x^2)$

"So **adding one more dimension is a key** and it's called the **kernel trick**."

### 2D to 3D Example

"So again, **this data is not separable in 2D using linear hyperplane**."

> **Slide Visualization**: 
> - 2D XOR pattern or concentric circles
> - No linear separator exists

"So what we do is we **add the third dimension**. **So this is a z by the way** and this is **maybe we can call it x and y**."

"**So we're going to introduce z** and maybe **x here and y here**."

**Transformation**: $(x, y) \to (x, y, z)$ where $z = f(x, y)$

"And now we can see that **this data is separable with the hyperplane like this**."

> **Slide Visualization**: 
> - 3D plot showing data lifted to higher dimension
> - Plane slicing through the 3D space
> - When projected back to 2D, creates non-linear boundary

### Mathematical Interpretation

"**Adding one more dimension means** that we want to **make a higher order terms in the function**."

**Example**: For 2D input $(x_1, x_2)$:

**Original features**: $x_1, x_2$

**With higher-order terms**: $x_1, x_2, x_1^2, x_2^2, x_1x_2, ...$

"Okay, so we have a **p number of features in the data**. And then we can **add a high order terms in order to make an extra dimensions** to separate the data point, which was **previously not separable in the linear function**."

### The Problem: Dimensionality Explosion

"So we can **add high order terms like this**."

For $p$ features with degree $d$, number of terms grows as $\binom{p+d}{d}$.

**Example**: $p=10$ features, $d=3$ degree
- Number of terms: $\binom{13}{3} = 286$

"But then now **really what happens is that now our optimization need to find all these parameter values for the high order terms**, which might be a lot if you **add even more high order terms**."

"And as well as if you have a **larger number of features**. **It's going to be a problem like we saw in the polynomial regression**."

**Problems**:
1. Computational cost (too many parameters)
2. Overfitting risk
3. Memory requirements

### The Solution: Kernel Functions

"So **instead of adding directly higher order terms**, we can **use a kernel trick instead**."

**Key Insight**: We don't need to explicitly compute high-dimensional features. We only need **inner products** in the higher-dimensional space, which can be computed efficiently using **kernel functions**.

"So let's **make a use of this inner product**, we can **create a function**, **kernel function k that has this form**."

---

## 6. Polynomial Kernels

### Definition

"So **this is the, that product** and then **represent first order terms**. And by **having a constant plus this**, **that product to the order of d**, we can **create the polynomial function for the high order terms**."

**Polynomial Kernel**:
$$K(x, x') = (c + \langle x, x' \rangle)^d$$

or equivalently:
$$K(x, x') = (c + x^T x')^d$$

where:
- $c$: Constant term (often $c=1$)
- $d$: Degree of polynomial
- $\langle x, x' \rangle$: Inner product

**Common Choices**:
- $d=2$: Quadratic kernel
- $d=3$: Cubic kernel

### Generalized Function

"And then we can **generalize our function to be a form that has its kernels**."

**SVM with Kernel**:
$$f(x) = \alpha_0 + \sum_{i' \in SV} \alpha_{i'} y_{i'} K(x_{i'}, x)$$

### Example: Polynomial Kernel Results

"So let's have a look. When we have **this type of data that might involve a non-linear decision boundary**. We can **use polynomial kernel that we just saw**."

> **Slide Visualization**: 
> - Left: Original 2D data with circular/curved pattern
> - Right: Decision boundary with polynomial kernel

"So by **having polynomial kernel**, we can have **this type of decision boundary**."

"**Shows the data result when we had the d=2 for polynomial kernels**. It **nicely separate these blue points and the red points** by **adding another dimension to the data**."

**Result**: Curved decision boundary that separates non-linearly separable data.

### Mathematical Example: Degree 2 in 2D

For $x = [x_1, x_2]^T$ and $x' = [x_1', x_2']^T$ with $c=1$, $d=2$:

$$K(x, x') = (1 + x_1x_1' + x_2x_2')^2$$

Expanding:
$$= 1 + 2x_1x_1' + 2x_2x_2' + x_1^2{x_1'}^2 + x_2^2{x_2'}^2 + 2x_1x_2x_1'x_2'$$

This is equivalent to inner product in 6D space:
$$\phi(x) = [1, \sqrt{2}x_1, \sqrt{2}x_2, x_1^2, x_2^2, \sqrt{2}x_1x_2]^T$$

**Key**: We compute $K(x, x')$ directly (3 operations) instead of computing $\phi(x)$ and $\phi(x')$ (6D vectors) then taking inner product!

---

## 7. Radial Basis Function (RBF) Kernels

### Definition

"**There are other types of kernels**. And **another very famous one is called the radial kernel** or sometimes called the **radial basis functional kernel or RBF for short**."

"And **takes this form**, this **kind of Gaussian shape kernel defines the RBF kernel**."

**RBF Kernel**:
$$K(x, x') = \exp\left(-\gamma ||x - x'||^2\right)$$

where:
- $\gamma > 0$: Kernel coefficient (controls width)
- $||x - x'||^2 = (x - x')^T(x - x')$: Squared Euclidean distance

**Alternative form**:
$$K(x, x') = \exp\left(-\frac{||x - x'||^2}{2\sigma^2}\right)$$

where $\gamma = \frac{1}{2\sigma^2}$

### Interpretation

**Gaussian Shape**: Kernel value decreases as distance between $x$ and $x'$ increases.

- $||x - x'|| = 0$: $K = 1$ (same point)
- $||x - x'|| \to \infty$: $K \to 0$ (far points)

**Effect of $\gamma$**:
- **Small $\gamma$**: Wide Gaussian, smooth decision boundary
- **Large $\gamma$**: Narrow Gaussian, complex decision boundary (risk of overfitting)

### Results

"And the **result is like this**. So it's like around the **shape Basis kernel will be able to separate this data into three blocks**."

> **Slide Visualization**: 
> - Data with multiple circular clusters
> - RBF kernel creates circular/radial decision boundaries
> - Each cluster separated by its own region

**Characteristics**:
- Creates localized, radial decision regions
- Very flexible (can model complex patterns)
- Infinite-dimensional feature space (implicitly)

---

## 8. Kernel Selection

### The Challenge

"So **having kernel is great**, you can **solve some complex data**. However, we need to **think ahead**, **what kind of kernels that we should use**?"

**Problem**: No universal best kernel. Choice depends on data structure.

### Linear Kernel

"So when it's a **linear separable**, you can see that we **don't need any fancy kernels**. **Just a linear kernel or linear SVM or SVC** that does not use the kernels at all **will solve perfectly**."

> **Slide Visualization**: 
> Two side-by-side plots:
> - Linear SVM: Perfect separation with straight line
> - RBF SVM: Also separates, but unnecessarily complex

"**Whereas RBF kernel is fancy kernel**, so the **radial basis kernel can also solve the problem** depending on **how the data look like**."

"So **this data was generated by a block data**. So and **linear is separable**. So it was **both linear SVM and RBF SVM worked well**."

**Recommendation**: For linearly separable data, use **linear kernel** (simpler, faster).

### Moon/Yin-Yang Shape Data

"**Different types of data like this shape**, like **yin and yang or moon shape in eskilon**, they **look like this type of data usually**."

> **Slide Visualization**: 
> Grid showing different kernels on crescent moon data:
> - Linear SVM: Poor performance, straight boundary
> - Polynomial: Depends on degree
> - RBF: Excellent performance, follows curve

"And the **linear SVM doesn't work very well**. However, the **radial basis kernel did well on this**. **Other kernels did not do well for this type of data**."

### Circular/Doughnut Shape Data

"**How about this circular doughnut shape of data**? **Linear SVM did not do very well** as you can **expect but radial kernel is perfect for this type of data** because **the data shape is radial**."

> **Slide Visualization**: 
> Concentric circles (doughnut pattern):
> - Linear SVM: Fails completely
> - RBF SVM: Perfect circular boundary

**Key Insight**: RBF excels at **radially symmetric patterns**.

### Summary

"So now you can see that the **choice of kernel strongly depend on the pattern of the data**."

| Data Pattern | Best Kernel | Reasoning |
|--------------|-------------|-----------|
| **Linearly separable** | Linear | Simple, fast, no overfitting |
| **Curved boundaries** | RBF or Polynomial | Captures non-linear patterns |
| **Radial patterns** | RBF | Matches circular structure |
| **Specific polynomial patterns** | Polynomial (choose $d$) | Degree matches data complexity |

"So although the **kernel is very convenient for this linear data**, it **requires the user to think about what the data looks like and guess what the best kernel would be**."

**Challenge**: Kernel selection is an **art + science**:
1. Visualize your data
2. Try multiple kernels with cross-validation
3. Consider computational cost
4. Avoid overfitting (especially with RBF and high $\gamma$)

---

## 9. Python Examples

### Example 1: Comparing Kernels on Different Data Patterns

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification, make_circles, make_moons

# Generate different data patterns
np.random.seed(42)

# 1. Linearly separable
X_linear, y_linear = make_classification(
    n_samples=200, n_features=2, n_redundant=0, n_informative=2,
    n_clusters_per_class=1, class_sep=2.0, random_state=42
)

# 2. Circular pattern
X_circles, y_circles = make_circles(n_samples=200, noise=0.1, factor=0.4, random_state=42)

# 3. Moon pattern
X_moons, y_moons = make_moons(n_samples=200, noise=0.1, random_state=42)

# Datasets
datasets = [
    ('Linear Separable', X_linear, y_linear),
    ('Circular Pattern', X_circles, y_circles),
    ('Moon Pattern', X_moons, y_moons)
]

# Kernels to test
kernels = ['linear', 'poly', 'rbf']
kernel_names = ['Linear', 'Polynomial (d=3)', 'RBF']

# Plot
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for row, (data_name, X, y) in enumerate(datasets):
    for col, (kernel, kernel_name) in enumerate(zip(kernels, kernel_names)):
        ax = axes[row, col]
        
        # Train SVM
        if kernel == 'poly':
            svm = SVC(kernel=kernel, degree=3, gamma='auto')
        else:
            svm = SVC(kernel=kernel, gamma='auto')
        svm.fit(X, y)
        
        # Create mesh
        h = 0.02
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # Predict
        Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='coolwarm', edgecolors='k')
        
        # Highlight support vectors
        ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
                  s=150, linewidths=2, facecolors='none', edgecolors='lime')
        
        # Accuracy
        acc = svm.score(X, y)
        
        if row == 0:
            ax.set_title(f'{kernel_name}\nAcc: {acc:.3f}', fontsize=12)
        else:
            ax.set_title(f'Acc: {acc:.3f}', fontsize=12)
        
        if col == 0:
            ax.set_ylabel(data_name, fontsize=12)
        
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.savefig('kernel_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nKERNEL SELECTION INSIGHTS:")
print("="*60)
print("Linear Separable Data: All kernels work, but linear is fastest")
print("Circular Pattern: RBF excels (radial symmetry)")
print("Moon Pattern: RBF and Polynomial work well, Linear fails")
```

### Example 2: Polynomial Kernel with Different Degrees

```python
# Polynomial kernel with varying degrees
from sklearn.datasets import make_classification

# Generate polynomial-separable data
np.random.seed(10)
X_poly = np.random.randn(200, 2)
y_poly = (X_poly[:, 0]**2 + X_poly[:, 1]**2 > 1.5).astype(int)

# Test different polynomial degrees
degrees = [1, 2, 3, 5]
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.ravel()

for idx, degree in enumerate(degrees):
    svm = SVC(kernel='poly', degree=degree, coef0=1, gamma='auto')
    svm.fit(X_poly, y_poly)
    
    # Create mesh
    xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    ax = axes[idx]
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X_poly[:, 0], X_poly[:, 1], c=y_poly, s=40, cmap='coolwarm', edgecolors='k')
    ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
              s=180, linewidths=2, facecolors='none', edgecolors='lime',
              label=f'SVs: {len(svm.support_vectors_)}')
    
    acc = svm.score(X_poly, y_poly)
    ax.set_title(f'Polynomial Kernel (degree={degree})\nAccuracy: {acc:.3f}', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('polynomial_degrees.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nPOLYNOMIAL DEGREE EFFECTS:")
print("="*60)
print("d=1: Linear boundary (equivalent to linear kernel)")
print("d=2: Quadratic boundary (good for circles/ellipses)")
print("d=3,5: Higher complexity (risk of overfitting)")
```

### Example 3: RBF Kernel with Different Gamma Values

```python
# RBF kernel with varying gamma
X_rbf, y_rbf = make_moons(n_samples=150, noise=0.15, random_state=42)

gammas = [0.1, 1.0, 10.0, 100.0]
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.ravel()

for idx, gamma in enumerate(gammas):
    svm = SVC(kernel='rbf', gamma=gamma)
    svm.fit(X_rbf, y_rbf)
    
    # Create mesh
    xx, yy = np.meshgrid(np.linspace(X_rbf[:, 0].min()-0.5, X_rbf[:, 0].max()+0.5, 200),
                         np.linspace(X_rbf[:, 1].min()-0.5, X_rbf[:, 1].max()+0.5, 200))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    ax = axes[idx]
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X_rbf[:, 0], X_rbf[:, 1], c=y_rbf, s=40, cmap='coolwarm', edgecolors='k')
    ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
              s=180, linewidths=2, facecolors='none', edgecolors='lime')
    
    acc = svm.score(X_rbf, y_rbf)
    ax.set_title(f'RBF Kernel (γ={gamma})\nSVs: {len(svm.support_vectors_)}, Acc: {acc:.3f}', 
                fontsize=12)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rbf_gamma.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nGAMMA PARAMETER EFFECTS:")
print("="*60)
print("Small γ (0.1): Smooth, simple boundary (underfitting risk)")
print("Medium γ (1.0): Balanced complexity")
print("Large γ (10-100): Complex, wiggly boundary (overfitting risk)")
print("\nRule: γ ≈ 1/(n_features * X.var())")
```

### Example 4: Time Complexity Comparison

```python
import time

# Compare SVC vs SVM with linear kernel
n_samples_list = [100, 500, 1000, 2000]
times_svc = []
times_svm_linear = []

for n in n_samples_list:
    X_temp, y_temp = make_classification(n_samples=n, n_features=20, random_state=42)
    
    # SVC (liblinear)
    start = time.time()
    svc = SVC(kernel='linear', max_iter=1000)
    svc.fit(X_temp, y_temp)
    times_svc.append(time.time() - start)
    
    # SVM with linear kernel (libSVM)
    start = time.time()
    svm_linear = SVC(kernel='linear', max_iter=1000)
    svm_linear.fit(X_temp, y_temp)
    times_svm_linear.append(time.time() - start)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(n_samples_list, times_svc, 'o-', linewidth=2, markersize=8, label='SVC O(n×p)')
plt.plot(n_samples_list, times_svm_linear, 's-', linewidth=2, markersize=8, 
         label='SVM Linear Kernel O(n²×p)')
plt.xlabel('Number of Samples (n)', fontsize=12)
plt.ylabel('Training Time (seconds)', fontsize=12)
plt.title('Time Complexity: SVC vs SVM with Linear Kernel', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('time_complexity.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nTIME COMPLEXITY RESULTS:")
print("="*60)
for i, n in enumerate(n_samples_list):
    print(f"n={n:4d}: SVC={times_svc[i]:.4f}s, SVM Linear={times_svm_linear[i]:.4f}s")
```

---

## 10. Practice Problems

### Problem 1: Kernel Function Calculation

**Question**: Given two 2D points $x = [2, 3]^T$ and $x' = [1, -1]^T$, calculate the kernel values for:

a) Linear kernel: $K(x, x') = x^T x'$

b) Polynomial kernel (degree 2, $c=1$): $K(x, x') = (1 + x^T x')^2$

c) RBF kernel ($\gamma = 0.5$): $K(x, x') = \exp(-\gamma ||x - x'||^2)$

**Solution**:

**Part a): Linear Kernel**

$$K(x, x') = x^T x' = [2, 3] \cdot [1, -1]$$
$$= 2(1) + 3(-1) = 2 - 3 = -1$$

**Answer**: $K(x, x') = -1$

**Part b): Polynomial Kernel (d=2, c=1)**

First, calculate inner product (from part a): $x^T x' = -1$

$$K(x, x') = (1 + x^T x')^2 = (1 + (-1))^2 = 0^2 = 0$$

**Answer**: $K(x, x') = 0$

**Interpretation**: These two points are "orthogonal" in the polynomial feature space.

**Part c): RBF Kernel ($\gamma = 0.5$)**

Calculate squared distance:
$$||x - x'||^2 = (2-1)^2 + (3-(-1))^2 = 1^2 + 4^2 = 1 + 16 = 17$$

Apply RBF formula:
$$K(x, x') = \exp(-\gamma ||x - x'||^2) = \exp(-0.5 \times 17)$$
$$= \exp(-8.5) \approx 0.000203$$

**Answer**: $K(x, x') \approx 0.0002$

**Interpretation**: Points are far apart, so kernel value is very small (near 0).

---

### Problem 2: Polynomial Kernel Feature Space

**Question**: For a 2D input $x = [x_1, x_2]^T$, the polynomial kernel with $c=0$ and $d=2$ is:
$$K(x, x') = (x^T x')^2$$

a) What is the explicit feature mapping $\phi(x)$ that this kernel corresponds to?

b) What is the dimensionality of the feature space?

c) Calculate $K(x, x')$ for $x = [1, 2]^T$ and $x' = [3, 1]^T$ using:
   - i) The kernel function directly
   - ii) The explicit feature mapping

**Solution**:

**Part a): Explicit Feature Mapping**

Expand $(x_1 x_1' + x_2 x_2')^2$:
$$= x_1^2 {x_1'}^2 + x_2^2 {x_2'}^2 + 2x_1 x_2 x_1' x_2'$$

This is equivalent to:
$$\phi(x)^T \phi(x')$$

where:
$$\phi(x) = [x_1^2, x_2^2, \sqrt{2} x_1 x_2]^T$$

**Answer**: $\phi(x) = [x_1^2, x_2^2, \sqrt{2} x_1 x_2]^T$

**Part b): Dimensionality**

The feature vector $\phi(x)$ has 3 components.

**Answer**: Feature space dimensionality = **3D**

(Original 2D → Transformed to 3D)

**Part c): Kernel Calculation**

**Method i): Direct kernel computation**

$$x^T x' = 1(3) + 2(1) = 3 + 2 = 5$$
$$K(x, x') = (x^T x')^2 = 5^2 = 25$$

**Answer**: $K(x, x') = 25$

**Method ii): Explicit feature mapping**

$$\phi([1, 2]) = [1^2, 2^2, \sqrt{2}(1)(2)] = [1, 4, 2\sqrt{2}]$$
$$\phi([3, 1]) = [3^2, 1^2, \sqrt{2}(3)(1)] = [9, 1, 3\sqrt{2}]$$

$$\phi(x)^T \phi(x') = 1(9) + 4(1) + 2\sqrt{2}(3\sqrt{2})$$
$$= 9 + 4 + 6(2) = 9 + 4 + 12 = 25$$

**Answer**: $K(x, x') = 25$ ✓ (Matches!)

**Key Insight**: Kernel computes in 1 operation, explicit mapping requires 3D vectors. This is the **power of the kernel trick**!

---

### Problem 3: Kernel Selection

**Question**: You have four datasets with the following characteristics. For each, recommend the most appropriate kernel and explain why.

a) Dataset A: 10,000 samples, 50 features, linearly separable based on visualization

b) Dataset B: 500 samples, 2 features, concentric circular pattern (inner circle class 0, outer ring class 1)

c) Dataset C: 1,000 samples, 5 features, decision boundary appears to follow a quadratic curve

d) Dataset D: 200 samples, 100 features (p >> n), unknown structure

**Solution**:

**Part a): Dataset A (Large, Linearly Separable)**

**Recommendation**: **Linear kernel** (or SVC without kernels)

**Reasoning**:
1. **Linearly separable**: No need for complex kernels
2. **Large n (10,000)**: Kernel methods have $O(n^2)$ complexity, too slow
3. **Efficiency**: SVC with linear kernel runs in $O(n \times p) = O(10,000 \times 50)$
4. **Simplicity**: Avoids overfitting

**Implementation**: `SVC(kernel='linear')` or `LinearSVC()`

**Part b): Dataset B (Concentric Circles)**

**Recommendation**: **RBF kernel**

**Reasoning**:
1. **Radial pattern**: RBF naturally handles circular/radial structures
2. **Formula match**: $K(x, x') = \exp(-\gamma ||x - x'||^2)$ depends only on distance
3. **Proven effective**: RBF excels on circular/doughnut patterns (from lecture)
4. **Small n (500)**: Computational cost is manageable

**Implementation**: `SVC(kernel='rbf', gamma='scale')`

**Tuning**: Use cross-validation to find optimal $\gamma$

**Part c): Dataset C (Quadratic Boundary)**

**Recommendation**: **Polynomial kernel with degree 2**

**Reasoning**:
1. **Quadratic curve**: Matches polynomial degree 2
2. **Interpretability**: Degree matches known structure
3. **Efficiency**: Lower degree = less overfitting than RBF
4. **Moderate size**: n=1,000 is reasonable for kernel methods

**Implementation**: `SVC(kernel='poly', degree=2, coef0=1)`

**Alternative**: Could also try RBF with small $\gamma$ for smooth boundary

**Part d): Dataset D (High-dimensional, p >> n)**

**Recommendation**: **Linear kernel**

**Reasoning**:
1. **Curse of dimensionality**: In high dimensions, data tends to be linearly separable
2. **Overfitting risk**: Complex kernels very likely to overfit with p >> n
3. **Computational advantage**: Linear is much faster
4. **Small sample size**: n=200 is small, need simple model

**Additional**: Consider regularization (small C) to prevent overfitting

**Implementation**: `SVC(kernel='linear', C=0.1)`

**Warning**: With 100 features and 200 samples, consider feature selection first!

---

### Problem 4: RBF Kernel Analysis

**Question**: An RBF SVM is trained on 100 samples with $\gamma = 2$.

a) For two points with squared distance $||x - x'||^2 = 0.5$, what is the kernel value?

b) At what squared distance does the kernel value drop to $e^{-1} \approx 0.368$?

c) If you increase $\gamma$ to 10, what happens to the decision boundary complexity? Will overfitting increase or decrease?

d) All 100 training samples become support vectors. What does this indicate about the model?

**Solution**:

**Part a): Kernel Value Calculation**

Given: $\gamma = 2$, $||x - x'||^2 = 0.5$

$$K(x, x') = \exp(-\gamma ||x - x'||^2)$$
$$= \exp(-2 \times 0.5) = \exp(-1) = e^{-1} \approx 0.368$$

**Answer**: $K(x, x') \approx 0.368$

**Part b): Distance for Kernel = $e^{-1}$**

We want: $K(x, x') = e^{-1}$

$$\exp(-\gamma ||x - x'||^2) = e^{-1}$$
$$-\gamma ||x - x'||^2 = -1$$
$$||x - x'||^2 = \frac{1}{\gamma} = \frac{1}{2} = 0.5$$

**Answer**: Squared distance = **0.5**

**General Rule**: Kernel drops to $e^{-1}$ at distance $||x - x'||^2 = \frac{1}{\gamma}$

**Part c): Effect of Increasing $\gamma$ (2 → 10)**

**Kernel Width**: $\frac{1}{\gamma} = \frac{1}{10} = 0.1$ (vs. 0.5 before)

**Interpretation**: With $\gamma = 10$:
- Kernel becomes **narrower** (more localized)
- Points must be **5× closer** to have same influence
- Each support vector affects smaller region

**Decision Boundary**:
- **More complex**: Can create tighter, more intricate boundaries
- **Higher capacity**: Fits training data more closely

**Overfitting**: **INCREASES**

**Reasoning**:
- Model becomes more sensitive to individual training points
- Less smoothing/averaging
- Higher variance, lower bias

**Answer**: Decision boundary becomes more complex, overfitting **increases**.

**Part d): All Samples Are Support Vectors**

**Observation**: 100/100 samples are support vectors (100%)

**Indicates**:

1. **Severe Overfitting**:
   - Model memorizes every training point
   - No generalization
   - Every point influences decision boundary

2. **Possible Causes**:
   - **$\gamma$ too large**: Kernel too narrow, treats each point individually
   - **C too large**: Allows complex boundary with no regularization
   - **Data too complex**: Inherently difficult to separate

3. **Poor Generalization Expected**:
   - Test accuracy likely much lower than training accuracy
   - Model not learning underlying pattern

**Recommendation**:
- **Decrease $\gamma$**: Smoother, simpler boundary
- **Decrease C**: More regularization
- **Try simpler kernel**: Perhaps linear or low-degree polynomial
- **More data**: If possible, to better represent true distribution

**Healthy SVM**: Typically only 10-40% of samples are support vectors

**Answer**: Model is severely overfitting. Reduce $\gamma$ and/or C.

---

### Problem 5: Kernel Trick Efficiency

**Question**: You're building an SVM for a 10-dimensional dataset ($p = 10$) and want to use a polynomial kernel with degree $d = 3$.

a) How many features would be in the explicit polynomial feature space?

b) Approximately how many operations does it take to compute the kernel $K(x, x')$ directly?

c) How many operations would it take to compute the same result using explicit feature mapping $\phi(x)^T \phi(x')$?

d) For $p = 100$ and $d = 3$, what's the dimensionality of the feature space? Is the kernel trick beneficial?

**Solution**:

**Part a): Feature Space Dimensionality ($p=10$, $d=3$)**

Number of features with degree up to $d$:
$$\text{dim} = \binom{p + d}{d} = \binom{10 + 3}{3} = \binom{13}{3}$$

$$= \frac{13!}{3! \cdot 10!} = \frac{13 \times 12 \times 11}{3 \times 2 \times 1} = \frac{1716}{6} = 286$$

**Answer**: **286 features** in the polynomial feature space

**Part b): Kernel Direct Computation**

$$K(x, x') = (c + x^T x')^d$$

**Operations**:
1. Inner product $x^T x'$: $p = 10$ multiplications + $(p-1) = 9$ additions = 19 ops
2. Add constant $c$: 1 addition
3. Raise to power $d$: 2 multiplications (for $d=3$: $x \times x \times x$)

**Total**: Approximately **22 operations**

**Answer**: ~**20-25 operations**

**Part c): Explicit Feature Mapping**

$$\phi(x)^T \phi(x')$$

**Operations**:
1. Compute $\phi(x)$: 286 feature values
2. Compute $\phi(x')$: 286 feature values
3. Inner product: 286 multiplications + 285 additions = 571 ops

**Total**: Approximately **570-580 operations** (ignoring feature computation overhead)

**Answer**: ~**570+ operations**

**Efficiency Gain**: $\frac{570}{22} \approx 26\times$ **faster** with kernel trick!

**Part d): High-Dimensional Case ($p=100$, $d=3$)**

$$\text{dim} = \binom{100 + 3}{3} = \binom{103}{3} = \frac{103 \times 102 \times 101}{6}$$

$$= \frac{1,061,106}{6} = 176,851$$

**Answer**: **176,851 features** in feature space!

**Is Kernel Trick Beneficial?**

**Absolutely YES!**

**Reasons**:
1. **Memory**: Storing 176,851-dimensional vectors is prohibitive
2. **Computation**: Inner product of 176,851-dim vectors = 353,702 operations
3. **Kernel**: Still only ~102 operations (100 for inner product + 2 for power)
4. **Speedup**: $\frac{353,702}{102} \approx 3,500\times$ **faster**!

**Key Insight**: As dimensionality grows, kernel trick becomes exponentially more beneficial. This is why kernels are essential for SVMs!

---

## 11. Key Takeaways

**1. Kernel Functions**: Replace explicit high-dimensional feature mapping with efficient computation
- Linear: $K(x, x') = x^T x'$
- Polynomial: $K(x, x') = (c + x^T x')^d$
- RBF: $K(x, x') = \exp(-\gamma ||x - x'||^2)$

**2. The Kernel Trick**: Compute inner products in high-dimensional space without explicitly transforming data
- **Efficiency**: Polynomial $(p+d) \choose d$ features → $O(p)$ kernel computation
- **Memory**: No need to store transformed features
- **Flexibility**: Can work in infinite-dimensional spaces (RBF)

**3. Kernel Selection**:
- **Linear**: Separable data, large datasets, high dimensions
- **Polynomial**: Specific polynomial patterns, degree matches structure
- **RBF**: General-purpose, radial patterns, flexible boundaries

**4. Hyperparameters**:
- **Polynomial**: degree $d$, coef0 $c$
- **RBF**: $\gamma$ (larger → more complex → overfit risk)

**5. SVC vs SVM**:
- **SVC** (liblinear): Fast for linear problems, $O(np)$
- **SVM** (libSVM): For kernel methods, $O(n^2p)$

**Next Lecture**: Advanced SVM topics and applications

---

## Glossary

- **Kernel Function**: Function computing inner product in (possibly infinite) high-dimensional space
- **Kernel Trick**: Computing $K(x, x') = \phi(x)^T \phi(x')$ without explicit $\phi$
- **Polynomial Kernel**: $K(x, x') = (c + x^T x')^d$, creates degree-$d$ decision boundaries
- **RBF Kernel**: Radial Basis Function, $K(x, x') = \exp(-\gamma ||x - x'||^2)$
- **Gamma ($\gamma$)**: RBF kernel parameter controlling width/complexity
- **Feature Mapping** ($\phi$): Transformation to higher-dimensional space
- **liblinear**: Efficient library for linear SVM
- **libSVM**: Library for kernel SVM methods
- **Support Vectors**: Training points with $\alpha_i > 0$ that define decision boundary
