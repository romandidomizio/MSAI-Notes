# Support Vector Machine: Performance

**Lecture**: Module 6, Lecture 4  
**Course**: CSCA5622  
**Topic**: SVM Performance, Comparison with Ensemble Methods, Best Practices

---

## Table of Contents
1. [Recap: Kernel Tricks](#1-recap-kernel-tricks)
2. [Properties of SVM](#2-properties-of-svm)
3. [Hinge Loss Function](#3-hinge-loss-function)
4. [sklearn Library Usage](#4-sklearn-library-usage)
5. [Performance Comparison Study](#5-performance-comparison-study)
6. [Training Time Analysis](#6-training-time-analysis)
7. [Model Selection Guidelines](#7-model-selection-guidelines)
8. [Python Examples](#8-python-examples)

---

## 1. Recap: Kernel Tricks

"**Last time**, we talked about **kernel tricks** which are used to **treat the non-linear data**. This is **not linearly separable**. Therefore, **SVM with the linear hyperplane wouldn't be able to separate**."

**Polynomial Kernel**: $K(x, x') = (c + x^T x')^d$
**RBF Kernel**: $K(x, x') = \exp(-\gamma ||x - x'||^2)$

"By **adding higher dimension**, we might be able to **separate the data points**, which was **impossible in a low dimension**."

---

## 2. Properties of SVM

### Feature Scaling Required

"**SVM needs a feature scaling**. That means we need to **normalize a feature by column** so that **all the features are more or less in the same range of the values**."

**Standard Scaling**: $x_{scaled} = \frac{x - \mu}{\sigma}$

### Time Complexity

"**Time complexities scales linearly to number of features**. That means **SVM will treat well when the number of features are many**."

**Complexity**:
- Features: $O(p)$ - linear scaling
- Samples: $O(n^2)$ to $O(n^3)$ - quadratic to cubic

"**SVM is usually good for small to medium-sized data** with a **large number of features**."

### Handles Sparse Features

"**SVM also works well on sparse features**. That means **even though the feature value has a lot of zeros**, **SVM will be able to handle gracefully**."

### Comparison with Random Forest

"**Random forest** can be **very slow if the feature values are all real values and it's dense**. **Whereas SVM**, it can **handle more or less a similarly two categorical variables**."

**Why**: RF must sort all real values for each split. SVM treats all feature types uniformly.

---

## 3. Hinge Loss Function

### C Parameter

"**Small c means that the model can tolerate small error**. That means **high-variance and low bias model**. **Whereas the largest C the model can tolerate more errors** and therefore a **higher bias, lower variance**."

### Hinge Loss

"**This loss function is called the hinge loss**."

$$L(z) = \max(0, 1 - z) \text{ where } z = y \cdot f(x)$$

"**This looks like a hinge**. Therefore, the **name is the hinge loss**."

**Complete Objective**:
$$\min_{w,b} \frac{\lambda}{2}||w||^2 + \frac{1}{n}\sum_{i=1}^{n}\max(0, 1 - y_i(w^Tx_i + b))$$

**Relationship**: $C \propto \frac{1}{\lambda}$

---

## 4. sklearn Library Usage

### LinearSVC

"**Linear SVC** uses a **lip linear algorithm** which **does not use kernel**. This **linear SVC function works better when the data is linearly separable**."

```python
from sklearn.svm import LinearSVC
model = LinearSVC(penalty='l2', loss='squared_hinge', C=1.0)
```

**Characteristics**:
- No kernels, linear only
- Time: $O(n \times p)$
- Squared hinge loss: $\max(0, 1-z)^2$
- Multiclass: One-vs-Rest

### SVC

"**SVC** is using **libSVM**. It **uses a kernel**. By **default it's using radial basis function**."

```python
from sklearn.svm import SVC
model = SVC(kernel='rbf', C=1.0, gamma='scale')
```

### Important: C Parameter in sklearn

"**Important thing to mention sklearn has a C parameter**. However, the **definition of this C hyperparameter is inverse to the textbook's notation**."

**sklearn**: $C = \frac{1}{\lambda}$
- Large C → Less regularization → Narrow margin
- Small C → More regularization → Wide margin

**Textbook**: C = error budget
- Large C → More violations allowed

### SVR

"**SVM model can also do the regression**, and that **function is called SVR**."

"In **SVR**, we **reverse that condition** that we **want them to be as close to as possible with this decision boundary**."

Points **inside** ε-tube have zero loss (opposite of classification).

---

## 5. Performance Comparison Study

"We **prepared five data** that are **similar or different to each other**."

### Datasets

**Dataset 1**: n=5000, p=100, **sparse categorical**, baseline=60%
**Dataset 2**: n=5000, p=20, **dense real-valued**, baseline=90%
**Dataset 3**: n=3000, p=150, **sparse categorical**, baseline=70%
**Dataset 4**: n=6000, p=300, **dense real-valued**, baseline=70%
**Dataset 5**: n=375 (train), p=1800, **93% categorical**, baseline=70%

### Results Summary

"**All of this accuracy values are from a five-fold cross-validation**."

**Key Findings**:

| Dataset | Best Model | SVM Rank | Notes |
|---------|------------|----------|-------|
| Data 1 | Logistic Reg (85%) | 3rd (80%) | Simple data |
| Data 2 | GBM (96%) | 3rd (94%) | Easy problem |
| Data 3 | SVM (85%) | 1st | Many features |
| Data 4 | RF (86%) | 3rd (84%) | Very many features |
| Data 5 | SVM (78%) | 1st | **p >> n** |

"**Tree ensembles and the SVM model worked much better**" especially for high-dimensional data.

---

## 6. Training Time Analysis

"**How about the training time**? If they **gave a similar performance**? However, **one model gives a much shorter training time**..."

### Results

"**Ensemble methods** took **over 100 millisecond to a little less than hundreds a second**. **SVM** takes **not only shorter than tree ensemble method usually**. It also **doesn't care whether the feature values are real valued or the categorical**."

**Typical Times**:
- Data 1: RF=5s, GBM=8s, SVM=0.5s
- Data 2 (dense): RF=150s, GBM=200s, SVM=2s
- Data 5: RF=20s, GBM=25s, SVM=3s

### Why Time Varies

"**Data 1 had sparse features**, whereas the **data 2 had a dense features**. **Data 1 had the most severe categorical features**, whereas the **data 2 had all real valued features**."

**Decision Tree Splitting**:
- Real-valued: Must sort all values, test all splits → $O(n \log n)$
- Categorical: Simple splits → Faster

"**Any tree-based methods can be slower when there is a lot of real valued features**."

"**SVM doesn't suffer from that problem**." Treats all feature types uniformly.

---

## 7. Model Selection Guidelines

### When to Use SVM

"We **recommend to use SVM model** if it has **large number of features** and **small to medium-sized data**, which means a **few hundreds to thousands**. And also, if the **data features are mostly real valued**, it's **likely that the SVM performance comparable to random forests and GBM**, and it **takes a much lesser training time**."

**SVM Sweet Spot**:
- Features: Large (p > 50)
- Samples: Small to medium (100 < n < 10,000)
- Special: p >> n (SVM excels)

### Start Simple

"**Always try simple model first** and **see how it goes**."

**Workflow**: Baseline (Logistic/Tree) → Complex if needed → Compare

### Occam's Razor

"**Occam's razor principle**, which tells that if the **model performances are similar**, the **simpler model is always better**."

### Context Matters

"**Choice of model can be depending on your goal** and also the **computation resource and the data size**."

**Competitions**: "**Try it fancy models at the expense of some training time**"
**Production**: "**Go with a simple model that takes less time**"

"**All these choices depend on the situation**."

---

## 8. Python Examples

### Complete Comparison Pipeline

```python
import numpy as np
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC

# Generate datasets
X1, y1 = make_classification(n_samples=5000, n_features=100, random_state=42)
X1[X1 < 0.5] = 0  # Make sparse

X2, y2 = make_classification(n_samples=5000, n_features=20, random_state=43)  # Dense

# Models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100),
    'SVM (RBF)': SVC(kernel='rbf'),
    'Linear SVC': LinearSVC(max_iter=1000)
}

for data_name, (X, y) in [('Sparse', (X1, y1)), ('Dense', (X2, y2))]:
    print(f"\n{data_name} Data:")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for name, model in models.items():
        start = time.time()
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - start
        
        acc = model.score(X_test_scaled, y_test)
        cv = cross_val_score(model, X_train_scaled, y_train, cv=5).mean()
        
        print(f"{name:20s}: Acc={acc:.3f}, CV={cv:.3f}, Time={train_time:.3f}s")
```

### Feature Scaling Impact

```python
# Without scaling
svm_unscaled = SVC(kernel='rbf')
svm_unscaled.fit(X_train, y_train)
acc_unscaled = svm_unscaled.score(X_test, y_test)

# With scaling
X_train_scaled = StandardScaler().fit_transform(X_train)
X_test_scaled = StandardScaler().fit_transform(X_test)
svm_scaled = SVC(kernel='rbf')
svm_scaled.fit(X_train_scaled, y_train)
acc_scaled = svm_scaled.score(X_test_scaled, y_test)

print(f"Unscaled: {acc_unscaled:.3f}, Scaled: {acc_scaled:.3f}")
# Scaling typically improves SVM significantly!
```

### SVR Example

```python
from sklearn.svm import SVR

X_reg = np.sort(5 * np.random.rand(100, 1), axis=0)
y_reg = np.sin(X_reg).ravel() + 0.1 * np.random.randn(100)

svr_rbf = SVR(kernel='rbf', C=100, gamma=0.5, epsilon=0.1)
svr_rbf.fit(X_reg, y_reg)

X_test = np.linspace(0, 5, 300)[:, np.newaxis]
y_pred = svr_rbf.predict(X_test)
```

---

## 9. Key Takeaways

**1. SVM Properties**:
- Requires feature scaling
- Time: $O(p)$ features, $O(n^2)$ to $O(n^3)$ samples
- Handles sparse features well
- Uniform treatment of feature types

**2. Performance**:
- Comparable to ensemble methods in accuracy
- Significantly faster training (5-100× in some cases)
- Excels when p >> n
- Best for small-to-medium n, large p

**3. sklearn C Parameter**: **Inverse of textbook**
- Large C → Less regularization → Narrow margin
- Small C → More regularization → Wide margin

**4. Training Time**:
- Tree ensembles slow on dense real-valued features
- SVM consistent across feature types
- Data 2 (dense): RF 150s vs SVM 2s

**5. Model Selection**:
- Start simple (Logistic Regression, Decision Tree)
- Use SVM for: large p, small-medium n, real-valued features
- Use ensembles for: complex patterns, large n, time available
- Context matters: competition vs production

**6. Occam's Razor**: Simpler model preferred if performance similar

**7. Trade-offs**: Accuracy vs Speed vs Interpretability

---

## Glossary

- **Hinge Loss**: $\max(0, 1-z)$, zero beyond margin
- **Squared Hinge**: $\max(0, 1-z)^2$, used by LinearSVC
- **liblinear**: Fast library for linear SVM, no kernels
- **libSVM**: Library for kernel SVM methods
- **ε-insensitive**: SVR loss, zero within ε-tube
- **One-vs-Rest**: Multiclass strategy, K binary classifiers
- **Sparse Features**: Many zero values
- **Dense Features**: Few/no zero values
- **Occam's Razor**: Prefer simpler model if equal performance
