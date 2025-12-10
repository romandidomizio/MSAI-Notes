# sklearn Usage, Decision Tree Hyperparameters, and Early Stopping
**CSCA5622 - Module 04**

---

## 📚 Overview

This document covers practical implementation of **Decision Trees in sklearn**, including model construction, visualization, **hyperparameters for preventing overfitting**, early stopping strategies, and **GridSearchCV** for hyperparameter tuning.

All concepts explained from the lecture transcript.

---

## 1. Quick Review: Decision Tree Splitting

From lecture:
> "As a quick review, here is how a **decision tree splitting works**. From the **root node**, it has samples and it's going to **pick a feature and it's threshold value** to **minimize the sum of the MSE** of the splitted node, like this and then you will **further split** and **pick another feature in threshold value** like this."

**Process:**
1. Start at root node with all samples
2. Find best (feature, threshold) pair that minimizes metric
3. Split into two child nodes
4. Recursively repeat for each child node

### 📊 Metrics by Task

From lecture:
> "We also talked about **different metrics for different tasks**. For the **regression trees**, we use **MSE, MAE or RSS** to split a node, and for **classification task**, the tree picks. The tree uses a **Gini and entropy or information gain** sometimes to split the node."

| Task | Metrics | Goal |
|------|---------|------|
| **Regression** | MSE, MAE, RSS | Minimize variance/error |
| **Classification** | Gini, Entropy, Information Gain | Minimize impurity/uncertainty |

---

## 2. Video Scope

From lecture:
> "In this video, we're going to talk about some **usage in sklearn**, how to **fit the models** and some **useful functions**, and we'll talk about **hyperparameters of the decision trees** that we need to **pick values** such that we **minimize overfitting**."

**Topics covered:**
1. sklearn implementation basics
2. Model fitting
3. Visualization functions
4. Hyperparameters for overfitting control
5. GridSearchCV for tuning

---

## 3. sklearn Implementation Basics

### 📦 Importing

From lecture:
> "We simply **import decision tree regressor in classifier** from **sklearn tree module**."

```python
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
```

### 🔧 Basic Usage: Classification Example

From lecture:
> "For example, if it was **classification task**, we can **construct a model** by just simply calling this **decision tree classifier**, and then **fit the data**, the **features and the labels**."

```python
from sklearn.tree import DecisionTreeClassifier

# Create model
clf = DecisionTreeClassifier()

# Fit model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)
```

### 📋 Available Options

From lecture:
> "Here are the **snapshots from the document** that shows that it has **many other options**. We'll talk about some of them."

**Key parameters** (will discuss in detail):
- `max_depth`: Maximum tree depth
- `min_samples_split`: Minimum samples to split
- `min_samples_leaf`: Minimum samples in leaf
- `max_features`: Number of features to consider
- `class_weight`: Class weighting ('balanced')
- `ccp_alpha`: Pruning parameter

---

## 4. Visualization with plot_tree

### 🎨 Basic Visualization

From lecture:
> "Another **useful function** that is also contained in that **sklearn tree module** is the **plot tree**. When we pass the **fitted object** to the **plot tree function**, it's going to return some **list of text objects** and then also the **visualization** of this."

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Fit model
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)

# Visualize
plt.figure(figsize=(20, 10))
plot_tree(clf, 
         feature_names=feature_names,
         class_names=class_names,
         filled=True,
         rounded=True,
         fontsize=10)
plt.title("Decision Tree Visualization")
plt.show()
```

**Returns:** List of text objects and visual plot

---

## 5. Advanced Visualization with Graphviz

### 🎨 Fancy Visualization

From lecture:
> "We can also use **export graphics function** from **sklearn tree** to make a **fancier visualization**. To do them, we're going to use **graphviz** and some other modules."

```python
from sklearn.tree import export_graphviz
import graphviz

# Export tree structure
dot_data = export_graphviz(
    clf,
    out_file=None,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    special_characters=True
)

# Create graph object
graph = graphviz.Source(dot_data)

# Display
graph
```

### 🎨 Color Interpretation

From lecture:
> "It will look like this. If you see **more red and more blue**, it means that the **node is more pure**, and if you see **white node**, that means it's **50/50 or mixed** there, so it's a **little bit fancier**, but **essentially, the same**."

**Color coding:**
- **Deep red/blue:** Pure nodes (one class dominates)
- **White/light colors:** Mixed nodes (50-50 or close)
- **Intensity:** Indicates purity level

---

## 6. Decision Tree Drawbacks

From lecture:
> "**Decision trees** while they are **easy and useful to understand**, they have some **drawbacks**, they are **very easy to overfit**, so we're going to talk about some **strategies to prevent overfitting**."

**Main drawback:** High tendency to overfit

### 🔧 Three Overfitting Prevention Strategies

From lecture:
> "First strategy is **stopping the tree to grow**. It's called **early stopping**, and second strategy is called **pruning**. We'll talk about that later, and another good strategy is **ensembling the trees**."

**Strategies:**
1. **Early Stopping:** Limit tree growth with hyperparameters (this lecture)
2. **Pruning:** Trim back after growing (next lecture)
3. **Ensembling:** Combine multiple trees (Random Forest, Boosting)

---

## 7. Early Stopping Hyperparameters

From lecture:
> "How do we **stop the tree grow early**? We have **bunch of hyperparameters** listed here, and we can **pick some values** such that we can **stop the tree grow**."

### 📊 Key Hyperparameters

#### 1. max_depth

From lecture:
> "For example, **max depths** will **limit the depth of the tree** so that it can **stop growing** when it reaches **certain depths**."

**Purpose:** Directly limits tree complexity

**Example:**
```python
clf = DecisionTreeClassifier(max_depth=5)
```

**Effect:**
- Small value (3-5): Simple tree, less overfitting, may underfit
- Large value (10+): Complex tree, more overfitting
- None: Grows until pure leaves (maximum overfitting)

#### 2. min_samples_split

From lecture:
> "**Min samples split** will make the **node stop splitting** when it has **less number of samples** are right in that node."

**Purpose:** Prevents splitting on very small groups

**Example:**
```python
clf = DecisionTreeClassifier(min_samples_split=20)
```

**Effect:**
- Node won't split if it has < 20 samples
- Larger value → simpler tree → less overfitting

#### 3. min_samples_leaf

From lecture:
> "**Min samples leaf** also can **stop tree grow further** or **node split further** when it has a **certain number of samples** in the **leaf node**, so they are **similar**."

**Purpose:** Requires minimum samples in each leaf

**Example:**
```python
clf = DecisionTreeClassifier(min_samples_leaf=10)
```

**Effect:**
- Each leaf must have ≥ 10 samples
- Prevents tiny, overfitted leaves

**Difference from min_samples_split:**
- `min_samples_split`: Minimum to allow split
- `min_samples_leaf`: Minimum in resulting leaves

#### 4. min_weight_fraction_leaf

From lecture:
> "**Min weight fraction leaf** are also **similar**. It is a **continuous version of min sample split**, so **instead of number of samples**, I will look for the **weight fraction** of the node."

**Purpose:** Like min_samples_leaf but uses fraction instead of absolute count

**Example:**
```python
clf = DecisionTreeClassifier(min_weight_fraction_leaf=0.01)
```

**Effect:** Each leaf must contain ≥ 1% of total samples

#### 5. min_impurity_decrease

From lecture:
> "**Min impurity decrease** also **stop splitting at that node** if the **impurity decrease** from that node is **negligible or less than certain number**."

**Purpose:** Only split if impurity reduction is significant

**Example:**
```python
clf = DecisionTreeClassifier(min_impurity_decrease=0.01)
```

**Effect:** Node won't split unless impurity decreases by ≥ 0.01

#### 6. max_features

From lecture:
> "**Max features** also can **help with the overfitting** because they can make them more the **less flexible** by **looking at the less number of features** when we have so many features."

**Purpose:** Limits features considered at each split

**Options:**
- `int`: Exact number (e.g., 5)
- `float`: Fraction (e.g., 0.5 = half)
- `'sqrt'`: √(n_features)
- `'log2'`: log₂(n_features)
- `None`: All features

**Example:**
```python
clf = DecisionTreeClassifier(max_features='sqrt')
```

**Effect:** Adds randomness, reduces overfitting (especially useful in ensembles)

### 📋 Additional Parameters

From lecture:
> "There are **more design parameters** in the **sklearn implementation** of distant trees, which you can also **look at the documentation**, but we'll **focus on just a few**."

---

## 8. Most Important Hyperparameters

### 🎯 Priority Ranking

From lecture:
> "The **most direct way** to prevent overfitting in decision trees or **max depths**, so by just **limiting the depths**, we can **directly make the tree not grow**."

**1. max_depth (Most Important)**

From lecture:
> "The **minimum samples leaf** is also **very useful**. The **smaller the number of the sample** of the leaf, that means the model is a **more flexible**. If you want to make the model **less flexible**, **less overfitting** then **increase this number**."

**2. min_samples_leaf (Very Useful)**

From lecture:
> "Another **good one to try** is the **impurity decrease**, however, you will have to **know some values**, so you will have to give some **trial and error**."

**3. min_impurity_decrease (Good, but requires tuning)**

---

## 9. Impurity Decrease Calculation

### 📐 Formula

From lecture:
> "**Impurity decrease** is calculated as this one. When there are **N samples** in the **parent node** and it **splits to an l and an r**, the **impurity decrease** or **information gain** is given by the **impurity of the original node** minus the **weighted sum** of the **impurity of the children node**."

\[
\Delta Impurity = I_{parent} - \left(\frac{n_L}{n} \times I_L + \frac{n_R}{n} \times I_R\right)
\]

Where:
- I_parent = impurity of parent node
- n_L, n_R = number of samples in left/right children
- n = total samples in parent
- I_L, I_R = impurity of left/right children

From lecture:
> "So the **weights will be the fraction of the sample** inverse times the **impurity of the left box**, and the **weight of the right box**, times **impurity of the right box**, so that's the **impurity decrease**. You will **pick some value of threshold** and see what happens."

### 📊 Example

**Parent node:**
- n = 100 samples
- Gini_parent = 0.5

**After split:**
- Left: n_L = 60, Gini_L = 0.2
- Right: n_R = 40, Gini_R = 0.3

**Impurity decrease:**
\[
\Delta = 0.5 - (0.6 \times 0.2 + 0.4 \times 0.3)
\]
\[
\Delta = 0.5 - (0.12 + 0.12) = 0.5 - 0.24 = 0.26
\]

**Decision:** If min_impurity_decrease = 0.1, split is allowed (0.26 > 0.1)

---

## 10. Other Useful Options

### 🔧 max_features

From lecture:
> "Other **useful options** that you can use when you build a model is a **max features**. It's going to **limit the feature number** and usually, **square root or log options are popular**, **square root is more popular**, by the way."

**Popular choices:**
- `'sqrt'`: √(n_features) — **Most popular**
- `'log2'`: log₂(n_features)

**Example:**
```python
clf = DecisionTreeClassifier(max_features='sqrt')
```

### 🔧 class_weight

From lecture:
> "**Class weights**, by **default it's known** [None], but if you use a **balance**, they usually give the **better performance**, **especially true** when you have **imbalanced labels**."

**Purpose:** Handle imbalanced datasets

**Options:**
- `None`: All classes equal weight (default)
- `'balanced'`: Automatically adjust weights inversely proportional to class frequencies

**Example:**
```python
# For imbalanced data
clf = DecisionTreeClassifier(class_weight='balanced')
```

**Formula for 'balanced':**
\[
weight_k = \frac{n_{samples}}{n_{classes} \times n_{samples\_in\_class\_k}}
\]

### 🔧 ccp_alpha

From lecture:
> "**CCP Alpha** is used when you use the **minimal complexity pruning**, so we'll **talk about this more in detail** in the **pruning video**."

**Purpose:** Controls cost-complexity pruning

**Example:**
```python
clf = DecisionTreeClassifier(ccp_alpha=0.01)
```

**Note:** Covered in detail in next lecture on pruning

---

## 11. Hyperparameter Selection

### 🤔 How to Choose Values?

From lecture:
> "How do we choose is a **hyperparameter values**. We might have some **heuristic values** or just **try a few values**, however, we can also do **pragmatic approach** like **grid search**."

**Three approaches:**
1. **Heuristic values:** Use common defaults (max_depth=5, min_samples_leaf=10)
2. **Manual experimentation:** Try a few values and compare
3. **Grid search:** Systematic search (best approach)

---

## 12. GridSearchCV

### 🔍 What Is It?

From lecture:
> "Unfortunately [Fortunately], **sklearn library** also have a **very convenient tool** called **GridSearchCV**. It does **grid search** as well as **cross-validation** so that it makes sure to **not just a one peak value** that was out of luck, but it does **cross validation**."

**Purpose:** 
- Systematically try combinations of hyperparameters
- Use cross-validation to get reliable performance estimates

### 🔄 How It Works

From lecture:
> "Which will **split the data by default, five chunks** and it's going to **fit the model** and then get the **accuracy from this chunk and this chunk** and then it will **average the result**, and it will give the **results which model hyperparameter get the best result**."

**Process:**
1. Split data into K folds (default K=5)
2. For each hyperparameter combination:
   - Train on K-1 folds
   - Validate on remaining fold
   - Repeat K times
   - Average the K results
3. Select hyperparameters with best average score

### 💻 Complete Example

From lecture:
> "From **model selection module**, we can call the **GridSearchCV** and this is individual **decision tree classifier**, I just happen to call **RF**, but you can call whatever, and then this **parameters are dictionary** that shows which **hyperparameters and which values** you want to change to, so I gave some **different options**, and then I **put these two objects** into our **GridSearchCV**."

```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create base model
clf = DecisionTreeClassifier(random_state=42)

# Define parameter grid
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': ['sqrt', 'log2', None],
    'criterion': ['gini', 'entropy']
}

# Create GridSearchCV object
grid_search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=5,                    # 5-fold cross-validation
    scoring='accuracy',       # Metric to optimize
    n_jobs=-1,               # Use all CPU cores
    verbose=1                # Print progress
)

# Fit (this tries all combinations)
grid_search.fit(X_train, y_train)

# Get results
print("Best parameters:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)
print("Best estimator:", grid_search.best_estimator_)

# Use best model for prediction
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test)

# Evaluate on test set
from sklearn.metrics import accuracy_score
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {test_accuracy:.4f}")
```

### 📊 Accessing Results

From lecture:
> "After **fitting the grid search object** with that data, we can call the **result by the best estimator**, it will return **what was the best estimator** and gives the **hyperparameter values** here, and that **best score** will give **what was the accuracy value** for the classification or when we use these hyperparameters."

**Key attributes:**
```python
# Best hyperparameter combination
best_params = grid_search.best_params_

# Best cross-validation score
best_score = grid_search.best_score_

# Best fitted model
best_model = grid_search.best_estimator_

# All results
results_df = pd.DataFrame(grid_search.cv_results_)
```

### 📋 Example Output

```python
Best parameters: {
    'criterion': 'gini',
    'max_depth': 5,
    'max_features': 'sqrt',
    'min_samples_leaf': 2,
    'min_samples_split': 10
}

Best CV score: 0.9524

Best estimator: DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    max_features='sqrt',
    min_samples_leaf=2,
    min_samples_split=10,
    random_state=42
)
```

---

## 13. Complete Working Example

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Load data
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train baseline model (no tuning)
baseline_clf = DecisionTreeClassifier(random_state=42)
baseline_clf.fit(X_train, y_train)
baseline_score = baseline_clf.score(X_test, y_test)
print(f"Baseline Test Accuracy: {baseline_score:.4f}")
print(f"Baseline Tree Depth: {baseline_clf.get_depth()}")
print(f"Baseline Leaf Count: {baseline_clf.get_n_leaves()}")

# 4. GridSearchCV for hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': [None, 'balanced']
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# 5. Get best model
best_clf = grid_search.best_estimator_
print("\n" + "="*50)
print("BEST MODEL FROM GRIDSEARCH")
print("="*50)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
print(f"Best Tree Depth: {best_clf.get_depth()}")
print(f"Best Leaf Count: {best_clf.get_n_leaves()}")

# 6. Evaluate on test set
y_pred = best_clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=data.target_names))

# 7. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=data.target_names)
disp.plot()
plt.title("Confusion Matrix - Tuned Model")
plt.show()

# 8. Feature Importances
importances = best_clf.feature_importances_
indices = np.argsort(importances)[::-1][:10]  # Top 10

plt.figure(figsize=(12, 6))
plt.title("Top 10 Feature Importances")
plt.barh(range(10), importances[indices])
plt.yticks(range(10), [data.feature_names[i] for i in indices])
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# 9. Visualize best tree (limited depth for readability)
plt.figure(figsize=(20, 10))
plot_tree(best_clf,
         max_depth=3,  # Only show first 3 levels
         feature_names=data.feature_names,
         class_names=data.target_names,
         filled=True,
         rounded=True,
         fontsize=10)
plt.title("Decision Tree (First 3 Levels)")
plt.show()

# 10. Compare baseline vs tuned
print("\n" + "="*50)
print("COMPARISON")
print("="*50)
print(f"Baseline Accuracy: {baseline_score:.4f}")
print(f"Tuned Accuracy:    {test_accuracy:.4f}")
print(f"Improvement:       {(test_accuracy - baseline_score):.4f}")
```

---

## 14. Summary

From lecture:
> "These are some **handy tools**, so we show how to **use sklearn library** for **constructing decision trees** and how to use a **grid search** to find the **hyperparameter values**."

### 🎯 Key Concepts

**1. sklearn Implementation:**
```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
```

**2. Visualization:**
- `plot_tree()`: Basic visualization
- `export_graphviz()`: Advanced with color coding

**3. Overfitting Prevention:**
- Early stopping (hyperparameters)
- Pruning (next lecture)
- Ensembling (Random Forest, Boosting)

**4. Key Hyperparameters (Priority Order):**
1. `max_depth`: Most direct control
2. `min_samples_leaf`: Very useful
3. `min_impurity_decrease`: Good but requires tuning
4. `max_features`: Helpful, especially in ensembles
5. `class_weight='balanced'`: Critical for imbalanced data

**5. GridSearchCV:**
- Systematic hyperparameter search
- Built-in cross-validation
- Returns best model automatically

### 📋 Best Practices

1. **Always split data** into train/test before tuning
2. **Use GridSearchCV** for systematic tuning
3. **Start with small parameter grids** to test quickly
4. **For imbalanced data**, always use `class_weight='balanced'`
5. **Limit max_depth** (3-10) to prevent overfitting
6. **Visualize trees** to understand decision logic
7. **Check feature importances** to understand model
8. **Compare to baseline** to verify improvement

### 🔄 Next Steps

From lecture:
> "In the next video, we're going to talk about **pruning the decision trees** as a **part of strategies** of **preventing overfitting**."

**Coming next:** Cost-complexity pruning with `ccp_alpha`

---

**End of Lecture Notes - Module 04, Document 4**
