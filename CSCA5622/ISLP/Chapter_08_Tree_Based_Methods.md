# Chapter 8 - Tree-Based Methods

## ISLP (Introduction to Statistical Learning with Python)

---

## Section 8.1 – The Basics of Decision Trees

Decision trees are methods that partition (segment) the **predictor space** into distinct regions, and then make predictions (in regression) using simple models (constant values) within each region. They are intuitive, interpretable, and nonparametric. ([Bookdown][1])

Key idea:

* Split the feature space (all possible values of predictors (X_1, X_2, \dots)) into regions (R_1, R_2, \dots, R_J).
* For regression, in each region (R_j), predict (\hat{y}) as the average (mean) of the training response values in that region.
* The partitioning is typically done by **binary splits** (one predictor at a time) in a top-down “greedy” fashion, called *recursive binary splitting*. ([math.hkust.edu.hk][2])

Advantages:

* Easy to interpret and visualize (the splits can be drawn as a tree)
* No need to specify functional form (non-linearities and interactions are captured automatically)
* Works with both categorical and numeric predictors

Disadvantages:

* Trees tend to suffer from **overfitting** (too many splits) unless pruned
* Unstable: small changes in data can lead to very different trees
* Often less predictive accuracy than more advanced methods (e.g. ensemble methods) ([Bookdown][1])

---

### 8.1.1 Regression Trees

This subsection discusses how to build decision trees when the response (Y) is quantitative (continuous).

#### Partitioning & Prediction

We aim to partition the predictor space into (J) disjoint regions (R_1, R_2, \dots, R_J), and in each region, make a constant prediction:

[
\hat{y}(x) = \sum_{j=1}^J \hat{c}*j , \mathbf{1}*{{x \in R_j}}
]

where (\hat{c}_j) is typically the **mean** of the (y_i) values for training observations that fall into (R_j). ([Amit Rajan][3])

Thus, the overall objective is to choose the regions and constants such that the **Residual Sum of Squares (RSS)** is minimized:

[
\min_{R_1, \dots, R_J, , {c_j}} ; \sum_{j=1}^J \sum_{i \colon x_i \in R_j} (y_i - c_j)^2
]

Given a fixed partition, the optimal (c_j) is:

[
\hat{c}*j = \frac{1}{|R_j|} \sum*{i: x_i \in R_j} y_i \quad (\text{the mean in that region})
]

So we need to choose the partitioning (i.e. splits) to minimize RSS.

#### Recursive Binary Splitting (Greedy Algorithm)

We typically don’t search over all possible partitions (that is combinatorially huge). Instead, we use a **top-down greedy** strategy:

1. Start with the entire predictor space as one region.
2. At each step, consider splitting a current region into two regions by choosing a predictor (X_j) and a split point (s) (i.e. splitting on (X_j < s) vs (X_j \ge s)).
3. Choose the split (choice of (j) and (s)) that **most reduces RSS** (i.e. the best improvement in fit).
4. Continue splitting until some stopping criterion (e.g. minimum node size, maximum depth). ([math.hkust.edu.hk][2])

This is *greedy* because at each step, the split chosen is the best at that moment, without considering its effect on future splits.

At each candidate split (region (R), feature (j), split point (s)), we evaluate:

[
\min_{c_1, c_2} \Bigl{ \sum_{i: x_i \in R,, x_{ij} < s} (y_i - c_1)^2 + \sum_{i: x_i \in R,, x_{ij} \ge s} (y_i - c_2)^2 \Bigr}
]

We pick the ((j, s)) combination giving the largest drop in RSS.

#### Example from Hitters / Baseball Salaries

In ISLR, the authors illustrate building a regression tree to predict the (log) salary of baseball players using predictors like **Years** and **Hits**. ([math.hkust.edu.hk][2])

* First split might be: “Years < 4.5?”
* Then maybe additionally split by “Hits < 117.5?” in one branch, etc.
* At the leaves (terminal nodes), you see the predicted constant value (mean log‑salary in that node). ([math.hkust.edu.hk][2])

Figure 8.1 in the ISLR text shows a tree with splits on Years and Hits, and predicted means in leaves. ([math.hkust.edu.hk][2])

Predictions: given new (x) (Years, Hits), you traverse the tree (making the splits) until you land in a leaf region (R_j), and output (\hat{c}_j).

#### Overfitting & Complexity Control

If allowed to split too much, you get many small regions (leaves) that overfit the training data (low RSS but poor generalization). Hence **pruning** (trimming back) is necessary (this is in a later section).

Also, you may limit tree growth by:

* Minimum number of observations required to split a node
* Maximum depth of the tree
* Minimum number of observations per terminal node

These act as **pre‑pruning** controls, but the book typically prefers to grow a large tree and prune it back.

---

#### Tree Pruning / Cost-Complexity Pruning

**(ISLR Section 8.1.1, pages 307-311)**

##### Motivation

* As we build a tree via recursive binary splitting, we can produce a very large tree, \( T_0 \), with many terminal nodes (leaves).
* That tree may **overfit** the training data: i.e. capture noise, have low training error but poor generalization.
* Therefore, instead of stopping early (pre‐pruning), a preferred technique is to **grow a large tree** first, then **prune** it back (post‑pruning) to find an optimal subtree that balances fit and complexity.
* Pruning ensures simpler models, lower variance, and better generalization.

##### The Problem with Simple Stopping Rules

* We could try stopping when RSS reduction falls below a threshold.
* However, a seemingly poor split early on might lead to very good splits later.
* Thus, we prefer to grow a full tree and then prune it back using a principled criterion.

##### Cost Complexity Pruning (Weakest Link Pruning)

The standard approach used in CART (and in ISLR) is **cost complexity pruning** (also called "α‑pruning").

We define, for a candidate subtree \( T \subseteq T_0 \), a **complexity penalty**:

\[
R_{\alpha}(T) = \underbrace{\sum_{m=1}^{|T|} \sum_{i: x_i \in R_m} (y_i - \hat{y}_{R_m})^2}_{\text{RSS of }T} \; + \; \alpha \, |T|
\]

**Where:**
* \( |T| \) = number of terminal nodes (leaves) in subtree \(T\)
* \(\alpha \ge 0\) is a **tuning parameter** (complexity parameter)
* The first term is the usual RSS (residual sum of squares) of the tree \( T \)
* The second term penalizes complexity (number of leaves)

**Goal:** For each \(\alpha\), we seek the subtree \( T(\alpha) \subseteq T_0 \) that minimizes \( R_{\alpha}(T) \).

##### Properties of α-Pruning

* **If \(\alpha = 0\):** No penalty on complexity → the best tree is \(T_0\) (the full grown tree) because only RSS matters
* **As \(\alpha\) increases:** The penalty on many leaves becomes more important, pushing the optimum subtree to be smaller (fewer leaves)
* There is a sequence of \(\alpha\) values (often finitely many) for which the minimizing tree changes
* This gives a **nested sequence** of subtrees as \(\alpha\) increases, from large to small
* The path of candidate trees forms a regularization path (analogous to lasso/ridge in linear regression)

##### Algorithm: Cost-Complexity Pruning with Cross-Validation

**Step-by-step process:**

1. **Grow a full tree** (\(T_0\)), possibly until a minimal node size or until no further splits reduce RSS significantly
2. **Obtain pruning sequence:** For a set of candidate \(\alpha\) values, compute (or approximate) \( T(\alpha) \) that minimizes \( R_{\alpha}(T) \)
3. **Cross-validation:** Use K-fold cross-validation (or a validation set) to estimate test MSE for each \( T(\alpha) \)
4. **Select optimal α:** Choose \(\alpha\) (hence \( T(\alpha) \)) that yields lowest cross-validated error
5. **Final model:** Prune \(T_0\) down to that optimal subtree and use it as the final model

This is effectively **regularization for trees**: balancing fit (low RSS) against complexity (many leaves).

##### Comparison to Linear Model Regularization

| Aspect | Trees | Linear Models |
|--------|-------|---------------|
| **Penalty term** | \(\alpha \|T\|\) (# of leaves) | \(\lambda \sum \beta_j^2\) (coefficient size) |
| **What's penalized** | Model complexity (splits) | Coefficient magnitude |
| **Tuning parameter** | \(\alpha\) | \(\lambda\) |
| **Selection method** | Cross-validation | Cross-validation |
| **Effect** | Fewer leaves, simpler tree | Shrink coefficients toward zero |

##### Implementation in scikit-learn

Modern libraries support cost complexity pruning directly. In **scikit-learn**, the `DecisionTreeRegressor` class supports a parameter `ccp_alpha` for *minimal cost complexity pruning*.

**Key sklearn features:**
* `ccp_alpha` parameter: Controls the complexity penalty
* `cost_complexity_pruning_path()` method: Returns effective α values and corresponding impurities
* Higher `ccp_alpha` → more aggressive pruning → simpler tree

**Reference:** [sklearn documentation on minimal cost-complexity pruning](https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning)

##### Complete Python Example: Cost-Complexity Pruning

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import fetch_california_housing

# Load data
data = fetch_california_housing()
X, y = data.data, data.target

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 1: Grow a full tree with minimal constraints
tree_full = DecisionTreeRegressor(random_state=42, min_samples_leaf=1)
tree_full.fit(X_train, y_train)

# Step 2: Get cost complexity pruning path
# This returns effective alphas and corresponding impurities
ccp_path = tree_full.cost_complexity_pruning_path(X_train, y_train)
alphas = ccp_path.ccp_alphas  # array of possible alpha values
impurities = ccp_path.impurities  # total leaf impurities for each alpha

print(f"Number of candidate alpha values: {len(alphas)}")
print(f"Alpha range: [{alphas.min():.6f}, {alphas.max():.6f}]")

# Step 3: For each alpha, fit tree and compute cross-validated error
train_scores = []
cv_scores = []
test_scores = []

for alpha in alphas:
    # Create tree with this alpha
    tree = DecisionTreeRegressor(random_state=42, ccp_alpha=alpha)
    
    # Fit on training data
    tree.fit(X_train, y_train)
    
    # Training score (MSE)
    train_pred = tree.predict(X_train)
    train_mse = np.mean((train_pred - y_train)**2)
    train_scores.append(train_mse)
    
    # Cross-validation score (5-fold)
    cv_score = cross_val_score(
        tree, X_train, y_train, 
        cv=5, 
        scoring='neg_mean_squared_error'
    )
    cv_scores.append(-np.mean(cv_score))  # Convert to positive MSE
    
    # Test score
    test_pred = tree.predict(X_test)
    test_mse = np.mean((test_pred - y_test)**2)
    test_scores.append(test_mse)

# Step 4: Find alpha with lowest CV error
best_idx = np.argmin(cv_scores)
best_alpha = alphas[best_idx]
best_cv_score = cv_scores[best_idx]

print(f"\nBest alpha: {best_alpha:.6f}")
print(f"Best CV MSE: {best_cv_score:.4f}")

# Step 5: Refit with best alpha on full training set
final_tree = DecisionTreeRegressor(random_state=42, ccp_alpha=best_alpha)
final_tree.fit(X_train, y_train)

# Evaluate on test set
y_pred = final_tree.predict(X_test)
test_mse = np.mean((y_pred - y_test)**2)
print(f"Final Test MSE: {test_mse:.4f}")
print(f"Number of leaves in final tree: {final_tree.get_n_leaves()}")
print(f"Depth of final tree: {final_tree.get_depth()}")

# Visualization: MSE vs Alpha
plt.figure(figsize=(12, 5))

# Plot 1: MSE vs alpha
plt.subplot(1, 2, 1)
plt.plot(alphas, train_scores, label='Training MSE', marker='o', alpha=0.7)
plt.plot(alphas, cv_scores, label='CV MSE', marker='s', alpha=0.7)
plt.plot(alphas, test_scores, label='Test MSE', marker='^', alpha=0.7)
plt.axvline(best_alpha, color='red', linestyle='--', label=f'Best α={best_alpha:.4f}')
plt.xlabel('Alpha (Complexity Parameter)')
plt.ylabel('Mean Squared Error')
plt.title('Model Performance vs Complexity Parameter')
plt.legend()
plt.xscale('log')
plt.grid(True, alpha=0.3)

# Plot 2: Number of leaves vs alpha
n_leaves = [DecisionTreeRegressor(random_state=42, ccp_alpha=a)
            .fit(X_train, y_train).get_n_leaves() 
            for a in alphas]

plt.subplot(1, 2, 2)
plt.plot(alphas, n_leaves, marker='o')
plt.axvline(best_alpha, color='red', linestyle='--', label=f'Best α={best_alpha:.4f}')
plt.xlabel('Alpha (Complexity Parameter)')
plt.ylabel('Number of Leaves')
plt.title('Tree Complexity vs Alpha')
plt.legend()
plt.xscale('log')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

##### Interpreting the Results

**What to observe:**

1. **Training MSE increases** with α (as tree becomes simpler, fits training data worse)
2. **CV/Test MSE decreases then increases** (U-shaped curve)
   - Initially: Pruning removes overfitting → better generalization
   - Eventually: Too much pruning → underfitting
3. **Optimal α** is at minimum of CV curve
4. **Number of leaves** decreases monotonically with α

**Typical pattern:**
```
α = 0.0:      Full tree (e.g., 500 leaves) - overfit
α = 0.001:    Large tree (e.g., 200 leaves) - still complex
α = 0.01:     Medium tree (e.g., 50 leaves) - balanced ← often optimal
α = 0.1:      Small tree (e.g., 10 leaves) - underfit
α = 1.0:      Tiny tree (e.g., 2 leaves) - severe underfit
```

##### Key Takeaways

1. **Pruning is essential:** Even if you grow the "full" tree, you likely need to cut it back to avoid overfitting
2. **Cost complexity pruning** introduces a penalty on tree size (number of leaves), yielding a nested sequence of subtrees as a function of α
3. **Use cross-validation** to choose the best α (hence the best subtree)
4. **Libraries make it practical:** R's `tree` package (`prune.tree`, `cv.tree`) and scikit-learn's `ccp_alpha` parameter handle the mechanics
5. **Regularization analogy:** Pruning parallels regularization in linear models—like ridge/lasso penalize large coefficients, pruning penalizes many leaves
6. **Balance interpretability and accuracy:** Pruned trees are simpler to interpret while maintaining good predictive performance

##### Alternative: Pre-Pruning vs Post-Pruning

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **Pre-pruning** | Stop splitting early (max_depth, min_samples_leaf) | Fast, simple | May stop too early, miss good later splits |
| **Post-pruning** | Grow full tree, then prune back with α | More principled, better results | More computationally expensive |

**ISLR recommendation:** Post-pruning (cost-complexity) is generally preferred for better model selection.

---

#### Advantages & Shortcomings

* **Advantages**: interpretability, handling interactions automatically, nonlinearity, no need for variable transformations.
* **Shortcomings**: high variance (unstable), tendency to overfit, less smooth predictions (step function), and often less predictive accuracy compared to ensemble methods (bagging, random forest, boosting). ([Bookdown][1])

### Example: Python Code

```python
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data
from sklearn.datasets import load_boston  # Deprecated, use housing datasets
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['MEDV'] = boston.target

# Define features and response
X = df[['RM', 'LSTAT']]  # Number of rooms and % lower status
y = df['MEDV']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit regression tree
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X_train, y_train)

# Predict
y_pred = tree.predict(X_test)

# Plot predicted vs actual
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Decision Tree Regression")
plt.show()
```

---

## Brief Notes on K-Nearest Neighbors (KNN)

Although KNN is a distinct method from trees, it is often discussed in contrast.

### What is KNN?

* KNN is a **nonparametric** method for classification or regression.
* For regression: to predict at a new point (x), find the (k) nearest training points (based on some distance metric, e.g. Euclidean distance), and take the **average** of their (y) values as the prediction.
* For classification: the predicted class is the **mode** (most frequent class) among the (k) neighbors. ([Wikipedia][4])

### Pros & Cons & Behavior

* **Pros**:

  * Very simple, easy to implement.
  * No training phase (just storing data); all work is at prediction time.
  * Flexible (adapts to local data structure).

* **Cons**:

  * Choice of (k) is crucial: small (k) → low bias but high variance; large (k) → smoother, but may underfit.
  * Sensitive to the scale of predictors (features should be normalized).
  * Computationally expensive at prediction time (must compute distances to all training points).
  * Doesn’t intrinsically handle irrelevant features well (noise dims degrade performance).

### Example (Regression KNN)

Suppose you have points ((x_i, y_i)). Given a new (x), find the 3 nearest (x_i) (by absolute or Euclidean distance), then average their (y_i).

E.g.:

Training points:
(x = [1,2,3,4,5]), (y = [2,2.5,3,3.5,4])
If (k = 3) and you want prediction at (x = 3.2), the three closest are (x = 3, 2, 4). Their (y) values are ([3,2.5,3.5]). The prediction is average = ((3 + 2.5 + 3.5)/3 = 3.0).

As (k) increases, the estimate becomes smoother but less flexible.

### Python Example for KNN Regression

```python
from sklearn.neighbors import KNeighborsRegressor

# Use same X_train, X_test, y_train, y_test
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict
y_knn_pred = knn.predict(X_test)

# Plot
plt.scatter(y_test, y_knn_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("KNN Regression (k=3)")
plt.show()
```

---

### Comparison: Trees vs. KNN

| Characteristic       | Trees                    | KNN                   |
| -------------------- | ------------------------ | --------------------- |
| Model Type           | Partition-based          | Instance-based (lazy) |
| Interpretability     | High (rules from splits) | Low                   |
| Handles Nonlinearity | Yes                      | Yes                   |
| Sensitive to Scaling | No                       | Yes                   |
| Prediction Type      | Step-wise constant       | Local averaging       |

### Relationship / Contrast: Trees vs KNN

* Decision trees partition space into rectangular regions and fit constant predictions per region; KNN uses local neighborhoods to smooth predictions.
* Trees produce stepwise piecewise constant functions; KNN gives more local averaging (so predictions vary smoother).
* KNN is sensitive to distance metrics and scaling; trees are invariant to monotonic transformations and scale of predictors.
* Trees are easier to interpret (you can read off rules); KNN is less interpretable.

---

## 8.1.2 – Classification Trees

When the response (Y) is **qualitative** (categorical), we use **classification trees** instead of regression trees. The structure is similar (partition the predictor space), but splits are chosen based on classification error and impurity measures rather than RSS. ([bookdown.org](https://bookdown.org/taylordunn/islr-tidy-1655226885741/tree-based-methods.html)) ([Bookdown][1])

### Prediction Rule

* After partitioning the predictor space into regions (R_1, R_2, \dots, R_J), in each region (R_m), we classify a new observation into the class with the **highest proportion** among training observations in that region.
* Let (\hat p_{mk}) be the proportion of training observations in (R_m) that belong to class (k). Then in region (R_m) the predicted class is:

[
\hat{y} = \arg\max_k , \hat p_{mk}
]

* We also often report the impurity or class probability distribution in each node (leaf), not just the predicted class.

### Splitting Criterion: Impurity Measures

We need a quantitative criterion to choose splits (which variable and cutoff). Because RSS is not appropriate for classification, we use **node impurity** measures. Common ones:

1. **Classification error rate**
2. **Gini impurity**
3. **Entropy (deviance / cross-entropy)**

#### 1. Classification Error Rate

In node (m), define:

[
E_m = 1 - \max_k \hat p_{mk}
]

This is the fraction of misclassified training observations in that node.

However, this measure is **not very sensitive** to improvements in purity during tree building, so it's rarely used as the splitting criterion (but might be used for pruning). ([Bookdown][1])

#### 2. Gini Impurity

Defined as:

[
G_m = \sum_{k=1}^K \hat p_{mk} , (1 - \hat p_{mk}) = \sum_{k=1}^K \hat p_{mk} - \sum_{k=1}^K \hat p_{mk}^2 = 1 - \sum_{k=1}^K \hat p_{mk}^2
]

* It measures the degree of class mixture (heterogeneity) in the node.
* (G_m) is small when one class dominates (i.e. node is “pure”).
* Used by the CART algorithm for classification tree splits. ([koalaverse.github.io][2])

#### 3. Entropy / Cross‑Entropy (Deviance)

Also known as information gain criterion:

[
D_m = - \sum_{k=1}^K \hat p_{mk} , \log (\hat p_{mk})
]

* Like Gini, (D_m) is small when the node is pure.
* Because of the log, it penalizes uncertainty more strongly.
* Splits are chosen to reduce the weighted sum of entropies in child nodes relative to parent. ([Bookdown][1])

These impurity criteria (Gini, entropy) are more sensitive to changes in purity than classification error, so they tend to produce better splits.

### Split Selection in Classification Trees

The process of recursive binary splitting is similar to regression trees, but with impurity:

* For each candidate split (variable (X_j), threshold (s)), compute left region (R_1) and right region (R_2).
* Compute impurity measure for each region (e.g. Gini or entropy), and compute the **weighted average impurity**:

[
\text{Impurity}_{\text{split}} = \frac{|R_1|}{|R|} , I(R_1) + \frac{|R_2|}{|R|} , I(R_2)
]

* Choose the split that **minimizes** this weighted impurity (i.e. maximizes impurity reduction).
* Continue splitting until stopping criteria (minimum node size, maximum depth, or until nodes are pure) or then prune back.

### Example & Code Sketch (Python with scikit-learn)

Here is how you might build and use a classification tree in Python, and inspect node class proportions etc.:

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Example: use the Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)  # classes 0, 1, 2

# Split into train / test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit classification tree
clf = DecisionTreeClassifier(criterion="gini", max_depth=3)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)

# Plot the tree
plt.figure(figsize=(10, 8))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()

# Inspect predicted class probabilities for a sample
print("Predicted probabilities for first test sample:", y_prob[0])
print("Predicted class:", iris.target_names[y_pred[0]])
```

**What to notice:**

* `criterion="gini"` (default) or you can use `criterion="entropy"` to use cross-entropy.
* `clf.predict_proba(X)` gives (\hat p_{mk}) for each class in the assigned leaf node.
* The tree plot shows splits, node class distributions, and majority classes.

### Additional Comments & Interpretation

* In classification trees, a terminal node is often labeled with the majority class, but you also care about the class proportions for uncertainty.
* As with regression trees, classification trees are prone to overfitting; hence pruning or regularization is needed (not covered here).
* The choice between Gini and entropy typically doesn’t make huge differences in practice, though entropy may be more expensive to compute. ([Quantdare][3])
* Because the splits are chosen greedily, the tree may not be globally optimal.

---

### sklearn DecisionTreeClassifier: Complete Reference

**Official Documentation:** [sklearn.tree.DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

#### Overview

`DecisionTreeClassifier` is a class in sklearn that builds a classification tree using the CART algorithm. It supports both binary and multiclass classification, handles missing values, and includes cost-complexity pruning.

**Import:**
```python
from sklearn.tree import DecisionTreeClassifier
```

---

#### Key Parameters

##### 1. criterion (str, default='gini')

**Purpose:** The function to measure the quality of a split.

**Options:**
- `'gini'`: Gini impurity (default)
- `'entropy'`: Shannon information gain (same as cross-entropy)
- `'log_loss'`: Same as entropy

**Formula reminders:**
```
Gini: G = 1 - Σ(p_k²)
Entropy: H = -Σ(p_k log(p_k))
```

**When to use which:**
- **Gini**: Faster to compute, slightly prefers larger partitions
- **Entropy**: More computationally expensive, slightly prefers balanced splits
- In practice: **differences are usually minor**

**Example:**
```python
clf_gini = DecisionTreeClassifier(criterion='gini')
clf_entropy = DecisionTreeClassifier(criterion='entropy')
```

##### 2. splitter (str, default='best')

**Purpose:** Strategy used to choose the split at each node.

**Options:**
- `'best'`: Choose the best split
- `'random'`: Choose the best random split

**Note:** Even with `splitter='best'`, features are randomly permuted at each split when `max_features < n_features`.

##### 3. max_depth (int, default=None)

**Purpose:** The maximum depth of the tree.

**Behavior:**
- `None`: Nodes are expanded until all leaves are pure or contain < `min_samples_split` samples
- `int`: Stop growing when depth reaches this value

**Use case:** Primary pre-pruning parameter to control overfitting

**Example:**
```python
# Limit tree depth to 5 levels
clf = DecisionTreeClassifier(max_depth=5)
```

##### 4. min_samples_split (int or float, default=2)

**Purpose:** Minimum number of samples required to split an internal node.

**Options:**
- `int`: Absolute number (e.g., 10 samples minimum)
- `float`: Fraction of total samples (e.g., 0.01 = 1%)

**Use case:** Prevents splits on very small groups (reduces overfitting)

**Example:**
```python
clf = DecisionTreeClassifier(min_samples_split=20)  # At least 20 samples
clf2 = DecisionTreeClassifier(min_samples_split=0.02)  # At least 2% of data
```

##### 5. min_samples_leaf (int or float, default=1)

**Purpose:** Minimum number of samples required to be at a leaf node.

**Options:**
- `int`: Absolute number
- `float`: Fraction of total samples

**Effect:** Smooths the model by requiring minimum leaf sizes

**Example:**
```python
clf = DecisionTreeClassifier(min_samples_leaf=10)  # Each leaf ≥ 10 samples
```

##### 6. max_features (int, float, str, or None, default=None)

**Purpose:** Number of features to consider when looking for the best split.

**Options:**
- `int`: Exact number of features
- `float`: Fraction of features (e.g., 0.5 = half)
- `'sqrt'`: √(n_features)
- `'log2'`: log₂(n_features)
- `None`: Use all features (default)

**Use case:** Adds randomness to reduce overfitting (especially useful in ensembles)

**Example:**
```python
clf = DecisionTreeClassifier(max_features='sqrt')  # Use sqrt(n_features)
```

##### 7. random_state (int, default=None)

**Purpose:** Controls the randomness of the estimator.

**Behavior:**
- Features are always randomly permuted at each split
- When `max_features < n_features`, features are randomly selected
- Setting this to an integer ensures reproducibility

**Example:**
```python
clf = DecisionTreeClassifier(random_state=42)  # Reproducible results
```

##### 8. max_leaf_nodes (int, default=None)

**Purpose:** Grow a tree with max_leaf_nodes in best-first fashion.

**Behavior:**
- `None`: Unlimited leaf nodes
- `int`: Stop when reaching this many leaves

**Use case:** Alternative to `max_depth` for controlling tree size

**Example:**
```python
clf = DecisionTreeClassifier(max_leaf_nodes=20)  # At most 20 leaves
```

##### 9. min_impurity_decrease (float, default=0.0)

**Purpose:** A node will be split if this split induces a decrease of the impurity ≥ this value.

**Formula:**
```
Weighted impurity decrease:
N_t/N * (impurity - N_t_R/N_t * right_impurity - N_t_L/N_t * left_impurity)
```

Where:
- N = total samples
- N_t = samples at current node
- N_t_L, N_t_R = samples in left/right child

**Use case:** Early stopping criterion based on impurity reduction

##### 10. class_weight (dict, list, 'balanced', or None, default=None)

**Purpose:** Weights associated with classes.

**Options:**
- `None`: All classes have weight 1
- `dict`: {class_label: weight} (e.g., {0: 1, 1: 5})
- `'balanced'`: Automatically adjust weights inversely proportional to class frequencies
  ```python
  weights = n_samples / (n_classes * np.bincount(y))
  ```

**Use case:** **Critical for imbalanced datasets**

**Example:**
```python
# Manual weights
clf = DecisionTreeClassifier(class_weight={0: 1, 1: 10})  # Class 1 is 10x more important

# Automatic balancing
clf_balanced = DecisionTreeClassifier(class_weight='balanced')
```

##### 11. ccp_alpha (float, default=0.0)

**Purpose:** Complexity parameter used for Minimal Cost-Complexity Pruning.

**Behavior:**
- `0.0`: No pruning (default)
- `> 0.0`: Prune subtrees with cost complexity > α

**Use case:** Post-pruning to prevent overfitting

**Reference:** See pruning section above for detailed explanation and usage

**Example:**
```python
clf = DecisionTreeClassifier(ccp_alpha=0.01)  # Prune with α = 0.01
```

---

#### Key Attributes (After Fitting)

##### classes_ (ndarray)
The class labels.

**Example:**
```python
clf.fit(X_train, y_train)
print(clf.classes_)  # e.g., array([0, 1, 2])
```

##### feature_importances_ (ndarray)
The importance of each feature (Gini importance / mean decrease in impurity).

**Formula:** Sum of weighted impurity decreases for all nodes where feature is used.

**Example:**
```python
importances = clf.feature_importances_
for i, imp in enumerate(importances):
    print(f"Feature {i}: {imp:.4f}")
```

##### max_features_ (int)
The inferred value of max_features.

##### n_classes_ (int or list)
Number of classes (single output) or list of classes (multi-output).

##### n_features_in_ (int)
Number of features seen during fit.

##### feature_names_in_ (ndarray)
Names of features (if X had feature names).

##### n_outputs_ (int)
The number of outputs when fit is performed.

##### tree_ (Tree object)
The underlying Tree object. Access internal structure with:
- `tree_.node_count`: Number of nodes
- `tree_.max_depth`: Maximum depth
- `tree_.children_left`, `tree_.children_right`: Child node indices
- `tree_.feature`, `tree_.threshold`: Split information

**Example:**
```python
print(f"Tree has {clf.tree_.node_count} nodes")
print(f"Max depth: {clf.tree_.max_depth}")
```

---

#### Key Methods

##### fit(X, y, sample_weight=None)
Build a decision tree classifier from training set.

**Parameters:**
- `X`: Training data (n_samples, n_features)
- `y`: Target values (n_samples,)
- `sample_weight`: Optional sample weights

**Returns:** self (fitted estimator)

**Example:**
```python
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)
```

##### predict(X)
Predict class for X.

**Returns:** Predicted class labels (n_samples,)

**Note:** Uses `np.argmax(predict_proba(X))`, so if probabilities are tied, predicts the class with lowest index.

**Example:**
```python
y_pred = clf.predict(X_test)
```

##### predict_proba(X)
Predict class probabilities for X.

**Returns:** Class probabilities (n_samples, n_classes)

**How it works:** The predicted class probability is the **fraction of samples of the same class in the leaf**.

**Example:**
```python
y_proba = clf.predict_proba(X_test)
# y_proba[0] = [0.1, 0.9] means 10% prob class 0, 90% prob class 1
```

##### predict_log_proba(X)
Predict class log-probabilities.

**Returns:** Log of class probabilities

**Use case:** Numerical stability in some calculations

##### score(X, y, sample_weight=None)
Return the accuracy on test data.

**Formula:** accuracy = (# correct predictions) / (# total predictions)

**Example:**
```python
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

##### apply(X)
Return the index of the leaf that each sample is predicted as.

**Returns:** Leaf indices (n_samples,)

**Use case:** Identify which leaf node each sample ends up in

**Example:**
```python
leaf_indices = clf.apply(X_test)
print(f"Sample 0 ends in leaf {leaf_indices[0]}")
```

##### decision_path(X)
Return the decision path in the tree.

**Returns:** Sparse matrix indicating which nodes each sample passes through

**Example:**
```python
node_indicator = clf.decision_path(X_test)
# node_indicator[i, j] = 1 if sample i goes through node j
```

##### cost_complexity_pruning_path(X, y, sample_weight=None)
Compute the pruning path during Minimal Cost-Complexity Pruning.

**Returns:** Bunch object with:
- `ccp_alphas`: Effective alphas of subtree during pruning
- `impurities`: Total impurities of subtree leaves for each alpha

**Use case:** Find optimal alpha for pruning (see pruning section above)

**Example:**
```python
path = clf.cost_complexity_pruning_path(X_train, y_train)
alphas = path.ccp_alphas[:-1]  # Exclude the last alpha (tree with 0 leaves)

# Train trees with different alphas
clfs = []
for alpha in alphas:
    clf_alpha = DecisionTreeClassifier(
        criterion='gini',
        ccp_alpha=alpha,
        random_state=42
    )
    clf_alpha.fit(X_train, y_train)
    clfs.append(clf_alpha)

# Evaluate
train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(alphas, train_scores, marker='o', label='Train', alpha=0.7)
ax.plot(alphas, test_scores, marker='s', label='Test', alpha=0.7)
ax.set_xlabel('Alpha (Complexity Parameter)')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy vs Alpha (Pruning)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

#### Complete Working Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Load data
iris = load_iris()
X, y = iris.data, iris.target

# For binary classification example
X_binary = X[y != 2]  # Remove class 2
y_binary = y[y != 2]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_binary, y_binary, test_size=0.3, random_state=42
)

# 2. Create and fit classifier
clf = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

clf.fit(X_train, y_train)

# 3. Make predictions
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

# 4. Evaluate
print("="*50)
print("PERFORMANCE METRICS")
print("="*50)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Tree Depth: {clf.get_depth()}")
print(f"Number of Leaves: {clf.get_n_leaves()}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=iris.target_names[:2]))

# 5. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                              display_labels=iris.target_names[:2])
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# 6. Feature Importances
importances = clf.feature_importances_
feature_names = iris.feature_names

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.tight_layout()
plt.show()

# 7. Visualize Tree
plt.figure(figsize=(20, 10))
plot_tree(clf, 
         feature_names=iris.feature_names,
         class_names=iris.target_names[:2],
         filled=True,
         rounded=True,
         fontsize=10)
plt.title("Decision Tree Structure")
plt.show()

# 8. Cross-validation
cv_scores = cross_val_score(clf, X_binary, y_binary, cv=5)
print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 9. Example: Cost-Complexity Pruning
path = clf.cost_complexity_pruning_path(X_train, y_train)
alphas = path.ccp_alphas[:-1]  # Exclude the last alpha (tree with 0 leaves)

# Train trees with different alphas
clfs = []
for alpha in alphas:
    clf_alpha = DecisionTreeClassifier(
        criterion='gini',
        ccp_alpha=alpha,
        random_state=42
    )
    clf_alpha.fit(X_train, y_train)
    clfs.append(clf_alpha)

# Evaluate
train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(alphas, train_scores, marker='o', label='Train', alpha=0.7)
ax.plot(alphas, test_scores, marker='s', label='Test', alpha=0.7)
ax.set_xlabel('Alpha (Complexity Parameter)')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy vs Alpha (Pruning)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

#### Best Practices

##### 1. For Imbalanced Data
```python
# Always use class_weight='balanced'
clf = DecisionTreeClassifier(class_weight='balanced')
```

##### 2. Prevent Overfitting
```python
# Combine multiple strategies
clf = DecisionTreeClassifier(
    max_depth=10,              # Limit depth
    min_samples_split=20,      # Require minimum samples to split
    min_samples_leaf=10,       # Require minimum samples in leaves
    ccp_alpha=0.01,           # Post-pruning
    random_state=42
)
```

##### 3. For Better Generalization
```python
# Use cross-validation to tune hyperparameters
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'criterion': ['gini', 'entropy']
}

grid = GridSearchCV(DecisionTreeClassifier(random_state=42), 
                   param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_:.4f}")
```

##### 4. Interpretability
```python
# Always visualize tree and check feature importances
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=feature_names)
plt.show()

# Check which features are most important
for name, importance in zip(feature_names, clf.feature_importances_):
    print(f"{name}: {importance:.4f}")
```

---

#### Common Pitfalls and Solutions

| Problem | Symptom | Solution |
|---------|---------|----------|
| **Overfitting** | High train acc, low test acc | Increase `min_samples_split`, `min_samples_leaf`; decrease `max_depth`; use `ccp_alpha` |
| **Imbalanced data** | Poor minority class prediction | Use `class_weight='balanced'` |
| **Large trees** | Slow, hard to interpret | Set `max_depth`, `max_leaf_nodes`, or use pruning |
| **Tied probabilities** | Unexpected predictions | Remember: `predict()` uses `argmax(predict_proba())` - lowest index wins |
| **No randomness** | Deterministic splits | Set `random_state` for reproducibility |

---

#### Comparison: DecisionTreeClassifier vs DecisionTreeRegressor

| Aspect | Classifier | Regressor |
|--------|-----------|-----------|
| **Output** | Class labels | Continuous values |
| **Splitting criterion** | Gini, Entropy | MSE, MAE, Poisson |
| **Prediction** | Majority class in leaf | Mean value in leaf |
| **Probability output** | Yes (`predict_proba`) | No |
| **Pruning** | Same (`ccp_alpha`) | Same (`ccp_alpha`) |

---

#### Key Takeaways

1. **DecisionTreeClassifier** implements CART algorithm for classification
2. **Two main impurity measures:** Gini (default, faster) and Entropy (information gain)
3. **Pre-pruning:** Control with `max_depth`, `min_samples_split`, `min_samples_leaf`
4. **Post-pruning:** Use `ccp_alpha` for cost-complexity pruning
5. **Imbalanced data:** Always use `class_weight='balanced'`
6. **Feature importances:** Accessible via `feature_importances_` attribute
7. **Tree visualization:** Use `plot_tree()` for interpretability
8. **Probability estimates:** Come from leaf node class proportions
9. **Reproducibility:** Set `random_state` parameter
10. **Model selection:** Use cross-validation and GridSearchCV for hyperparameter tuning

---

### 8.1.3 - Trees Versus Linear Models
*[Content to be added]*

### 8.1.4 - Advantages and Disadvantages of Trees
*[Content to be added]*

---

## Section 8.2 - Bagging, Random Forests, Boosting, and Bayesian Additive Regression Trees

### 8.2.1 - Bagging
*[Content to be added]*

### 8.2.2 - Random Forests
*[Content to be added]*

### 8.2.3 - Boosting
*[Content to be added]*

### 8.2.4 - Bayesian Additive Regression Trees
*[Content to be added]*

### 8.2.5 - Summary of Tree Ensemble Methods
*[Content to be added]*

---

## Section 8.3 - Lab: Tree-Based Methods

### 8.3.1 - Fitting Classification Trees
*[Content to be added]*

### 8.3.2 - Fitting Regression Trees
*[Content to be added]*

### 8.3.3 - Bagging and Random Forests
*[Content to be added]*

### 8.3.4 - Boosting
*[Content to be added]*

### 8.3.5 - Bayesian Additive Regression Trees
*[Content to be added]*

---

## Section 8.4 - Exercises
*[Content to be added]*

---

## Notes
*[Add your notes here]*
