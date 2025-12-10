# Ensemble Methods Introduction: Random Forest

**Lecture**: Module 5, Lecture 1  
**Course**: CSCA5622  
**Topic**: Introduction to Ensemble Methods, Bagging, and Random Forest

---

## Table of Contents
1. [Introduction to Ensemble Methods](#1-introduction-to-ensemble-methods)
2. [The Musical Analogy](#2-the-musical-analogy)
3. [Why Diversity Matters](#3-why-diversity-matters)
4. [Bagging: Bootstrap Aggregation](#4-bagging-bootstrap-aggregation)
5. [Random Forest](#5-random-forest)
6. [Out-of-Bag Error](#6-out-of-bag-error)
7. [Feature Importance](#7-feature-importance)
8. [Performance Analysis](#8-performance-analysis)
9. [Python Implementation](#9-python-implementation)
10. [Practice Problems](#10-practice-problems)

---

## 1. Introduction to Ensemble Methods

### What is an Ensemble?

An **ensemble** in machine learning refers to a collection of multiple models working together to make predictions. The fundamental idea is that combining multiple "weak learners" can create a single "strong learner" with better predictive performance.

**Key Insight**: Individual models may have limitations, but when aggregated properly, they can overcome these limitations and achieve superior performance.

> **Slide Visualization**: 
> The opening slide likely shows the word "Ensemble" with images of multiple decision trees or models arranged together, possibly with arrows showing how they combine into a final prediction.

---

## 2. The Musical Analogy

### Individual Instruments vs. Orchestra

The lecturer uses a powerful analogy to explain ensemble methods:

**Individual Instrument**:
*   A single instrument player can produce sound and music.
*   However, the **sound characteristics** and **spectrum** are **limited** by that one instrument alone.
*   Example: A violin can play beautiful melodies but cannot produce the depth of a cello or the brightness of a trumpet.

**Musical Ensemble (Orchestra)**:
*   A **collection of different types of instruments** playing together.
*   Creates **very rich and flavorful musical sound**.
*   Each instrument contributes its unique characteristics.
*   The combination produces something far more complex and beautiful than any single instrument could achieve.

> **Slide Visualization**: 
> The slide likely shows:
> - Left side: A single musician with one instrument
> - Right side: A full orchestra with multiple sections (strings, brass, woodwinds, percussion)
> - Possibly with sound wave visualizations showing limited vs. rich frequency spectrum

### Application to Machine Learning

**Single Model (e.g., Decision Tree)**:
*   Can be a **weak learner**.
*   Limited by its individual biases and variance.
*   May overfit or underfit depending on the data.

**Ensemble of Models**:
*   If models are **aggregated in certain ways**, they can be **much better**.
*   Each model contributes its strengths.
*   Errors of individual models can cancel out.
*   The combined prediction is more robust and accurate.

---

## 3. Why Diversity Matters

### The Public Decision Analogy

The lecturer provides another powerful analogy about decision-making in a community:

#### Scenario A: Homogeneous Group
*   **Sample people** who have:
    *   Same race
    *   Same gender
    *   Same age group
    *   Same background
*   **Problem**: This group is **likely to represent only those kinds of people**.
*   **Result**: Limited perspectives, potential biases, narrow decision-making.

#### Scenario B: Diverse Group
*   **Sample people** who have:
    *   Different genders
    *   Different age groups
    *   Different races
    *   Different backgrounds
*   **Advantage**: **Likely to be more representative of different groups**.
*   **Result**: Therefore, you're **likely to make a better decision**.

**Key Principle**: **"Diversity is great."**

> **Slide Visualization**: 
> The slide likely shows two panels:
> - Left panel: Uniform icons representing similar people (same color/shape)
> - Right panel: Diverse icons representing different demographics
> - Possibly with a scale or metric showing "Better Decision Making" on the right side

### Applying Diversity to Machine Learning Models

**Question**: How can we make our models diverse?

**Answer**: Train models on different aspects of the data:
1.  **Different subsets of data** (Bagging)
2.  **Different subsets of features** (Random Forest)
3.  Different algorithms (General ensemble approach)
4.  Different hyperparameters

This lecture focuses on approaches 1 and 2.

---

## 4. Bagging: Bootstrap Aggregation

### What is Bagging?

**Bagging** = **B**ootstrap **Agg**regation

**Core Idea**: Train models on **different random subsets of data**.

### Etymology of "Bagging"

*   You might think: "Putting different datasets into bags" 
*   **Actually**: The name comes from **"Bootstrap Aggregation"**
*   Not literally about bags!

> **Slide Visualization**: 
> The slide might show:
> - A large dataset represented as a grid or collection of points
> - Multiple "bags" or subsets drawn from it with arrows
> - Each bag feeding into a separate decision tree

### The Bootstrap Aggregation Algorithm

#### Step 1: Bootstrap Sampling

**Process**: Randomly sample a subset of training data **with replacement**.

**"With Replacement" Explained**:
*   After selecting a data point, we "put it back" into the pool.
*   This means **we can sample the same data point multiple times**.
*   Some data points may appear multiple times in a bootstrap sample.
*   Some data points may not appear at all.

**Mathematical Formulation**:
Given original training data $D = \{(x_1, y_1), (x_2, y_2), ..., (x_N, y_N)\}$

For each bootstrap sample $D_b$ (where $b = 1, 2, ..., B$):
*   Randomly draw $N$ samples from $D$ with replacement
*   $D_b = \{(x_{i_1}, y_{i_1}), (x_{i_2}, y_{i_2}), ..., (x_{i_N}, y_{i_N})\}$
*   where each $i_j$ is randomly selected from $\{1, 2, ..., N\}$

> **Slide Visualization**: 
> The slide likely shows:
> - Original dataset as a complete grid
> - Multiple colored "selections" or highlights showing different bootstrap samples
> - Overlapping regions showing that samples can include the same data points
> - The lecturer mentions "yellow sections" being sampled with overlaps

**Example**:
```
Original Data: [A, B, C, D, E]
Bootstrap Sample 1: [A, A, C, D, E]  (A appears twice, B not selected)
Bootstrap Sample 2: [B, C, C, D, D]  (C and D appear twice, A and E not selected)
Bootstrap Sample 3: [A, B, D, E, E]  (E appears twice, C not selected)
```

#### Step 2: Grow Trees

**Process**: For each bootstrap sample, train a decision tree.

*   Tree 1 trained on Bootstrap Sample 1
*   Tree 2 trained on Bootstrap Sample 2
*   Tree 3 trained on Bootstrap Sample 3
*   ...
*   Tree B trained on Bootstrap Sample B

**Important Note on Pruning**: 
*   "In general, we **don't let these trees prune**."
*   **Reason**: If we prune them, "they may become similar to each other."
*   We want to maintain diversity, so we grow **deep, fully-grown trees**.
*   **However**: "It is also possible in practice that we can grow, prune the tree, and then ensemble them."
*   The standard approach is to grow trees fully without pruning.

#### Step 3: Ensemble (Aggregate)

**Process**: Combine predictions from all trees.

**For Regression**:
*   **Method**: **Averaging**
*   Final prediction: 
    $$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} \hat{y}_b$$
*   where $\hat{y}_b$ is the prediction from tree $b$

**For Classification**:
*   **Method**: **Voting** (Majority vote)
*   Final prediction: The class that receives the most votes
    $$\hat{y} = \text{mode}\{\hat{y}_1, \hat{y}_2, ..., \hat{y}_B\}$$

**Example (Classification)**:
```
Tree 1 predicts: Class A
Tree 2 predicts: Class B
Tree 3 predicts: Class A
Tree 4 predicts: Class A
Tree 5 predicts: Class B

Final Prediction: Class A (3 votes vs 2 votes)
```

### Why Bagging Works

**Variance Reduction**:
*   Individual decision trees have **high variance** (they overfit easily).
*   By averaging many trees trained on different samples:
    *   Random errors cancel out
    *   Systematic patterns are reinforced
*   **Result**: Lower overall variance, more stable predictions

**Mathematical Intuition**:
If we have $B$ independent models, each with variance $\sigma^2$, the variance of the average is:
$$\text{Var}(\bar{y}) = \frac{\sigma^2}{B}$$

The variance decreases as we add more trees!

---

## 5. Random Forest

### Beyond Bagging: Adding Decorrelation

**Random Forest** = Bagging + **Random Feature Sampling**

The lecturer states: "Random forest has another added idea to the bagging."

### The Decorrelation Process

**Problem with Bagging Alone**:
*   Even with different data subsets, trees might still look similar.
*   If all trees use all features, they might:
    *   Split on the same features
    *   In the same order
    *   Create similar tree structures

**Example of Correlated Trees**:
```
Tree 1: Split on Feature 1 → Feature 2 → Feature 3
Tree 2: Split on Feature 1 → Feature 2 → Feature 4
Tree 3: Split on Feature 1 → Feature 3 → Feature 2
```
All trees start with Feature 1 because it's the strongest predictor!

### Random Feature Sampling

**Solution**: At each split, **randomly sample a subset of features** to consider.

**Process**:
1.  At each node in the tree-building process
2.  Randomly select a subset of features
3.  Choose the best split only from these selected features
4.  Different nodes may consider different feature subsets

**Example**:
```
Tree 1: 
  - Root node: Consider features [1, 3, 5] → Split on Feature 1
  - Left child: Consider features [2, 4, 5] → Split on Feature 2

Tree 2:
  - Root node: Consider features [2, 4, 7] → Split on Feature 2 (Feature 1 not available!)
  - Left child: Consider features [1, 3, 6] → Split on Feature 4
```

The lecturer explains: "If you have a random sample of features, so maybe the second tree didn't have feature 1, so it **forced to split on feature 2**, then it will split on some other feature, maybe feature 4 and so on."

### Why It's Called "Decorrelation"

**Without Feature Sampling**:
*   Trees are **correlated** (similar structure)
*   They make similar errors
*   Averaging doesn't help as much

**With Feature Sampling**:
*   Trees have **different structures**
*   They make **different errors** (errors are uncorrelated)
*   Averaging is much more effective

The lecturer states: "If you have a random sampling of features, individual trees grown on the subset of the data and subset of features will be **likely to have a different structure from each other**. That is why it is called the **decorrelation**."

### The Square Root Rule

**Question**: How many features should we randomly sample?

**Answer**: **Rule of Thumb: Square Root Method**

**Formula**:
$$m = \sqrt{p}$$

where:
*   $p$ = total number of features in the dataset
*   $m$ = number of features to sample at each split

**Example**:
*   If we have **100 features** in the data
*   We select $\sqrt{100} = 10$ features at each split

**Why This Works**:
*   Enough features to find good splits
*   Few enough to force diversity
*   Empirically proven to work well across many problems

**Alternative Rules**:
*   For classification: $m = \sqrt{p}$ (standard)
*   For regression: $m = p/3$ (sometimes used)
*   Can be tuned as a hyperparameter

> **Slide Visualization**: 
> The slide likely shows a comparison:
> - Left: Bagging trees (similar structure, all using same features)
> - Right: Random Forest trees (different structures, using different feature subsets)
> - Arrows or highlights showing how different features are selected at each split

### Random Forest Algorithm Summary

**Complete Algorithm**:

```
For b = 1 to B:
    1. Draw a bootstrap sample D_b of size N from training data D
    2. Grow a tree T_b using D_b:
        a. At each node:
            i. Randomly select m features from p total features
            ii. Find the best split using only these m features
            iii. Split the node
        b. Grow tree to maximum depth (no pruning)
    3. Store tree T_b

For prediction on new data point x:
    - Regression: Return average of all tree predictions
    - Classification: Return majority vote of all tree predictions
```

---

## 6. Out-of-Bag Error

### What is Out-of-Bag (OOB) Error?

**Definition**: A validation error metric that comes as a **bonus** from bagging.

The lecturer states: "Another bonus by doing bagging is we can use the **out-of-bag error**."

### How Bootstrap Sampling Creates OOB Data

**Key Observation**: When we draw a bootstrap sample with replacement:
*   Some data points are selected (possibly multiple times)
*   Some data points are **not selected at all**

**Probability Analysis**:
For a dataset of size $N$, when drawing $N$ samples with replacement:
*   Probability a specific point is NOT selected in one draw: $(1 - 1/N)$
*   Probability it's NOT selected in any of $N$ draws: $(1 - 1/N)^N$
*   As $N \to \infty$: $(1 - 1/N)^N \to e^{-1} \approx 0.368$

**Result**: Approximately **36.8%** of the original data is **not used** in each bootstrap sample!

### Using OOB Data for Validation

**Process**:

1.  For each tree $T_b$ trained on bootstrap sample $D_b$:
    *   Identify which original data points were **not** in $D_b$
    *   These are the **out-of-bag samples** for tree $T_b$

2.  For each original data point $(x_i, y_i)$:
    *   Find all trees that did NOT include this point in training
    *   Use only these trees to make a prediction for $(x_i, y_i)$
    *   Compare prediction to true label $y_i$

3.  Calculate OOB error across all predictions

The lecturer explains: "Out-of-bag error is validation error that we can test our fitted tree that was trained on this yellow chunks, then we can **test on the rest of the data that we didn't select to train on**."

> **Slide Visualization**: 
> The slide likely shows:
> - A dataset represented as a complete set of points
> - One tree's bootstrap sample highlighted (e.g., in yellow)
> - The remaining points (not yellow) labeled as "OOB data for this tree"
> - Arrows showing how these OOB points are used for validation

### Advantages of OOB Error

**Benefits**:
1.  **No need for separate validation set**: We get validation error "for free"
2.  **Uses all data for training**: Each point is OOB for ~37% of trees
3.  **Efficient**: No computational overhead
4.  **Similar to cross-validation**: But automatically built into the bagging process

**Formula**:
$$\text{OOB Error} = \frac{1}{N} \sum_{i=1}^{N} L(y_i, \hat{y}_i^{\text{OOB}})$$

where $\hat{y}_i^{\text{OOB}}$ is the prediction for point $i$ using only trees that didn't include it in training, and $L$ is the loss function.

---

## 7. Feature Importance

### Built-in Feature Importance in Random Forest

The lecturer states: "Random forests also have a cool feature. It has a **built-in feature importance**."

### How Feature Importance is Calculated

**Method**: Measure how much each feature decreases the impurity (e.g., Gini impurity or entropy) across all trees.

**Process**:
1.  For each tree in the forest:
    *   Track which features are used for splitting
    *   Record the decrease in impurity at each split
    
2.  For each feature:
    *   Sum the impurity decreases across all trees
    *   Normalize by the number of trees
    
3.  Rank features by their total importance

**Mathematical Formulation**:

For feature $j$, the importance is:
$$\text{Importance}(j) = \frac{1}{B} \sum_{b=1}^{B} \sum_{t \in T_b} \mathbb{1}(v_t = j) \cdot \Delta i_t$$

where:
*   $B$ = number of trees
*   $T_b$ = set of all nodes in tree $b$
*   $v_t$ = feature used at node $t$
*   $\mathbb{1}(v_t = j)$ = indicator (1 if feature $j$ used at node $t$, 0 otherwise)
*   $\Delta i_t$ = decrease in impurity at node $t$

### Accessing Feature Importance in Scikit-learn

The lecturer notes: "In sklearn library, you can **pull out feature importance** after feeding the random forest model in the data."

**Code Example**:
```python
from sklearn.ensemble import RandomForestClassifier

# Train model
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Get feature importance
importances = rf.feature_importances_
```

### Using Feature Importance for Feature Selection

**Practical Application**:

The lecturer explains: "Oftentimes, this is useful because you can **figure out some feature importance** and then **use it as a feature selection**."

**Workflow**:
1.  Train a Random Forest on all features
2.  Extract feature importances
3.  Select top-k most important features
4.  Train a new model (can be different algorithm!) using only selected features

**Advantage**: "Even if you want to use some different model, you can still **use random forest to do the feature selection** and then you can **build some more serious model on top of it**."

**Example Use Cases**:
*   Use RF for feature selection, then train a Linear SVM
*   Use RF for feature selection, then train a Neural Network
*   Use RF for feature selection, then train a simpler interpretable model

The lecturer concludes: "That can be a **handy tool**."

---

## 8. Performance Analysis

### Comparison of Methods

The lecture presents results comparing different approaches.

> **Slide Visualization 1**: 
> The slide likely shows a graph with:
> - X-axis: Number of features selected (or method type)
> - Y-axis: Performance metric (e.g., accuracy or error rate)
> - Green line: Random Forest with square root feature selection
> - Red line: Random Forest with all features (essentially bagging)

#### Result 1: Feature Selection Benefits

"Here are the result of random forest classifiers that the **green line** shows that we had a **square root method for selecting features** versus the **red curve** in the random forest with **using all samples**. Essentially, that's the bagging."

**Observation**: "You can see some **increased performance** in the decorrelated trees or use a smaller number of features."

**Interpretation**:
*   Green line (square root method) performs better than red line (all features)
*   Decorrelation through feature sampling improves performance
*   Using fewer features at each split actually helps!

> **Slide Visualization 2**: 
> The slide likely shows a graph with:
> - X-axis: Number of trees in ensemble
> - Y-axis: Test error rate
> - Green star: Single tree performance (baseline)
> - Blue curve: Random Forest test error
> - Red curve: Bagging test error
> - Dashed lines: Out-of-bag test errors

#### Result 2: Ensemble Size and Performance

"Here's another result showing the **power of ensemble**."

**Single Tree Baseline**:
"This **green star point** is actually a **single tree test performance**."

**Effect of Adding Trees**:
"Then as you can see, as we **increase the number of trees in the ensemble**, it generally **goes up** and then at **certain point, they behave similar**."

**Interpretation**:
*   Performance improves dramatically from single tree to ensemble
*   Improvement continues as we add more trees
*   Eventually plateaus (diminishing returns)
*   More trees don't hurt (no overfitting), but give diminishing benefits

**Comparing Ensemble Methods**:

"This **blue curve**, for example, is a **random forest test error**, and this **red curve** is a **bagging test error**."

**Key Findings**:

1.  **Both methods beat single tree**: 
    "Both the random forest and the bagging method, they are ensembling method and they **increase the performance a lot** compared to just a single tree."

2.  **Random Forest beats Bagging**:
    "However, as you can see, **decorrelating trees make it a little better than just bagging**, just random sampling the data."

3.  **Out-of-Bag Error**:
    "You can also see the **out-of-bag test error**. These are **validation error during the training process**."

**Performance Hierarchy** (from worst to best):
1.  Single Decision Tree (green star)
2.  Bagging (red curve)
3.  Random Forest (blue curve)

**Why Random Forest Wins**:
*   Combines benefits of bootstrap sampling (bagging)
*   Adds decorrelation through feature sampling
*   Results in more diverse, less correlated trees
*   Better at reducing variance

---

## 9. Python Implementation

### Complete Random Forest Example

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import make_classification, make_regression

# Set random seed for reproducibility
np.random.seed(42)

# ============================================
# Example 1: Classification with Random Forest
# ============================================

# Generate synthetic classification data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                          n_redundant=5, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 1. Single Decision Tree
single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)
single_tree_acc = accuracy_score(y_test, single_tree.predict(X_test))

print("=" * 50)
print("CLASSIFICATION RESULTS")
print("=" * 50)
print(f"Single Tree Accuracy: {single_tree_acc:.4f}")

# 2. Bagging (Random Forest with all features)
bagging = RandomForestClassifier(n_estimators=100, max_features=X.shape[1],
                                 random_state=42, oob_score=True)
bagging.fit(X_train, y_train)
bagging_acc = accuracy_score(y_test, bagging.predict(X_test))
print(f"Bagging Accuracy:     {bagging_acc:.4f}")
print(f"Bagging OOB Score:    {bagging.oob_score_:.4f}")

# 3. Random Forest (with sqrt feature selection)
rf = RandomForestClassifier(n_estimators=100, max_features='sqrt',
                           random_state=42, oob_score=True)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"Random Forest Accuracy: {rf_acc:.4f}")
print(f"Random Forest OOB Score: {rf.oob_score_:.4f}")

# ============================================
# Feature Importance Analysis
# ============================================

print("\n" + "=" * 50)
print("FEATURE IMPORTANCE (Top 10)")
print("=" * 50)

# Get feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Print top 10 features
for i in range(min(10, len(importances))):
    print(f"Feature {indices[i]:2d}: {importances[indices[i]]:.4f}")

# Visualize feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xlabel('Feature Index (sorted by importance)')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# Performance vs Number of Trees
# ============================================

print("\n" + "=" * 50)
print("PERFORMANCE VS NUMBER OF TREES")
print("=" * 50)

n_trees_range = [1, 5, 10, 20, 50, 100, 200, 500]
rf_scores = []
bagging_scores = []

for n_trees in n_trees_range:
    # Random Forest
    rf_temp = RandomForestClassifier(n_estimators=n_trees, max_features='sqrt',
                                     random_state=42)
    rf_temp.fit(X_train, y_train)
    rf_scores.append(accuracy_score(y_test, rf_temp.predict(X_test)))
    
    # Bagging
    bag_temp = RandomForestClassifier(n_estimators=n_trees, max_features=X.shape[1],
                                      random_state=42)
    bag_temp.fit(X_train, y_train)
    bagging_scores.append(accuracy_score(y_test, bag_temp.predict(X_test)))
    
    print(f"Trees: {n_trees:3d} | RF Acc: {rf_scores[-1]:.4f} | Bag Acc: {bagging_scores[-1]:.4f}")

# Plot performance curves
plt.figure(figsize=(10, 6))
plt.plot(n_trees_range, rf_scores, 'b-o', label='Random Forest', linewidth=2)
plt.plot(n_trees_range, bagging_scores, 'r-s', label='Bagging', linewidth=2)
plt.axhline(y=single_tree_acc, color='g', linestyle='--', label='Single Tree', linewidth=2)
plt.xlabel('Number of Trees')
plt.ylabel('Test Accuracy')
plt.title('Ensemble Performance vs Number of Trees')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ensemble_performance.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# Example 2: Regression with Random Forest
# ============================================

print("\n" + "=" * 50)
print("REGRESSION EXAMPLE")
print("=" * 50)

# Generate synthetic regression data
X_reg, y_reg = make_regression(n_samples=1000, n_features=20, n_informative=15,
                               noise=10, random_state=42)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42)

# Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, max_features='sqrt',
                              random_state=42, oob_score=True)
rf_reg.fit(X_train_reg, y_train_reg)

# Predictions
y_pred_train = rf_reg.predict(X_train_reg)
y_pred_test = rf_reg.predict(X_test_reg)

# Metrics
mse_train = mean_squared_error(y_train_reg, y_pred_train)
mse_test = mean_squared_error(y_test_reg, y_pred_test)

print(f"Training MSE: {mse_train:.2f}")
print(f"Test MSE:     {mse_test:.2f}")
print(f"OOB Score (R²): {rf_reg.oob_score_:.4f}")

# ============================================
# Bootstrap Sampling Demonstration
# ============================================

print("\n" + "=" * 50)
print("BOOTSTRAP SAMPLING DEMONSTRATION")
print("=" * 50)

# Small dataset to visualize
small_data = np.array([1, 2, 3, 4, 5])
print(f"Original Data: {small_data}")

# Create 5 bootstrap samples
for i in range(5):
    bootstrap_sample = np.random.choice(small_data, size=len(small_data), replace=True)
    oob_mask = ~np.isin(small_data, bootstrap_sample)
    oob_data = small_data[oob_mask]
    print(f"Bootstrap {i+1}: {bootstrap_sample} | OOB: {oob_data}")

# Calculate OOB probability
n_simulations = 10000
oob_counts = []
for _ in range(n_simulations):
    sample = np.random.choice(range(100), size=100, replace=True)
    oob_count = len(set(range(100)) - set(sample))
    oob_counts.append(oob_count)

avg_oob = np.mean(oob_counts)
print(f"\nAverage OOB samples: {avg_oob:.2f} out of 100 ({avg_oob:.1f}%)")
print(f"Theoretical: 36.8%")
```

### Feature Selection with Random Forest

```python
from sklearn.feature_selection import SelectFromModel

# ============================================
# Feature Selection Example
# ============================================

print("\n" + "=" * 50)
print("FEATURE SELECTION WITH RANDOM FOREST")
print("=" * 50)

# Train Random Forest
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selector.fit(X_train, y_train)

# Select features based on importance
selector = SelectFromModel(rf_selector, threshold='mean', prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

print(f"Original number of features: {X_train.shape[1]}")
print(f"Selected number of features: {X_train_selected.shape[1]}")

# Train a different model on selected features (e.g., Logistic Regression)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=42, max_iter=1000)

# Compare performance with all features vs selected features
lr.fit(X_train, y_train)
acc_all = accuracy_score(y_test, lr.predict(X_test))

lr.fit(X_train_selected, y_train)
acc_selected = accuracy_score(y_test, lr.predict(X_test_selected))

print(f"\nLogistic Regression Performance:")
print(f"  With all features:      {acc_all:.4f}")
print(f"  With selected features: {acc_selected:.4f}")
print(f"  Speedup: {X_train.shape[1] / X_train_selected.shape[1]:.2f}x fewer features")
```

### Understanding Decorrelation

```python
# ============================================
# Visualizing Decorrelation Effect
# ============================================

from sklearn.tree import DecisionTreeClassifier

print("\n" + "=" * 50)
print("DECORRELATION EFFECT DEMONSTRATION")
print("=" * 50)

# Function to get first split feature
def get_first_split_feature(tree, X):
    tree_structure = tree.tree_
    return tree_structure.feature[0]

# Train multiple trees with all features (highly correlated)
n_trees = 50
first_features_all = []
for i in range(n_trees):
    # Bootstrap sample
    indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
    X_boot = X_train[indices]
    y_boot = y_train[indices]
    
    # Train tree with all features
    tree = DecisionTreeClassifier(max_features=None, random_state=i)
    tree.fit(X_boot, y_boot)
    first_features_all.append(get_first_split_feature(tree, X_boot))

# Train multiple trees with sqrt features (decorrelated)
first_features_sqrt = []
for i in range(n_trees):
    # Bootstrap sample
    indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
    X_boot = X_train[indices]
    y_boot = y_train[indices]
    
    # Train tree with sqrt features
    tree = DecisionTreeClassifier(max_features='sqrt', random_state=i)
    tree.fit(X_boot, y_boot)
    first_features_sqrt.append(get_first_split_feature(tree, X_boot))

# Analyze diversity
from collections import Counter

print("First Split Feature Distribution:")
print("\nWith ALL features (Bagging):")
all_counts = Counter(first_features_all)
for feat, count in all_counts.most_common(5):
    print(f"  Feature {feat}: {count}/{n_trees} ({100*count/n_trees:.1f}%)")

print("\nWith SQRT features (Random Forest):")
sqrt_counts = Counter(first_features_sqrt)
for feat, count in sqrt_counts.most_common(5):
    print(f"  Feature {feat}: {count}/{n_trees} ({100*count/n_trees:.1f}%)")

# Calculate diversity metric (entropy)
def entropy(counts, total):
    probs = [c/total for c in counts]
    return -sum([p * np.log2(p) for p in probs if p > 0])

entropy_all = entropy(all_counts.values(), n_trees)
entropy_sqrt = entropy(sqrt_counts.values(), n_trees)

print(f"\nDiversity (entropy):")
print(f"  Bagging (all features):  {entropy_all:.3f}")
print(f"  Random Forest (sqrt):    {entropy_sqrt:.3f}")
print(f"  Higher entropy = More diverse trees")
```

---

## 10. Practice Problems

### Problem 1: Understanding Bootstrap Sampling

**Question**: You have a dataset with 100 samples. You create a bootstrap sample by randomly selecting 100 samples with replacement.

a) What is the probability that a specific sample is **not** selected in your bootstrap sample?

b) Approximately what percentage of the original data will be out-of-bag?

c) Why is this useful for validation?

**Solution**:

**Part a):**

The probability that a specific sample is selected in one draw:
$$P(\text{selected in one draw}) = \frac{1}{100}$$

The probability that it is NOT selected in one draw:
$$P(\text{not selected in one draw}) = 1 - \frac{1}{100} = \frac{99}{100}$$

Since we draw 100 times with replacement, the probability it is NOT selected in ANY of the 100 draws:
$$P(\text{not selected at all}) = \left(\frac{99}{100}\right)^{100}$$

Calculating:
$$\left(\frac{99}{100}\right)^{100} = \left(1 - \frac{1}{100}\right)^{100} \approx 0.366$$

**Answer**: Approximately **36.6%** or **1/e**

**Part b):**

From part a), each sample has a 36.6% chance of being out-of-bag.

**Answer**: Approximately **36.8%** of the original data will be OOB.

**Part c):**

**Why it's useful**:
1.  We can use OOB samples as a validation set without setting aside separate data
2.  Every sample is OOB for ~37% of trees, so we get predictions for all training data
3.  This gives us an unbiased estimate of the test error
4.  We don't lose training data (all data is used in ~63% of trees)
5.  Similar to cross-validation but computed automatically during training

---

### Problem 2: Feature Sampling in Random Forest

**Question**: You are building a Random Forest classifier for a dataset with 64 features.

a) Using the square root rule, how many features should be considered at each split?

b) Why might using all 64 features at each split (i.e., bagging) lead to correlated trees?

c) Suppose the most important feature is Feature 7. How does feature sampling help?

**Solution**:

**Part a):**

Using the square root rule:
$$m = \sqrt{p} = \sqrt{64} = 8$$

**Answer**: **8 features** should be randomly selected at each split.

**Part b):**

**Why bagging leads to correlated trees**:

If all 64 features are available at every split:
*   The algorithm will always choose Feature 7 (the most important) at the root if it's dominant
*   Even with different bootstrap samples, trees will have similar structure
*   All trees split on the same features in similar order
*   Trees make similar predictions
*   Averaging similar predictions doesn't reduce variance much

**Mathematical Intuition**:

If trees are perfectly correlated, variance of average is:
$$\text{Var}(\bar{y}) = \sigma^2$$
(no reduction!)

If trees are independent, variance of average is:
$$\text{Var}(\bar{y}) = \frac{\sigma^2}{B}$$
(significant reduction!)

**Part c):**

**How feature sampling helps**:

With feature sampling (8 features per split):
*   Probability Feature 7 is in the random sample: $\frac{8}{64} = 0.125 = 12.5\%$
*   About 87.5% of splits will NOT have Feature 7 available
*   Trees are forced to find alternative features for splitting
*   Different trees split on different features
*   Trees have diverse structures
*   Trees make different errors
*   Averaging diverse predictions reduces variance effectively

---

### Problem 3: Comparing Ensemble Methods

**Question**: You train three models on the same dataset:
*   Model A: Single decision tree (Test Error = 0.25)
*   Model B: Bagging with 100 trees (Test Error = 0.15)
*   Model C: Random Forest with 100 trees (Test Error = 0.12)

a) Explain why Model B outperforms Model A.

b) Explain why Model C outperforms Model B.

c) Would you expect Model C's performance to improve significantly if you increased to 1000 trees? Why or why not?

**Solution**:

**Part a): Why Bagging beats Single Tree**

**Variance Reduction**:
*   A single decision tree has high variance (overfits easily)
*   Different bootstrap samples lead to different trees
*   Errors of individual trees are partially independent
*   Averaging predictions cancels out random errors
*   Systematic patterns are reinforced

**Mathematical**:
$$\text{Error}_{\text{ensemble}} < \text{Error}_{\text{single tree}}$$
because averaging reduces variance while maintaining bias.

**Part b): Why Random Forest beats Bagging**

**Additional Decorrelation**:
*   Bagging trees are still somewhat correlated (use same features)
*   Random Forest adds feature sampling
*   Trees become more diverse (less correlated)
*   Independent errors average out more effectively
*   Greater variance reduction

**Key Principle**:
The effectiveness of averaging depends on how uncorrelated the errors are:
$$\text{Var}(\bar{y}) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$
where $\rho$ is correlation between trees.

*   Bagging: Higher $\rho$ (more correlated trees)
*   Random Forest: Lower $\rho$ (less correlated trees)
*   Lower $\rho$ → Lower variance → Better performance

**Part c): Would 1000 trees help?**

**Expected Result**: **Marginal improvement** or **plateau**

**Reasoning**:
1.  **Diminishing Returns**: The graph shown in the lecture indicates performance plateaus
2.  **Already Sufficient**: 100 trees typically captures most of the benefit
3.  **Asymptotic Behavior**: As $B \to \infty$, we approach the limit:
    $$\lim_{B \to \infty} \text{Var}(\bar{y}) = \rho\sigma^2$$
    (The correlated part cannot be removed by adding more trees)
4.  **Computational Cost**: 10x more trees, but minimal accuracy gain
5.  **No Overfitting**: More trees don't hurt, but diminishing benefits

**Practical Advice**: 100-500 trees is usually sufficient. Beyond that, it's rarely worth the computational cost.

---

### Problem 4: Feature Importance Application

**Question**: You train a Random Forest on a medical diagnosis dataset with 50 features (various test results, patient demographics, etc.). The feature importance scores reveal:

*   Top 3 features: Blood pressure (0.25), Age (0.18), Cholesterol (0.15)
*   Remaining 47 features: All below 0.02

a) What does this tell you about your data?

b) How could you use this information to build a better model?

c) What are potential risks of relying too heavily on feature importance for feature selection?

**Solution**:

**Part a): What this tells us**

**Insights**:
1.  **Clear Signal**: Three features are highly informative for the target
2.  **Potential Redundancy**: 47 features contribute very little (< 2% each)
3.  **Possible Noise**: Many features may be irrelevant or noisy
4.  **Simplification Opportunity**: A much simpler model might work well
5.  **Domain Validation**: The important features (BP, age, cholesterol) make medical sense

**Mathematical Interpretation**:
$$\sum_{j=1}^{3} \text{Importance}(j) = 0.58$$

Just 3 features (6% of features) account for 58% of the total importance!

**Part b): How to use this information**

**Strategy 1: Feature Selection**
```python
# Keep only important features
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
important_features = rf.feature_importances_ > 0.05  # threshold
X_train_reduced = X_train[:, important_features]
X_test_reduced = X_test[:, important_features]
```

**Benefits**:
*   Faster training (fewer features)
*   More interpretable model
*   Reduced risk of overfitting
*   Lower memory requirements

**Strategy 2: Try simpler models**
```python
# Use RF for feature selection, then try simpler model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train_reduced, y_train)
```

**Benefits**:
*   Logistic regression is more interpretable
*   Can understand exact coefficient values
*   Easier to deploy in production
*   Medical professionals may prefer interpretable models

**Strategy 3: Feature Engineering**
*   Focus data collection efforts on important features
*   Create interaction terms between important features
*   Investigate why certain features are unimportant (data quality issues?)

**Part c): Risks of relying on feature importance**

**Risk 1: Correlated Features**
*   If two features are highly correlated, RF may pick one arbitrarily
*   The other appears unimportant, but it's actually redundant
*   Example: "Systolic BP" vs "Diastolic BP" might split importance

**Risk 2: Feature Interactions**
*   A feature might be important only in combination with others
*   Individual importance doesn't capture this
*   Example: A medication feature might only matter for certain age groups

**Risk 3: Bias in Tree-based Importance**
*   Tree-based importance favors high-cardinality features
*   Features with many unique values get more split opportunities
*   Can artificially inflate importance

**Risk 4: Model-Specific**
*   Random Forest importance is specific to tree-based models
*   A different algorithm might find different features important
*   Linear models might prioritize different features

**Risk 5: Overfitting to Training Data**
*   Importance is calculated on training data
*   May not reflect importance on test/real-world data
*   Should validate on held-out set

**Best Practice**:
*   Use multiple feature selection methods
*   Validate with domain experts
*   Test performance on held-out data
*   Consider using permutation importance (more robust)
*   Cross-validate feature selection process

---

## 11. Key Takeaways and Summary

### Main Concepts Covered

The lecturer concludes: "So far we talked about some basics of random forest, what their definition is and why they are useful, and what kind of case can they work on."

**1. Ensemble Methods**:
*   Collection of models working together
*   "Wisdom of crowds" principle
*   Individual weak learners → Strong collective learner

**2. Diversity is Crucial**:
*   Homogeneous models → Correlated errors
*   Diverse models → Uncorrelated errors → Better averaging

**3. Bagging (Bootstrap Aggregation)**:
*   Random sampling with replacement
*   Train separate trees on each sample
*   Aggregate via averaging (regression) or voting (classification)
*   Reduces variance

**4. Random Forest**:
*   Bagging + Random feature sampling
*   Decorrelates trees
*   Square root rule: $m = \sqrt{p}$
*   Superior to bagging alone

**5. Out-of-Bag Error**:
*   Free validation set (~37% of data per tree)
*   No need for separate validation
*   Similar to cross-validation

**6. Feature Importance**:
*   Built-in to Random Forest
*   Useful for feature selection
*   Can be used with other models

**7. Performance**:
*   Single Tree < Bagging < Random Forest
*   Performance plateaus with more trees
*   Typically 100-500 trees sufficient

### Next Steps

The lecturer previews: "Next video, we're going to talk about **another ensemble method called boosting**."

**Coming Up**:
*   Boosting algorithms (AdaBoost, Gradient Boosting)
*   Sequential vs. parallel ensemble methods
*   Bias-variance tradeoff in boosting
*   Comparison with Random Forest

---

## 12. Additional Resources and Notes

### Recommended Reading
*   "The Elements of Statistical Learning" - Chapter 15 (Random Forests)
*   Breiman, L. (2001). "Random Forests". Machine Learning, 45(1), 5-32.
*   Hastie, T., Tibshirani, R., & Friedman, J. (2009). ESL II

### Scikit-learn Documentation
*   [`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
*   [`RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
*   [Ensemble Methods Guide](https://scikit-learn.org/stable/modules/ensemble.html)

### Key Parameters to Tune

```python
RandomForestClassifier(
    n_estimators=100,        # Number of trees (100-500 typical)
    max_features='sqrt',     # Features per split ('sqrt' for classification)
    max_depth=None,          # Tree depth (None = fully grown)
    min_samples_split=2,     # Min samples to split node
    min_samples_leaf=1,      # Min samples in leaf
    bootstrap=True,          # Use bootstrap sampling
    oob_score=True,          # Calculate out-of-bag score
    n_jobs=-1,               # Parallel processing (use all cores)
    random_state=42          # Reproducibility
)
```

### Computational Considerations
*   **Parallelization**: Trees are independent, can be trained in parallel
*   **Memory**: Stores all trees (can be large for deep trees)
*   **Training Time**: Linear in number of trees
*   **Prediction Time**: Must query all trees (can be slow for real-time)

### When to Use Random Forest
**Good For**:
*   Tabular data with many features
*   Both classification and regression
*   When you need feature importance
*   When interpretability is moderate concern
*   Robust to outliers and missing data

**Not Ideal For**:
*   Very high-dimensional sparse data (text, images)
*   When you need probability calibration
*   When prediction speed is critical
*   When model size must be minimal

---

## Glossary

*   **Ensemble**: Collection of multiple models whose predictions are combined
*   **Weak Learner**: A model that performs slightly better than random guessing
*   **Strong Learner**: A model with high predictive accuracy
*   **Bootstrap Sample**: A sample drawn with replacement from the original data
*   **Bagging**: Bootstrap Aggregation - training models on bootstrap samples
*   **Decorrelation**: Making trees different from each other to reduce correlation
*   **Out-of-Bag (OOB)**: Samples not included in a particular bootstrap sample
*   **Feature Importance**: Measure of how much each feature contributes to predictions
*   **Aggregation**: Combining predictions (averaging for regression, voting for classification)
*   **Variance Reduction**: Decreasing prediction variability through averaging
*   **Random Forest**: Bagging with random feature sampling at each split
*   **Square Root Rule**: Selecting $\sqrt{p}$ features at each split for classification

---

**End of Lecture 1**
