# Minimal Cost-Complexity Pruning
**CSCA5622 - Module 04**

---

## 📚 Overview

This document covers **tree pruning** as a strategy to prevent overfitting in decision trees, focusing on **Minimal Cost-Complexity Pruning** algorithm. Topics include limitations of early stopping, the pruning concept, α-effective calculation, and the iterative pruning process.

All concepts explained from the lecture transcript.

---

## 1. Review: Early Stopping Strategies

From lecture:
> "Last time we talked about some **ways to prevent overfitting** in decision trees. **Decision trees are very easy to overfit**. So to mitigate, we talked about all these **tapping last time**. We talked about **number of hyperparameters** that can be used for **stopping growing trees awry**."

### 📊 Three Early Stopping Examples

**1. Maximum Depth:**

From lecture:
> "For example we can set the **maximum depth of the tree**. So after that **certain depths**, the **tree stops growing**."

**2. Minimum Samples in Leaves:**

From lecture:
> "Another example was set the **minimum sample leaves**. That means we set some **threshold** such that the **number of samples need to be in the node** in order to **split further**."

**3. Information Gain Threshold:**

From lecture:
> "Another strategy was the **information gain**. We look at the **information gain** and if the **gain is not enough**, by **splitting the node**, then we **stops bleeding** [splitting] there."

---

## 2. Limitations of Early Stopping

### ❌ The Problem

From lecture:
> "These strata as it can be **effective for preventing overfitting** but it **doesn't guarantee** that the **performance of the tree will be better**."

### 🔍 Why Early Stopping Can Fail

From lecture:
> "The issue is that we can have some **goods split after the tree stop growing**, or maybe we **locally to get some node** and **stop splitting from that node** because we may be, we saw the **information gain wasn't enough**. However, **further split** can have some **huge reduction in impurities** for example. We **never know what's going to happen after certain point**."

**The problem:** Early stopping is greedy and local
- May stop at a node with low immediate gain
- But future splits from that node might have high gain
- Cannot see "beyond" the current split

**Example scenario:**
```
Node A: Split gain = 0.05 (below threshold, stops)
  ↓
  If allowed to continue:
  Node B: Split gain = 0.40 (huge gain!)
  
Problem: Never reach Node B because stopped at Node A
```

---

## 3. The Pruning Solution

### 💡 Better Idea

From lecture:
> "Another idea that we can try is maybe we can **let the tree grow fully** and then **prune back** because it's **behind the site** [hindsight], we can **make sure the prune tree is good enough** put performance and overfitting."

**Pruning approach:**
1. Grow tree fully (until pure leaves)
2. Look at entire tree structure
3. Prune back weak branches
4. Keeps strong branches even if they start with weak splits

**Advantage:** Can see the full picture before making decisions

### 🔧 Implementation

From lecture:
> "How are we going to do that? We're going to use an **algorithm called the minimal cost complexity pruning** and this feature is **implemented** since two burdens are going **SKLearn library**."

**Available since:** sklearn version 0.22+

---

## 4. Minimal Cost-Complexity Pruning Concept

### 🌳 Setup

From lecture:
> "Here's a **big tree**. We **grow the tree fully** and we will call it **T0**."

**Notation:**
- T₀ = Full grown tree (before pruning)
- T = Pruned subtree
- t = A specific node in the tree

### 📊 Node Impurity with Penalty

From lecture:
> "And then a certain point, maybe **pick this point** that this is the **node T** and the **impurity can be measured as RT**. **Impurity can be Gini index or entropy** for classification task, but it could be **something else like RSS or mean squared error** if it's a regression so our **RT really means there's some error measure** of the node **before the splitting**."

**R(t) = Impurity at node t before split**

**For classification:** R(t) = Gini or Entropy
**For regression:** R(t) = RSS or MSE

### 🔧 Adding Complexity Penalty

From lecture:
> "Then we can add some **additional penalty**, **Alpha t**, which term is a **measure of complexity** by **splitting further**, this **Alpha t is a measure of complexity**, so **proportional to a complexity parameter** and also **proportional to the number of terminal nodes** from the node t."

**Formula:**
\[
R_{\alpha}(T_t) = R(T_t) + \alpha |T_t|
\]

Where:
- R_α(T_t) = Cost-complexity measure
- R(T_t) = Total impurity of subtree
- α = Complexity parameter (penalty)
- |T_t| = Number of terminal nodes in subtree

From lecture:
> "We define a **sub tree**, **everything below this node t** and we count the **number of terminal nodes**, in this case **three** and the **bigger dose or three**, that means we **penalize more**. That means we add **more term into our error term** so effectively the **error term is larger** when we add this **penalization term or regularization term**."

---

## 5. Detailed Formula Breakdown

### 📐 Key Components

From lecture:
> "We talked about that this **Alpha is complex a parameter** and this **size T is number of leaf nodes or terminal nodes** in the subtree and this is again, the **grey area is a sub tree** from that node t."

**Visual:**
```
        [Node t] ← Focus node
           |
      _____|_____
     |           |
  [Node]      [Node]
    |            |
  __|__        __|__
 |     |      |     |
[L1] [L2]   [L3] [L4] ← Terminal nodes

Subtree T_t: Everything in grey
|T_t| = 4 (number of leaf nodes)
```

### 🔍 Impurity Comparison

From lecture:
> "Let's say this is **node t** and these are the **leaf nodes** and as you might guess, the **impurity at the node T before the split** is **larger than** the **impurity of the sub tree** otherwise they **own split**, so this is **generally larger** than the **impurity of the subtree**."

**Relationship:**
\[
R(t) > R(T_t)
\]

**Why:** Splitting reduces impurity (that's why we split in first place!)

### 📊 Calculating Subtree Impurity

From lecture:
> "How do we **calculate the impurity of sub tree**? It's just a **sum of all the impurities** in the **leaf node** of that subtree."

**Formula:**
\[
R(T_t) = \sum_{\ell \in \text{leaves of } T_t} R(\ell)
\]

**Example:**
```
Subtree with 3 leaves:
- Leaf 1: Gini = 0.1
- Leaf 2: Gini = 0.2
- Leaf 3: Gini = 0.15

R(T_t) = 0.1 + 0.2 + 0.15 = 0.45
```

From lecture:
> "So far these were the **pure impurities** that the **node T** and the **sub tree**. **Some of the impurities** it to the **leaf nodes**."

---

## 6. Effective Error Calculation

### 📊 Adding Complexity Terms

From lecture:
> "Then, now let's think about **what happens** if you **add this complexity term**, or **regularization term**. Each case, we can **add this regularization term**."

### 🔍 Node t (Before Split)

From lecture:
> "For **node t**, we can say the **effective error** at the **node t** is its **plane impurity** plus the **complexity term** but remember, it was **before the split**. Our **complex that term**, the **number of terminal node is just one** here so we're going to just **add Alpha** here."

**Formula for node t (before split):**
\[
R_{\alpha}(t) = R(t) + \alpha \times 1 = R(t) + \alpha
\]

**Explanation:** 
- Node t itself is a terminal node (if we don't split)
- So |T_t| = 1
- Complexity penalty = α × 1 = α

### 🌳 Subtree T_t (After Split)

From lecture:
> "Let's think about the **subtree hall itself**. **Sub tree**, the **effective area of the subtree**, it's going to be the **impurity of the sub-tree** which again is a **sum of all the impurities at the terminal nodes** plus the **complexity parameter Alpha** times the **complexity of the tree** of that subtree, which is **number of leaf nodes**, in this case is **three**."

**Formula for subtree T_t:**
\[
R_{\alpha}(T_t) = R(T_t) + \alpha |T_t|
\]

**Example with 3 leaves:**
\[
R_{\alpha}(T_t) = R(T_t) + \alpha \times 3
\]

---

## 7. Finding α-Effective

### 🎯 The Equality Point

From lecture:
> "At certain point if we **pick the Alpha carefully** here, then we may be able to **set these two numbers to be equal**. **Error of this node before split** and **error of the entire sub-tree** below that node so the **Alpha that makes this possible** is called **Alpha effective**."

**Concept:** Find α where it's equally good to:
- Keep node t as leaf (don't split)
- Keep entire subtree (do split)

**Equation:**
\[
R_{\alpha}(t) = R_{\alpha}(T_t)
\]

Substituting:
\[
R(t) + \alpha = R(T_t) + \alpha |T_t|
\]

### 📐 Solving for α-Effective

From lecture:
> "This **Alpha effective** is actually a **number** that will **set the threshold** when we can **split further**. If you do the **algebra using this formula**, then we get **this formula**."

**Derivation:**
\[
R(t) + \alpha = R(T_t) + \alpha |T_t|
\]
\[
R(t) - R(T_t) = \alpha |T_t| - \alpha
\]
\[
R(t) - R(T_t) = \alpha (|T_t| - 1)
\]
\[
\alpha_{eff}(t) = \frac{R(t) - R(T_t)}{|T_t| - 1}
\]

**Final formula:**
\[
\alpha_{eff}(t) = \frac{R(t) - R(T_t)}{|T_t| - 1}
\]

### 🔍 Interpretation

From lecture:
> "So great so we can **define a threshold** at the **node t** that tells **whether we should split or not**."

**Meaning of α_eff(t):**
- Measures the "strength" of the split at node t
- **Small α_eff:** Weak split (candidate for pruning)
- **Large α_eff:** Strong split (keep this branch)

---

## 8. Numerical Example

### 📊 Example Calculation

**Given:**
- Node t before split: R(t) = 0.5 (Gini impurity)
- After split creates subtree with 3 leaves:
  - Leaf 1: R = 0.1
  - Leaf 2: R = 0.2  
  - Leaf 3: R = 0.15
  
**Calculate R(T_t):**
\[
R(T_t) = 0.1 + 0.2 + 0.15 = 0.45
\]

**Calculate α_eff:**
\[
\alpha_{eff}(t) = \frac{0.5 - 0.45}{3 - 1} = \frac{0.05}{2} = 0.025
\]

**Interpretation:** 
- The split at node t reduces impurity by 0.05
- But creates 2 additional terminal nodes (3 - 1 = 2)
- Impurity reduction per additional terminal node = 0.025
- This is the "cost" of complexity at this node

---

## 9. The Pruning Algorithm

### 🔄 Iterative Process

From lecture:
> "With that, how do we do the **pruning**? We can **calculate all Alpha effective** for **intermediate nodes**. **Alpha effective** here, here and **every intermediate node** except **terminal node**, will have its **own Alpha effective**, and their **numbers can be different**."

**Step 1:** Calculate α_eff for all internal nodes

```
Tree T₀:
  Node A: α_eff = 0.015
  Node B: α_eff = 0.032
  Node C: α_eff = 0.008  ← Smallest!
  Node D: α_eff = 0.021
  ...
```

### 🎯 Selecting Nodes to Prune

From lecture:
> "We have that **list of that Alpha effective** for all the intermediate nodes and then we **pick the one that's smallest** and then **remove it** and we can **iteratively remove** the **smallest or Alpha effective**."

**Step 2:** Find minimum α_eff
\[
\alpha_{min} = \min_{t} \alpha_{eff}(t)
\]

**Step 3:** Prune that node and its subtree

From lecture:
> "For example if this **node had a smallest Alpha effective** among these all other intermediate nodes, then we can **remove this node** as well as **its subtree** like that and let's say this one wants the **next term**. We **get rid of that**. **Get rid of this** and we **repeat** until we **meet some criteria**."

### 🔄 Iteration

**Process:**
```
Iteration 1: Remove node with smallest α_eff
Iteration 2: Recalculate α_eff for remaining nodes
Iteration 3: Remove next smallest α_eff
...
Continue until stopping criterion met
```

---

## 10. Stopping Criterion: CCP Alpha

### 🛑 When to Stop Pruning

From lecture:
> "When do we **stop the pruning**? We set some **threshold called Alpha CCP** or **CCP Alpha**, such that we **start pruning** when all of the **Alpha effectives are bigger** than this number, then means the **link strength is strong enough** that we **don't need to prune anymore**."

**Stopping rule:**
\[
\text{Stop when: } \min(\alpha_{eff}) > \alpha_{CCP}
\]

From lecture:
> "Again, these are **social value** is called the **CCP Alpha in the SKLearn library**."

**In sklearn:** Parameter name is `ccp_alpha`

### 🔍 Interpretation of CCP Alpha

From lecture:
> "Again, this **Alpha, its activity is a measure of strength** of that link. If the **Alpha effective is bigger** than means the **split at the node was worth** so we **don't prune that link**. If the **Alpha effective is smaller** than **certain threshold**, that means it was **not worth bleeding** [splitting] so we just **proved** [pruned] that branch."

**Decision rule:**
- **α_eff(t) > α_CCP:** Keep the split (strong link)
- **α_eff(t) < α_CCP:** Prune the split (weak link)

**Effect of α_CCP value:**
```
α_CCP = 0:      No pruning (keep full tree)
α_CCP = 0.001:  Light pruning (remove weakest branches)
α_CCP = 0.01:   Moderate pruning
α_CCP = 0.1:    Heavy pruning (keep only strongest branches)
```

---

## 11. sklearn Implementation

### 💻 Using ccp_alpha Parameter

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model with pruning
clf = DecisionTreeClassifier(ccp_alpha=0.01, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)

print(f"Tree depth: {clf.get_depth()}")
print(f"Number of leaves: {clf.get_n_leaves()}")
print(f"Train accuracy: {train_score:.4f}")
print(f"Test accuracy: {test_score:.4f}")
```

### 📊 Finding Optimal α_CCP

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# Get pruning path
clf = DecisionTreeClassifier(random_state=42)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
impurities = path.impurities

# Train trees with different alphas
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

# Evaluate
train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

# Plot
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Scores vs alpha
ax[0].plot(ccp_alphas, train_scores, marker='o', label='Train', drawstyle="steps-post")
ax[0].plot(ccp_alphas, test_scores, marker='s', label='Test', drawstyle="steps-post")
ax[0].set_xlabel('Alpha (CCP)')
ax[0].set_ylabel('Accuracy')
ax[0].set_title('Accuracy vs Alpha')
ax[0].legend()
ax[0].grid(True)

# Plot 2: Tree complexity vs alpha
node_counts = [clf.tree_.node_count for clf in clfs]
depth_counts = [clf.tree_.max_depth for clf in clfs]

ax[1].plot(ccp_alphas, node_counts, marker='o', label='Total Nodes', drawstyle="steps-post")
ax[1].plot(ccp_alphas, depth_counts, marker='s', label='Max Depth', drawstyle="steps-post")
ax[1].set_xlabel('Alpha (CCP)')
ax[1].set_ylabel('Count')
ax[1].set_title('Tree Complexity vs Alpha')
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()

# Select best alpha
best_idx = np.argmax(test_scores)
best_alpha = ccp_alphas[best_idx]
print(f"Best alpha: {best_alpha:.6f}")
print(f"Best test accuracy: {test_scores[best_idx]:.4f}")
```

---

## 12. Complete Example with Comparison

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 1. Full unpruned tree
print("="*50)
print("UNPRUNED TREE")
print("="*50)
unpruned = DecisionTreeClassifier(random_state=42)
unpruned.fit(X_train, y_train)

print(f"Depth: {unpruned.get_depth()}")
print(f"Leaves: {unpruned.get_n_leaves()}")
print(f"Train accuracy: {unpruned.score(X_train, y_train):.4f}")
print(f"Test accuracy: {unpruned.score(X_test, y_test):.4f}")

# 2. Get pruning path
path = unpruned.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]  # Remove last alpha (empty tree)

# 3. Train with each alpha
print(f"\nTesting {len(ccp_alphas)} different alpha values...")
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

# 4. Compute scores
train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]
depths = [clf.get_depth() for clf in clfs]
leaves = [clf.get_n_leaves() for clf in clfs]

# 5. Find best alpha
best_idx = np.argmax(test_scores)
best_alpha = ccp_alphas[best_idx]
best_clf = clfs[best_idx]

print("\n" + "="*50)
print("PRUNED TREE (BEST ALPHA)")
print("="*50)
print(f"Best alpha: {best_alpha:.6f}")
print(f"Depth: {best_clf.get_depth()}")
print(f"Leaves: {best_clf.get_n_leaves()}")
print(f"Train accuracy: {best_clf.score(X_train, y_train):.4f}")
print(f"Test accuracy: {best_clf.score(X_test, y_test):.4f}")

# 6. Comparison
print("\n" + "="*50)
print("COMPARISON")
print("="*50)
print(f"Depth reduction: {unpruned.get_depth()} → {best_clf.get_depth()}")
print(f"Leaves reduction: {unpruned.get_n_leaves()} → {best_clf.get_n_leaves()}")
print(f"Test accuracy change: {unpruned.score(X_test, y_test):.4f} → {best_clf.score(X_test, y_test):.4f}")

# 7. Visualize results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Accuracy vs Alpha
axes[0, 0].plot(ccp_alphas, train_scores, marker='o', label='Train', alpha=0.7)
axes[0, 0].plot(ccp_alphas, test_scores, marker='s', label='Test', alpha=0.7)
axes[0, 0].axvline(best_alpha, color='r', linestyle='--', label=f'Best α={best_alpha:.4f}')
axes[0, 0].set_xlabel('Alpha (CCP)')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('Model Performance vs Alpha')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xscale('log')

# Plot 2: Tree Complexity vs Alpha
axes[0, 1].plot(ccp_alphas, depths, marker='o', label='Depth', alpha=0.7)
axes[0, 1].plot(ccp_alphas, leaves, marker='s', label='Leaves', alpha=0.7)
axes[0, 1].axvline(best_alpha, color='r', linestyle='--', label=f'Best α')
axes[0, 1].set_xlabel('Alpha (CCP)')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Tree Complexity vs Alpha')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xscale('log')

# Plot 3: Unpruned Tree (first 3 levels)
plot_tree(unpruned, max_depth=3, ax=axes[1, 0],
         feature_names=data.feature_names,
         class_names=data.target_names,
         filled=True, fontsize=8)
axes[1, 0].set_title('Unpruned Tree (First 3 Levels)')

# Plot 4: Pruned Tree (first 3 levels)
plot_tree(best_clf, max_depth=3, ax=axes[1, 1],
         feature_names=data.feature_names,
         class_names=data.target_names,
         filled=True, fontsize=8)
axes[1, 1].set_title(f'Pruned Tree (α={best_alpha:.4f}, First 3 Levels)')

plt.tight_layout()
plt.show()
```

---

## 13. Summary

### 🎯 Key Concepts

**1. Pruning vs Early Stopping:**
- Early stopping: Greedy, may miss good future splits
- Pruning: Grow full tree, then remove weak branches (better!)

**2. Cost-Complexity Measure:**
\[
R_{\alpha}(T) = R(T) + \alpha |T|
\]
- Balances impurity and complexity
- Similar to regularization in other models

**3. α-Effective Formula:**
\[
\alpha_{eff}(t) = \frac{R(t) - R(T_t)}{|T_t| - 1}
\]
- Measures "strength" of split
- Small α_eff = weak split (prune it)
- Large α_eff = strong split (keep it)

**4. Pruning Algorithm:**
1. Grow full tree T₀
2. Calculate α_eff for all internal nodes
3. Prune node with smallest α_eff
4. Repeat until min(α_eff) > α_CCP

**5. CCP Alpha Parameter:**
- `ccp_alpha=0`: No pruning
- `ccp_alpha>0`: More pruning
- Find optimal via cross-validation

### 📋 Advantages of Pruning

1. **Better than early stopping:** Can keep branches that start weak but become strong
2. **Principled approach:** Based on cost-complexity tradeoff
3. **Tunable:** Single parameter (α_CCP) to control
4. **Automatic:** sklearn handles the complex calculations

### 🔧 Best Practices

1. **Always try pruning** for decision trees
2. **Use cross-validation** to find optimal α_CCP
3. **Visualize pruning path** (accuracy vs alpha)
4. **Compare to unpruned** to verify improvement
5. **Check both train and test** scores

---

**End of Lecture Notes - Module 04, Document 5**
