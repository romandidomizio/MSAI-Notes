# Decision Tree Classifier and Metrics (Gini and Entropy)
**CSCA5622 - Module 04**

---

## 📚 Overview

This document covers **Decision Tree Classifiers** and their splitting criteria, focusing on **Gini impurity** and **Entropy/Information Gain**. Topics include metric formulas, intuitive explanations, and detailed calculation examples.

All concepts explained from the lecture transcript.

---

## 1. Decision Tree Classifier Structure

### 🌳 Visual Representation

From lecture:
> "**Decision tree classifier look exactly like decision tree regressor**... it's **binary class classification**... in the **terminal node** will have a **few samples**."

**Key similarity:** Structure identical to regressor (root, intermediate nodes, leaves)

**Key difference:** Leaf nodes contain class labels instead of continuous values

### 🔍 Tree Growth

From lecture:
> "When we **don't stop growing tree** in the middle, it will just **fully grow** until it has a **pure node**, **everything pure** in the terminal node."

**Pure node:** All samples in leaf belong to same class

---

## 2. Splitting Criteria

### 🔄 Regressor vs Classifier

From lecture:
> "So it works **very similar to diary regressors**... except that now the **metric that we use** to calculate this **left box and right box** result is a **gini instead of MSE**."

| Model | Metric | Goal |
|-------|--------|------|
| **Regressor** | MSE, RSS, MAE | Minimize variance |
| **Classifier** | Gini, Entropy | Minimize impurity |

### 📊 Classification Metrics

From lecture:
> "And for **classification**, we have **three choices**... these **three are most popular**."

1. **Gini Impurity:** Measures impurity (most popular)
2. **Entropy:** Measures uncertainty
3. **Information Gain:** Reduction in entropy

---

## 3. Gini Impurity

### 📐 Formula

**Binary case:**
\[
Gini = 2p(1-p)
\]

**General (K classes):**
\[
Gini = 1 - \sum_{k=1}^{K} p_k^2
\]

### 📊 Properties

- **Range:** [0, 0.5] for binary
- **Gini = 0:** Pure node
- **Gini = 0.5:** Maximum impurity (50-50)

From lecture:
> "**Gini function**... in the **binary case**, it's a **symmetric**... **maximum at 50 50 mixture** and then at a **0, at the pure node**."

---

## 4. Entropy

### 📐 Formula

\[
Entropy = -\sum_{k=1}^{K} p_k \log(p_k)
\]

From lecture:
> "**Entropy** is a measure of **uncertainty**... If we **don't know fully** the **uncertainty is maximized**."

### 📊 Properties

- **Using log₂:** Range [0, 1] for binary
- **Using ln:** Range [0, 0.693] for binary
- **Entropy = 0:** Pure node
- **Entropy = max:** Maximum uncertainty (50-50)

---

## 5. Information Gain

### 📐 Formula

\[
IG = Entropy_{parent} - \left(\frac{n_{left}}{n} Entropy_{left} + \frac{n_{right}}{n} Entropy_{right}\right)
\]

**Goal:** Maximize information gain (find split that most reduces entropy)

---

## 6. Intuitive Marble Example

From lecture:
> "**Gini and entropy**, they measure the **same kind of property**, **purity**... you have some **bag**... **goal is to separate them**... **all blues in one bag** and **all red in the other bag**."

### Three Scenarios:

**Perfect Separation (Pure):**
```
Bag 1: [B B B B B] ← 100% Blue
Bag 2: [R R R R R] ← 100% Red
Gini = 0, Entropy = 0
```

**Good Separation:**
```
Bag 1: [B B B B R] ← 80% Blue
Bag 2: [R R R R B] ← 80% Red
Gini ≈ 0.32, Entropy ≈ 0.5
```

**No Separation (Impure):**
```
Bag 1: [B B R R B] ← 50-50
Bag 2: [R R B B R] ← 50-50
Gini = 0.5, Entropy = 1 (max)
```

---

## 7. Gini Calculation Example

### Example 1: Fully Mixed

**Given:** 5 Cats, 5 Tigers (10 total)

\[
Gini = 0.5(1-0.5) + 0.5(1-0.5) = 0.5
\]

### Example 2: After Split

**Left:** 4 Cats, 0 Tigers
\[
Gini_{left} = 1.0(1-1.0) + 0(1-0) = 0
\]

**Right:** 1 Cat, 5 Tigers
\[
Gini_{right} = \frac{1}{6}(1-\frac{1}{6}) + \frac{5}{6}(1-\frac{5}{6}) = \frac{5}{18} \approx 0.278
\]

**Total:**
\[
Gini_{total} = \frac{4}{10}(0) + \frac{6}{10}(0.278) = 0.167
\]

**Improvement:** 0.5 → 0.167 (67% reduction!)

---

## 8. Entropy Calculation Example

### Original Box: 5 Cats, 5 Tigers

**Using log₂:**
\[
Entropy = -0.5\log_2(0.5) - 0.5\log_2(0.5) = 1.0
\]

### After Split

**Left:** 4 Cats, 0 Tigers → Entropy = 0

**Right:** 1 Cat, 5 Tigers
\[
Entropy_{right} = -\frac{1}{6}\log_2(\frac{1}{6}) - \frac{5}{6}\log_2(\frac{5}{6}) \approx 0.65
\]

### Information Gain

\[
IG = 1.0 - (0.4 \times 0 + 0.6 \times 0.65) = 1.0 - 0.39 = 0.61
\]

---

## 9. Practice Problem

### Setup

**Original:** 10 samples (5 Cats, 5 Tigers)

**Three split options:**
- Red: 4 left, 6 right
- Green: 1 left, 9 right  
- Blue: 3 left, 7 right

### Intuitive Answer

From lecture:
> "The **answer is red**... it's going to give the **most number in the pure node in the left box**."

**Red Split:**
- Left: 4 Cats, 0 Tigers (PURE, 4 samples)
- Right: 1 Cat, 5 Tigers (mostly pure)

**Green Split:**
- Left: 0 Cats, 1 Tiger (pure but only 1 sample)
- Right: 5 Cats, 4 Tigers (still very mixed)

**Blue Split:**
- Left: 0 Cats, 3 Tigers (pure, 3 samples)
- Right: 5 Cats, 2 Tigers (moderately mixed)

### Quantitative Results

| Split | Information Gain | Rank |
|-------|-----------------|------|
| **Red** | 0.61 | 1st (Best) |
| **Blue** | 0.40 | 2nd |
| **Green** | 0.11 | 3rd |

**Winner:** Red split maximizes information gain!

---

## 10. Summary

From lecture:
> "We talked about **different metrics** for **regression task and classification task** in the **decision tree split**."

### Metrics Comparison

| Task | Metrics | Purpose |
|------|---------|---------|
| **Regression** | MSE, RSS, MAE | Minimize variance |
| **Classification** | Gini, Entropy | Minimize impurity |

### Key Formulas

**Gini (binary):**
\[
Gini = 2p(1-p)
\]

**Entropy:**
\[
Entropy = -\sum p_k \log(p_k)
\]

**Information Gain:**
\[
IG = Entropy_{parent} - \text{(weighted entropy of children)}
\]

### Decision Process

1. Try all features and thresholds
2. Calculate Gini or Entropy for each split
3. Choose split with minimum Gini or maximum Information Gain
4. Recursively apply to child nodes

---

**End of Lecture Notes - Module 04, Document 3**
