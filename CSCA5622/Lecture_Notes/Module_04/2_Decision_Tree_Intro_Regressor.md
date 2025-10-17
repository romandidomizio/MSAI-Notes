# Decision Tree Introduction and Decision Tree Regressor
**CSCA5622 - Module 04**

---

## 📚 Overview

This document provides a comprehensive introduction to **Decision Trees**, focusing on how they work and specifically on **Decision Tree Regressors**. Topics include tree structure, terminology, splitting criteria, MSE minimization, and practical examples.

All concepts explained from the lecture transcript.

---

## 1. Review: Parametric vs Non-Parametric Models

### 🔍 Context

From lecture:
> "So far we talked about some **examples of parametric models** such as **linear regression and logistic regression** which they have **parameters or coefficients inside** and we used **different metrics to optimize those**, and then we had the **kNN**, for example of **non-parametric model** which does **not have parameter inside**."

### 📊 Model Types Summary

**Parametric Models:**
- Linear Regression: Has β parameters, minimizes MSE
- Logistic Regression: Has weights, minimizes cross-entropy

**Non-Parametric Models:**
- kNN: No parameters, uses distance metric
- Decision Trees: No parameters, uses splitting criteria

### 🎯 Decision Trees Position

From lecture:
> "However we use **distance metric to make a decision** and **decision trees** are another **non-parametric method** which is a **little bit more complex than kNN**."

---

## 2. What Is a Decision Tree?

### 🍄 The Mushroom Example

From lecture:
> "Let's take an example. These are the **photos of two different mushroom** and **one of them is edible** and the **other one is really poisonous**, so **which one do you think it's edible**? It is **difficult to tell** because they **look very similar** and in fact the **upper one is edible** and the **lower one is called the Death Cap**."

**Problem:** Need to distinguish between edible and poisonous mushrooms that look similar

**Challenge:** Visual similarity makes direct classification difficult

### 🌳 Tree Structure for Mushroom Classification

From lecture:
> "A **Decision Tree may look like this**. Let's say we have **different samples of mushroom data**, and then from this **first node** it's asking some **criteria** whether it's **large or not**, so let's say this one is large and then **classify to largely equals yes**, this one as well and these two are **not large**, so we will **arrive to this node**."

**Tree structure:**
```
                    [Root Node]
                   Is it large?
                   /          \
              YES /            \ NO
                 /              \
          [Node 2]           [Node 3]
        Is it yellow?      Is it spotted?
         /        \           /        \
    YES /          \ NO   YES /          \ NO
       /            \        /            \
  Poisonous      Edible  [Node 4]      Edible
                        Foul smell?
                         /        \
                    YES /          \ NO
                       /            \
                  Poisonous      Edible
```

### 📋 Step-by-Step Classification

From lecture:
> "Let's say we call this **node 2** and this **node 3**. From the **Node 2**, there is another criteria where we ask questions whether it's **yellow or not**. This one is **not yellow**, so end up the **edible** and this one is **yellow**, so therefore it's **poisonous**."

**Example 1: Large mushroom**
```
Start → Large? YES → Yellow? NO → Classification: Edible
```

**Example 2: Large yellow mushroom**
```
Start → Large? YES → Yellow? YES → Classification: Poisonous
```

From lecture:
> "From the **Node 3**, both of them are **spotted**, so we'll go to **Node 4** asking whether they have **foul smell or not** and let's say this one had a **foul smell**, therefore it's **poisonous** and this one **did not**, so therefore it is **edible**."

**Example 3: Small mushroom with foul smell**
```
Start → Large? NO → Spotted? YES → Foul smell? YES → Classification: Poisonous
```

**Example 4: Small mushroom, no foul smell**
```
Start → Large? NO → Spotted? YES → Foul smell? NO → Classification: Edible
```

### 🎯 How Decision Trees Work

From lecture:
> "Decision Tree works like this. It **splits the samples** from each node **depending on their criteria**."

**Key principle:** Sequentially ask yes/no questions about features until reaching a classification

---

## 3. Decision Tree Terminology

### 📊 Node Types

**1. Root Node:**

From lecture:
> "**Node at the top** is called the **root node** and contains **all the samples to begin it**."

**Definition:** The starting point of the tree containing all data

**2. Leaf Nodes (Terminal Nodes):**

From lecture:
> "And then as the samples **travel through these different node splits**, when it **arrives at the terminal nodes**, now that **doesn't split anymore**, those nodes are called the **leaf nodes** and they are **highlighted as green** here."

**Definition:** Final nodes that contain predictions (no further splits)

**3. Intermediate Nodes (Decision Nodes):**

From lecture:
> "And **all other nodes between** they're called the **intermediate nodes**, including **root node**, they also have a **decision criteria** therefore they are **decision nodes**."

**Definition:** Nodes between root and leaves that contain splitting criteria

### 🌳 Visual Structure

```
                [Root Node] ← Decision node (top)
               /           \
              /             \
      [Intermediate]    [Intermediate] ← Decision nodes (middle)
         /    \            /    \
        /      \          /      \
    [Leaf]  [Leaf]   [Leaf]  [Leaf] ← Terminal nodes (bottom)
```

**Summary:**
- **Root node:** Top node with all samples
- **Decision nodes:** Any node with a split criterion (including root)
- **Intermediate nodes:** Decision nodes between root and leaves
- **Leaf nodes:** Terminal nodes with predictions

---

## 4. How Different Models Learn

### 📊 Comparison Across Model Types

From lecture:
> "How the model learns to make a decision? As we mentioned, **linear regression minimize the MSE** to allow to make a decision by **optimizing their parameter values**. Same goes for **logistic regression** except that the **criteria is now cross-entropy**."

**Linear Regression:**
- Has parameters: β₀, β₁, ..., βₚ
- Optimization metric: MSE (Mean Squared Error)
- Method: Gradient descent / closed-form solution

**Logistic Regression:**
- Has parameters: weights and bias
- Optimization metric: Cross-entropy (BCE)
- Method: Gradient descent / Newton's method

From lecture:
> "**kNN has no parameters**, therefore **no optimization**, however uses a **distance metric** to make a decision."

**kNN:**
- Has parameters: None
- Has hyperparameters: K (number of neighbors)
- Decision method: Distance metric (Euclidean, Manhattan)
- No training/optimization needed

From lecture:
> "**Decision Tree** similarly **doesn't have parameters**, however uses **other metrics** such as **MSE for regression task** and **entropy or Gini for classification task**, and they **split nodes** as we've seen before."

**Decision Tree:**
- Has parameters: None
- Has hyperparameters: max_depth, min_samples_split, etc.
- Decision method: Splitting criteria
- Metrics: MSE (regression), Entropy/Gini (classification)

### 📋 Summary Table

| Model | Parameters? | Optimization Metric | Method |
|-------|-------------|---------------------|--------|
| **Linear Regression** | Yes (β) | MSE | Gradient descent |
| **Logistic Regression** | Yes (weights) | Cross-entropy | Gradient/Newton |
| **kNN** | No | Distance | Distance calculation |
| **Decision Tree** | No | MSE/Entropy/Gini | Node splitting |

---

## 5. Decision Tree Regressor: Core Concept

### 🎯 The Goal

From lecture:
> "This **Decision Tree Regressor work like this**, so the goal is to **split the samples into two boxes** such that the **MSE is minimized** as a result of the split."

**Objective:** Partition data into regions where MSE within each region is minimized

### 📊 Basic Setup

From lecture:
> "Let's say I have **different options to split** or more this **different features**. Let's say I have data that has **two features** on me and **six data points** and then I want to **split this into two boxes**, and I **don't know how yet**. They will **minimize the total sum of MSE**."

**Example setup:**
```
Data: 6 samples
Features: X₁, X₂
Target: y (continuous)
Goal: Split into 2 boxes minimizing total MSE
```

### 🔀 Split Options

From lecture:
> "I have a **choice of splitting along X1** or I have a **choice of splitting along X2**, so on **other choice** that we should make is that, let's say I chose **X2 to split**, then **which value of X2 should I split**? Should I split **here or here or here or here**? These all our **decisions will be made** by looking at the **MSE of each split**."

**Two decisions to make:**
1. **Which feature** to split on? (X₁ or X₂)
2. **What threshold** to use? (which value)

---

## 6. Split Selection Process

### 📊 Splitting Along X₁

From lecture:
> "Again, we have **different choices** for making split along **X1**, for example I can split this way and maybe **left and right** and **measure the MSE**, and let's say this split criteria was **A**, then I can see if **split by this split criteria** and measure the **MSE here** and then **sum them up**. I recorded."

**Process for X₁:**

**Split criterion A:** X₁ ≤ a
```
Left box: samples where X₁ ≤ a
Right box: samples where X₁ > a

MSE_left = Σ(yᵢ - ȳ_left)² / n_left
MSE_right = Σ(yᵢ - ȳ_right)² / n_right

Total MSE_A = MSE_left + MSE_right
```

From lecture:
> "And then now I'm going to move **different split criteria**. Let's say this is called **b**, then **x_1 is less than or equal to b**. Then I'm going to **measure this MSE** for **left-hand right boxes** and then **record a thorough MSE**, and I keep this procedure. Let's say this one is **c, d, e**."

**Multiple thresholds for X₁:**
```
Split A: X₁ ≤ a → Total MSE_A
Split B: X₁ ≤ b → Total MSE_B
Split C: X₁ ≤ c → Total MSE_C
Split D: X₁ ≤ d → Total MSE_D
Split E: X₁ ≤ e → Total MSE_E
```

From lecture:
> "Then I have **five different split criteria along x_1 feature**, and then I also **record this MSE**."

### 📊 Splitting Along X₂

From lecture:
> "That's for **x_1**, and I can do the **same for x_2**. Again, it also have **five different scholarly criteria**, and then we can call it **a, b, c, d, e**. Something like that along the **feature x_2**."

**Multiple thresholds for X₂:**
```
Split A: X₂ ≤ a → Total MSE_A'
Split B: X₂ ≤ b → Total MSE_B'
Split C: X₂ ≤ c → Total MSE_C'
Split D: X₂ ≤ d → Total MSE_D'
Split E: X₂ ≤ e → Total MSE_E'
```

### 🎯 Selecting the Best Split

From lecture:
> "As a result, we have **10 different values for MSE** as a result of **10 different split options**. What we want to do is to **inspect these MSE values** and then **pick the one** that makes the **minimized MSE**."

**Decision process:**
```
Compare all 10 MSE values:
MSE_A, MSE_B, MSE_C, MSE_D, MSE_E (for X₁)
MSE_A', MSE_B', MSE_C', MSE_D', MSE_E' (for X₂)

Select: min(all MSE values)
```

From lecture:
> "Let's say this **split criterion gave the smallest MSE** among these **10 different choices**, then, now it becomes my **split criterion for my root node**."

**Example result:**
```
If MSE_C is smallest:
Root node criterion: X₁ ≤ c
```

---

## 7. Tree Building Process

### 🌳 After First Split

From lecture:
> "By **root node**, I mean the **first box** that we're given. This is my **root node**, and then as we just saw, let's say this was the **best split**, the **minimized MSE**, then now these **two boxes becomes the splitted node**."

**Before split:**
```
[Root Node]
All 6 samples
```

**After split:**
```
        [Root Node: X₁ ≤ c]
           /          \
          /            \
  [Left Node]      [Right Node]
  3 samples         3 samples
```

From lecture:
> "This node, the **root node had six data points**, and now we have our **split into two boxes**; **left and right**. Each of them has **three samples**, and the **decision criteria at the root node** was **x_1 is less than equal to c**."

### 🔄 Recursive Process

From lecture:
> "If we keep doing this procedure, we're going to reach to some **terminal node** or **stopping criteria**, then the **tree stops there**."

**Process:**
1. Start with all data in root node
2. Find best split (feature + threshold) that minimizes MSE
3. Create two child nodes
4. **Repeat steps 2-3** for each child node
5. Stop when reaching stopping criteria

**Stopping criteria:**
- Maximum depth reached
- Minimum samples per node
- No further MSE reduction
- All samples in node have same target value

---

## 8. Decision Tree Classifier Preview

From lecture:
> "This is **how the decision tree regressor works** and the **decision tree classifier works similar way**, except that it's **not MSE**, but uses some **other metric**. We'll talk about that later."

**Key difference:**
- **Regressor:** Uses MSE to evaluate splits
- **Classifier:** Uses Entropy or Gini impurity

**Similarity:** Both use recursive binary splitting to build tree

---

## 9. Real Example: Faculty Salary Dataset

### 📊 Dataset Description

From lecture:
> "This dataset is called **faculty salary dataset**, **recording faculty salary at all the '90s**, and that has **codes to predict the assistant professor's salary**. It has **four features and 50 samples**, both for **simplicity to visualize**. I only use the **two features**, and then **depth equals two**."

**Dataset characteristics:**
- Target: Assistant professor salary
- Original features: 4
- Used features: 2 (for visualization)
- Samples: 50
- Tree depth: 2 (limited)

### 🌳 Tree Depth Concept

From lecture:
> "**Depth in the tree** means that **how many levels to recall to grow the tree**. This is **depth equals zero** at the root node, and this is **depth 1**, and this is **depth 2**."

**Depth levels:**
```
Depth 0:     [Root]                    ← 1 node
            /      \
Depth 1:  [N1]    [N2]                 ← 2 nodes
         /  \     /  \
Depth 2: [L1][L2][L3][L4]              ← 4 leaf nodes
```

**Depth limit:**

From lecture:
> "If you **don't specify the depth's limit**, the tree will **grow until** it has **only one sample at the leaf node**, or if there is **another stoping criteria**, it will **stop there**."

**Without depth limit:**
- Tree grows until each leaf has 1 sample
- Or until no further splits improve MSE
- Can lead to overfitting

---

## 10. Faculty Salary Example: Step-by-Step

### 📊 Root Node (Depth 0)

From lecture:
> "Anyway, this is the **original data**, that our **salary mean value 43,000**, and then you had a **50 samples**. We have **50 samples** and our value here at the **root node**."

**Root node state:**
```
[Root Node]
n_samples: 50
mean_salary: $43,000
```

### 🔍 Finding First Split

From lecture:
> "As we saw before, we will **find all the split criteria**. That means that the decision tree will **inspect all these split points** along **x_0**, and then along **x_1**, so it's going to **split in the middle** between the **two samples**. In the middle here, **all the way to here**. Then it's going to **measure MSEs** and then we'll **figure out which one to split**."

**Process:**
1. For feature x₀: Try all possible split points
2. For feature x₁: Try all possible split points
3. Calculate MSE for each split
4. Select split with minimum MSE

From lecture:
> "That's **how it measures MSE** here."

### 🎯 Best Split at Root

From lecture:
> "Then as a result, it found that the **splitting x_1, 54.95**. At this point, we'll make the **MSE the lowest** from the root node."

**Selected split:**
```
Feature: x₁
Threshold: 54.95
Criterion: x₁ ≤ 54.95?
```

### 🌳 After First Split (Depth 1)

From lecture:
> "Depending on the **answer to this criteria**, we'll have these **two resulting children nodes**. Therefore, it looks like this. There's **two nodes**. This is **R1T**, and this is **R1F**."

**Tree structure:**
```
        [Root: x₁ ≤ 54.95?]
           /          \
      YES /            \ NO
         /              \
    [R1T]              [R1F]
  (True branch)    (False branch)
```

From lecture:
> "Then each of the boxes have **mean values at this value**; **41,049.8** and then have a **simple number**. **Eight samples and 12 samples** here."

**Node details:**
```
[R1T] (x₁ ≤ 54.95)          [R1F] (x₁ > 54.95)
n_samples: 8                 n_samples: 42
mean_salary: $41,049.80      mean_salary: [not stated]
```

### 🔍 Second Level Splits (Depth 2)

From lecture:
> "From **each node** we'll also **find another split criteria** for the next split. For example, this **R1F box split criteria** was **splitting at feature x_0** at the value **84.25**, and then it's going to give **split**, and then it **leads to these two boxes**. Each of them have a **mean value** of **46,000 and 42,000**."

**R1F node splits:**
```
    [R1F: x₀ ≤ 84.25?]
       /          \
  YES /            \ NO
     /              \
[Leaf]            [Leaf]
n: ?              n: ?
mean: $46,000     mean: $42,000
```

From lecture:
> "If we further do that, the **same procedure** for this node will **end up with this result**."

**Complete tree structure:**
```
                [Root: x₁ ≤ 54.95?]
                   /          \
              YES /            \ NO
                 /              \
          [R1T: split]      [R1F: x₀ ≤ 84.25?]
            /    \              /          \
           /      \        YES /            \ NO
     [Leaf]    [Leaf]        /              \
                        [Leaf]            [Leaf]
                     mean: $46K        mean: $42K
```

---

## 11. How MSE Is Calculated

### 📐 MSE Formula

**For a single node:**
\[
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2
\]

Where:
- n = number of samples in node
- yᵢ = actual value of sample i
- ȳ = mean value in node

**For a split:**
\[
MSE_{total} = \frac{n_{left}}{n} \times MSE_{left} + \frac{n_{right}}{n} \times MSE_{right}
\]

Or equivalently (sum of MSEs):
\[
MSE_{total} = MSE_{left} + MSE_{right}
\]

### 📊 Example Calculation

**Before split:**
```
Node has 6 samples with values: [10, 12, 15, 20, 22, 25]
Mean: ȳ = 17.33
MSE = [(10-17.33)² + (12-17.33)² + (15-17.33)² + 
       (20-17.33)² + (22-17.33)² + (25-17.33)²] / 6
    = [53.78 + 28.41 + 5.43 + 7.11 + 21.78 + 58.78] / 6
    = 175.29 / 6
    = 29.22
```

**After split (X ≤ 15):**
```
Left node: [10, 12, 15]
Mean: ȳ_left = 12.33
MSE_left = [(10-12.33)² + (12-12.33)² + (15-12.33)²] / 3
         = [5.43 + 0.11 + 7.11] / 3
         = 4.22

Right node: [20, 22, 25]
Mean: ȳ_right = 22.33
MSE_right = [(20-22.33)² + (22-22.33)² + (25-22.33)²] / 3
          = [5.43 + 0.11 + 7.11] / 3
          = 4.22

Total MSE = 4.22 + 4.22 = 8.44
```

**Improvement:** 29.22 → 8.44 (reduced by 71%)

---

## 12. Prediction with Decision Tree Regressor

### 🔍 How Predictions Are Made

**For a new sample:**
1. Start at root node
2. Check split criterion
3. Go to left or right child based on answer
4. Repeat until reaching a leaf node
5. Predict the mean value of that leaf

### 📊 Example Prediction

**Tree structure:**
```
        [Root: x₁ ≤ 54.95?]
           /          \
      YES /            \ NO
         /              \
    [Leaf]          [Node: x₀ ≤ 84.25?]
  mean: $41K           /          \
                  YES /            \ NO
                     /              \
                [Leaf]            [Leaf]
              mean: $46K        mean: $42K
```

**New sample:** x₁ = 60, x₀ = 90

**Prediction process:**
```
Step 1: x₁ ≤ 54.95? → NO (60 > 54.95)
        Go right →

Step 2: x₀ ≤ 84.25? → NO (90 > 84.25)
        Go right →

Step 3: Reach leaf node
        Prediction: $42,000
```

---

## 13. Key Properties of Decision Tree Regressor

### ✅ Advantages

1. **No assumptions about data distribution**
   - Non-parametric
   - Can capture non-linear relationships

2. **Interpretable**
   - Easy to visualize
   - Can explain decisions

3. **Handles mixed data types**
   - Numerical and categorical features
   - No need for normalization

4. **Automatic feature interaction**
   - Captures interactions naturally through splits

5. **Fast prediction**
   - O(log n) time complexity

### ❌ Disadvantages

1. **Prone to overfitting**
   - Especially with deep trees
   - Need to limit depth or prune

2. **High variance**
   - Small changes in data can change tree structure
   - Unstable

3. **Greedy algorithm**
   - Locally optimal splits
   - May not find globally optimal tree

4. **Prediction is piecewise constant**
   - Cannot extrapolate beyond training range
   - Creates step-like predictions

---

## 14. Hyperparameters for Control

### 🔧 Important Hyperparameters

**1. max_depth**
- Maximum depth of tree
- Controls complexity directly
- Typical values: 3-10

**2. min_samples_split**
- Minimum samples required to split a node
- Prevents splitting on very small groups
- Typical values: 2-20

**3. min_samples_leaf**
- Minimum samples required in a leaf node
- Smooths the model
- Typical values: 1-10

**4. max_leaf_nodes**
- Maximum number of leaf nodes
- Alternative to max_depth
- Limits total complexity

**5. min_impurity_decrease**
- Minimum decrease in MSE required for split
- Split only if MSE reduction exceeds this
- Helps prevent overfitting

---

## 15. Summary

From lecture:
> "But this was a **simple example** for how a **decision tree regressor works**. In the next video, we'll talk about **decision tree classifier**."

### 🎯 Key Concepts

**1. Decision Tree Structure**
- Root node: All samples start here
- Decision nodes: Contain split criteria
- Leaf nodes: Contain predictions (mean values)

**2. Non-Parametric Nature**
- No learnable parameters (β, weights, etc.)
- Uses splitting criteria instead
- Hyperparameters control tree growth

**3. Splitting Process**
- Try all features and thresholds
- Calculate MSE for each potential split
- Choose split that minimizes total MSE
- Recursively apply to child nodes

**4. MSE Minimization**
- Goal: Minimize within-node variance
- Formula: MSE = Σ(yᵢ - ȳ)² / n
- Total MSE = MSE_left + MSE_right

**5. Prediction**
- Traverse tree based on feature values
- Predict mean of leaf node reached
- Fast: O(log n) for balanced trees

**6. Tree Depth**
- Depth 0: Root node
- Depth k: k levels from root
- Deeper trees = more complex = more prone to overfit

### 📋 Comparison with Other Models

| Aspect | Decision Tree | Linear Regression | kNN |
|--------|--------------|-------------------|-----|
| **Parameters** | None | Yes (β) | None |
| **Metric** | MSE (for splits) | MSE (for optimization) | Distance |
| **Interpretability** | High (visual tree) | Medium (coefficients) | Low |
| **Handling non-linearity** | Excellent | Poor | Good |
| **Overfitting risk** | High (deep trees) | Low (simple model) | High (small K) |

### 🔧 Best Practices

1. **Limit tree depth** to prevent overfitting (start with 3-5)
2. **Use validation set** to tune hyperparameters
3. **Visualize tree** to understand decisions
4. **Consider ensemble methods** (Random Forest, Gradient Boosting) for better performance
5. **Check feature importance** to understand which features drive splits

### 🎓 Next Steps

The next lecture will cover **Decision Tree Classifiers**, which use:
- **Entropy** or **Gini impurity** instead of MSE
- **Majority voting** in leaves instead of mean
- Similar recursive splitting process

---

**End of Lecture Notes - Module 04, Document 2**
