# Boosting Introduction

**Lecture**: Module 5, Lecture 2  
**Course**: CSCA5622  
**Topic**: Introduction to Boosting, Sequential Ensemble Methods

---

## Table of Contents
1. [Review: The Problem with Trees](#1-review-the-problem-with-trees)
2. [Parallel vs Sequential Ensembling](#2-parallel-vs-sequential-ensembling)
3. [The Core Idea of Boosting](#3-the-core-idea-of-boosting)
4. [The Scientist Analogy](#4-the-scientist-analogy)
5. [The Boosting Algorithm](#5-the-boosting-algorithm)
6. [Graphical Representation](#6-graphical-representation)
7. [Shrinkage and Learning Rate](#7-shrinkage-and-learning-rate)
8. [Variants: AdaBoost and Gradient Boosting](#8-variants-adaboost-and-gradient-boosting)
9. [Python Implementation](#9-python-implementation)
10. [Practice Problems](#10-practice-problems)

---

## 1. Review: The Problem with Trees

### Decision Trees as Weak Learners

The lecturer begins: "Previously we talked about the trees that have a problem, that they are **weak learner** and they can **overfit very easily**."

**Problems with Single Trees**:
- Weak learner (not very accurate)
- High variance (overfits easily)
- Unstable (sensitive to noise)

### Previous Solution: Parallel Ensembling

**Bagging**: "The first idea we use to address this issue was **let's try to untangle them by introducing diversity tree**, which were trained on **different subsets of data**."

**Random Forest**: "On top of that, we also add an idea that we can **further decorrelate the trees**."

**Key Differences**:
- **Bagging**: Random sample **data** (rows)
- **Random Forest**: Random sample **data AND features** (rows and columns)

**Both are Parallel**: "They are **parallel ensembling method**, which means the training of each tree on different subsets of data, **they can be trained at the same time**."

### Performance Hierarchy

"We also showed that the **performance increased dramatically by ensembling trees**."

**Results**:
- Single tree → Bagging: "**I can make this huge difference**"
- Bagging → Random Forest: "**another performance increase**"

---

## 2. Parallel vs Sequential Ensembling

### Introducing Sequential Ensembling

"We're going to introduce a **second ensembling method**, which is a **sequential ensembling**. Whereas previously we talked about **parallel ensembling**, so the **sequential ensembling is called boosting**."

**Key Difference**:
- **Parallel**: Trees trained independently, simultaneously
- **Sequential**: Trees trained one after another, each depending on previous

### Different Philosophy

"Boosting also **solve the same problem** that trees are weak learner and trees overfit. But instead of **diversifying and averaging those different many trees**, we're going to **make single tree stronger learner**."

---

## 3. The Core Idea of Boosting

### Growing Trees Incrementally

**The Strategy**: "We're going to **grow a small stump at a time to fit the error from the previous stage**. Then we're going to **grow another tree in the next stage to fit the error from the previous stage**."

**Key Terms**:
- **Stump**: Very small tree (1-2 levels deep)
- **Error/Residual**: Difference between prediction and true value
- **Stage**: Each iteration where we add a new tree

**Fundamental Principle**: Rather than one large tree solving everything, build many small trees where each corrects mistakes of previous trees.

---

## 4. The Scientist Analogy

### The Big Problem

"You can think about this analogy. When we have a **big problem like this**..."

### Sequential Problem Solving

**First Scientist**: "Maybe the **first scientists will look at it and quickly solve the problem by this much** and leave this problem."
- Solves partial problem
- Leaves remaining error

**Second Scientist**: "The **second scientists will ignore all this**, but only will get this part. **Will focus on this part**."
- Ignores solved part
- Focuses only on remaining error

**Third Scientist**: "The **third scientists or investigator will look at it and then solve more problems** and then **reduce the gap of this error gradually**."

### Applying to Trees

"We can do the same with the **small tree** instead of growing **large tree that try to solve this big problem all at once**."

**Process**:
1. **First tree**: Solves part of problem, leaves residual
2. **Second tree**: "**only look at this error and try to solve it**"
3. **Third tree**: "**even further reduce the gap of the error**"

### Definition

"This process is called **boosting**. **Boosting just means** that we will **make one single tree to strong learner** by **growing the tree slowly a link or two at a time**."

**Contrast**:
- **Single Tree**: "grown to the **maximum depth**...try to **solve the problem all at once**"
- **Boosting**: "grow **very simple and very little one or two depths at a time**"

**Final Model**: "Our **final model will be sum of these small trees**."

---

## 5. The Boosting Algorithm

### Step 0: Initialization

"We're going to **initialize our model to zero**. That means our **model doesn't know anything about our data**."

$$f_0(x) = 0$$

"Let's say our **error is as big as the label**."

$$r_0 = y - f_0(x) = y$$

### Step 1: Iteration Loop

"Then we're going to **iterate for B times**."

#### Substep 1a: Fit Stump to Residuals

"They will try to **fit a stump in the stage B to train data**. The data x and then the **label is now the residual**."

$$h_b = \text{fit}(X, r_{b-1})$$

"In the first iteration, **this residual is the same as y**, so we tried to **fit the y first**."

#### Substep 1b: Update Model with Shrinkage

"We're going to have **our model equals our stump times some constant**."

$$f_b(x) = f_{b-1}(x) + \nu \cdot h_b(x)$$

"This **constant is less than one**." ($0 < \nu < 1$)

"That means we will **add the stump model to our whole model by certain fraction**. The reason why is that we want to **consider our new model conservatively**."

#### Substep 1c: Update Residuals

"We're going to **update the residual in the current stage** that our **residual is also from the previous residual minus the shrink the prediction**."

$$r_b = r_{b-1} - \nu \cdot h_b(x)$$

Or equivalently: $r_b = y - f_b(x)$

### Step 2: Final Model

"After we **repeat B times**, the **final output model will be the sum of this shrinked stump models**."

$$f(x) = \sum_{b=1}^{B} \nu \cdot h_b(x)$$

### Algorithm Summary

```
Algorithm: Generic Boosting

Initialize: f_0(x) = 0, r_0 = y

For b = 1 to B:
  1. Fit stump: h_b = fit(X, r_{b-1})
  2. Update: f_b(x) = f_{b-1}(x) + ν·h_b(x)
  3. Residuals: r_b = y - f_b(x)

Output: f(x) = Σ ν·h_b(x)
```

---

## 6. Graphical Representation

### Stage-by-Stage Flow

"Graphically it looks like this."

#### Stage 1

"**Here's the data** and then it **feeds to our first stump model**. Then the **stump model will predict the prediction**. In the first stage it will be **compared against the label**."

```
Data (X) → Stump 1 → Prediction (ŷ₁)
                           ↓
True Label (y) → Comparison → Residual (r₁)
```

"Then it's **difference**...we will get the **residual from the first stage**."

$$r_1 = y - \nu \cdot h_1(X)$$

#### Stage 2

"From the second stage, we **build on other stump model and try to fit the data**."

```
Data (X), Residual (r₁) → Stump 2 → Prediction (r̂₁)
                                          ↓
                          r₁ → Comparison → Residual (r₂)
```

#### Stage 3 and Beyond

"We'll **continue that with the third stump model**...and we're going to have **another residual from the third stage and so on**."

**Complete Flow**:
```
Stage 1: X,y → h₁ → r₁
Stage 2: X,r₁ → h₂ → r₂
Stage 3: X,r₂ → h₃ → r₃
  ...
Stage B: X,r_{B-1} → h_B → r_B

Final: f(x) = ν·Σh_b(x)
```

---

## 7. Shrinkage and Learning Rate

### Purpose of Shrinkage

"This **helps our learning slow**. It is something **similar to learning rate**."

**Why Shrinkage Works**:
- Prevents aggressive overfitting
- Conservative updates lead to better generalization
- Allows many weak learners to contribute

### Choosing Learning Rate

**Small ν (e.g., 0.01)**:
- Pros: Better generalization, smoother model
- Cons: Needs many iterations (large B)

**Large ν (e.g., 0.5)**:
- Pros: Faster convergence (fewer trees needed)
- Cons: Risk of overfitting, less smooth

### Mathematical Evolution

Starting with $r_0 = y$:

$$r_b = y - \nu \sum_{i=1}^{b} h_i(x)$$

As $b$ increases, residual should decrease.

---

## 8. Variants: AdaBoost and Gradient Boosting

### Generic Boosting Framework

"We just have showed the **generic boosting algorithm**, which I **iteratively fit the small model to residuals from the previous stage** and then we **add up all these small models with some shrinkage**."

### AdaBoost (Adaptive Boosting)

"**AdaBoost use exponential loss instead of just the residual**."

**Exponential Loss**: $L(y, f(x)) = \exp(-y \cdot f(x))$ for $y \in \{-1, +1\}$

**Sample Weighting**: "**AdaBoost to also use different weighting to the data points**. You cannot do **better performance by weighting more to the data points or data sample that were previously misclassified**."

**How It Works**:
- Start with equal weights
- After each iteration:
  - Increase weights for misclassified samples
  - Decrease weights for correct samples
- Next tree focuses on hard examples

**Characteristics**:
- Primarily for classification
- Adapts weights based on errors
- Focuses on difficult examples

### Gradient Boosting

"Another popular method is called **gradient boost**. Gradient boost method try to **fit the gradient term rates to instead of rates to itself**."

**The Gradient Perspective**: Instead of fitting residuals directly, fit the negative gradient of a loss function.

**For Squared Loss** $L = \frac{1}{2}(y - f)^2$:
$$-\frac{\partial L}{\partial f} = y - f = r$$

The residual IS the negative gradient for squared loss!

**For Other Losses**:
- Absolute loss: gradient is $\text{sign}(y - f)$
- Log loss (classification): gradient involves probabilities
- Any differentiable loss function can be used

**Characteristics**:
- General framework for any differentiable loss
- Works for classification and regression
- Most flexible boosting variant

**Future Discussion**: "We're going to **talk about this method in the next videos**."

---

## 9. Python Implementation

### Basic Boosting from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(42)
X, y = make_regression(n_samples=1000, n_features=10, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

class SimpleGradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        
    def fit(self, X, y):
        # Initialize: f_0(x) = 0
        f_pred = np.zeros(len(y))
        
        for b in range(self.n_estimators):
            # Calculate residuals: r = y - f(x)
            residuals = y - f_pred
            
            # Fit tree to residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=b)
            tree.fit(X, residuals)
            self.trees.append(tree)
            
            # Update: f(x) = f(x) + ν·h(x)
            f_pred += self.learning_rate * tree.predict(X)
    
    def predict(self, X):
        predictions = np.zeros(len(X))
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions

# Train and evaluate
model = SimpleGradientBoosting(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Test MSE: {mean_squared_error(y_test, y_pred):.2f}")
```

### Using Scikit-learn

```python
from sklearn.ensemble import GradientBoostingRegressor

gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X_train, y_train)
print(f"Test MSE: {mean_squared_error(y_test, gb.predict(X_test)):.2f}")
print(f"R² Score: {gb.score(X_test, y_test):.4f}")
```

### Visualizing Residual Reduction

```python
# Generate 1D data for visualization
X_viz = np.linspace(0, 10, 100).reshape(-1, 1)
y_viz = np.sin(X_viz).ravel() + np.random.normal(0, 0.3, 100)

f_pred = np.zeros(len(y_viz))
learning_rate = 0.3

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for stage in range(5):
    residuals = y_viz - f_pred
    tree = DecisionTreeRegressor(max_depth=2, random_state=stage)
    tree.fit(X_viz, residuals)
    f_pred += learning_rate * tree.predict(X_viz)
    
    axes[stage].scatter(X_viz, y_viz, alpha=0.5, label='Data')
    axes[stage].plot(X_viz, f_pred, 'r-', linewidth=2, label=f'Stage {stage+1}')
    axes[stage].set_title(f'MSE: {np.mean(residuals**2):.3f}')
    axes[stage].legend()

plt.tight_layout()
plt.show()
```

---

## 10. Practice Problems

### Problem 1: Residual Updates

**Question**: With $\nu = 0.2$, $y = [10, 20, 30, 40]$, $h_1(x) = [8, 22, 28, 35]$:

a) Calculate residuals after stage 1
b) What does stage 2 tree predict?
c) If $h_2(x) = [1.5, -1.0, 2.5, 4.0]$, find $f_2(x)$
d) Calculate residuals after stage 2

**Solution**:

a) $f_1(x) = 0.2 \times [8, 22, 28, 35] = [1.6, 4.4, 5.6, 7.0]$
   $r_1 = [10, 20, 30, 40] - [1.6, 4.4, 5.6, 7.0] = [8.4, 15.6, 24.4, 33.0]$

b) Stage 2 tree predicts $r_1 = [8.4, 15.6, 24.4, 33.0]$

c) $f_2(x) = [1.6, 4.4, 5.6, 7.0] + 0.2 \times [1.5, -1.0, 2.5, 4.0]$
   $= [1.6, 4.4, 5.6, 7.0] + [0.3, -0.2, 0.5, 0.8] = [1.9, 4.2, 6.1, 7.8]$

d) $r_2 = [10, 20, 30, 40] - [1.9, 4.2, 6.1, 7.8] = [8.1, 15.8, 23.9, 32.2]$

### Problem 2: Comparing Shrinkage

**Question**: Compare Model A ($\nu=0.01$, $B=500$) vs Model B ($\nu=0.5$, $B=50$):

a) Which trains faster?
b) Which is more likely to overfit?
c) Which likely has better test error?
d) Total learning for each?

**Solution**:

a) **Model B** trains faster (50 vs 500 trees)

b) **Model B** more likely to overfit (large $\nu$ = aggressive updates)

c) **Model A** likely better test error (smaller $\nu$ = better regularization)

d) Model A: $0.01 \times 500 = 5$; Model B: $0.5 \times 50 = 25$

**Key Insight**: Small $\nu$ with large $B$ provides gradual, stable learning and better generalization.

### Problem 3: Sequential vs Parallel

**Comparison Table**:

| Aspect | Bagging/Random Forest | Boosting |
|--------|----------------------|----------|
| **Training** | Independent, parallel | Sequential, dependent |
| **Tree Depth** | Deep (fully grown) | Shallow (stumps, 1-3 levels) |
| **Aggregation** | Average/Vote | Weighted sum: $\sum \nu \cdot h_b$ |
| **Focus** | Reduce variance | Reduce bias |
| **Parallelization** | Easy | Sequential only |
| **Philosophy** | "Wisdom of crowds" | "Learn from mistakes" |

### Problem 4: Scientist Analogy

**Explanation**:
"Imagine multiple scientists solving a problem sequentially. Each scientist ignores what's been solved and focuses only on the remaining problem. First scientist solves 40%, second solves 25% of original, third continues. Combined solutions give the complete answer."

**Mapping**:
- Each scientist = Small tree $h_b$
- Remaining problem = Residual error $r_b$
- Ignoring solved parts = Training on residuals, not original labels
- Shrinkage $\nu$ = Accepting only a fraction of each solution (conservativeness)

### Problem 5: Gradient Intuition

**Question**: For squared loss $L = \frac{1}{2}(y - f)^2$, show gradient equals residual.

**Solution**:
$$\frac{\partial L}{\partial f} = \frac{\partial}{\partial f}\left[\frac{1}{2}(y - f)^2\right] = (y - f) \cdot (-1) = f - y$$

$$-\frac{\partial L}{\partial f} = y - f = r$$

**The negative gradient equals the residual!** This is why fitting residuals works - we're doing gradient descent in function space.

---

## 11. Key Takeaways

**1. Boosting = Sequential Ensembling**:
- Trees trained one after another
- Each corrects mistakes of previous trees

**2. Core Algorithm**:
- Initialize: $f_0(x) = 0$
- Iterate: Fit to residuals, update with shrinkage
- Final: $f(x) = \sum \nu \cdot h_b(x)$

**3. Key Parameters**:
- **B**: Number of trees (more = better, diminishing returns)
- **ν**: Learning rate (smaller = better generalization, needs more trees)
- **Depth**: Shallow trees (1-3 levels) typical

**4. vs Random Forest**:
- RF: Parallel, deep trees, reduce variance
- Boosting: Sequential, shallow trees, reduce bias

**5. Variants**:
- **AdaBoost**: Exponential loss, sample weighting
- **Gradient Boosting**: Fits gradients, any differentiable loss

**Next**: Detailed gradient boosting algorithms in upcoming lectures.

---

## Glossary

- **Boosting**: Sequential ensemble method that builds trees iteratively
- **Stump**: Very shallow tree (1-3 levels deep)
- **Residual**: Error to be corrected ($r = y - f(x)$)
- **Shrinkage**: Learning rate parameter $\nu$ controlling update size
- **Sequential Ensembling**: Training models one after another, each depending on previous
- **AdaBoost**: Adaptive Boosting with sample reweighting
- **Gradient Boosting**: Fits gradients of loss function
- **Forward Stagewise**: Building model by adding one component at a time

