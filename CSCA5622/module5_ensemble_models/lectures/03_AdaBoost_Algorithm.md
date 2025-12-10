# AdaBoost Algorithm

**Lecture**: Module 5, Lecture 3  
**Course**: CSCA5622  
**Topic**: Adaptive Boosting (AdaBoost) for Classification and Regression

---

## Table of Contents
1. [Review: Generic Boosting](#1-review-generic-boosting)
2. [Introduction to AdaBoost](#2-introduction-to-adaboost)
3. [The AdaBoost Algorithm](#3-the-adaboost-algorithm)
4. [Detailed Example Walkthrough](#4-detailed-example-walkthrough)
5. [Implementation in Scikit-learn](#5-implementation-in-scikit-learn)
6. [AdaBoost for Regression](#6-adaboost-for-regression)
7. [Performance Analysis](#7-performance-analysis)
8. [Python Examples](#8-python-examples)
9. [Practice Problems](#9-practice-problems)

---

## 1. Review: Generic Boosting

### Quick Recap

The lecturer begins: "Previously we talked about **generic boosting algorithm**, which **iteratively fit the stump tree to the data to predict the residue**."

**Generic Boosting Key Points**:
1. Fit stump trees iteratively
2. Each tree predicts the **residual** (error from previous stage)
3. Trees are added together with shrinkage parameter

**Mathematical Form**:
$$f(x) = \sum_{b=1}^{B} \lambda \cdot h_b(x)$$

"Then each stump from each iteration is **added together with some shrink parameter Lambda** here."

### Purpose of Shrinkage

"This **Lambda help the model to learn slowly** so that we can **avoid overfitting**."

**Shrinkage Benefits**:
- Prevents aggressive overfitting
- Conservative, gradual learning
- Better generalization

### Boosting Variants

"There are **many variants over boosting algorithms**. However, **these two are mostly used and most popular**. We'll talk about those."

**Two Popular Variants**:
1. **AdaBoost** (this lecture)
2. **Gradient Boosting** (next lecture)

---

## 2. Introduction to AdaBoost

### What is AdaBoost?

**AdaBoost** = **Ada**ptive **Boost**ing

"**AdaBoost is originally developed for classification**. However, **later it was developed to also do regression as well**."

### What Makes AdaBoost Interesting?

"What makes AdaBoost interesting is that **it uses weights to data samples**."

**Key Innovation: Sample Weighting**

"That means it will **make some more emphasis on the misclassified samples** so that it can **learn more from this errors**."

**How It Works**:
- Start with equal weights for all samples
- After each iteration:
  - **Increase** weights for **misclassified** samples
  - Keep weights small for correctly classified samples
- Next tree focuses more on the hard examples

> **Slide Visualization**: 
> The slide likely shows:
> - Data points with varying sizes (representing weights)
> - Misclassified points growing larger
> - Correctly classified points staying small
> - Visual showing the adaptive focus

### Key Differences from Generic Boosting

**Three Main Differences**:

1. **Target Variable**: "Each stump **fits to y instead of residue**."
   - Generic Boosting: Fit residuals $r = y - f(x)$
   - AdaBoost: Fit original labels $y$ (but with weights)

2. **Label Encoding**: "Because it's a **classification**, it gives **discrete values**. But instead of 0, 1, we're going to use **-1 or 1**."
   - Standard classification: $y \in \{0, 1\}$
   - AdaBoost: $y \in \{-1, +1\}$
   - Why? Mathematical convenience for exponential loss

3. **Weighting Mechanism**: "Then it uses **exponential weight to update the data sample weights**."
   - Sample weights updated using exponential function
   - Misclassified examples get exponentially more weight

---

## 3. The AdaBoost Algorithm

### Model Form

"**AdaBoost algorithm**, we want to have a **classifier that gives a -1 or 1**."

**Final Model**:
$$f(x) = \sum_{b=1}^{B} \lambda_b \cdot h_b(x)$$

"This model is a **linear combination of the stump model** and b is the iteration."

### Important Note on Lambda

"A little difference from the **generic boosting algorithm**, this **Lambda_b now in AdaBoost is not the shrinkage parameter**, but it's **representing this modeling patterns from each iteration**."

**Key Distinction**:
- Generic Boosting: $\lambda$ is fixed shrinkage (e.g., 0.1)
- AdaBoost: $\lambda_b$ varies per iteration, represents model confidence

### Algorithm Steps

#### Step 0: Initialize Sample Weights

"This algorithm **start by initializing all the sample weights to one over N**, which means that **all the data points are equally important**."

$$w_i^{(1)} = \frac{1}{N} \quad \text{for } i = 1, 2, ..., N$$

where $N$ is the number of training samples.

**Interpretation**: Equal weights = equal importance initially.

#### Step 1: Iterate for B Times

"Then we're going to **repeat for b times**..."

**For each iteration b = 1 to B:**

##### Substep 1a: Fit Stump with Sample Weights

"We **fit the stump tree to the training data to predict the label instead of a register with a sample weight w**."

$$h_b = \text{fit}(X, y, w^{(b)})$$

**Important Detail**: "If you remember, the **stump model is actually decision tree**. Decision tree can **use a sample weight when calculating the split criteria**."

**How Weights Affect Training**:
- When finding best split, weighted samples contribute more to impurity calculation
- Tree focuses on getting high-weight samples correct
- Effectively, hard examples get more "votes" in determining splits

##### Substep 1b: Calculate Error

"After fitting the stump model using this stumper model. Here is that, and then we **compare how much accurate it is**."

$$\epsilon_b = \frac{\sum_{i=1}^{N} w_i^{(b)} \cdot \mathbb{1}(y_i \neq h_b(x_i))}{\sum_{i=1}^{N} w_i^{(b)}}$$

The lecturer explains the indicator function: "This i is the **identity function which will give zero when it's correctly classified** and which will **give one when it's misclassified**."

**Breakdown**:
- $\mathbb{1}(y_i \neq h_b(x_i))$: 
  - Returns 1 if prediction is wrong
  - Returns 0 if prediction is correct
- Numerator: Sum of weights for misclassified samples
- Denominator: Sum of all weights (normalization)

"This means that we **calculate the error only using misclassified examples**."

**First Iteration Note**: "The first iteration, **this weight are all equal**. However, it's **going to be updated as we go**."

##### Substep 1c: Calculate Model Coefficient

"Using this error, we're going to **calculate the model coefficient Lambda_b**, which again tells us **how much we should include this stump model into the total model**."

$$\lambda_b = \log\left(\frac{1 - \epsilon_b}{\epsilon_b}\right)$$

"This **Lambda is given by this formula**."

**Convention Note**: "Sometimes you're **going to see 1/2 in front of this formula**, which is also **popular convention**, but **with or without it's fine**."

Alternative formulation (also valid):
$$\lambda_b = \frac{1}{2}\log\left(\frac{1 - \epsilon_b}{\epsilon_b}\right)$$

**Understanding Lambda**:
- If $\epsilon_b \to 0$ (perfect classification): $\lambda_b \to +\infty$ (high confidence)
- If $\epsilon_b = 0.5$ (random guessing): $\lambda_b = 0$ (no confidence)
- If $\epsilon_b \to 1$ (worse than random): $\lambda_b \to -\infty$ (negative, flip prediction)

**Range**: "By the way, this function can **go from minus infinity to infinity**. This **parameter doesn't have to be somewhere between zero and one** unlike the shrinkage parameter."

##### Substep 1d: Update Sample Weights

"In using this **model coefficients**, we're going to **update the sample weight**."

$$w_i^{(b+1)} = w_i^{(b)} \cdot \exp(\lambda_b \cdot \mathbb{1}(y_i \neq h_b(x_i)))$$

Or equivalently:
$$w_i^{(b+1)} = w_i^{(b)} \cdot \exp(\lambda_b \cdot y_i \cdot h_b(x_i))$$

(since $y_i \cdot h_b(x_i) = -1$ when misclassified, $+1$ when correct)

"This sample again, when there was a **misclassification**, the weight of that sample **becomes larger by this exponential factor**."

**Effect**:
- **Misclassified**: $w_i$ multiplied by $\exp(\lambda_b)$ (increases)
- **Correctly classified**: $w_i$ stays same (or decreases if using the $y_i \cdot h_b(x_i)$ formulation)

**Normalization** (optional but common):
$$w_i^{(b+1)} = \frac{w_i^{(b+1)}}{\sum_{j=1}^{N} w_j^{(b+1)}}$$

This ensures weights sum to 1.

#### Step 2: Final Prediction

"After we do the **iteration for b times**, we finally get our **output model that looks like this**, the **linear combination of this stump model**."

$$f(x) = \sum_{b=1}^{B} \lambda_b \cdot h_b(x)$$

"Then the **final sign is given by that**."

$$\text{Prediction} = \text{sign}(f(x)) = \text{sign}\left(\sum_{b=1}^{B} \lambda_b \cdot h_b(x)\right)$$

**Interpretation**: 
- Each stump votes with weight $\lambda_b$
- High-confidence stumps (low error) vote more strongly
- Final class is the weighted majority vote

### Algorithm Summary

```
Algorithm: AdaBoost

Input: Training data (X, y) where y ∈ {-1, +1}
       Number of iterations B

Initialize: w_i^(1) = 1/N for all i

For b = 1 to B:
  1. Fit stump h_b to (X, y) with weights w^(b)
  2. Calculate weighted error:
     ε_b = Σ w_i^(b) · 𝟙(y_i ≠ h_b(x_i)) / Σ w_i^(b)
  3. Calculate model coefficient:
     λ_b = log((1 - ε_b) / ε_b)
  4. Update weights:
     w_i^(b+1) = w_i^(b) · exp(λ_b · 𝟙(y_i ≠ h_b(x_i)))
  5. Normalize weights (optional):
     w_i^(b+1) = w_i^(b+1) / Σ w_j^(b+1)

Output: f(x) = sign(Σ λ_b · h_b(x))
```

---

## 4. Detailed Example Walkthrough

### Setup

The lecturer provides: "Here's a **brief example with the picture**."

**Dataset**:
- 10 samples
- Features: $x_1, x_2, ...$
- Target: $y \in \{-1, +1\}$

> **Slide Visualization**: 
> The slide likely shows:
> - A table with columns: [Sample ID, Features, y, w^(1)]
> - Or a 2D scatter plot with + and - symbols for the two classes
> - Sample weights displayed as point sizes

### Iteration 1: Initialization

"**Initialize the sample weights w**. In this data, these are the **features** and this is the **target y**, and this is the **initial weight**."

**Initial Weights**:
$$w_i^{(1)} = \frac{1}{10} = 0.1 \quad \text{for all } i$$

| Sample | Features | y | w^(1) |
|--------|----------|---|-------|
| 1      | ...      | +1| 0.1   |
| 2      | ...      | -1| 0.1   |
| ...    | ...      |...| 0.1   |
| 10     | ...      | +1| 0.1   |

### Iteration 1: Fit Model

"Therefore, **this iteration**, we're going to **fit the stump model to training data**. It's **some sample weight** and it's **going to give some output like this**."

**Result**: Stump $h_1$ is trained and makes predictions.

### Iteration 1: Calculate Error

"Then we notice that **these two samples are misclassified**."

**Assumption**: Samples 3 and 7 are misclassified.

"Therefore, when we **calculate the error**, it's **going to give a 0.2**. **Two misclassification out of 10 examples**."

$$\epsilon_1 = \frac{w_3^{(1)} + w_7^{(1)}}{\sum_{i=1}^{10} w_i^{(1)}} = \frac{0.1 + 0.1}{10 \times 0.1} = \frac{0.2}{1.0} = 0.2$$

### Iteration 1: Calculate Model Coefficient

"Then we **further calculate the model coefficients and gives this value**."

$$\lambda_1 = \log\left(\frac{1 - 0.2}{0.2}\right) = \log\left(\frac{0.8}{0.2}\right) = \log(4) \approx 1.386$$

**Interpretation**: 
- Error rate is 20% (good performance)
- Model gets high confidence weight (1.386)

### Iteration 1: Update Weights

"Then using this **model coefficients**, we're going to **update the weights using this exponential factor that gives this weight**."

**For misclassified samples** (3 and 7):
$$w_i^{(2)} = w_i^{(1)} \cdot \exp(\lambda_1) = 0.1 \cdot \exp(1.386) = 0.1 \cdot 4 = 0.4$$

"This **misclassified example receives more weight**, **four times more than the others** and this one as well, it **receives a bigger weight**."

**For correctly classified samples** (all others):
$$w_i^{(2)} = w_i^{(1)} = 0.1$$

**Before Normalization**:
```
Samples 1, 2, 4, 5, 6, 8, 9, 10: w = 0.1 each (8 samples)
Samples 3, 7: w = 0.4 each (2 samples)

Sum = 8×0.1 + 2×0.4 = 0.8 + 0.8 = 1.6
```

### Iteration 1: Normalize Weights

"Then we can **normalize this weight** so that these **all examples, some of these weights becomes one**."

$$w_i^{(2)} = \frac{w_i^{(2)}}{\text{Sum}} = \frac{w_i^{(2)}}{1.6}$$

**After Normalization**:
- Correct samples: $w = 0.1 / 1.6 = 0.0625$
- Misclassified samples: $w = 0.4 / 1.6 = 0.25$

**Summary Table**:

| Sample | y | Classified? | w^(1) | w^(2) (before norm) | w^(2) (after norm) |
|--------|---|-------------|-------|---------------------|---------------------|
| 1      | +1| ✓ Correct   | 0.1   | 0.1                 | 0.0625              |
| 2      | -1| ✓ Correct   | 0.1   | 0.1                 | 0.0625              |
| 3      | +1| ✗ Wrong     | 0.1   | 0.4                 | 0.25                |
| ...    |   | ✓ Correct   | 0.1   | 0.1                 | 0.0625              |
| 7      | -1| ✗ Wrong     | 0.1   | 0.4                 | 0.25                |
| ...    |   | ✓ Correct   | 0.1   | 0.1                 | 0.0625              |

**Key Observation**: Misclassified samples now have **4× the weight** of correctly classified samples (0.25 vs 0.0625).

### Iteration 2: Effect of Updated Weights

In the next iteration:
- Stump $h_2$ will focus heavily on samples 3 and 7
- Their high weights make them more influential in split decisions
- The model adaptively learns from its previous mistakes

---

## 5. Implementation in Scikit-learn

### AdaBoost Classifier

"Let's have a look at some **usage**. **AdaBoost is available in sklearn ensemble module**."

```python
from sklearn.ensemble import AdaBoostClassifier
```

"**AdaBoost in sklearn**, both have a **classifier and regressor**."

### Key Parameters

"**Classifier has these options**..."

#### base_estimator

"**base estimator which is not specified**, then it's a **decision tree classifier with maximum depth equals 1**. That means it's **just stump**."

**Default**:
```python
from sklearn.tree import DecisionTreeClassifier
base_estimator = DecisionTreeClassifier(max_depth=1)
```

**Customization**: You can use any classifier as base estimator:
```python
# Use deeper trees
AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3))

# Or even other classifiers
AdaBoostClassifier(base_estimator=LogisticRegression())
```

#### learning_rate

"Then you can also see this **learning rate on top of this Lambda_b**, which was the **weight to the model**. There's also **learning rate as a hyperparameter**."

$$f(x) = \sum_{b=1}^{B} \nu \cdot \lambda_b \cdot h_b(x)$$

where $\nu$ is the learning_rate parameter.

"You can **reduce the learning rate** if you want to **make the AdaBoost classifier learn slowly**."

**Effect**:
- Smaller learning_rate → More conservative → Better generalization (usually)
- Larger learning_rate → Faster convergence → Risk of overfitting
- Trade-off with n_estimators (number of iterations)

#### algorithm

"By default, the **SAMME.R algorithm is used** to the **real AdaBoost**. This **R comes from real AdaBoost**."

**Two Options**:
1. **SAMME**: Original discrete AdaBoost (uses class predictions)
2. **SAMME.R**: Real AdaBoost (uses class probabilities)

"They make use of **predicted probability**. The **probability of being each class** instead of using those **binarized classifier**."

**SAMME.R Advantages**: "**SAMME.R is advanced version** of original AdaBoost algorithm **semi**. It is **good for a multi-class classifier**, but it also **works out better for the binary class classification**."

**Recommendation**: "You can **just leave it as is and use it**."

### Resources

"Here are some **more resources**. How this **real AdaBoost algorithm**, just the **SAMME.R is a little bit better** than the original **discrete AdaBoost algorithm**."

The SAMME.R paper: *"Multi-class AdaBoost"* by Hastie et al. (2009)

### Performance Comparison

"Again, this **boosting algorithm gives much better performance** than just the **one-stop as well as the fully grown decision tree**."

**Performance Hierarchy**:
```
Single Stump < Fully Grown Tree < AdaBoost
```

---

## 6. AdaBoost for Regression

### Can AdaBoost Do Regression?

"Being **AdaBoost originally developed for the classification problem**, can **AdaBoost also do regression**? The **answer is yes**."

### Implementation

"You just need to **call the AdaBoost regressor in sklearn ensemble module**."

```python
from sklearn.ensemble import AdaBoostRegressor
```

"**Everything is very similar**. It also **accepts a learning rate**."

### Key Difference: Loss Function

"The **only difference in the regressor** is that we can **specify the loss function** which by **default is a linear loss**."

**Available Loss Functions**:
1. **'linear'** (default): $L = |y - f(x)|$
2. **'square'**: $L = (y - f(x))^2$
3. **'exponential'**: $L = 1 - \exp(-|y - f(x)|)$

**How It Works for Regression**:
- Instead of sample reweighting based on misclassification
- Weights are updated based on prediction error magnitude
- Samples with large errors get higher weights

---

## 7. Performance Analysis

### Experimental Setup

"Let's talk about **how good is the AdaBoost**. I **picked two different data sets**, each of which have about **5,000 samples** and then **20 features**."

**Two Datasets**:
- Dataset 1: Harder problem
- Dataset 2: Easier problem

### Results

"As you can see, **depending on the problem difficulty**, the **absolute accuracy can be different**. However, **regardless of its difficulty**, the **boosting algorithm is always better** than **fully grown decision trees**."

> **Slide Visualization**: 
> The slide likely shows two graphs side-by-side:
> - Left graph: Harder dataset
> - Right graph: Easier dataset
> - Both showing: Decision Tree baseline vs AdaBoost curves
> - X-axis: Number of estimators (trees)
> - Y-axis: Accuracy

**Observations**:

**Hard Dataset (Left Graph)**:
"This **left graph being more difficult case**. **AdaBoost accuracy isn't too good**, but..."
- Fully grown tree: ~60% accuracy
- AdaBoost: ~75% accuracy
- Still significant improvement despite low absolute accuracy

**Easy Dataset (Right Graph)**:
"This **right one is a little easier data**. They had a **higher accuracy**."
- Fully grown tree: ~85% accuracy
- AdaBoost: ~95% accuracy
- Clear performance gain

### Important Warning: Overfitting

"As you can see here, **boosting algorithm can have overfitting as well**."

**Causes of Overfitting**:
1. "If the **learning rate is too big**..."
2. "The **number of trees are too big as well**..."

**Trade-off**: "There's a **trade-off between the learning rate and the number of trees**."

**Guidelines**:
- Small learning_rate + Large n_estimators = Better generalization
- Large learning_rate + Small n_estimators = Faster but risky
- Monitor validation error to find sweet spot

---

## 8. Python Examples

### Complete Classification Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Generate synthetic data
np.random.seed(42)
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                          n_redundant=5, random_state=42)

# Convert to {-1, +1} labels (AdaBoost convention)
y = 2 * y - 1

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("=" * 60)
print("ADABOOST CLASSIFICATION")
print("=" * 60)

# 1. Single Stump (baseline)
stump = DecisionTreeClassifier(max_depth=1, random_state=42)
stump.fit(X_train, y_train)
stump_acc = accuracy_score(y_test, stump.predict(X_test))
print(f"Single Stump Accuracy:        {stump_acc:.4f}")

# 2. Fully Grown Tree (baseline)
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
tree_acc = accuracy_score(y_test, tree.predict(X_test))
print(f"Fully Grown Tree Accuracy:    {tree_acc:.4f}")

# 3. AdaBoost with default settings
ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
ada.fit(X_train, y_train)
ada_acc = accuracy_score(y_test, ada.predict(X_test))
print(f"AdaBoost (50 trees) Accuracy: {ada_acc:.4f}")

# 4. AdaBoost with reduced learning rate
ada_slow = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)
ada_slow.fit(X_train, y_train)
ada_slow_acc = accuracy_score(y_test, ada_slow.predict(X_test))
print(f"AdaBoost (lr=0.5) Accuracy:   {ada_slow_acc:.4f}")

print("\nImprovement over single stump: {:.1f}%".format((ada_acc - stump_acc) * 100))
print("Improvement over full tree:    {:.1f}%".format((ada_acc - tree_acc) * 100))
```

### Visualizing Learning Curves

```python
# Performance vs number of estimators
n_estimators_range = [1, 5, 10, 20, 50, 100, 200]
train_scores = []
test_scores = []

for n_est in n_estimators_range:
    ada_temp = AdaBoostClassifier(n_estimators=n_est, learning_rate=1.0, random_state=42)
    ada_temp.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, ada_temp.predict(X_train))
    test_acc = accuracy_score(y_test, ada_temp.predict(X_test))
    
    train_scores.append(train_acc)
    test_scores.append(test_acc)
    
    print(f"Trees: {n_est:3d} | Train: {train_acc:.4f} | Test: {test_acc:.4f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, train_scores, 'b-o', label='Train Accuracy', linewidth=2)
plt.plot(n_estimators_range, test_scores, 'r-s', label='Test Accuracy', linewidth=2)
plt.axhline(y=stump_acc, color='g', linestyle='--', label='Single Stump', linewidth=2)
plt.axhline(y=tree_acc, color='orange', linestyle='--', label='Full Tree', linewidth=2)
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('AdaBoost Performance vs Number of Trees')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Sample Weights Visualization

```python
# Track sample weights evolution
from sklearn.tree import DecisionTreeClassifier

# Simple 2D dataset for visualization
X_viz, y_viz = make_classification(n_samples=100, n_features=2, n_informative=2,
                                   n_redundant=0, n_clusters_per_class=1, random_state=42)
y_viz = 2 * y_viz - 1  # Convert to {-1, +1}

# Initialize weights
weights = np.ones(len(X_viz)) / len(X_viz)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for iteration in range(4):
    # Fit stump with current weights
    stump = DecisionTreeClassifier(max_depth=1, random_state=iteration)
    stump.fit(X_viz, y_viz, sample_weight=weights)
    predictions = stump.predict(X_viz)
    
    # Calculate error
    incorrect = (predictions != y_viz)
    error = np.sum(weights * incorrect) / np.sum(weights)
    
    # Calculate alpha (model coefficient)
    alpha = np.log((1 - error) / error) if error > 0 and error < 1 else 0
    
    # Update weights
    weights = weights * np.exp(alpha * incorrect)
    weights = weights / np.sum(weights)  # Normalize
    
    # Plot
    ax = axes[iteration]
    scatter = ax.scatter(X_viz[:, 0], X_viz[:, 1], c=y_viz, s=weights*5000, 
                        alpha=0.6, cmap='coolwarm', edgecolors='black')
    ax.set_title(f'Iteration {iteration+1}\nError: {error:.3f}, α: {alpha:.3f}')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

plt.tight_layout()
plt.show()

print("\nFinal weight distribution:")
print(f"  Min: {weights.min():.6f}")
print(f"  Max: {weights.max():.6f}")
print(f"  Ratio (max/min): {weights.max()/weights.min():.1f}x")
```

### Regression Example

```python
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score

# Generate regression data
X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=20, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42)

print("\n" + "=" * 60)
print("ADABOOST REGRESSION")
print("=" * 60)

# Compare different loss functions
for loss in ['linear', 'square', 'exponential']:
    ada_reg = AdaBoostRegressor(n_estimators=50, learning_rate=1.0, 
                               loss=loss, random_state=42)
    ada_reg.fit(X_train_r, y_train_r)
    
    y_pred = ada_reg.predict(X_test_r)
    mse = mean_squared_error(y_test_r, y_pred)
    r2 = r2_score(y_test_r, y_pred)
    
    print(f"\nLoss: {loss:12s} | MSE: {mse:8.2f} | R²: {r2:.4f}")
```

---

## 9. Practice Problems

### Problem 1: Manual AdaBoost Calculation

**Question**: You have a binary classification problem with 5 samples. After the first iteration:
- Samples 1, 2, 4, 5 are correctly classified
- Sample 3 is misclassified
- Current weights: all equal at 0.2

a) Calculate the weighted error $\epsilon_1$
b) Calculate the model coefficient $\lambda_1$
c) Update the weights for iteration 2 (before normalization)
d) Normalize the weights

**Solution**:

**Part a): Weighted Error**

$$\epsilon_1 = \frac{\sum w_i \cdot \mathbb{1}(\text{wrong})}{\sum w_i} = \frac{w_3}{\sum_{i=1}^{5} w_i} = \frac{0.2}{5 \times 0.2} = \frac{0.2}{1.0} = 0.2$$

**Answer**: $\epsilon_1 = 0.2$ (20% error rate)

**Part b): Model Coefficient**

$$\lambda_1 = \log\left(\frac{1 - \epsilon_1}{\epsilon_1}\right) = \log\left(\frac{1 - 0.2}{0.2}\right) = \log\left(\frac{0.8}{0.2}\right) = \log(4) \approx 1.386$$

**Answer**: $\lambda_1 \approx 1.386$

**Part c): Update Weights (Before Normalization)**

For **correctly classified** samples (1, 2, 4, 5):
$$w_i^{(2)} = w_i^{(1)} = 0.2$$

For **misclassified** sample (3):
$$w_3^{(2)} = w_3^{(1)} \cdot \exp(\lambda_1) = 0.2 \cdot \exp(1.386) = 0.2 \cdot 4 = 0.8$$

**Answer**: 
- Samples 1, 2, 4, 5: $w = 0.2$
- Sample 3: $w = 0.8$

**Part d): Normalize Weights**

Sum of weights:
$$\text{Sum} = 4 \times 0.2 + 1 \times 0.8 = 0.8 + 0.8 = 1.6$$

Normalized weights:
$$w_i^{(2)} = \frac{w_i^{(2)}}{\text{Sum}}$$

**Answer**:
- Samples 1, 2, 4, 5: $w = 0.2 / 1.6 = 0.125$
- Sample 3: $w = 0.8 / 1.6 = 0.5$

**Interpretation**: Sample 3 now has **4× the weight** of other samples (0.5 vs 0.125), so it will be 4× more influential in the next iteration.

---

### Problem 2: Understanding Alpha Values

**Question**: Consider three stumps with different error rates:
- Stump A: $\epsilon_A = 0.1$ (10% error)
- Stump B: $\epsilon_B = 0.5$ (50% error, random)
- Stump C: $\epsilon_C = 0.9$ (90% error, worse than random)

a) Calculate $\lambda$ for each stump
b) Which stump will have the most influence in the final model?
c) What happens with Stump C?

**Solution**:

**Part a): Calculate Lambda**

**Stump A**:
$$\lambda_A = \log\left(\frac{1 - 0.1}{0.1}\right) = \log(9) \approx 2.197$$

**Stump B**:
$$\lambda_B = \log\left(\frac{1 - 0.5}{0.5}\right) = \log(1) = 0$$

**Stump C**:
$$\lambda_C = \log\left(\frac{1 - 0.9}{0.9}\right) = \log\left(\frac{1}{9}\right) = -\log(9) \approx -2.197$$

**Part b): Most Influential**

**Answer**: **Stump A** will have the most influence ($|\lambda_A| = 2.197$ is largest positive value).

**Reasoning**:
- High $\lambda$ means high confidence
- Stump A has low error (good performance) → high positive $\lambda$
- Its predictions will be weighted heavily in the final vote

**Part c): What Happens with Stump C?**

**Answer**: Stump C has **negative** $\lambda$ (≈ -2.197).

**Interpretation**:
- Negative $\lambda$ means we **flip** the stump's predictions
- If Stump C predicts +1, we actually use -1 in the ensemble
- A stump worse than random is still useful—just use the opposite of what it says!
- This is equivalent to having a good classifier

**Mathematical Insight**:
$$f(x) = \lambda_C \cdot h_C(x) = (-2.197) \cdot h_C(x)$$

If $h_C(x) = +1$, contribution is $-2.197$ (strong vote for -1)
If $h_C(x) = -1$, contribution is $+2.197$ (strong vote for +1)

---

### Problem 3: Learning Rate Trade-off

**Question**: You're training an AdaBoost classifier and must choose between:
- **Config A**: learning_rate=1.0, n_estimators=50
- **Config B**: learning_rate=0.1, n_estimators=500

Both configurations have similar total "learning budget" (1.0×50 ≈ 0.1×500).

a) Which will train faster?
b) Which is more likely to overfit?
c) Which would you choose for a production model?

**Solution**:

**Part a): Training Speed**

**Answer**: **Config A** will train faster.

**Reasoning**:
- Training time is primarily proportional to n_estimators
- Config A: 50 trees
- Config B: 500 trees (10× slower)
- Learning rate doesn't significantly affect per-tree training time

**Part b): Overfitting Risk**

**Answer**: **Config A** is more likely to overfit.

**Reasoning**:
- Large learning rate (1.0) means aggressive updates
- Each tree has strong influence
- Can quickly memorize training data
- Less regularization

Config B with smaller learning rate:
- Conservative updates (0.1 weight per tree)
- More gradual learning
- Acts as implicit regularization
- Better generalization

**Part c): Production Choice**

**Answer**: **Config B** (learning_rate=0.1, n_estimators=500)

**Justification**:
1. **Better Generalization**: Smaller learning rate typically gives better test performance
2. **More Robust**: Less sensitive to noise in training data
3. **Standard Practice**: Industry convention favors smaller learning rates
4. **Training Time**: One-time cost; prediction speed is same for both

**Exception**: Choose Config A if:
- Training time is critical
- You have very limited data
- You've validated it doesn't overfit on your specific problem

---

### Problem 4: SAMME vs SAMME.R

**Question**: You're implementing AdaBoost for a 3-class classification problem.

a) What's the key difference between SAMME and SAMME.R algorithms?
b) Why does SAMME.R generally perform better?
c) When might you prefer SAMME over SAMME.R?

**Solution**:

**Part a): Key Difference**

**SAMME** (Stagewise Additive Modeling using Multi-class Exponential loss):
- Uses **discrete class predictions**: $h(x) \in \{1, 2, 3\}$
- Updates based on whether prediction is correct or wrong (binary)
- Original AdaBoost approach

**SAMME.R** ("Real" SAMME):
- Uses **class probabilities**: $P(y=k|x)$ for each class $k$
- Updates based on probability distribution
- More nuanced information about prediction confidence

**Example**:
```
Sample with true class = 1

SAMME sees: h(x) = 2 → Wrong (binary feedback)

SAMME.R sees: P(y=1) = 0.45, P(y=2) = 0.40, P(y=3) = 0.15
            → Almost correct (soft feedback)
```

**Part b): Why SAMME.R Performs Better**

**Reasons**:

1. **Richer Information**:
   - SAMME.R uses full probability distribution
   - Knows how confident (or uncertain) each prediction is
   - Can distinguish between "barely wrong" and "completely wrong"

2. **Better Weight Updates**:
   - Samples with confident misclassifications get more weight
   - Samples with uncertain predictions treated more carefully
   - More nuanced adaptation

3. **Faster Convergence**:
   - Typically reaches good performance with fewer iterations
   - More efficient use of each stump

4. **Multi-class Friendly**:
   - Naturally handles multiple classes
   - SAMME needs modification for multi-class

**Part c): When to Prefer SAMME**

**Scenarios**:

1. **Base Estimator Limitations**:
   - If your base estimator doesn't support `predict_proba()`
   - Some classifiers only give hard predictions

2. **Computational Efficiency**:
   - SAMME is slightly faster (no probability calculation)
   - Matters only for very large-scale problems

3. **Interpretability**:
   - SAMME's binary feedback is simpler to understand
   - Easier to explain to non-technical stakeholders

4. **Legacy Systems**:
   - Matching behavior of existing implementations
   - Reproducing published results

**Recommendation**: Default to SAMME.R unless you have a specific reason to use SAMME.

---

### Problem 5: Exponential Weight Growth

**Question**: A sample is misclassified in 3 consecutive AdaBoost iterations with:
- Iteration 1: $\epsilon_1 = 0.2$, initial weight $w = 0.1$
- Iteration 2: $\epsilon_2 = 0.3$
- Iteration 3: $\epsilon_3 = 0.25$

Track this sample's weight evolution (ignore normalization for simplicity).

**Solution**:

**Iteration 1**:

Calculate $\lambda_1$:
$$\lambda_1 = \log\left(\frac{1 - 0.2}{0.2}\right) = \log(4) \approx 1.386$$

Update weight (misclassified):
$$w^{(2)} = w^{(1)} \cdot \exp(\lambda_1) = 0.1 \cdot \exp(1.386) = 0.1 \cdot 4 = 0.4$$

**Iteration 2**:

Calculate $\lambda_2$:
$$\lambda_2 = \log\left(\frac{1 - 0.3}{0.3}\right) = \log\left(\frac{7}{3}\right) \approx 0.847$$

Update weight (misclassified again):
$$w^{(3)} = w^{(2)} \cdot \exp(\lambda_2) = 0.4 \cdot \exp(0.847) = 0.4 \cdot 2.333 \approx 0.933$$

**Iteration 3**:

Calculate $\lambda_3$:
$$\lambda_3 = \log\left(\frac{1 - 0.25}{0.25}\right) = \log(3) \approx 1.099$$

Update weight (misclassified third time):
$$w^{(4)} = w^{(3)} \cdot \exp(\lambda_3) = 0.933 \cdot \exp(1.099) = 0.933 \cdot 3 \approx 2.8$$

**Summary**:

| Iteration | Error | Lambda | Weight |
|-----------|-------|--------|--------|
| Initial   | -     | -      | 0.1    |
| After 1   | 0.2   | 1.386  | 0.4    |
| After 2   | 0.3   | 0.847  | 0.933  |
| After 3   | 0.25  | 1.099  | 2.8    |

**Interpretation**:
- Sample weight grew from 0.1 to 2.8 (28× increase!)
- This sample is now extremely influential
- Next tree will prioritize getting this sample correct
- Demonstrates AdaBoost's adaptive focus on hard examples

**Key Insight**: Consistently misclassified samples get exponentially increasing attention. This is both a strength (focuses on hard cases) and potential weakness (can overfit to outliers).

---

## 10. Key Takeaways

**1. AdaBoost's Core Innovation**:
- Sample weighting: Adaptively focuses on misclassified examples
- Fits to original labels (not residuals), but with varying sample importance

**2. Algorithm Mechanics**:
- Initialize: Equal weights (1/N)
- Iterate: Fit → Error → Alpha → Update weights
- Final: Weighted vote $\text{sign}(\sum \lambda_b h_b(x))$

**3. Key Differences from Generic Boosting**:
- Targets: Original labels (with weights) vs residuals
- Coefficients: Variable $\lambda_b$ (model confidence) vs fixed $\nu$ (shrinkage)
- Mechanism: Sample reweighting vs residual fitting

**4. Model Coefficient ($\lambda_b$)**:
- $\lambda = \log\left(\frac{1-\epsilon}{\epsilon}\right)$
- High accuracy → Large positive $\lambda$ → Strong influence
- Random (50%) → $\lambda = 0$ → No influence
- Worse than random → Negative $\lambda$ → Flip predictions

**5. Implementation**:
- `AdaBoostClassifier`: Default uses stumps (depth=1), SAMME.R algorithm
- `AdaBoostRegressor`: Supports linear, square, exponential losses
- learning_rate parameter provides additional regularization

**6. Performance**:
- Consistently beats single stumps and full trees
- Trade-off: learning_rate vs n_estimators
- Can overfit with too many trees or high learning rate

**7. SAMME.R vs SAMME**:
- SAMME.R uses probabilities (better performance, default)
- SAMME uses discrete predictions (simpler, legacy)

**Next**: Gradient Boosting in the final lecture of this module.

---

## Glossary

- **AdaBoost**: Adaptive Boosting using sample reweighting
- **Sample Weight**: Importance assigned to each training example
- **Lambda (λ)**: Model coefficient representing stump confidence
- **Epsilon (ε)**: Weighted error rate
- **SAMME**: Stagewise Additive Modeling (discrete AdaBoost)
- **SAMME.R**: Real AdaBoost using probability estimates
- **Exponential Weight Update**: Mechanism to increase weights for errors
- **Base Estimator**: Weak learner (typically depth-1 decision tree)

