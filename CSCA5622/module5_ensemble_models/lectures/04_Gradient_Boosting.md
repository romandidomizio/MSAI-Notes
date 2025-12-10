# Gradient Boosting

**Lecture**: Module 5, Lecture 4  
**Course**: CSCA5622  
**Topic**: Gradient Boosting Machines, Performance Comparisons, and Module Recap

---

## Table of Contents
1. [Review: Generic Boosting](#1-review-generic-boosting)
2. [Introduction to Gradient Boosting](#2-introduction-to-gradient-boosting)
3. [The Gradient Boosting Algorithm](#3-the-gradient-boosting-algorithm)
4. [Why Gradients Over Residuals?](#4-why-gradients-over-residuals)
5. [Performance Analysis: Gradient Boosting vs AdaBoost](#5-performance-analysis-gradient-boosting-vs-adaboost)
6. [Performance Analysis: Boosting vs Random Forest](#6-performance-analysis-boosting-vs-random-forest)
7. [Other Useful Packages](#7-other-useful-packages)
8. [Module Recap: Ensemble Methods](#8-module-recap-ensemble-methods)
9. [Python Implementation](#9-python-implementation)
10. [Practice Problems](#10-practice-problems)

---

## 1. Review: Generic Boosting

### The Core Process

"So previously, we talked about **generic boosting algorithm** that we **iterative add a stump model to our initial model**."

**Key Steps**:
1. Start with initial model
2. Calculate residuals
3. Fit stump to residuals
4. Add stump with shrinkage
5. Repeat

"And each **stamp model fits the data to predict the residue from each stage**."

### The Shrinkage Mechanism

"And with the **shrink cut parameter**, we **add this dump model iteratively** and also the **residue gets smaller and smaller** as we go through this iteration."

**Mathematical Form**:
$$f_b(x) = f_{b-1}(x) + \nu \cdot h_b(x)$$

where:
- $f_b(x)$: Model at stage $b$
- $\nu$: Shrinkage parameter (learning rate)
- $h_b(x)$: Stump fitted to residuals $r_{b-1}$

### Final Output

"And then as an **output**, we're going to have the **combined model**."

$$f(x) = \sum_{b=1}^{B} \nu \cdot h_b(x)$$

---

## 2. Introduction to Gradient Boosting

### Generalization of Boosting

"**Gradient boosting is a generalization of this boosting algorithm**."

**Key Innovation**: "Instead of feeding the **residue which is Y-F(x)** at each stage, we're going to use **gradient of a loss function**."

### Understanding Loss Functions

"So if you remember **loss function is some generalization form of measuring some error**."

**Definition**: A loss function measures the discrepancy between predictions and true values.

"So we're going to **measure an error by having a data X and Y** and our **prediction YP which is essentially the FX**."

$$L(y, f(x))$$

where:
- $y$: True label
- $f(x)$: Predicted value

### Examples of Loss Functions

#### Regression

"So this can be **MSE or RSS in the regression**."

**Mean Squared Error (MSE)**:
$$L_{\text{MSE}} = \frac{1}{2}(y - f(x))^2$$

**Residual Sum of Squares (RSS)**:
$$L_{\text{RSS}} = \sum_{i=1}^{N} (y_i - f(x_i))^2$$

#### Classification

"So for example, something like this, or **some other function if it's classification**."

**Log Loss (Cross-Entropy)**:
$$L_{\text{log}} = -[y \log(p) + (1-y) \log(1-p)]$$

**Exponential Loss** (AdaBoost):
$$L_{\text{exp}} = \exp(-y \cdot f(x))$$

### Generalization Power

"So this **loss function can be a very general form**, and this can be a **measure of error**. But **loss function is more generalized form**."

**Why This Matters**:
- Can optimize for different objectives
- More flexibility than just squared error
- Better suited to specific problem types

### The Gradient Approach

"So by measuring the **gradient of loss function with respect to our change of model in each iteration**, we can **measure the gradient of the loss function**."

**Mathematical Formulation**:

The gradient at stage $b$:
$$g_b = -\frac{\partial L(y_i, f_{b-1}(x_i))}{\partial f_{b-1}(x_i)}$$

This is the **negative gradient** (direction of steepest descent).

### The Goal

"And the goal is to **feed our tree to predict the negative gradient minus G** instead of just **pure residue**."

**Key Difference**:
- **Generic Boosting**: Fit tree to residuals $r = y - f(x)$
- **Gradient Boosting**: Fit tree to negative gradients $g = -\frac{\partial L}{\partial f}$

"So that's a **little bit different from our previous model** and **everything else is the same**."

---

## 3. The Gradient Boosting Algorithm

### Algorithm Overview

"So we're going to **see more in detail here**."

#### Step 0: Initialize Model

"So we **start by fitting our initial model to minimize the loss function**."

$$f_0(x) = \arg\min_c \sum_{i=1}^{N} L(y_i, c)$$

For MSE loss, this is typically:
$$f_0(x) = \bar{y} = \frac{1}{N}\sum_{i=1}^{N} y_i$$

"This is something **similar to minimizing entropy or minimizing MSC loss for regression in decision tree**."

#### Step 1: Iterate for B Stages

**For b = 1 to B:**

##### Substep 1a: Calculate Negative Gradient

"So we're going to have some split, and then for **each iteration we're going to do calculate the negative gradient**, which is again a **gradient of loss function with respect to the change of this function**."

$$g_{ib} = -\frac{\partial L(y_i, f_{b-1}(x_i))}{\partial f_{b-1}(x_i)}$$

Compute this for all training samples $i = 1, ..., N$.

**Example for Squared Loss**:
$$L = \frac{1}{2}(y - f)^2$$
$$g = -\frac{\partial L}{\partial f} = -(f - y) = y - f = r$$

The gradient equals the residual for squared loss!

##### Substep 1b: Fit Tree to Gradients

"And with this **gradient value**, we're going to **fit the stump tree to this training data to predict negative gradient value**."

$$h_b = \arg\min_h \sum_{i=1}^{N} (g_{ib} - h(x_i))^2$$

Fit a regression tree to predict the negative gradients.

##### Substep 1c: Update Parameters

"And this will give **some set of parameters while it's fitting** and then we will **update our loss function using these updated parameter values**."

For each terminal node (leaf) $j$ in tree $h_b$, find the optimal value:
$$\gamma_{jb} = \arg\min_\gamma \sum_{x_i \in R_{jb}} L(y_i, f_{b-1}(x_i) + \gamma)$$

where $R_{jb}$ is the region (set of samples) in leaf $j$.

##### Substep 1d: Update Model

"And also we're going to **update the function**."

$$f_b(x) = f_{b-1}(x) + \nu \sum_{j=1}^{J_b} \gamma_{jb} \mathbb{1}(x \in R_{jb})$$

Or more simply:
$$f_b(x) = f_{b-1}(x) + \nu \cdot h_b(x)$$

where $\nu$ is the learning rate (shrinkage parameter).

#### Step 2: Final Model

"And as we go **this iteration**, we're going to have **this editing model as a result**."

$$f(x) = f_0(x) + \sum_{b=1}^{B} \nu \cdot h_b(x)$$

### Algorithm Summary

```
Algorithm: Gradient Boosting

Input: Training data (X, y), Loss function L, Number of iterations B

1. Initialize: f_0(x) = argmin_c Σ L(y_i, c)

2. For b = 1 to B:
   a. Calculate negative gradients:
      g_ib = -∂L(y_i, f_{b-1}(x_i))/∂f_{b-1}(x_i) for all i
   
   b. Fit tree h_b to (X, g_b)
   
   c. For each leaf j, compute optimal value:
      γ_jb = argmin_γ Σ L(y_i, f_{b-1}(x_i) + γ) for x_i in leaf j
   
   d. Update model:
      f_b(x) = f_{b-1}(x) + ν · h_b(x)

3. Output: f(x) = f_B(x)
```

---

## 4. Why Gradients Over Residuals?

### The Steepest Descent Argument

"So let's talk about **why we want to use a gradient instead of just a residue**."

#### Generic Boosting as Greedy Algorithm

"So if you use just **generic pushing algorithm**, which is a **greedy algorithm**, which will **look into all the possible split** of a stump or small tree. And then it will **pick one that gave the best split**."

**Greedy Approach**: At each step, make the locally optimal choice.

"That means you will **choose the parameters such that the reduction in residual is the biggest**."

#### Gradient as Steepest Descent

"So **measuring gradient of this multidimensional space is very similar to this greedy approach**, but **it's even better** because it's going to **choose the direction that the steepest descent**."

**Steepest Descent**: Move in the direction that decreases the loss function most rapidly.

"So **steepest descent in terms of reducing the loss function**."

**Mathematical Intuition**:

In function space, we're doing gradient descent:
$$f_{b}(x) = f_{b-1}(x) - \nu \cdot \nabla_f L(f_{b-1})$$

The gradient $\nabla_f L$ points in the direction of steepest increase, so the negative gradient points toward steepest decrease.

### Better for Classification

"And when you think about **classification problem**, where we **chose some different function like entropy or genie** in decision tree classifier instead of **whether it's right or wrong, which is residue in classifier**, that is **more true to how the decision tree split happens**."

**For Classification**:
- **Residual approach**: Just binary right/wrong
- **Gradient approach**: Uses log loss or other smooth loss functions

"So **having those function is more expressive in that way**."

**Example**:

Consider log loss for binary classification:
$$L = -[y \log(p) + (1-y)\log(1-p)]$$

Gradient provides probabilistic information:
$$-\frac{\partial L}{\partial f} = y - p(x)$$

This is richer than just "correct" or "incorrect".

### Theoretical Superiority

"Okay, so **I'm trying to convince you that the gradient boosting should in theory work better** than **four step wise** or **generic boosting algorithm**."

**Key Advantages**:
1. **Optimization**: Directly minimizes any differentiable loss
2. **Expressiveness**: Captures probabilistic information
3. **Flexibility**: Works with various loss functions
4. **Efficiency**: Steepest descent path

---

## 5. Performance Analysis: Gradient Boosting vs AdaBoost

### Experimental Setup

"So let's have some **comparison**. So I **prepared two data**, each of which are **very similar to each other**."

**Dataset Specifications**:
- **Common**: "They have a **small number of features**, certain features versus **20 features** and they have approximately **5000 or more samples**."

**Data 1 (Difficult)**:
"The **data one is a little bit difficult**, so having **fully grown decision tree will give about 61% accuracy**..."

**Data 2 (Easier)**:
"...whereas **data two**, it's a **little easier**. So the **decision tree fully grown will give performance of 89% accuracy**."

### Why Difficulty Differs

"So even though a **number of features and the number of samples are similar**, sometimes depending on **how one or more features are a good predictor of the target variable**, things can be different."

**Factors Affecting Difficulty**:
- Quality of predictive features
- Signal-to-noise ratio
- Class separability
- Feature interactions

> **Slide Visualization**: 
> The slide likely shows two graphs side-by-side:
> - X-axis: Number of estimators
> - Y-axis: Accuracy
> - Multiple lines: Decision Tree (baseline), AdaBoost, Gradient Boosting

### Results: Data 1 (Difficult Case)

"But as you can **imagine**, the **gradient boosting is much better than decision three already**."

**Performance Comparison**:
- Single Decision Tree: ~61%
- AdaBoost: Significantly better
- Gradient Boosting: Significantly better

"So if you compare to **other boost**, the **data one on the data one gives similar result**, both of **other boost and gradient boosting give a much better results** than just a **decision tree**."

**Key Finding**: Both boosting methods dramatically improve over single tree, and perform similarly to each other on difficult data.

### Results: Data 2 (Easier Case)

"In **data two**, **much better result than distant alone**."

"However, you can see the **gradient boosting works slightly better than a boost**."

**Performance Hierarchy** (Data 2):
```
Decision Tree (89%) < AdaBoost < Gradient Boosting
```

### Conclusion on Gradient Boosting vs AdaBoost

"So the **conclusion** is that **whether the gradient boosting is always better than other boost, it depends on the data**. But **most of the time it is likely to be better performing** than other boost."

**General Guideline**: Gradient Boosting ≥ AdaBoost (usually slightly better)

### Robustness to Mislabeled Data

"Also **gradient boosting is less sensitive to mislabel data**."

**Reason**: "So for example, **other boosts are sensitive to mislabel data** because it uses **[INAUDIBLE] to each data samples**."

*[Note: The lecturer is referring to sample weighting in AdaBoost]*

"Therefore, **if the label is wrong**, it's **likely to suffer**."

**AdaBoost Problem**: Mislabeled points get increasingly high weights, causing the model to focus heavily on noise.

"However, **gradient boosting doesn't have that problem**."

**Why Gradient Boosting is Robust**:
- Doesn't use explicit sample reweighting
- Gradients for mislabeled points don't dominate as much
- More stable optimization procedure

---

## 6. Performance Analysis: Boosting vs Random Forest

### Learning Rate and Overfitting

"**How about some other aspects**? So these **graphs were generated at a different learning rate**."

"As you know from **previous video**, any **boosting algorithm can deteriorate** if **learning rate is too high** and **number of trees are too many**."

**Overfitting Risk**: High learning rate + Many trees = Overfitting

"So in order to **prevent overfitting**, when we have a **larger number of trees in additive model like boosting algorithm**, we need to **reduce the learning rate**."

**Trade-off Rule**:
$$\text{More Trees} \Rightarrow \text{Smaller Learning Rate}$$

"So this **graphic shows that** and then you can see that both the **boosting gradient boosting**, they **require smaller learning rate** as the **number of trees increases**."

> **Slide Visualization**: 
> Likely shows learning curves with different learning rates:
> - Multiple curves for different learning rates (e.g., 0.01, 0.1, 1.0)
> - Shows how high learning rate curves start to overfit with many trees
> - Shows how low learning rate curves are more stable

### Training Time Comparison

"**This one is time**, so I ran a **fivefold cross validation for each model**."

"In this case, **gradient boosting was time efficient than other boost**, but **just empirical speaking, it depends on the data**."

**Observation**: Training time varies by dataset characteristics.

### Important Implementation Detail

"And also you have to **keep in mind that other boost uses a stump** which means the **max steps equals one**. Whereas a **gradient boosting Sklearn library**, they by **default use max steps equals three**."

**Default Parameters**:
- **AdaBoost**: `max_depth=1` (stumps)
- **GradientBoosting**: `max_depth=3` (small trees)

**Implication**: Fair comparison requires matching tree depths, or acknowledging that deeper trees affect both performance and speed.

### Data 1 & 2: Few Features

"Now let's talk about **performance comparison with the random forest** even. So the **data one**, which was a **difficult case**, we saw that the **random forest didn't do much better than decision tree**."

**Performance on Difficult Data (Data 1)**:
- Decision Tree: ~61%
- Random Forest: Slightly better than single tree
- Boosting Algorithms: Much better than Random Forest

"And **boosting algorithms were much better than random forest**."

**Performance on Easier Data (Data 2)**:
"Whereas this a **little bit easier data with the data two**, **all of the ensemble algorithm did better much better** than just a **decision tree**."

**Key Finding**: With few features, boosting methods generally outperform Random Forest.

### When Does Random Forest Shine?

"So can you say **random forest which is a parallel ensemble algorithm** versus **boosting algorithm**, **which one would be better**? It is **difficult to tell** when we have a **small number of features**."

**Critical Insight**: "Because when the **random forest really shines** is when the **number of features are a lot**."

### Data 3: Many Features

"So I **prepared the data three** which has **145 features** which is a **lot more features** than previous data and has **3000 samples**, and **single sent performance is almost a 70%** which is kind of **medium difficulty**."

**Dataset 3 Specs**:
- Features: 145 (vs 20 in Data 1 & 2)
- Samples: 3000
- Difficulty: Medium (~70% baseline)

"And then **run three different ensemble models**. And as you can see **random forest did better than boosting algorithm**."

**Performance Ranking** (145 features):
```
Boosting < Random Forest
```

### General Guidelines: When to Use Which

"So now you have some **sense of when to use which algorithm**."

**Rule of Thumb**:

"So when you have **a lot of features**, a **random forest will work better**. And when you have a **smaller number of features**, usually the **gradient boosting will do better**."

**Summary Table**:

| Number of Features | Best Choice | Reason |
|-------------------|-------------|---------|
| Small (< ~50)     | Gradient Boosting | Examines all features carefully |
| Large (> ~100)    | Random Forest | Feature subsampling provides diversity |

"And as I **mentioned before**, **it all depends on data too**. But **in general, that's the trend**."

### Training Time with Many Features

"**We can also think about the time**. So as you can see the **boosting algorithm takes much longer time than random forest**."

**Why Boosting is Slower** (with many features):

"And it's **not surprising** because we have a **lot of number of features**, all the **gradient boosting algorithm**, they **inspect all the features** whereas a **random forest will take only subset of features** and **by default it a square root**."

**Feature Inspection**:
- **Random Forest**: Uses $\sqrt{p}$ features per split
  - Example: $\sqrt{145} \approx 12$ features
- **Gradient Boosting (default)**: Uses all $p$ features
  - Example: All 145 features

"So about **12 features they will only look at** and the **other boosting algorithm they will look at all 145 features here**."

### Max_features Option in Gradient Boosting

"So there is an **interesting feature in gradient boosting in Sklearn library** that it can take an **option called max features**."

**Solution**: "So you can **actually set it to random sample the features**."

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(max_features='sqrt')  # Sample sqrt(p) features
```

**Comparison**:
- **AdaBoost**: "Whereas **other boost algorithm doesn't have this option**. So it will **consider all the number of features**..."
- **Gradient Boosting**: "...whereas **gradient boosting**, if you **set to do something similar to random forest**, it will **run faster**."

### Trade-off with Feature Sampling

"So you can **save some time** at the **expense of a slight performance drop**."

**Cost-Benefit**:
- ✅ Faster training
- ❌ Slightly lower accuracy

"However, I think **if you have a lot of features**, it can be **worthwhile**."

**Recommendation**: With high-dimensional data, use `max_features` to speed up Gradient Boosting.

---

## 7. Other Useful Packages

### XGBoost

"So let me just **mention briefly other useful packages**. **XGBoost is external library**. So it's **not part of Sklearn**."

**Full Name**: "**XGBoost is an acronym for extreme gradient boost**..."

#### What Makes XGBoost Different?

"...and **nothing very different from gradient boosting**. But they **implement some other tricks**..."

**Additional Features**:
1. **Regularization**: "such as a **regularization**..."
2. **Data Sampling**: "...and **random sampling of the data**..."
3. **Feature Sampling**: "...and **random sampling of features like random forest do**."

#### Advantages

"So **XGBoost is a time efficient**, also **provide a good performance because of built in regularization**."

**Key Benefits**:
- Fast (optimized C++ implementation)
- Built-in regularization (L1 and L2)
- Handles missing values
- Parallel processing
- Cross-validation built-in

**Usage**:
```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,  # Row sampling
    colsample_bytree=0.8,  # Column sampling
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=1.0   # L2 regularization
)
```

### LightGBM

"**lightGBM is another external package** that's **not part of Sklearn**..."

#### How It Works

"...and it **makes the boosting faster by binning the value of each feature**."

**Binning Concept**: "So if the **feature has some continuous values a lot**, instead of **looking into all this chart values**, it can **be in larger size like this**, so it can **split faster**."

**Example**:
```
Original feature values: [0.1, 0.15, 0.2, 0.23, 0.25, ...]
After binning: [Bin1, Bin1, Bin2, Bin2, Bin2, ...]
```

Instead of sorting all unique values, just consider bin boundaries.

"So **that way it can be useful**."

**Advantages**:
- Very fast training
- Memory efficient
- Handles large datasets well
- Good accuracy

### Sklearn's Histogram-based Gradient Boosting

"**Sklearn also has a counter part to this one**."

```python
from sklearn.ensemble import HistGradientBoostingClassifier
```

"So I think you can **get similar results from Sklearn library**."

**HistGradientBoosting** (added in sklearn 0.21):
- Similar to LightGBM
- Uses histogram-based algorithm
- Much faster than standard GradientBoosting on large datasets
- Native to sklearn (no external dependency)

### ExtraTree (Extremely Randomized Trees)

"**ExtraTree is similar to random forest**. It's also **part of Sklearn library**."

**Full Name**: "**ExtraTree means extreme randomized tree**."

#### How It Works

"It **works very similar to random forest in Sklearn**. And the **only difference** is that it **doesn't do bugging**, so **no bugging**, but it **still randomly sampled the features**..."

**Key Differences from Random Forest**:

1. **No Bootstrap Sampling**: Uses entire dataset (no bagging)
2. **Random Splits**: "...and also **why it's extreme randomized** because it **picks split value randomly** instead of **doing the best split**."

**Process**:
- Random Forest: Find *best* threshold for selected features
- ExtraTree: Pick *random* threshold for selected features

**Trade-offs**:
- Faster training (no need to find optimal splits)
- More randomness (may need more trees)
- Sometimes comparable or better performance

**Usage**:
```python
from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier(
    n_estimators=100,
    max_features='sqrt'
)
```

### Complete List in Sklearn

"**Here is a full list of ensemble methods in Sklearn libraries**."

> **Slide Visualization**: 
> A table or list showing:
> - AdaBoostClassifier / AdaBoostRegressor
> - BaggingClassifier / BaggingRegressor
> - RandomForestClassifier / RandomForestRegressor
> - GradientBoostingClassifier / GradientBoostingRegressor
> - HistGradientBoostingClassifier / HistGradientBoostingRegressor
> - ExtraTreesClassifier / ExtraTreesRegressor
> - VotingClassifier / VotingRegressor
> - StackingClassifier / StackingRegressor

#### Understanding the Categories

"So we **talked about other boost** and they have **both classification and regression**..."

"...and **bugging classifier would be for something like random forest**. We are **random sampling on features**. So it has **just a bugging part**..."

**BaggingClassifier**: Generic bagging with any base estimator (not necessarily trees).

"...and **ExtraTree classifier it's the opposite**. So it does **not have bugging** but it **random samples on the features**."

**Comparison**:
- **BaggingClassifier**: Bagging ✓, Feature sampling ✗
- **RandomForest**: Bagging ✓, Feature sampling ✓
- **ExtraTreesClassifier**: Bagging ✗, Feature sampling ✓

"And **gradient boosting we talked about it** and **random forest we also mentioned** and there are **some other more complicated stuff**..."

"...and this **hist gradient boosting would be something equivalent to lightGBM**."

---

## 8. Module Recap: Ensemble Methods

### Summary Introduction

"So as a **recap**, we **talked about ensemble method in this module**."

### Strengthening Decision Trees

"So **ensemble methods are ways to strengthen the decision tree model**, **decision tree model is a weak longer**."

**The Problem**: "So it can **overfeed** and overall it's a **performance isn't very good**..."

**The Solution**: "...however, by taking **parallel ensemble** or **serial ensemble** we can **make the performance better**."

### Parallel Ensembling: Random Forest

"So **parallel ensemble we talked about random forest**. So **random forest is a parallel method on sampling method**."

**How It Works**: "And **we also talked about boosting method** which is a **serial ensembling method**. So this is just a **growing different trees randomized**, they **look different** because we **random sample data and features** and then you just **average them** right..."

**Key Features**:
- Independent trees
- Bootstrap sampling (data)
- Feature subsampling
- Averaging (regression) or Voting (classification)

**Formula**:
$$f(x) = \frac{1}{B}\sum_{b=1}^{B} h_b(x)$$

### Sequential Ensembling: Boosting

"...and **boosting**, we use a **smaller tree like stump** and try to **fit to the residue**. And then we **additively at these small models to create a stronger model**."

**Key Features**:
- Sequential trees (each depends on previous)
- Fit to residuals or gradients
- Small trees (stumps or shallow)
- Additive combination

**Formula**:
$$f(x) = \sum_{b=1}^{B} \nu \cdot h_b(x)$$

### When to Use Random Forest vs Boosting

"And **he also talked about when to use this random forest versus boosting**."

#### Random Forest: Many Features

"So **random forest usually works better** when there is a **larger number of features**. So **number of features is large**..."

**Reason**: Feature subsampling provides diversity when many features available.

#### Boosting: Fewer Features

"...whereas **boosting can take longer** because it's **additive**. So we **prefer using when the number of features are smaller**."

**Reason**: Sequential nature allows careful examination of all features.

#### Hybrid Approach

"However, it can also **take advantage of random subsampling of features** by using the **max features option**, okay."

**Best of Both Worlds**: Use Gradient Boosting with `max_features` parameter when you have many features but want boosting's sequential power.

### Module Conclusion

"So this is the **end of three ensemble models** and we'll **talk about color method in the next module**."

*[Note: "color method" likely refers to "kernel method" - next module topic]*

---

## 9. Python Implementation

### Complete Gradient Boosting Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import (GradientBoostingClassifier, 
                              GradientBoostingRegressor,
                              RandomForestClassifier,
                              AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import time

np.random.seed(42)

# ===============================================
# EXPERIMENT 1: Few Features (like Data 1 & 2)
# ===============================================

print("=" * 70)
print("EXPERIMENT 1: FEW FEATURES (20 features, 5000 samples)")
print("=" * 70)

# Generate data with few features
X_few, y_few = make_classification(
    n_samples=5000, n_features=20, n_informative=15,
    n_redundant=5, random_state=42
)

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_few, y_few, test_size=0.3, random_state=42
)

# Models
models_few = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                     max_depth=3, random_state=42)
}

print("\nPerformance Comparison (Few Features):")
print("-" * 70)

results_few = {}
for name, model in models_few.items():
    start = time.time()
    model.fit(X_train_f, y_train_f)
    train_time = time.time() - start
    
    train_acc = accuracy_score(y_train_f, model.predict(X_train_f))
    test_acc = accuracy_score(y_test_f, model.predict(X_test_f))
    
    results_few[name] = {'train': train_acc, 'test': test_acc, 'time': train_time}
    
    print(f"{name:20s} | Train: {train_acc:.4f} | Test: {test_acc:.4f} | Time: {train_time:.2f}s")

# ===============================================
# EXPERIMENT 2: Many Features (like Data 3)
# ===============================================

print("\n" + "=" * 70)
print("EXPERIMENT 2: MANY FEATURES (145 features, 3000 samples)")
print("=" * 70)

# Generate data with many features
X_many, y_many = make_classification(
    n_samples=3000, n_features=145, n_informative=100,
    n_redundant=45, random_state=42
)

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_many, y_many, test_size=0.3, random_state=42
)

# Models (adding GB with max_features)
models_many = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                     max_depth=3, random_state=42),
    'GB (max_features)': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                     max_depth=3, max_features='sqrt',
                                                     random_state=42)
}

print("\nPerformance Comparison (Many Features):")
print("-" * 70)

results_many = {}
for name, model in models_many.items():
    start = time.time()
    model.fit(X_train_m, y_train_m)
    train_time = time.time() - start
    
    train_acc = accuracy_score(y_train_m, model.predict(X_train_m))
    test_acc = accuracy_score(y_test_m, model.predict(X_test_m))
    
    results_many[name] = {'train': train_acc, 'test': test_acc, 'time': train_time}
    
    print(f"{name:20s} | Train: {train_acc:.4f} | Test: {test_acc:.4f} | Time: {train_time:.2f}s")

print("\n" + "=" * 70)
print("KEY OBSERVATIONS:")
print("=" * 70)
print("Few Features:  Gradient Boosting > AdaBoost > Random Forest")
print("Many Features: Random Forest > Gradient Boosting")
print("Speed:         Random Forest fastest with many features")
print("               GB with max_features helps speed")
```

### Visualizing Learning Rate Effects

```python
# ===============================================
# LEARNING RATE EFFECTS
# ===============================================

print("\n" + "=" * 70)
print("LEARNING RATE vs NUMBER OF TREES")
print("=" * 70)

learning_rates = [0.01, 0.1, 1.0]
n_estimators_range = [10, 20, 50, 100, 200]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, lr in enumerate(learning_rates):
    train_scores = []
    test_scores = []
    
    for n_est in n_estimators_range:
        gb = GradientBoostingClassifier(
            n_estimators=n_est,
            learning_rate=lr,
            max_depth=3,
            random_state=42
        )
        gb.fit(X_train_f, y_train_f)
        
        train_scores.append(accuracy_score(y_train_f, gb.predict(X_train_f)))
        test_scores.append(accuracy_score(y_test_f, gb.predict(X_test_f)))
    
    ax = axes[idx]
    ax.plot(n_estimators_range, train_scores, 'b-o', label='Train', linewidth=2)
    ax.plot(n_estimators_range, test_scores, 'r-s', label='Test', linewidth=2)
    ax.set_xlabel('Number of Trees')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Learning Rate = {lr}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Highlight overfitting
    if test_scores[-1] < max(test_scores):
        ax.axvline(x=n_estimators_range[test_scores.index(max(test_scores))],
                   color='green', linestyle='--', alpha=0.5, label='Optimal')

plt.tight_layout()
plt.savefig('learning_rate_comparison.png', dpi=300)
plt.show()

print("\nNote: High learning rate (1.0) shows overfitting with many trees")
print("      Low learning rate (0.01) is more stable but needs more trees")
```

### Comparing Loss Functions

```python
# ===============================================
# DIFFERENT LOSS FUNCTIONS
# ===============================================

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Generate regression data
X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=20, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

print("\n" + "=" * 70)
print("DIFFERENT LOSS FUNCTIONS (Regression)")
print("=" * 70)

loss_functions = ['squared_error', 'absolute_error', 'huber']

for loss in loss_functions:
    gb_reg = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        loss=loss,
        random_state=42
    )
    gb_reg.fit(X_train_r, y_train_r)
    
    y_pred = gb_reg.predict(X_test_r)
    mse = mean_squared_error(y_test_r, y_pred)
    mae = mean_absolute_error(y_test_r, y_pred)
    
    print(f"\nLoss Function: {loss:15s}")
    print(f"  Test MSE: {mse:8.2f}")
    print(f"  Test MAE: {mae:8.2f}")
```

### Feature Importance Analysis

```python
# ===============================================
# FEATURE IMPORTANCE
# ===============================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE COMPARISON")
print("=" * 70)

# Train models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

rf.fit(X_train_f, y_train_f)
gb.fit(X_train_f, y_train_f)

# Get importances
rf_importances = rf.feature_importances_
gb_importances = gb.feature_importances_

# Plot top 10 features
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (model_name, importances) in enumerate([('Random Forest', rf_importances),
                                                   ('Gradient Boosting', gb_importances)]):
    indices = np.argsort(importances)[::-1][:10]
    
    axes[idx].bar(range(10), importances[indices])
    axes[idx].set_xlabel('Feature Rank')
    axes[idx].set_ylabel('Importance')
    axes[idx].set_title(f'{model_name} - Top 10 Features')
    axes[idx].set_xticks(range(10))
    axes[idx].set_xticklabels([f'F{i}' for i in indices], rotation=45)

plt.tight_layout()
plt.savefig('feature_importance_comparison.png', dpi=300)
plt.show()
```

---

## 10. Practice Problems

### Problem 1: Gradient Calculation

**Question**: For the following loss functions, calculate the negative gradient $g = -\frac{\partial L}{\partial f}$:

a) Squared loss: $L = \frac{1}{2}(y - f)^2$
b) Absolute loss: $L = |y - f|$
c) Log loss (binary): $L = -[y \log(p) + (1-y)\log(1-p)]$ where $p = \sigma(f) = \frac{1}{1+e^{-f}}$

**Solution**:

**Part a): Squared Loss**

$$L = \frac{1}{2}(y - f)^2$$

Calculate derivative:
$$\frac{\partial L}{\partial f} = \frac{1}{2} \cdot 2(y - f) \cdot (-1) = -(y - f) = f - y$$

Negative gradient:
$$g = -\frac{\partial L}{\partial f} = -(f - y) = y - f$$

**Answer**: $g = y - f$ (the residual!)

**Part b): Absolute Loss**

$$L = |y - f|$$

Calculate derivative:
$$\frac{\partial L}{\partial f} = -\text{sign}(y - f)$$

where $\text{sign}(x) = \begin{cases} +1 & \text{if } x > 0 \\ 0 & \text{if } x = 0 \\ -1 & \text{if } x < 0 \end{cases}$

Negative gradient:
$$g = -\frac{\partial L}{\partial f} = \text{sign}(y - f)$$

**Answer**: $g = \text{sign}(y - f)$

**Interpretation**: For absolute loss, the gradient only indicates direction (+1 or -1), not magnitude. This makes the algorithm robust to outliers.

**Part c): Log Loss (Binary Classification)**

$$L = -[y \log(p) + (1-y)\log(1-p)]$$

where $p = \sigma(f) = \frac{1}{1+e^{-f}}$

**Step 1**: Find $\frac{\partial p}{\partial f}$:
$$\frac{\partial p}{\partial f} = \frac{\partial}{\partial f}\left[\frac{1}{1+e^{-f}}\right] = \frac{e^{-f}}{(1+e^{-f})^2} = p(1-p)$$

**Step 2**: Calculate $\frac{\partial L}{\partial f}$ using chain rule:
$$\frac{\partial L}{\partial f} = \frac{\partial L}{\partial p} \cdot \frac{\partial p}{\partial f}$$

$$\frac{\partial L}{\partial p} = -\left[\frac{y}{p} - \frac{1-y}{1-p}\right]$$

$$\frac{\partial L}{\partial f} = -\left[\frac{y}{p} - \frac{1-y}{1-p}\right] \cdot p(1-p)$$

$$= -[y(1-p) - (1-y)p] = -[y - yp - p + yp] = -(y - p) = p - y$$

Negative gradient:
$$g = -\frac{\partial L}{\partial f} = -(p - y) = y - p$$

**Answer**: $g = y - p$ (difference between true label and predicted probability)

---

### Problem 2: When to Use Which Ensemble Method

**Question**: You're working on three different projects:

**Project A**: Predicting house prices with 15 features (size, bedrooms, location, age, etc.)

**Project B**: Image classification with 2048 features (extracted from a neural network)

**Project C**: Fraud detection with 50 features, known to have ~5% mislabeled data

For each project, recommend an ensemble method and explain why.

**Solution**:

**Project A: House Prices (15 features)**

**Recommendation**: **Gradient Boosting**

**Reasoning**:
1. **Few features** (15): Gradient Boosting can examine all features carefully
2. **Regression task**: GradientBoostingRegressor handles this well
3. **Sequential learning**: Can capture subtle interactions between features
4. **Accuracy**: Generally higher accuracy than Random Forest with few features

**Implementation**:
```python
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8
)
```

**Project B: Image Classification (2048 features)**

**Recommendation**: **Random Forest**

**Reasoning**:
1. **Many features** (2048): Random Forest excels with high dimensionality
2. **Feature subsampling**: Uses $\sqrt{2048} \approx 45$ features per split
3. **Speed**: Much faster than Gradient Boosting with 2048 features
4. **Parallelization**: Can train trees in parallel
5. **Overfitting protection**: Feature subsampling prevents overfitting

**Alternative**: Gradient Boosting with `max_features='sqrt'` if you need slightly better accuracy and can afford longer training time.

**Implementation**:
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    max_features='sqrt',
    n_jobs=-1  # Use all CPU cores
)
```

**Project C: Fraud Detection with Mislabeled Data (50 features)**

**Recommendation**: **Gradient Boosting** (NOT AdaBoost)

**Reasoning**:
1. **Medium number of features** (50): Both methods could work
2. **Mislabeled data** (5%): Critical factor!
   - **AdaBoost**: Sample reweighting will amplify mislabeled points → Bad
   - **Gradient Boosting**: More robust to mislabeled data → Good
3. **Accuracy**: Gradient Boosting typically performs better
4. **Fraud detection**: High stakes require robustness

**Implementation**:
```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.05,  # Lower LR for stability
    max_depth=3,
    subsample=0.8,  # Additional regularization
    min_samples_split=5  # Avoid overfitting to noise
)
```

**Additional Consideration**: Use cross-validation with stratified folds to ensure mislabeled samples don't all end up in training set.

---

### Problem 3: Learning Rate Trade-off

**Question**: You're training a Gradient Boosting model. After experimentation:
- Config A: `learning_rate=0.1`, `n_estimators=100` → Test Accuracy: 0.85
- Config B: `learning_rate=0.01`, `n_estimators=1000` → Test Accuracy: 0.88

Training time for Config B is 10× longer. Your boss wants faster predictions.

a) Does using Config B (more trees) make predictions slower?
b) Which configuration should you choose and why?
c) What's a compromise solution?

**Solution**:

**Part a): Prediction Speed**

**Answer**: **Yes**, Config B will make predictions slower (though only ~10× for this specific case).

**Reasoning**:
- Prediction time is proportional to number of trees
- Config A: 100 trees
- Config B: 1000 trees
- Each prediction must query all trees and sum results

**Calculation**:
$$\text{Prediction Time}_B \approx 10 \times \text{Prediction Time}_A$$

However, this is still typically very fast (milliseconds even for 1000 trees).

**Part b): Which to Choose?**

**Answer**: **It depends on the constraint**, but likely **Config B**.

**Analysis**:

| Aspect | Config A (lr=0.1, n=100) | Config B (lr=0.01, n=1000) |
|--------|--------------------------|----------------------------|
| Test Accuracy | 0.85 | 0.88 (3% better) |
| Training Time | Fast | 10× slower |
| Prediction Time | Fast | 10× slower |
| Generalization | Good | Better |

**If training is one-time**: Choose **Config B**
- Training time is a one-time cost
- Better accuracy (0.88 vs 0.85) is worth it
- Prediction time increase is usually acceptable (still very fast)

**If training repeatedly** (e.g., online learning): Consider **Config A**
- Retraining is frequent
- Speed matters more
- 3% accuracy loss might be acceptable

**If prediction speed is critical** (e.g., high-frequency trading): Choose **Config A**
- Sub-millisecond predictions required
- 3% accuracy trade-off acceptable

**Part c): Compromise Solution**

**Solution**: Use **Config C**: `learning_rate=0.05`, `n_estimators=200`

**Expected Results**:
- Training time: ~2× Config A (much better than 10×)
- Prediction time: ~2× Config A
- Accuracy: ~0.86-0.87 (between A and B)

**Additional Optimizations**:

1. **Use `max_features`**:
```python
model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_features='sqrt'  # Faster training
)
```

2. **Early Stopping** (validation-based):
```python
model = GradientBoostingClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    validation_fraction=0.1,
    n_iter_no_change=50,  # Stop if no improvement for 50 rounds
    tol=0.0001
)
```
This automatically finds the right number of trees.

3. **Model Deployment Optimization**:
- Train with Config B (best accuracy)
- Use tree pruning or distillation to create smaller model for production
- Best of both worlds: high accuracy training, fast inference

---

### Problem 4: Gradient Boosting vs Random Forest Trade-offs

**Question**: Complete the following comparison table:

| Aspect | Random Forest | Gradient Boosting |
|--------|---------------|-------------------|
| Training Paradigm | ? | ? |
| Base Learner Depth | ? | ? |
| Feature Handling | ? | ? |
| Overfitting Risk | ? | ? |
| Training Speed | ? | ? |
| Interpretability | ? | ? |
| Hyperparameter Sensitivity | ? | ? |
| Best Use Case | ? | ? |

**Solution**:

| Aspect | Random Forest | Gradient Boosting |
|--------|---------------|-------------------|
| **Training Paradigm** | Parallel (independent trees) | Sequential (each tree depends on previous) |
| **Base Learner Depth** | Deep trees (often fully grown) | Shallow trees (depth 2-5 typical) |
| **Feature Handling** | Samples √p features per split | Uses all p features (unless max_features set) |
| **Overfitting Risk** | Low (averaging reduces variance) | Higher (needs careful tuning of learning_rate) |
| **Training Speed** | Fast (parallelizable) | Slower (sequential, but can parallelize tree-building) |
| **Interpretability** | Moderate (feature importance available) | Moderate (feature importance + partial dependence plots) |
| **Hyperparameter Sensitivity** | Low (robust to default settings) | High (learning_rate, n_estimators, max_depth critical) |
| **Best Use Case** | Many features (high-dimensional data) | Few-to-medium features, need max accuracy |

**Additional Insights**:

**When Random Forest Wins**:
- High-dimensional data (p > 100)
- Need fast training/predictions
- Robust to defaults (less tuning needed)
- Can tolerate slightly lower accuracy

**When Gradient Boosting Wins**:
- Low-to-medium dimensions (p < 50)
- Need maximum accuracy
- Have time for hyperparameter tuning
- Data is clean (no/few mislabeled samples)

**Hybrid Strategy**:
1. Start with Random Forest (fast baseline)
2. If accuracy isn't sufficient, try Gradient Boosting
3. For high dimensions, use GB with `max_features='sqrt'`

---

### Problem 5: Understanding the Algorithm

**Question**: Trace through one iteration of Gradient Boosting (stage b=2) with:
- Loss: Squared error $L = \frac{1}{2}(y - f)^2$
- Current model: $f_1(x) = [3.0, 5.0, 2.0]$
- True labels: $y = [3.5, 6.0, 2.5]$
- Learning rate: $\nu = 0.3$
- New tree predicts: $h_2(x) = [0.5, 0.8, 0.6]$

Calculate:
a) The negative gradients
b) The updated model $f_2(x)$
c) The new residuals

**Solution**:

**Part a): Calculate Negative Gradients**

For squared loss, the negative gradient is:
$$g = -\frac{\partial L}{\partial f} = y - f_1(x)$$

For each sample:
- Sample 1: $g_1 = 3.5 - 3.0 = 0.5$
- Sample 2: $g_2 = 6.0 - 5.0 = 1.0$
- Sample 3: $g_3 = 2.5 - 2.0 = 0.5$

**Answer**: Gradients $g_2 = [0.5, 1.0, 0.5]$

**Interpretation**: These are the residuals—how much each prediction is off.

**Part b): Update Model**

The update rule:
$$f_2(x) = f_1(x) + \nu \cdot h_2(x)$$

For each sample:
- Sample 1: $f_2(x_1) = 3.0 + 0.3 \times 0.5 = 3.0 + 0.15 = 3.15$
- Sample 2: $f_2(x_2) = 5.0 + 0.3 \times 0.8 = 5.0 + 0.24 = 5.24$
- Sample 3: $f_2(x_3) = 2.0 + 0.3 \times 0.6 = 2.0 + 0.18 = 2.18$

**Answer**: $f_2(x) = [3.15, 5.24, 2.18]$

**Part c): Calculate New Residuals**

New residuals:
$$r_2 = y - f_2(x)$$

For each sample:
- Sample 1: $r_1 = 3.5 - 3.15 = 0.35$
- Sample 2: $r_2 = 6.0 - 5.24 = 0.76$
- Sample 3: $r_3 = 2.5 - 2.18 = 0.32$

**Answer**: New residuals $r_2 = [0.35, 0.76, 0.32]$

**Comparison with Previous Residuals**:

| Sample | Old Residual $r_1$ | New Residual $r_2$ | Reduction |
|--------|-------------------|-------------------|-----------|
| 1      | 0.50              | 0.35              | 30% |
| 2      | 1.00              | 0.76              | 24% |
| 3      | 0.50              | 0.32              | 36% |

**Key Observations**:
1. All residuals decreased (good!)
2. Reduction is partial, not complete (due to $\nu = 0.3 < 1$)
3. Tree $h_3$ will now try to predict these new residuals $[0.35, 0.76, 0.32]$
4. Over many iterations, residuals approach zero

**If we used $\nu = 1.0$** (no shrinkage):
- $f_2 = [3.5, 5.8, 2.6]$
- Residuals would be reduced more aggressively
- But risk of overfitting increases

---

## 11. Key Takeaways

**1. Gradient Boosting = Generalized Boosting**:
- Fits trees to gradients of any differentiable loss function
- Residuals are special case (gradient of squared loss)

**2. Why Gradients?**:
- Steepest descent in function space
- More expressive for classification (uses probabilities, not just correct/wrong)
- Flexibility to use different loss functions

**3. Performance Hierarchy**:
- Gradient Boosting ≥ AdaBoost > Single Tree
- GB more robust to mislabeled data

**4. Gradient Boosting vs Random Forest**:
- **Few features** (<50): Gradient Boosting usually better
- **Many features** (>100): Random Forest usually better
- **Speed**: RF faster with many features; GB can use `max_features` to speed up

**5. Important Packages**:
- **XGBoost**: External, fast, regularization built-in
- **LightGBM**: External, histogram-based, very fast
- **HistGradientBoosting**: Sklearn's LightGBM equivalent
- **ExtraTree**: Sklearn, extremely randomized (no bagging, random splits)

**6. Ensemble Methods Recap**:
- **Parallel** (RF): Independent trees, bootstrap+feature sampling, average
- **Sequential** (Boosting): Dependent trees, fit residuals/gradients, additive

**7. Choosing the Right Method**:
- High-dimensional → Random Forest
- Max accuracy with few features → Gradient Boosting
- Clean data → Any ensemble method
- Mislabeled data → Gradient Boosting (NOT AdaBoost)
- Speed critical → Random Forest or HistGB

---

## Glossary

- **Gradient Boosting**: Sequential ensemble fitting trees to negative gradients of loss
- **Loss Function**: General measure of prediction error (MSE, log loss, etc.)
- **Negative Gradient**: Direction of steepest descent for loss minimization
- **Steepest Descent**: Optimization direction that reduces loss most rapidly
- **HistGradientBoosting**: Histogram-based GB for faster training on large datasets
- **XGBoost**: External library with regularization and additional optimizations
- **LightGBM**: External library using histogram binning for speed
- **ExtraTree**: Extremely randomized trees with random split points
- **max_features**: Parameter to randomly subsample features per split

---

**End of Module 5: Ensemble Methods**

