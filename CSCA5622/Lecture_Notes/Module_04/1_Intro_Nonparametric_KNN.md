# Introduction to Non-parametric Models and K-Nearest Neighbors
**CSCA5622 - Module 04**

---

## 📚 Overview

This document provides a comprehensive introduction to **non-parametric models**, with focus on **K-Nearest Neighbors (KNN)** as the simplest machine learning algorithm. Topics include parametric vs non-parametric comparison, KNN mechanics, distance metrics, bias-variance tradeoff, and the curse of dimensionality.

All concepts explained from the lecture transcript.

---

## 1. Parametric vs Non-Parametric Models

### 🔍 Parametric Models

From lecture:
> "Parametric models that look like this, the **model has the parameters** and then the model may or may not have a **hyper parameters**. And the model takes the **features** and make a **prediction**."

**Structure:**
```
Features (X) → Model(θ, hyperparameters) → Prediction (ŷ)
                     ↑                            ↓
                     └──── Loss(ŷ, y) ← Error ───┘
```

**Process:**

From lecture:
> "And it's going to compare the prediction value with the target value and then produce this **error**. And this error will **optimize the parameter values** so that this prediction value will be **as close as possible** to the target value."

**Key characteristic:** Has learnable parameters (θ) optimized during training

**Examples:**
- Linear Regression: β₀, β₁, ..., βₚ
- Logistic Regression: weights and bias
- Neural Networks: weights and biases

### 🔍 Non-Parametric Models

From lecture:
> "In **non-parametric models**, the **parameter doesn't exist**. Therefore, the question is, well, **how do we optimize the model** such that this prediction value gets **as close as possible** to the target value?"

**Key differences:**

From lecture:
> "The model has a **hyper parameters usually**, they may not have, but usually they should have. And then these non-parametric models, **sometimes they use it as an error or sometimes they don't**, but uses some **other quantity** to optimize the model."

**Characteristics:**
- ✗ No learnable parameters to optimize
- ✓ Have hyperparameters (user-specified)
- ✓ Use different optimization approaches
- ✓ Often "memorize" training data

### 📊 Comparison Table

| Aspect | Parametric | Non-Parametric |
|--------|------------|----------------|
| **Parameters** | Yes (learned from data) | No |
| **Hyperparameters** | Optional | Usually yes |
| **Optimization** | Via loss function gradient | Via other methods |
| **Training data storage** | Not needed after training | Often needed for prediction |
| **Examples** | Linear/Logistic Regression | KNN, Decision Trees |

---

## 2. Examples of Non-Parametric Models

From lecture:
> "So examples of non-parametric models are **K-Nearest neighbor** which is the **simplest motion learning algorithm**. And the **decision trees** that uses a tree like model, we will get to that later and **support vector machine** which uses **distance between the points** and the **decision boundary or hyperplane**."

### 📋 Three Main Non-Parametric Models

**1. K-Nearest Neighbors (KNN)**
- Simplest ML algorithm
- Uses distance to nearest neighbors
- No training phase (lazy learning)

**2. Decision Trees**
- Tree-like model
- Splits based on feature values
- Interpretable rules

**3. Support Vector Machine (SVM)**
- Uses distance to decision boundary/hyperplane
- Maximizes margin between classes
- Can use kernel trick

---

## 3. How K-Nearest Neighbors Works

### 🔍 The Basic Idea

From lecture:
> "So imagine I have **training data** that looks like this **red dots and blue dots** and the task is to **classify my data points**, whether it's red or blue. And let's say I have **data points to classify here** and I don't know whether it's red or blue."

**KNN Principle:**

From lecture:
> "And K-Nearest neighbor says that just **take K numbers of nearest neighbor** and **classify to the majority of them**."

### 📊 Example: Different K Values

**K = 1 (1 nearest neighbor):**

From lecture:
> "So let's say if I have **K=1**, I take the **closest one**, I take one nearest neighbor, which is **red**, so my **green point is going to be red**."

```
Green point → Find 1 nearest neighbor → Red
Prediction: Red
```

**K = 3 (3 nearest neighbors):**

From lecture:
> "In this case, if I had **three neighbors**, then I have **two blues and one red**. Therefore, my green points will be **classified as two** by the **majority rule of voting rule**."

```
Green point → Find 3 nearest neighbors → [Blue, Blue, Red]
Majority vote: Blue (2 out of 3)
Prediction: Blue
```

**K = 5 (5 nearest neighbors):**

From lecture:
> "If I had 5, if I had a **K=5**, then now I have a **three red neighbors and two blue neighbors**. So it's going to be **classified as red now**."

```
Green point → Find 5 nearest neighbors → [Red, Red, Red, Blue, Blue]
Majority vote: Red (3 out of 5)
Prediction: Red
```

### 🎯 Key Observations

**1. Why Odd K Values?**

From lecture:
> "You might have noticed two things, first, this green points kind of **flips between red and blue** and second, the **choice of K number is odd number**. Why is that? Because if I have an **even number**, then I might have a **tie**, I might have just **two red and two blues** and I don't know what to choose then. So that's why we **usually use all the number for the K values** for KNN model."

**Reason:** Avoid ties in majority voting

**Example of tie problem:**
```
K = 4: [Red, Red, Blue, Blue] → Tie! Which class?
K = 5: [Red, Red, Red, Blue, Blue] → Red wins (no tie)
```

**2. Points on Decision Boundary**

From lecture:
> "Another thing you might have noticed is that **why is this green swing between red and blue**? It is just happened to be that this **green sits on the distant boundary**, for example, let's say **this side is red, and this side is blue**, and this **green just at the right in between**, so you can swing. But that's not very important, I just wanted to show it **can happen**."

---

## 4. KNN for Regression

From lecture:
> "And another question you might ask is that **can, KNN do other than classification**. **Yes, you can**, you can also do the **regression**. The difference would be that **instead of taking the majority rule** here, if it's a regression, it's going to **take the average** of these five values, for example, when the K=5."

### 📊 Classification vs Regression

| Task | Output | Aggregation Method | Example |
|------|--------|-------------------|---------|
| **Classification** | Class label | Majority vote | [Red, Red, Blue, Red, Red] → Red |
| **Regression** | Continuous value | Mean (average) | [2.5, 3.1, 2.8, 3.0, 2.9] → 2.86 |

**Regression example:**
```python
# K = 5 nearest neighbors with values
neighbors = [12.5, 13.2, 11.8, 12.9, 13.1]

# Prediction = average
prediction = mean(neighbors) = 12.7
```

---

## 5. Distance Metrics

From lecture:
> "KNN uses **distance metric**, for example, **Manhattan distance and Euclidean distance**, **Euclidean distance** is a **simple distance** between these two points. Whereas the **Manhattan distance** would be the **delta X + delta Y**, for example. There are **more distance metrics** that you can use, but **these two are pretty popular**."

### 📐 Euclidean Distance

**Definition:** Straight-line distance between two points

**Formula (2D):**
\[
d_{Euclidean} = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
\]

**General formula (n dimensions):**
\[
d_{Euclidean} = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
\]

**Example:**
```
Point A = (1, 2)
Point B = (4, 6)

d = √[(4-1)² + (6-2)²]
  = √[9 + 16]
  = √25
  = 5
```

### 📐 Manhattan Distance

**Definition:** Sum of absolute differences (like walking on a grid)

**Formula (2D):**
\[
d_{Manhattan} = |x_2 - x_1| + |y_2 - y_1|
\]

**General formula (n dimensions):**
\[
d_{Manhattan} = \sum_{i=1}^{n} |x_i - y_i|
\]

**Example:**
```
Point A = (1, 2)
Point B = (4, 6)

d = |4-1| + |6-2|
  = 3 + 4
  = 7
```

### 🎯 Visual Comparison

```
     B(4,6)
      •
      |
      |      Euclidean: straight line (5.0)
    / |      Manhattan: along grid (7.0)
   /  |
  /   |
 /    |
•─────┘
A(1,2)
```

**When to use which:**
- **Euclidean:** Most common, works well in many cases
- **Manhattan:** Better when features are not comparable scales, or movement is grid-based

---

## 6. Decision Boundaries with Different K Values

### 🔍 Iris Dataset Example

From lecture:
> "This is from **famous Irish data set** to display more conveniently, I only use **two features** and then **two classes of Iris**. So you can see some **blue points and red points** are kind of **mixed in some area**. So it's hard to separate."

From lecture:
> "So this **two graph shows** that the **decision boundary KNN model**, each case **K values are different**. And now I have a **question for you**, **which of this case have a smaller K number**?"

### 📊 Answer and Explanation

From lecture:
> "The answer was the **left one has a smaller K value**, in fact, it was **K=1**."

### 🎯 How K Affects Decision Boundaries

From lecture:
> "And as you can see here, as the **K increases**, the **decision boundary becomes smoother and smoother**. When the K is a **small**, let's say 1, then I only have to **consider just one neighbor**. So if my data point is here, I only consider this one and the next time my data points here, then I consider this one. Therefore, the **system boundary can be very granular**. Therefore, it can **fit to the very complex data** like this."

**Small K (e.g., K=1):**
```
Decision boundary: Very wiggly, complex
Behavior: Follows training data closely
Result: Can capture complex patterns
Risk: Overfitting
```

**Large K (e.g., K=61):**

From lecture:
> "Whereas if I have to **consider many neighbors**, when I'm here, I will have to consider **61 neighbors** and then **count red versus blue** and decide which one is more dominant here, in this case, red. So the **decision boundary can be very smooth** in this way because I'm kind of **averaging out a lot of data points**."

```
Decision boundary: Smooth, simple
Behavior: Averages over many neighbors
Result: Simpler patterns
Risk: Underfitting
```

### 📊 Visual Summary

```
K = 1:    Highly complex, wiggly boundary
K = 3:    Moderately complex boundary
K = 11:   Smooth boundary
K = 51:   Very smooth, simple boundary
```

---

## 7. Bias-Variance Tradeoff in KNN

From lecture:
> "All right, so this might remind you the **concept of bias and variants**. So how is the **bias and variance in KNN**?"

### 🔍 Question 1: Which K Has Larger Bias?

From lecture:
> "So here are some keys which model has a **larger bias** when the **K is small** or when the **K is large**? The answer is when we have a **larger k**, we have a **larger bias**."

**Answer: Larger K → Larger Bias**

**Reasoning:**

From lecture:
> "Why is that? Because a k in the model with a **larger k is a simpler model** and it's **less flexible** as you saw in the previous slides that the **decision boundaries are much smoother** for when k is larger. The **simpler model** which is a **less flexible model** has a **larger bias** because it **simplifies the real world data**. Therefore, it introduces **more bias** and **more assumption** about the data."

### 🔍 Question 2: Which K Has Larger Variance?

From lecture:
> "All right, another question which model has a **larger variance** when the k is small or when the k is large? So **larger variances happens** when the model is **more flexible**. Therefore, we can guess that the **small k KNN** should have a **larger variance**."

**Answer: Smaller K → Larger Variance**

### 📊 Bias-Variance Summary Table

| K Value | Model Complexity | Flexibility | Bias | Variance | Risk |
|---------|-----------------|-------------|------|----------|------|
| **Small (e.g., 1)** | High | Very flexible | Low | High | Overfitting |
| **Medium (optimal)** | Moderate | Balanced | Moderate | Moderate | Best generalization |
| **Large (e.g., 100)** | Low | Less flexible | High | Low | Underfitting |

### 🎯 Key Insight

```
K ↓ (smaller) → More complex → Lower bias, Higher variance → Overfitting
K ↑ (larger)  → Simpler     → Higher bias, Lower variance → Underfitting
```

---

## 8. Determining Optimal K Value

### 📊 Training vs Test Error

From lecture:
> "So how do we **determine the optimal K value**? As you saw previously that the **training error goes down** as the **model complexity increases**. **Test error goes down** in the beginning, but then it **has an optimal value** and it **goes up again** as the model complexity increases because the model is **too complex to the data**. Therefore, it's **overfitting**, it's **not generalizing very well**."

### 📈 Error Curves

```
Error
  |
  |  Test Error
  |     \
  |      \___/←── Optimal K (minimum test error)
  |          \
  |           \
  |            Training Error
  |             \___________
  |_________________________ K
 small                    large
(complex)              (simple)
```

### 🎯 Finding Optimal K

From lecture:
> "So that point happens here. So around the **K=21 the optimal value happens** and the **test error is the minimized** whereas it can go up if it keep increasing the motor complexity."

**Process:**
1. Try different K values (e.g., 1, 3, 5, 7, ..., 51)
2. Compute test error (or cross-validation error) for each K
3. Select K with minimum test error

### 📊 Relationship Summary

From lecture:
> "So you can see that **this side is more complex model**. When the **K value gets smaller**, the model gets **more complex**, **more flexible**. And the **other side becomes simpler**, it has a **larger bias**, **larger variance**."

**Note:** Typo in original - should be "larger bias, **smaller** variance" for large K

```
Small K ←────────────────→ Large K
Complex                    Simple
Flexible                   Rigid
Low bias                   High bias
High variance              Low variance
Overfit                    Underfit
```

---

## 9. Time Complexity of KNN

### 📊 Computational Cost

From lecture:
> "As we saw, it's a **simple and memory-based algorithm**, **memory-based means** that it does need **all the training data in order to influence**. And its **time complexity** is when the **order of number of samples times the number of features**."

**Formula:**
\[
\text{Time Complexity} = O(n \times m)
\]

Where:
- n = number of samples
- m = number of features

From lecture:
> "There can be **K here as well**, but if you had to **rank K neighbors anyway**, then there are some **clever algorithms**. So, it's not when you measure the time actually. It **doesn't go very linearly** because there are some **better sorting algorithm**. But anyway, time complexity is **roughly nm**, where this is number of samples and this is number of features."

### 🧪 Experimental Validation

**Experiment 1: Varying Number of Samples**

From lecture:
> "So this data comes from **ML** and this has more than **90,000 samples** with the **100 features**. And the task is to **classify binary class**. So at **K=9** which was the optimal value. The **train time for KNN linearly increases** as the **number of sample increases**."

**Result:** Training time ∝ number of samples (linear relationship)

**Experiment 2: Varying Number of Features**

From lecture:
> "Another data supports that as well. Instead of **increasing the number of samples** this time by **increasing number of features**, the **train time also goes linearly**. So this data was classifying three different **boundary types of the gene sequences**. And has **180 binary features**."

**Result:** Training time ∝ number of features (linear relationship)

### 📊 KNN vs Logistic Regression Speed

From lecture:
> "Training the **logistic regression** on the same data set and measuring the training time can give some **comparison**. Surprisingly, this **KNN model is very efficient**. Well, it is usually said the **KNN is a slow** because it has to **measure all the distance** between the points in the training data set. So it's said to be slow, but with this **number of samples**, it's **not terribly bad**. It has a training time, **very small** and compared to the regression, it's **surprisingly fast**."

**Why LR might be slower:**

From lecture:
> "And logistic regression might be slow just because it uses **offensive second derivative optimization algorithm** that can **run many times** as well."

**Key insight:** KNN training is just storing data (fast!), but prediction is slow. Logistic regression training is slow (optimization), but prediction is fast.

---

## 10. Optimal K Example

From lecture:
> "This graph shows that there is an **optimal K value** for the KNN model. The **test accuracy** has some **optimal value at certain K value** which is **7 in this case**."

### 📊 Finding K=7 as Optimal

**Process:**
1. Train KNN with different K values (1, 3, 5, 7, 9, ...)
2. Evaluate test accuracy for each K
3. Plot accuracy vs K
4. Identify maximum accuracy

**Typical pattern:**
```
Accuracy
   |
   |        ___/‾‾\___  ← Peak at K=7
   |      /           \
   |    /               \
   |  /                   \
   |/_______________________\
                              K
  1  3  5  7  9 11 13 15 17
```

---

## 11. Curse of Dimensionality

### 🔍 What Is It?

From lecture:
> "So another property that KNN has is that it **suffer severely from curse of dimensionality**. What is the **curse of dimensionality** course of dimensionality is that the model **performs very poorly** when we have a **lot of features**. So that there is a **curse when the dimension is a high**."

### 🧪 Experimental Setup

From lecture:
> "To see that we're going to do some experiment. And I just applied to **explain the variance ratio from PCA**. By transforming our **180 features** using **principal component analysis** that it's going to rank the **combination of these 180 features** in order of importance that is called **explained variance ratio**."

### 📊 Feature Importance Analysis

From lecture:
> "And this **gradual increase** of this explained variance ratio tells that **a lot of these features are all important**. If only **a few of these features were important**, then this explained variance ratio graph would look like this like **very sharply increase to point**. That means **more than 90% of variants** will be explained by **just only a few features**."

**Two patterns:**
```
Pattern 1: Few important features
Explained
Variance
   100% |___/‾‾‾‾‾‾‾‾‾‾  ← Sharp rise (good!)
        |  /
     50%| /
        |/_______________
           Features →


Pattern 2: All features important
Explained  
Variance
   100% |            /‾  ← Gradual rise (problem!)
        |          /
     50%|       /
        |    /
        |/_____________
           Features →
```

From lecture:
> "However, this graph shows that it **gradually increases** that means **all features are kind of important**."

### 📊 Performance Degradation

**Logistic Regression (handles high dimensions well):**

From lecture:
> "So with that in mind, let's have a look and compare with the **logistic regression**. So because all, most of all, because **most of these features are important** in logistic regression, you can see that the **test accuracy still increases** as the **number of features increases**."

**KNN (suffers from curse):**

From lecture:
> "However, in **KNN**, as you can see with the **various values of K** for the various K values, it has some **peak value at very small dimension of features** and then it **sharply decrease the performance**."

### 📈 Visualization: K=7 Example

From lecture:
> "So let's fix it to our **optimum value K=7**. And as you can see, the **test accuracy dropped very sharply** as we **increase the number of features** that are including the model."

```
Test
Accuracy
    |  ___
    | /   \___
    |/         \___
    |               \____
    |                    \______
    |_____________________________
         # Features →
      10   50   100   150   180
```

---

## 12. Why Curse of Dimensionality Happens

### 🔍 Intuitive Explanation

From lecture:
> "So why does **dimension that it happened** here? It happens because intuitively the **number of data points** in the **given volume** of this **high dimension sharply decreases** when this **dimension becomes high**. Therefore, we need **more data points** in order to have the **same level of accuracy**. However, with the **fixed data size**, the **concentration of data decreases dramatically**. Therefore we have **degradation of performance in accuracy** when the dimension is too high."

### 📊 Mathematical Insight

**The problem:** As dimensions increase, the volume of space increases exponentially, but we have fixed number of data points.

**Example:**
```
1D space: Need ~10 points to cover line
2D space: Need ~100 points to cover area (10×10)
3D space: Need ~1000 points to cover volume (10×10×10)
nD space: Need ~10^n points to cover space
```

**Result:** Data points become sparse in high dimensions, distances become less meaningful

### 🎯 Special Cases

From lecture:
> "But it's **not that simple**, researchers have found that if the **features are highly correlated** to each other, it may **suffer less** because the **effective dimension is less** than the number of features. But anyway, still **canon suffers from course of dimensionality**."

**Correlated features:** Effective dimensionality is reduced
- Example: height_cm and height_inches are perfectly correlated
- They represent 1 dimension, not 2

---

## 13. Solutions and When to Use KNN

### 🔧 Dealing with Curse of Dimensionality

From lecture:
> "So when this happens, you want to **use smaller number of features** and **avoid it from being high dementia** when you're using KNN."

**Solutions:**
1. **Feature selection:** Keep only most important features
2. **Dimensionality reduction:** Use PCA, t-SNE, etc.
3. **Feature engineering:** Create better features
4. **Use different model:** Switch to models that handle high dimensions better

### 🎯 Model Choice in High Dimensions

From lecture:
> "Also, **not only the KNN** other **machine learning models** they use to **descent the metric** in their algorithm can **suffer from cross of dimensionality**. So you might **choose wisely which model to use** when your **dimension is too high**, unless you can or you want to **reduce the number of features**."

**Models affected by curse:**
- KNN (severely)
- Distance-based clustering
- Some kernel methods

**Models that handle high dimensions better:**
- Logistic Regression (with regularization)
- Decision Trees / Random Forests
- Neural Networks (with proper architecture)

---

## 14. Summary

From lecture:
> "All right, so in this video, we talked about **KNN as an example of a non-parametric model**, which is the **simplest machine learning model**. And we talked about **its property**. It's a **biased variance**, it's a **hyper parameter K** and **how it behaves** when the K increases or K decreases. And it's a **property such as cos of dimensionality**."

### 🎯 Key Concepts

**1. Non-Parametric Models**
- No learnable parameters
- Have hyperparameters
- Examples: KNN, Decision Trees, SVM

**2. KNN Algorithm**
- Find K nearest neighbors
- Classification: Majority vote
- Regression: Average
- Use odd K to avoid ties

**3. Distance Metrics**
- Euclidean: √[(x₂-x₁)² + (y₂-y₁)²]
- Manhattan: |x₂-x₁| + |y₂-y₁|

**4. K Value Effects**
- Small K: Complex, flexible, low bias, high variance, overfit
- Large K: Simple, rigid, high bias, low variance, underfit
- Optimal K: Found via cross-validation (minimize test error)

**5. Bias-Variance Tradeoff**
- K ↓ → Complexity ↑ → Bias ↓, Variance ↑
- K ↑ → Complexity ↓ → Bias ↑, Variance ↓

**6. Computational Properties**
- Time complexity: O(n × m)
- Memory-based (stores all training data)
- Fast "training" (just stores data)
- Slow prediction (must compute distances)

**7. Curse of Dimensionality**
- KNN performs poorly with many features
- Data becomes sparse in high dimensions
- Solution: Reduce features or use different model

### 📋 Decision Guide: When to Use KNN

**✅ Use KNN when:**
- Small to medium number of features (< 20-30)
- Non-linear decision boundaries
- No clear parametric form
- Interpretability is important (can visualize neighbors)
- Small datasets

**❌ Avoid KNN when:**
- High-dimensional data (> 50 features)
- Very large datasets (slow prediction)
- Need fast predictions
- Features have very different scales (need normalization)
- Data has many irrelevant features

### 🔧 Best Practices

1. **Always normalize/standardize features** (especially for Euclidean distance)
2. **Use cross-validation** to find optimal K
3. **Try odd K values** to avoid ties
4. **Reduce dimensionality** if features > 30
5. **Remove irrelevant features** before training
6. **Consider computational cost** for large datasets

---

**End of Lecture Notes - Module 04, Document 1**
