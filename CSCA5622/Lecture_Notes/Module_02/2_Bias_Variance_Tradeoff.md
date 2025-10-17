# Bias-Variance Trade-Off - Detailed Lecture Notes
**CSCA5622 - Module 02**

---

## 📚 Overview

This document explores the **bias-variance trade-off**, explaining why test error behaves the way it does as model complexity changes. Topics include:

- Understanding the U-shaped test error curve
- Graphical explanation using target analogy
- Bias: error from simplification
- Variance: model variability
- The fundamental trade-off relationship
- Mathematical decomposition of test error
- Practical error curve shapes
- Universal applicability across loss functions

All concepts explained with examples from the lecture transcript.

---

## 1. Motivation: The U-Shaped Test Error

### 🔍 The Observed Pattern

From polynomial regression, we saw:

```
Test    │    ╱
Error   │   ╱
        │  ╱╲
        │ ╱  ╲___
        │╱
        └────────────────> Model Complexity
```

**Why does test error go down then up?**

From the lecture:
> "The behavior of test error that goes down first and goes up later as we increase the model flexibility can be explained by **bias-variance trade-off**."

---

## 2. Graphical Explanation: The Target Analogy

### 🎯 Understanding Through Bullets on Target

Imagine shooting bullets at a target. Four scenarios:

#### Scenario 1: Low Bias, Low Variance ✓ (Ideal)

```
        Target
          ☉
        • •
       • ☉ •
        • •
```

From lecture:
> "When the bullets are **well-centered and well-grouped**, they are called the **low bias and low variance**."

**Characteristics:**
- Centered on target (low bias)
- Tightly grouped (low variance)
- This is what we want!

#### Scenario 2: High Bias, Low Variance

```
        Target
          ☉


              • •
             • • •
              • •
```

From lecture:
> "When the bullets are **well-grouped but far away from the target center**, then it has a **high bias** because it's far away from the center or true value, but it has a **low variance** because they are well grouped."

**Characteristics:**
- Far from center (high bias)
- Tightly grouped (low variance)
- Consistent but wrong

#### Scenario 3: Low Bias, High Variance

```
        Target
    •     ☉
              •
       • ☉
  •       
            •
```

From lecture:
> "If the bullets are **quite spread, but it is still well-centered** around the target, then we can say it has a **low bias and high variance**."

**Characteristics:**
- Centered on average (low bias)
- Widely scattered (high variance)
- Correct on average but inconsistent

#### Scenario 4: High Bias, High Variance (Worst)

```
        Target
          ☉

                •
              •
                  •
             •
```

From lecture:
> "If bullets are **not close to the center, but it also has a large spread**, then we say it's a **high bias and high variance**."

---

## 3. Bias and Variance in Machine Learning

### 🔍 The ML Context

From lecture:
> "In machine learning, we have data from real life and this data can be very complex and **we don't know what the true model is**."

### 📐 Bias: Error from Simplification

From lecture:
> "By making a model, we **introduce some assumption**, and there's an **error that's caused by a simplification**, by choosing our model, and this error is called **bias**."

**What is bias:**
- Error from model being too simple
- Systematic mistakes
- Unable to capture true patterns
- Underfitting

**Example:** Using straight line for curved data

### 📊 Variance: Model Variability

From lecture:
> "**Variance** in machine learning means **variability of the model**."

More specifically:
> "This **variability of the model** is called the **variance of the model**."

**What is variance:**
- How much model changes with different training data
- Sensitivity to specific training samples
- Overfitting to noise

---

## 4. Model Variability: The Experiment

### 🔬 The Setup

From lecture: We have data, fit a simple model AND a complex model. Then repeat with different dataset.

### 📊 Simple Model (Low Variance)

```
Dataset 1:           Dataset 2:
    •  •                 • •
   •    •               •   •
  •      •      →      •     •
 •        •           •       •

Simple fit:          Simple fit:
─────────────        ─────────────
   Similar!
```

From lecture:
> "If we chose different data set like this, for example, then if we fit the **simple model** again, **they will be very similar**."

**Result:** Low variance - consistent across datasets

### 📊 Complex Model (High Variance)

```
Dataset 1:           Dataset 2:
    •  •                 • •
   •    •               •   •
  •      •      →      •     •
 •        •           •       •

Complex fit:         Complex fit:
 ╱╲  ╱╲               ╱╲╱╲
╱  ╲╱  ╲             ╱    ╲
  Different!
```

From lecture:
> "But if we fit the **complex model**, now it's going to be **little different** from the previous."

**Result:** High variance - changes significantly

### 🧠 Key Principle

From lecture:
> "If the model is simple, they tend to have **low variance**. They don't change much even though we changed the training data, but when we have more flexibility in the model, it may change quite a bit depending on how we choose the training samples, so they tend to have **high variance**."

---

## 5. The Trade-Off Relationship

### ⚖️ The Fundamental Pattern

**Simple Models:**
- ❌ High bias (too simple to capture patterns)
- ✅ Low variance (consistent)

**Complex Models:**
- ✅ Low bias (can capture patterns)
- ❌ High variance (inconsistent)

### 📊 Mapping Models to Target Scenarios

From lecture:
> "**Simpler model** tends to have a high bias and low variance, so it will correspond to this one, and **more complex or flexible model** tend to have low bias but has a higher variance, so it will be this case."

```
Simple Model:           Complex Model:
Target ☉                Target ☉
                              •
        • •               • ☉
       • • •                •
        • •                   •

High Bias              Low Bias
Low Variance           High Variance
```

### 💡 Why "Trade-Off"?

From lecture:
> "In machine learning, a lot of models are either this case or this case. **There is a trade-off between the two**. That's where the **bias-variance trade-off** coming from."

**You cannot minimize both:**
- Reduce bias → Increase variance
- Reduce variance → Increase bias
- Must find balance

### 🎯 The Four Quadrants

| Bias | Variance | Common? | Model Type |
|------|----------|---------|------------|
| High | Low | ✓ | Too simple (underfit) |
| Low | High | ✓ | Too complex (overfit) |
| High | High | Sometimes | Poor model |
| Low | Low | Rare | Ideal (deep learning with tricks) |

From lecture:
> "Sometimes if the model is not very good, then you may encounter this case [high bias, high variance]. Also some type of model such as a **deep neural network** with some other tricks, they may have **low bias and low variance**. But most of cases, we have the **trade-off** between the bias and variance."

---

## 6. Mathematical Decomposition

### 🔍 How Components Change

From lecture:
> "This is model complexity, this is test error or error in general. The **bias goes down** as our model complexity increases, and the **model variability goes up** as our model complexity goes up."

```
         │  Bias ↓
         │╲
         │ ╲___
         │     ╲___
         └────────────────> Complexity


         │      Variance ↑
         │          ╱╱
         │       ╱╱
         │    ╱╱
         └────────────────> Complexity
```

### 🧮 The MSE Formula

From lecture:
> "When you use a squared error, you can actually derive the general relationship between bias and variance to the test error. Test error MSE can be written as a **variance of the model** or estimated model, and then the **bias of the estimated model also in squared** plus some **irreducible error**, the variance of the residuals."

**Formula:**
```
Test Error (MSE) = Variance + Bias² + Irreducible Error
```

More formally:
```
E[(y - f̂(x))²] = Var[f̂(x)] + Bias²[f̂(x)] + σ²
```

**Components:**
- `Var[f̂(x)]`: Model variance (increases with complexity)
- `Bias²[f̂(x)]`: Squared bias (decreases with complexity)
- `σ²`: Irreducible error (constant, inherent noise)

### 📈 The Resulting Curve

From lecture:
> "According to this, the test error is the **sum of this variance of the model and bias squared** of the model. In the end, our **test error will have a shape of this** because it adds to this too, and then there is some irreducible error from the residuals."

```
Error   │
        │     Total = Var + Bias² + σ²
        │    ╱‾‾╲
        │   ╱    ╲___
        │  ╱ Variance ╱╱
        │ ╱╱╱╱╱╱
        │╱ Bias²
        │─────────────  Irreducible (σ²)
        └────────────────────> Complexity
```

**Result:** U-shaped test error curve!

- **Left:** Bias² dominates → high error
- **Middle:** Optimal balance → **minimum error**
- **Right:** Variance dominates → high error

---

## 7. Practical Error Curve Shapes

### 🌐 Real-World Variation

From lecture:
> "However, in reality, depending on your model and data, your test error may just **go down and then flattens** and that's very common, whereas your training error goes down and down."

### 📊 Common Scenarios

#### Scenario 1: Flattening (Very Common)

```
Test:   │╲
        │ ╲____
        │      ──────
        │
        └────────────────> Complexity

Train:  │╲
        │ ╲___
        │     ╲___
        └────────────────> Complexity
```

**Characteristics:**
- Test error plateaus instead of rising
- Training continues decreasing
- Very common in practice

#### Scenario 2: Good Simple Model

From lecture:
> "Sometimes, the **simple model fits well** to the data already in the case you may have already **good test error for the simple model** as well like this, and then it goes up like this."

```
        │╲
        │ ╲___
        │     ╲___
        │
        └────────────────> Complexity
            ↑
    Already good!
```

**When this happens:**
- Data has simple structure
- Don't need complex models
- Occam's Razor applies

#### Scenario 3: Classic U-Shape

```
        │   ╱
        │  ╱╲
        │ ╱  ╲
        │╱    ╲
        └────────────────> Complexity
```

**Textbook case:**
- Clear minimum
- Visible overfitting region
- Model selection critical

### 🎯 Universal Pattern

**Training error ALWAYS decreases:**
```
        │╲
        │ ╲___
        │     ╲___
        └────────────────> Complexity
```

**Test error varies** but shows different behavior than training!

---

## 8. Beyond Squared Error

### 🔍 Universal Principle

From lecture:
> "Also note that **this doesn't have to be squared error**. It is very **general behavior** no matter which loss function or error function you have."

### 🎯 Applies To All Metrics

The bias-variance trade-off applies to:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Cross-entropy loss
- Any error metric

### 💡 Why It's Universal

The fundamental concepts hold:
1. **Underfitting:** Model too simple → high error
2. **Overfitting:** Model too complex → poor generalization
3. **Balance:** Need optimal complexity

Whether MSE, MAE, or other metric, the principle remains!

---

## 9. Summary

### 📋 From the Lecture

From lecture:
> "In summary, we talked about **what happens if we add more complexity to our model**, we talked about **polynomial regression** and **where we stop adding more terms to the model** by monitoring training and test error, and we also talked about the **bias-variance trade-off principle**."

### 🎯 Key Takeaways

**1. Bias and Variance Definitions**
- **Bias:** Error from oversimplification
- **Variance:** Model variability across datasets
- Visualized through target/bullet analogy

**2. The Trade-Off**
- Simple: High bias, Low variance
- Complex: Low bias, High variance
- Cannot minimize both simultaneously

**3. Mathematical Decomposition**
```
MSE = Variance + Bias² + Irreducible Error
```
- Creates U-shaped test error curve
- Optimal complexity at minimum

**4. Practical Implications**
- Test error varies (U-shape, flatten, etc.)
- Training error always decreases
- Gap indicates overfitting
- Applies to any loss function

**5. Model Selection Strategy**
- Monitor training AND test error
- Find minimum test error
- Understand bias-variance balance
- Choose optimal complexity
- Prefer simpler when similar

### 🔄 Complete Picture

**Connection to polynomial regression:**
1. Add complexity (polynomial degrees)
2. Training error ↓ monotonically
3. Test error shows U-shape
4. **Explanation:** Bias-variance trade-off

**Decision framework:**
```
Complexity  →  Bias ↓  Variance ↑  Test Error
Low            High   Low          High (underfit)
Optimal        Med    Med          Low ← Choose!
High           Low    High         High (overfit)
```

### ✅ Best Practices

1. Always split data (train/validation/test)
2. Calculate both error types
3. Plot error curves
4. Understand bias-variance balance
5. Select at minimum test error
6. Apply Occam's Razor for ties
7. Test on final held-out set

---

**End of Lecture Notes - Module 02, Document 2**
