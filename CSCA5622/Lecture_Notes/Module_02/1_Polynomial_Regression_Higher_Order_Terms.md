# Polynomial Regression & Higher-Order Terms - Detailed Lecture Notes
**CSCA5622 - Module 02**

---

## 📚 Overview

This document explores **multi-linear regression** and **polynomial regression**, focusing on adding model complexity through higher-order terms and additional features. Key topics include:

- Extending simple linear regression to multiple features
- Polynomial regression with higher-order terms
- Feature engineering with domain knowledge
- Critical scaling issues with polynomial terms
- Model complexity and overfitting
- Training vs. validation error curves
- Bias-variance tradeoff
- Model selection and Occam's Razor

All concepts are explained with examples from the lecture transcript.

---

## 1. From Simple to Multi-Linear Regression

### 🔍 Simple Linear Regression Review

**Simple linear regression** uses a single predictor variable:

```
y = a₀ + a₁x₁
```

Where:
- `y`: target variable (output we want to predict)
- `x₁`: single predictor feature (input)
- `a₀`: intercept coefficient
- `a₁`: slope coefficient

### 💡 Example: House Price Prediction

**Simple model with one feature:**
```
Price = a₀ + a₁ × HouseSize
```

This model only considers house size. But what about other factors that affect price?

### 🏘️ Adding More Features: Multi-Linear Regression

We can extend the model by adding more features:

```
Price = a₀ + a₁×HouseSize + a₂×LotSize + a₃×NumBedrooms + ...
```

**Example from lecture:**
- `x₁`: Size of the house (square feet)
- `x₂`: Size of the lot
- `x₃`: Number of bedrooms
- And so on...

Each feature gets its own coefficient, allowing the model to account for multiple factors simultaneously.

### 🧮 General Multi-Linear Regression Formula

For `p` features:

```
y = a₀ + a₁x₁ + a₂x₂ + a₃x₃ + ... + aₚxₚ
```

**This is called multi-linear regression** because it has multiple features.

### 🧠 Key Point

Multi-linear regression is still **linear** in the parameters (coefficients), even though we have multiple features. We're still finding the best linear combination of our input features.

---

## 2. Polynomial Regression - Adding Higher-Order Terms

### 🔍 Core Concept

Instead of (or in addition to) adding different features, we can add **powers** of the same feature:

```
y = a₀ + a₁x₁ + a₂x₁²
```

Or even higher orders:

```
y = a₀ + a₁x₁ + a₂x₁² + a₃x₁³
```

**This is called polynomial regression.**

### 📐 Polynomial Order (m)

The **order** `m` represents the highest power in the model:

- **m = 1**: `y = a₀ + a₁x` (simple linear regression - straight line)
- **m = 2**: `y = a₀ + a₁x + a₂x²` (quadratic)
- **m = 3**: `y = a₀ + a₁x + a₂x² + a₃x³` (cubic)
- And so on...

### 💡 Why Use Polynomial Terms?

Real-world relationships are often **curved**, not straight lines. Polynomial terms allow us to model these curves:

- **Quadratic (m=2)**: Can model U-shapes or inverted U-shapes
- **Cubic (m=3)**: Can model S-curves with one inflection point
- **Higher orders**: Can model more complex curves

### 📊 Visual Understanding

As we add higher-order terms:
- The fitted line becomes more **flexible**
- It can have different **shapes** and **curves**
- The model can fit more complex patterns

### 🧠 Important Note

Polynomial regression is **still a form of multi-linear regression**! We're just creating new features (x², x³, etc.) and fitting them linearly.

---

## 3. Feature Engineering with Domain Knowledge

### 🔍 Core Concept

**Feature engineering** means creating new features from existing ones using domain knowledge or intuition about the problem.

### 🏥 Example from Lecture: BMI in Diabetes Prediction

**Original approach** (separate features):
```
P(diabetes) = a₀ + a₁×Height + a₂×Weight + a₃×LabTest + ...
```

**Engineered feature approach** (combining features):

Instead of using height and weight separately, we can create **BMI (Body Mass Index)**:

```
BMI = Weight / Height²
```

Then our model becomes:
```
P(diabetes) = a₀ + a₁×BMI + a₂×LabTest + ...
```

This new feature `BMI` is:
- A **function** of existing features (weight and height)
- Based on **domain knowledge** (BMI is medically meaningful)
- More **interpretable** than separate height/weight coefficients
- Often **more predictive** because it captures the relationship that matters

### 🔧 Key Principle

You can construct features like `x'` (x-prime) that are functions of your original features `x₁` and `x₂`:

```
x' = f(x₁, x₂)
```

For example: `BMI ∝ Weight / Height²`

### 🧠 Flexibility of Linear Models

By engineering features, **linear models become really flexible**:
- You're not restricted to just adding raw features
- You're not restricted to just polynomial terms
- You can create **any features** that make sense for your problem
- Domain knowledge helps you choose relevant feature transformations

### 💡 Benefits

1. **Reduces features**: Instead of height AND weight, just use BMI
2. **Improves interpretability**: BMI has clear medical meaning
3. **Captures relationships**: Combines features in meaningful ways
4. **Better predictions**: Often performs better than raw features

---

## 4. The Scaling Problem with Polynomial Features

### 🚨 Critical Issue

When you add polynomial terms to features with **large values**, you encounter a serious numerical problem.

### 📐 The Problem Illustrated

**Example from lecture:** House size as a feature

If house size is measured in square feet:
- Original value: `x = 1000, 2000, 3000` (on the order of thousands)
- Price: `y` on the order of millions

When we create polynomial terms:
- `x²` = on the order of **millions**
- `x³` = on the order of **billions**
- `x⁶` = on the order of **10¹⁸** (a quintillion!)

### ⚠️ Why This Is a Disaster

**Coefficient magnitude mismatch:**
- `a₁` might be on the order of 1000
- `a₂` might be on the order of 1
- `a₆` would need to be on the order of **10⁻¹⁸** (incredibly tiny!)

**Computer precision fails:**
- Computers have a hard time calculating coefficients when the scales are so different
- Some coefficients are enormous, others microscopic
- Numbers can become **10¹⁸**, which is extremely large and causes numerical instability
- **The fitting may not work very well** - it can even fail completely!

### 🎯 What Happens

From the lecture:
> "By the time I have the size of the house is 6 power, this could be 10 to 18, which is a really big number and the coefficient to match this number should be very small. That means the computer has a hard time to calculate all these coefficients. Therefore, **the fitting may not work very well**."

The fitting can fail, giving you bad results or even crashing.

### ✅ The Solution: Feature Scaling

**Scale your features to be on the order of 1** instead of thousands!

**Simple approach:**
```
x_scaled = x / 1000
```

Now your values are:
- `x_scaled = 1, 2, 3` (on the order of 1)
- `x_scaled² = 1, 4, 9` (still manageable!)
- `x_scaled⁶ = 1, 64, 729` (much better than 10¹⁸!)

### 📊 Python Example: Demonstrating the Scaling Problem

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Data with LARGE values (house size in sq ft)
house_size = np.array([[1000], [2000], [3000], [4000], [5000]])
price = np.array([200000, 350000, 500000, 650000, 800000])

print("="*60)
print("WITHOUT SCALING - Can cause problems!")
print("="*60)

# Try to fit polynomial degree 6 WITHOUT scaling
poly = PolynomialFeatures(degree=6, include_bias=True)
X_poly = poly.fit_transform(house_size)

print(f"Feature ranges:")
print(f"  x¹: {house_size.min():.0f} to {house_size.max():.0f}")
print(f"  x²: {(house_size**2).min():.0f} to {(house_size**2).max():.0f}")
print(f"  x⁶: {(house_size**6).min():.2e} to {(house_size**6).max():.2e}")

try:
    model = LinearRegression().fit(X_poly, price)
    print(f"\nCoefficients (absolute values):")
    for i, coef in enumerate(model.coef_):
        if i <= 6:
            print(f"  a_{i}: {abs(coef):.2e}")
    print("\nWarning: Coefficients span many orders of magnitude!")
    print("This indicates numerical instability.")
except Exception as e:
    print(f"\nFitting FAILED: {e}")

print("\n" + "="*60)
print("WITH SCALING - Works properly!")
print("="*60)

# NOW scale to reasonable values
house_size_scaled = house_size / 1000  # Divide by 1000

poly = PolynomialFeatures(degree=6, include_bias=True)
X_poly_scaled = poly.fit_transform(house_size_scaled)

print(f"Feature ranges after scaling:")
print(f"  x¹: {house_size_scaled.min():.1f} to {house_size_scaled.max():.1f}")
print(f"  x²: {(house_size_scaled**2).min():.1f} to {(house_size_scaled**2).max():.1f}")
print(f"  x⁶: {(house_size_scaled**6).min():.1f} to {(house_size_scaled**6).max():.1f}")

model_scaled = LinearRegression().fit(X_poly_scaled, price)
print(f"\nCoefficients (absolute values):")
for i, coef in enumerate(model_scaled.coef_):
    if i <= 6:
        print(f"  a_{i}: {abs(coef):.2e}")

print("\nMuch better! Coefficients are in manageable ranges.")
print("The model fits properly without numerical issues.")

# Compare predictions
print("\n" + "="*60)
print("Predictions:")
print("="*60)
test_size = np.array([[2500]])
test_size_scaled = test_size / 1000

print(f"For house size = {test_size[0,0]:.0f} sq ft:")
print(f"  Actual price in data: ~$525,000")
try:
    pred_unscaled = model.predict(poly.transform(test_size))
    print(f"  Without scaling: ${pred_unscaled[0]:,.0f}")
except:
    print(f"  Without scaling: FAILED")
    
pred_scaled = model_scaled.predict(poly.transform(test_size_scaled))
print(f"  With scaling: ${pred_scaled[0]:,.0f}")
```

### 🧮 Mathematical Insight

When you scale `x' = x/c`:
- `(x')² = x²/c²` 
- `(x')³ = x³/c³`
- `(x')⁶ = x⁶/c⁶`

This keeps all powers in a similar range!

### 🧠 Key Takeaways

1. **Always scale features** before creating polynomial terms
2. Scale to values on the **order of 1** (not thousands or millions)
3. **Prevents numerical instability** in fitting
4. Allows you to add more high-order terms if needed
5. **Critical for model success** - not optional!

---

## 5. Model Complexity and the Risk of Overfitting

### 🔍 The Central Question

> "Where do we want to stop adding high order terms?"

As you add more complexity to your model (higher polynomial degrees), your model fit will keep improving on the training data. But is that always good?

### 📈 What Happens as Complexity Increases

**Model fitness will go up and up** as you add more model complexity.

You could have a really high-order polynomial that fits everything perfectly:

```
Data points:  • • • •  •  • •
Crazy model:  ╱╲╱╲╱╲╱╲╱╲╱╲
```

### ⚠️ The Problem with Over-Complex Models

From the lecture, this highly complex model is **not very good** for two reasons:

#### 1. Not Interpretable
A polynomial with many terms (like degree 10 or 15) is hard to understand and explain.

#### 2. Vulnerable to New Data Points

**Example from lecture:**
Imagine you have a complex, wiggly fitted curve. Now you get a new data point:
- If it falls in an unexpected place, your model might predict a **huge error**
- The extreme curves make predictions unreliable

**However**, if you have a **simpler model**:
- It will have a **smaller error** at new data points
- More stable and reliable predictions
- Better **generalization** to unseen data

### 🎯 The Core Insight

> "That's the motivation, how do we determine where to stop when we add model complexity: we want to monitor error that's introduced when we introduce new data points."

**Key principle:** Don't just look at training error - look at how well your model performs on **new, unseen data**!

---

## 6. Training Data vs. Test/Validation Data

### 🔍 The Setup

**Your dataset** has both features and labels (inputs and outputs).

**Split it into two parts:**

1. **Training data**: Used to fit the model (find coefficients)
2. **Test data** (also called **validation data**): Set aside, NOT used for training

### 📝 Terminology Note

From the lecture:
> "Another name for test data that's used while we are training is called the validation. So we can call them interchangeably. But in machine learning community, **validation error** is a more used term for the data set that's set aside for the purpose of testing while you're training the model."

**Key terms:**
- **Test data** = **Validation data** (used interchangeably)
- Used to measure performance on unseen data
- Helps us choose the best model complexity

### 📊 Measuring Errors

**Training Error:**
```
MSE_train = (1/n_train) Σ (y_train_i - ŷ_train_i)²
```

- Use the trained model to predict on **training data**
- Compare predictions to actual training labels
- Measures how well model fits the data it was trained on

**Test/Validation Error:**
```
MSE_test = (1/n_test) Σ (y_test_i - ŷ_test_i)²
```

- Use the trained model to predict on **test data** (held-out data)
- Compare predictions to actual test labels
- Measures how well model **generalizes** to new data

### 🔄 Process for Different Model Complexities

For each model complexity (e.g., m=1, m=2, m=3, ...):

1. Train the model on training data
2. Calculate training error (MSE on training set)
3. Calculate test error (MSE on test set)
4. Record both errors

This gives you error measurements for each model complexity level.

---

## 7. The Error Curves: Training vs. Validation

### 📉 Typical Behavior

From the lecture:
> "The exact shape of the curve for training error and test error will be different depending on your number of data and the data itself that you randomly sample. And also it will depend on your model complexity and so on. However, **in general, you're going to see this type of error curves**."

### 📊 Training Error Curve

**Training error will go down as you increase your model complexity.**

```
Training
Error  │╲
       │ ╲
       │  ╲___
       │      ╲____
       │           ╲___
       └────────────────> Model Complexity
```

This makes sense: more complex models can fit training data better and better.

### 📊 Test/Validation Error Curve  

**Test error has a different pattern:**

```
Test    │
Error   │    ╱
        │   ╱
        │  ╱
        │ ╱╲
        │╱  ╲___
        └────────────────> Model Complexity
           ↑
        Sweet spot!
```

**The U-shaped curve:**
- **Goes down initially**: Model captures true patterns
- **Reaches minimum**: Optimal complexity
- **Goes up again**: Model starts overfitting to noise

### 🎯 Finding the Sweet Spot

From the lecture:
> "Then we can find the **sweet spot** here that the test error is minimized. So we can pick our best model complexity."

**Example:** If test error is minimized at m=2, that's your best model.

### 📋 Complete Comparison

```
Model       Training    Test
Complexity  Error       Error       Status
─────────────────────────────────────────
m = 1       High        High        Underfitting
m = 2       Medium      Low         ✓ OPTIMAL
m = 3       Low         Low         Also good
m = 5       Very Low    Medium      Starting to overfit
m = 10      Near Zero   High        Overfitting
```

### 🧠 Key Insight from Lecture

> "You can also see this model complexity [3] because the [model with complexity] 3 is also comparably good. And in some cases, depending on your data draw, it can show you actually slightly better results than model complexity equals 2. However, **if they are similar, then you want to still choose the simpler model**."

This leads us to an important principle...

---

## 8. Occam's Razor Principle

### 🔍 The Principle

From the lecture:
> "And this kind of principle is called **Occam's Razor**, consistently telling that **if the model performances are similar for simpler model and complex model, we prefer choosing simpler model**."

### 🎯 Application

**Scenario:** You're comparing models:
- Model A (m=2): Test MSE = 10.5
- Model B (m=3): Test MSE = 10.3

Model B is *slightly* better, but Model A is simpler.

**Occam's Razor says:** Choose Model A (the simpler one)!

### 💡 Why Prefer Simpler Models?

1. **More interpretable**: Easier to understand and explain
2. **More stable**: Less sensitive to small changes in data
3. **Less prone to overfitting**: More likely to generalize
4. **Computationally cheaper**: Faster to train and predict
5. **Fewer parameters**: Less risk of capturing noise

### 🧮 The Trade-off Decision

**Choose the more complex model** only if:
- Performance improvement is **substantial**
- The added complexity is **justified** by better results

**Choose the simpler model** if:
- Performance is **comparable**
- Difference is small or within noise level
- You value **interpretability and stability**

### 🧠 Practical Guideline

Plot your test errors:
1. Find the model with lowest test error
2. Check nearby complexities (one simpler, one more complex)
3. If they perform similarly, choose the simplest one
4. This is your final model choice!

---

## 9. Summary & Key Takeaways

### 🎯 Main Concepts

**1. Multi-Linear Regression**
- Extends simple regression by adding multiple features
- Form: `y = a₀ + a₁x₁ + a₂x₂ + ... + aₚxₚ`
- Each feature gets its own coefficient

**2. Polynomial Regression**
- Adds higher-order terms of the same feature
- Form: `y = a₀ + a₁x + a₂x² + a₃x³ + ...`
- Order `m` determines complexity
- Still linear in parameters!

**3. Feature Engineering**
- Create new features from existing ones
- Use domain knowledge (e.g., BMI = Weight/Height²)
- Can improve model performance and interpretability
- Makes linear models very flexible

**4. Scaling is Critical**
- Large feature values cause numerical problems with polynomials
- Scale features to order of 1 before creating polynomial terms
- Prevents coefficient magnitude mismatch
- Essential for successful fitting!

**5. Model Complexity Trade-off**
- More complexity → better training fit
- BUT too much complexity → poor generalization
- Need to find the "sweet spot"

**6. Training vs. Validation Error**
- Training error: always decreases with complexity
- Validation error: U-shaped curve with complexity
- Use validation error to choose optimal model
- Don't trust training error alone!

**7. Occam's Razor**
- When models perform similarly, choose the simpler one
- Simpler = more interpretable, stable, and generalizable
- Only add complexity if substantially better

### 📋 Model Selection Workflow

```
1. Split data → Training + Validation sets
2. For each complexity level (m = 1, 2, 3, ...):
   a. Train model on training data
   b. Calculate training error
   c. Calculate validation error
3. Plot both error curves
4. Find minimum validation error
5. Check nearby complexities
6. Choose simplest model with good validation error
7. Apply Occam's Razor if performance is similar
```

### ⚠️ Common Pitfalls to Avoid

1. ❌ Using only training error to select model
2. ❌ Forgetting to scale features before polynomial features
3. ❌ Adding complexity indefinitely without validation
4. ❌ Choosing overly complex model for marginal gains
5. ❌ Not holding out validation data

### ✅ Best Practices

1. ✅ Always split data (train/validation/test)
2. ✅ Scale features before polynomial transformation
3. ✅ Monitor both training and validation errors
4. ✅ Plot error curves to visualize tradeoff
5. ✅ Apply Occam's Razor when choosing final model
6. ✅ Test final model on completely held-out test set

---

**End of Lecture Notes - Module 02, Document 1**
