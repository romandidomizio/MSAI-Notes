# Logistic Regression Optimization - Detailed Lecture Notes
**CSCA5622 - Module 03**

---

## 📚 Overview

This document covers **optimization in logistic regression** - how we determine the optimal coefficient values. Topics include maximum likelihood estimation, deriving the loss function, gradient descent, and Newton's method.

All concepts explained from the lecture transcript.

---

## 1. Maximum Likelihood Estimation

### 🔍 Introduction

From lecture:
> "We're going to start by introducing a new concept called the **maximum likelihood**."

### 📐 What Is the Likelihood Function?

From lecture:
> "The likelihood function is a **product of all the probabilities** that correspond to labels. For each sample, the **probability of classifying the label correctly** and **multiplying all these probabilities** for each sample is called the **likelihood function**."

**Definition:** Likelihood = product of probabilities that the model assigns to the correct labels

### 🎯 The Goal

From lecture:
> "By **maximizing this likelihood**, we can **determine the coefficient values** for the logic in the logistic regression."

**Principle:** Find parameters that make observed data most probable

### 🌍 Universality of Maximum Likelihood

From lecture:
> "By the way, this likelihood function is **not only for the logistic regression**, but it **occurs again and again** in machine learning, and it's a **common theme**. That this **principle applies to our parametric models**. If we **maximize the likelihood** of the parameters get determined."

---

## 2. Deriving the Likelihood Function

### 📋 Example Setup

From lecture:
> "Here are some example where we have **y_1**, and we set it to **one**, and **y_2=1**. **Y_3** value for the third example would be **zero**. **Y_4 is zero** and **y_5 is one**."

**Example labels:**
```
Sample 1: y₁ = 1
Sample 2: y₂ = 1
Sample 3: y₃ = 0
Sample 4: y₄ = 0
Sample 5: y₅ = 1
```

### 🔢 Model Output: Probabilities

From lecture:
> "Let's say we are using logistic regression and feed our features to get the predicted value y hat, but actually what **logistic regression father produced at the output is the probability**, so this is a **sigmoid function** that we saw before."

**Sigmoid outputs probabilities:**
```
σ(z₁) = P(y₁=1|x₁) = p₁
σ(z₂) = P(y₂=1|x₂) = p₂
σ(z₃) = P(y₃=1|x₃) = p₃
σ(z₄) = P(y₄=1|x₄) = p₄
σ(z₅) = P(y₅=1|x₅) = p₅
```

### 🔄 Matching Probabilities to Labels

From lecture:
> "We do like to construct the likelihood. That means when the probability represent the **probability of the label being one**, we're going to have to **flip some of them** like this one, so that it has the **probability that represent y=0**."

**For y=0, we need P(y=0):**
```
If y = 1: use p (already correct)
If y = 0: use (1 - p) (flip probability)
```

### 📊 Constructing Total Probability

From lecture:
> "To do that, we're going to **change the sign** and **multiply all of the probabilities together**. This quantity becomes a **total probability** of having all of these labels classified correctly."

**Likelihood function:**
```
L = P(y₁=1) × P(y₂=1) × P(y₃=0) × P(y₄=0) × P(y₅=1)
  = p₁ × p₂ × (1-p₃) × (1-p₄) × p₅
```

### 🎯 Maximization Objective

From lecture:
> "This says that the correct probability, that means that we would like to **maximize this probability** so that our model can **correctly classify all the examples**. Let's maximize this. That's why this is called the **maximum likelihood**."

---

## 3. General Form and Log-Likelihood

### 📐 General Likelihood Formula

From lecture:
> "Likelihood function is the **total probability of classifying everything correctly**. So it takes this form and you'd like to maximize."

**For n samples:**
```
L(β) = ∏ᵢ₌₁ⁿ P(yᵢ|xᵢ, β)
```

### ⚠️ The Multiplication Problem

From lecture:
> "Now, we have some **trouble** because there are **so many multiplication here**. This is **not very easy to calculate**."

**Problems with products:**
- Numerical underflow (very small numbers)
- Difficult to differentiate
- Computationally expensive

### ✅ Solution: Take the Log

From lecture:
> "So we're going to **take a log** to the entire term, so that we **change this to the summation**."

**Log-likelihood:**
```
log L(β) = log[∏ᵢ P(yᵢ|xᵢ)] = Σᵢ log P(yᵢ|xᵢ)
```

**Why this works:** log converts products to sums, maximizing log L is same as maximizing L

---

## 4. Binary Cross-Entropy Derivation

### 📋 Step-by-Step Derivation

From lecture:
> "Summing up all the examples that has **y_i label is one**. Actually, this has to be **log p_i**. Sum when the case is **y=0**."

**Initial form (two separate sums):**
```
log L = Σ(yᵢ=1) log(pᵢ) + Σ(yᵢ=0) log(1-pᵢ)
```

### 🧹 Cleaning Up the Formula

From lecture:
> "We can even make it more clean by having this **one summation** instead of two. Instead of having when the case is zero or one, I'm going to just **set everything has to be one**."

**Combining with clever use of yᵢ:**

From lecture:
> "Then I'm going to add **y_i here**, so it becomes the multiplication. Plus, if I change the variable here to be **1-y_i**, then when this is one, and this quantity becomes zero, so we can **combine these two terms together**."

**Final log-likelihood:**
```
log L = Σᵢ [yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]
```

**How it works:**
- When yᵢ = 1: First term active, second term vanishes
- When yᵢ = 0: First term vanishes, second term active

### 🔄 From Maximum Likelihood to Loss Function

From lecture:
> "This is our final form for the **log-likelihood**. This is the log-likelihood, and we want to **maximize** this quantity. Actually, **maximizing log-likelihood is the same as minimizing the loss function**. We can define a loss function as the **inverse** of this. Take a **minus sign** here."

**Binary Cross-Entropy (BCE) Loss:**
```
L_BCE = -Σᵢ [yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]
```

Or per sample:
```
L_BCE(y, p) = -[y log(p) + (1-y) log(1-p)]
```

### 📝 Common Name

From lecture:
> "This is called **binary cross-entropy**. This binary cross entropy is very common in binary codes classification. We'll use this **cross-entropy loss function very often**."

---

## 5. General Cross-Entropy

### 🌍 Generalized Form

From lecture:
> "**Across entropy** again is a generalized form as this. These two are **probability distribution**."

**General cross-entropy:**
```
H(p, q) = -Σₖ p(k) log q(k)
```

### 🎯 Interpretation

From lecture:
> "Usually the one that's here means that the **probability distribution that's a label or true value**. It can come from the data or it can come from the labels. Whereas this one, the **probability that goes into the log is predicted value**."

**Components:**
- **p**: True distribution (labels)
- **q**: Predicted distribution (model output)

From lecture:
> "Essentially we are **measuring the difference** between the **true labels** and the **predicted probability**. That's the meaning of the cross-entropy."

### 🔢 Multiclass Extension

From lecture:
> "I omitted category, but if you have **more than two categories**, they have **index for the category**."

**Multiclass cross-entropy:**
```
L = -Σᵢ Σₖ yᵢₖ log(pᵢₖ)
```

Where k indexes classes

---

## 6. Optimization Overview

### 🔄 The Optimization Cycle

From lecture:
> "**Searching parameters** involves the optimization. Again, the **feature goes into model**, and the model has **parameters**, and the model will **predict a value**. With a **target value**, the **loss function** will compare this prediction and target and produce some **error**."

**Cycle:**
```
Features (X) → Model(β) → Prediction (ŷ) → Loss(y, ŷ) → Error
                ↑                                        ↓
                └────────── Update Parameters ──────────┘
```

### 🎯 Parameter Update Logic

From lecture:
> "If the **error is bigger** then it's going to **change the parameter value more**, and we need to do this **cycle multiple times**, we're going to get the parameter values, **optimal value**. That's optimization and this is **parameter update procedure**."

---

## 7. Gradient Descent

### 🏔️ The Mountain Analogy

From lecture:
> "This error surface is actually from MSE loss, which is from linear regression. But the reason why I just draw here is that it's **easier to draw** than cross-entropy."

From lecture:
> "Let's say **you're scared** and you're on Ottomans, scared that you're not afraid to go to **steep slope**, and let's say you want to get to the **base as soon as possible**. What is your strategy here? You're going to **follow steepest slope**."

**Intuition:** Go downhill in the direction of steepest descent

### 📐 The Gradient

From lecture:
> "Let's **measure a gradient** along the coefficient a, and let's measure the **gradient** along the coefficient b. We're going to **update our weight**, which is the parameter values for a and b **according to this gradient**."

**Gradient:** Vector of partial derivatives showing steepest direction

### 🔧 Derivative Calculation

**For MSE example:**

From lecture:
> "Loss functions for MSE looks like this. It has a **residual squared**, and then the **gradient** along a coefficient is a **partial derivative of loss function** with respect to a."

```
L = (y - ŷ)²
∂L/∂w = ∂L/∂ŷ × ∂ŷ/∂w
```

### 📋 Chain Rule

From lecture:
> "The function here takes the form of **f squared**. It's the square root of dx. It will be **2F times df dx**. This is called the **chain rule**."

**Chain rule formula:**

From lecture:
> "If you have a function that's a **function of some other function** which is a **function of something else** like this, you can take a **chain**, making derivative of this."

```
If f(g(x)), then: df/dx = (df/dg) × (dg/dx)
```

**For nested functions:**
```
f(g(h(x))): df/dx = (df/dg) × (dg/dh) × (dh/dx)
```

From lecture:
> "When there is a **multiple nested function**, you take the derivatives convenient to it is this. I'll take the chain rule."

---

## 8. Weight Update Rule

### 📐 The Formula

From lecture:
> "**Weight update rule** says the weight is updated such that it's the **old value of the weight** minus some constant **Alpha** times the **gradient** of the loss function with respect to that weight."

**Update equation:**
```
w_new = w_old - α × (∂L/∂w)
```

### 🎚️ Learning Rate (α)

From lecture:
> "This is called the **learning rate** by the way. The bigger the value, the **bigger the step size**. If the learning rate is big, then the **step is bigger**."

**Effects of learning rate:**

**Too large:**

From lecture:
> "Be careful if it's **too big**, then it can **pass the solution** like this."

```
     /\
    /  \___  ← Overshoots minimum
   /       \
```

**Too small:**

From lecture:
> "If the learning rate is very small, we're going to take a **small step** toward a goal like this. If the learning rate is **too small**, it's going to take **a lot of steps and longer time**."

```
  \_
   \_  ← Many tiny steps
    \_
     \_ (slow convergence)
```

### 🎛️ Hyperparameter

From lecture:
> "Usually when you do the gradient descent optimization, you will have to **choose this learning rate**, therefore, this learning rate is a **hyperparameter**. **Hyperparameter** means that some parameter that the **user will have to choose**. Learning rate isn't one of them."

**Note for logistic regression:**

From lecture:
> "We don't have to worry about it for the logistic regression, because the logistic regression uses **another form of gradient descent**. Actually, that uses **second derivatives** rather than first derivative like in the gradient descent. That is called the **Newton method**."

---

## 9. BCE Gradient Calculation

### 📐 Full Derivation

From lecture:
> "Gradient descent for **binary cross-entropy loss**. Let's calculate this."

**Loss function:**
```
L = -[y log(σ(z)) + (1-y) log(1-σ(z))]
```

Where:
```
σ(z) = 1/(1 + e^(-z))
z = w·x + b
```

### 🔗 Chain Rule Application

From lecture:
> "We're going to take a derivative of loss function with respect to w and **z is as a function of w.x+b**, so **dL/dw** is going to be the **dL/dSigma** because loss is a function of Sigma. Then it's going to be **dSigma/dz** because Sigma is a function of z and then **dz/dw**."

**Chain:**
```
∂L/∂w = (∂L/∂σ) × (∂σ/∂z) × (∂z/∂w)
```

### 📊 Step 1: ∂L/∂σ

From lecture:
> "This value is going to be, **y/Sigma**. This is Sigma, by the way. This is Sigma and derivative of **log (x) is 1/x**. **Minus sign** is from here and then **-1**."

```
∂L/∂σ = -y/σ - (-(1-y))/(1-σ) = -y/σ + (1-y)/(1-σ)
```

### 📊 Step 2: ∂σ/∂z

From lecture:
> "Then **dSigma/dz is Sigma*1-Sigma**. I'm not going to show here, but you can **prove this easily**."

**Special property of sigmoid:**
```
∂σ/∂z = σ(z) × (1 - σ(z))
```

From lecture:
> "This is very good formula, that **sigmoid function is very convenient**, that its **derivative is itself times one minus itself**. That's why it's **used in the many gradient descent application**."

### 📊 Step 3: ∂z/∂w

From lecture:
> "Then **dz/dw is simply x**."

```
z = w·x + b
∂z/∂w = x
```

### ✅ Final Result

From lecture:
> "Combining all these together, we're gonna get **dL/dw** is going to be **(-y/Sigma-(1-y)/(1-Sigma)) *Sigma*1-x**."

**Simplified:**
```
∂L/∂w = (σ(z) - y) × x
```

**For bias:**

From lecture:
> "Similarly, we can do the derivative for the bias. You will take the same thing, except this part. There is just **one here**."

```
∂L/∂b = (σ(z) - y)
```

### 🔄 Weight Update

From lecture:
> "We're going to **apply the same principle** to update our weights. This could be either w or bias. They're the coefficients and then **dl/dw times the learning rate**."

```
w = w - α × (∂L/∂w)
b = b - α × (∂L/∂b)
```

---

## 10. Newton's Method

### 🔍 Extension of Gradient Descent

From lecture:
> "**Newton's method**, it's an **extension** to gradient descent method. Gradient descent method only use the **first derivative** of loss function and it's updated rule was this; gradient with respect to w of the loss function. Whereas the **Newton's method** will use **both the first and second derivative**."

**Gradient descent (first-order):**
```
w = w - α × (∂L/∂w)
```

**Newton's method (second-order):**
```
w = w - α × H^(-1) × (∂L/∂w)
```

### 📐 The Hessian Matrix

From lecture:
> "First derivative here, and then **second derivative here**. By the way, this term is called the **Hessian**."

**Hessian (H):** Matrix of second partial derivatives

```
H = [∂²L/∂w₁² ...]
    [∂²L/∂w₁∂w₂ ...]
```

### 🎯 Why Newton's Method Can Be Better

From lecture:
> "The reason why Newton's method can be good is when we have a **better flat gradient**, it can be very **slowly converting**. However, if there's a **Hessian** that's **dividing this small gradient**, then Hessian is also smaller then this can **boost the speed** of the convergence when the gradient is very small."

**Key advantage:** Accelerates when gradient is small (near minimum)

From lecture:
> "The **Newton's method converges faster** than means it **requires less number of iterations** given the same learning rate for the gradient and the Newton's method."

---

## 11. Computational Complexity Comparison

### 📊 Per Iteration Costs

**Memory:**

From lecture:
> "However, it has a **drawback**. The **memory that requires** for one iteration of Newton's method is scales as **n squared**, whereas the gradient method only it takes **O(n)**."

```
Gradient Descent: O(n)
Newton's Method:  O(n²)
```

**Time:**

From lecture:
> "For the **time complexity**, Newton's method per iteration, it's going to be **n cubed**, whereas gradient method is **n**."

```
Gradient Descent: O(n)
Newton's Method:  O(n³)
```

From lecture:
> "This is **more expensive per iteration**. It's great that you can **require less iteration**, but given the **same number of iterations**, gradient descent requires **less memory and less time**."

### 🧮 What is n?

From lecture:
> "This **n is the number of parameters**."

### 🔍 When to Use Each Method

**For large models (many parameters):**

From lecture:
> "If you have a **lot of parameters**, like in the **neural network**. Neural networks typically have **millions or billions of parameters**. Newton's method or similarly second derivative method will be **very slow**. Usually the **neural network optimization** utilize a **gradient descent method** rather than the second derivative method."

**For smaller models:**

From lecture:
> "**Logistic regression** and other **parametric models** in machine learning where the **number of parameters are smaller**, we don't have to worry about that. That's why **I scale on other similar packages** using **Newton's method** over similar methods to optimize the parameters."

---

## 12. Visual Comparison

### 🎬 Simulation

From lecture:
> "Let's have a look at some **simulation** here. This is a **gradient descent** and this is **Newton method**. Then they **start from the same place**."

**Newton's method:**

From lecture:
> "Then you're going to see shortly that they **shoot very fast** to the bottom and then it will go towards the goal."

**Gradient descent:**

From lecture:
> "Compared to the gradient descent method, which goes **very slowly when the gradient is small** at the bottom, Newton's method is **faster**."

### 💡 Why Faster?

From lecture:
> "That's because again, because this **small gradient is divided by small Hessian** so it gains **more boost** when the gradient is flat here."

**Mathematical reason:**
```
When gradient ≈ 0 and Hessian ≈ 0:
Step size = H^(-1) × gradient ≈ large value
```

---

## 13. Summary

### 🎯 Key Concepts

**Maximum Likelihood:**
- Product of probabilities for correct labels
- Maximize to find optimal parameters
- Universal principle in ML

**Log-Likelihood:**
- Take log to convert products to sums
- Easier to compute and differentiate

**Binary Cross-Entropy:**
```
L = -[y log(p) + (1-y) log(1-p)]
```
- Derived from maximum likelihood
- Standard loss for binary classification

**Gradient Descent:**
```
w = w - α × (∂L/∂w)
```
- First-order optimization
- α is learning rate (hyperparameter)
- O(n) memory and time per iteration

**Newton's Method:**
```
w = w - α × H^(-1) × (∂L/∂w)
```
- Second-order optimization
- Faster convergence (fewer iterations)
- O(n²) memory, O(n³) time per iteration
- Used in sklearn for logistic regression

### 📋 Trade-offs

| Method | Iterations | Memory/Iteration | Time/Iteration | Best For |
|--------|-----------|------------------|----------------|----------|
| Gradient Descent | More | O(n) | O(n) | Large models, neural networks |
| Newton's Method | Fewer | O(n²) | O(n³) | Small models, logistic regression |

---

**End of Lecture Notes - Module 03, Document 3**
