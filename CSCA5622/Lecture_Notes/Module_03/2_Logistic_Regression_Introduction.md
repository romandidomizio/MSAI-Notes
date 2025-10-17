# Logistic Regression Introduction - Detailed Lecture Notes
**CSCA5622 - Module 03**

---

## 📚 Overview

This document provides a comprehensive introduction to **logistic regression**, a fundamental classification algorithm. Despite its name, logistic regression is used for **classification**, not regression tasks.

Topics covered:
- Overview of machine learning problem types
- Binary classification definition and examples
- The logistic (sigmoid) function
- Decision boundaries (univariate, bivariate, multivariate)
- Why linear regression fails for classification
- Multiclass classification approaches (softmax and OvR)
- Multi-label vs multi-class problems

All concepts explained from the lecture transcript.

---

## 1. Machine Learning Problem Taxonomy

### 🔍 Three Main Categories

From lecture:
> "Periphery view of machine learning problems. In machine learning, we have **supervised learning**, we do labels, and **unsupervised learning**, which doesn't have label and **reinforcement learning** with feedback signals."

### 📊 The Hierarchy

```
Machine Learning
├── Supervised Learning (with labels)
│   ├── Regression (continuous output)
│   └── Classification (discrete output)
│       ├── Binary Classification (2 classes)
│       └── Multiclass Classification (3+ classes)
├── Unsupervised Learning (no labels)
└── Reinforcement Learning (feedback signals)
```

### 🎯 Course Focus

From lecture:
> "We're going to focus on **supervised learning**. Largely, it has **two tasks regression and classification**."

**Two main supervised tasks:**
1. **Regression:** Predict continuous values (e.g., house prices)
2. **Classification:** Predict discrete categories (e.g., spam/not spam)

---

## 2. Logistic Regression: Classification Not Regression

### 🔍 The Name Confusion

From lecture:
> "**Logistic regression**, we're going to talk about in this video. Although its **name says regression**, it's **actually for classification**, especially to useful for **binary class classification**."

**Important distinction:**
- Name: "Logistic **Regression**"
- Actual use: **Classification**
- Best for: **Binary classification** (2 classes)

### 🔧 Multiclass Capability

From lecture:
> "There are some ways to do the **multiclass classification** with the logistic regression method, but it's going to require some **engineering** to do that."

**Key point:** Logistic regression excels at binary classification but can be extended to multiclass with additional techniques.

---

## 3. Comparison with Other ML Models

### 📊 Model Capabilities Summary

From lecture:
> "Other models that we're going to talk about it later, and some of them will not talk about it in this course, they can do different things."

| Model | Regression | Binary Class | Multiclass | Categorical Features | Notes |
|-------|------------|--------------|------------|---------------------|-------|
| **Linear Regression** | ✓ | ✗ | ✗ | Limited | Continuous output only |
| **Logistic Regression** | ✗ | ✓✓ | ✓ (with engineering) | Yes | Best for binary |
| **Support Vector Machine** | ✓ | ✓✓ | ✓ (with engineering) | Yes | Similar to LogReg |
| **Decision Trees** | ✓ | ✓ | ✓✓ | ✓✓ | Handles everything |
| **Neural Networks** | ✓ | ✓ | ✓ | ✓ | Very flexible |

### 💡 Key Insights

**Support Vector Machine (SVM):**

From lecture:
> "Support vector machine can do both regression and classification. Similar to logistic regression it's **usually good for binary class** rather than multi-class. But it **can work on multi class**."

**Decision Trees:**

From lecture:
> "Decision trees can do everything. You can do regression, and binary class, multi-class **without any problem**. Also, it's nice that it can take **categorical variable very efficiently**."

**Neural Networks:**

From lecture:
> "Neural Network, same thing, can do everything."

---

## 4. Binary Classification Defined

### 🔍 What Is Binary Classification?

From lecture:
> "Let's talk about what is the **binary class classification**? It is essentially **Yes or No problem**. The **label is binary**."

**Definition:** Binary classification is the task of predicting one of **two possible categories** (classes).

**Mathematical representation:**
- Class labels: {0, 1} or {-1, +1} or {False, True}
- Output: Discrete, not continuous

---

## 5. Real-World Binary Classification Examples

From the lecture, here are practical applications:

### 💳 Credit Card Default

From lecture:
> "For example, **credit card default**, whether this customer that uses a credit card will likely to **default** on that given some historic data."

**Problem:** Predict if customer will default
- Class 0: Will not default
- Class 1: Will default

### 🚨 Insurance Fraud

From lecture:
> "Maybe there is a **insurance claims** and some insurance claim can be **fradulant**."

**Problem:** Detect fraudulent claims
- Class 0: Legitimate claim
- Class 1: Fraudulent claim

### 📧 Spam Filtering

From lecture:
> "**Spam filtering**. Given this email texts, this is a spam or not."

**Problem:** Filter unwanted emails
- Class 0: Not spam (ham)
- Class 1: Spam

### 🏥 Medical Diagnosis

From lecture:
> "**Medical diagnosis**. Given this patient's information and lab tests and data, is this person have **disease or not**?"

**Problem:** Diagnose presence of disease
- Class 0: Healthy
- Class 1: Disease present

### ⏰ Survival Prediction

From lecture:
> "**Survivor prediction** given this patient's information and history and things like that, whether these patients will **survive for next five years or not**?"

**Problem:** Predict survival outcome
- Class 0: Will not survive
- Class 1: Will survive

### 📉 Customer Churn

From lecture:
> "How about **customer retention**? Is this customer behavior is likely to **churn or not**? Then **marketing action** can be taken."

**Problem:** Predict customer retention
- Class 0: Will stay (retain)
- Class 1: Will leave (churn)

### 🖼️ Image Recognition

From lecture:
> "**Image recognition**, various kinds can also be binary class classification. For example, is this animal, **dog or cat**?"

**Problem:** Classify images
- Class 0: Dog
- Class 1: Cat

### 💬 Sentiment Analysis

From lecture:
> "**Sentiment analysis**, given this texts or Twitters sentences, what is the sentiment? Is it **negative or positive**?"

**Problem:** Classify text sentiment
- Class 0: Negative
- Class 1: Positive

### 📊 Data Types for Binary Classification

From lecture:
> "As you can see, binary class classification can have a **variety of different types of data input**. It could be **tabulated data**, it could be **image**, it could be **text**, it could be even **speeches**."

**Key insight:**

From lecture:
> "That determines the binary class or not, is actually entirely for the **label** instead of the data itself, or the features itself."

**What makes it binary:** The number of possible output classes (2), not the type of input data!

---

## 6. Example: Breast Cancer Diagnosis

### 🏥 Problem Setup

From lecture:
> "Brief example, we can talk about some **breast cancer diagnosis** problem. This is one of the features that can determine whether this **tumor is malignant or not**. It can be a binary class classification problem."

**Classes:**
- **Malignant** (cancerous) vs **Benign** (non-cancerous)

### 📏 Univariate Case (One Feature)

From lecture:
> "We want to have some **threshold** or some **decision value** that **above this value**, maybe we are more sure that this is going to be malignant. Maybe **below this certain value**, maybe it's less likely to be malignant."

**Visualization:**
```
Feature Value
    |
    |     Benign          |    Malignant
    |   (Class 0)         |   (Class 1)
    |--------------------[T]-------------------
    0                threshold              max
```

**Decision boundary = threshold value**

From lecture:
> "Building a logistic regression model, will help us to find this **threshold value**, which is called the **decision boundary** by the way."

### 📊 Bivariate Case (Two Features)

From lecture:
> "If you have **more than one features**, let's say we have a **two features**, it can be shown as a **2D diagram** like here. Our decision boundary will likely to be a **line** instead of a threshold value."

**Visualization:**
```
Feature 2
    |
    |      Benign    /
    |              /
    |            /  Decision
    |          /    Boundary
    |        /      (Line)
    |      /
    |    /   Malignant
    |  /
    |/________________ Feature 1
```

From lecture:
> "Maybe this side is malignant and this side is likely to be benign."

**Decision boundary = line separating two regions**

---

## 7. The Logistic Function (Sigmoid)

### 🔍 Why Logistic Function?

From lecture:
> "**Logistic function** provides some convenient way to construct a model like this."

### 📐 Function Properties

From lecture:
> "Loss function look like this. It's **between 0-1** and it's **smoothly connect** the line between 0 and 1. There is a **sharp transition** around the certain special value. Let's say this is zero, but it could be any other value."

**Visual shape:**
```
Probability
   1.0 |              ___________
       |            /
       |          /
   0.5 |________/________________  ← Sharp transition
       |      /
       |    /
   0.0 |___/____________________
       |
       -∞         0         +∞
             Input (z)
```

### 🎯 Probability Interpretation

From lecture:
> "This represents because it's **between 0 and 1**. Logistic function can be a **probability function**."

**Key property:** Output range [0, 1] = valid probability

### 📝 Another Name: Sigmoid

From lecture:
> "Actually the logistic function has another name called the **Sigmoid**. This is also called the **Sigmoid function**."

### 🧮 Mathematical Formula

From lecture:
> "The form takes this one. The **z is the linear combination** of the features with the weight and bias, like we did in the linear regression. Then this z goes through our **nonlinear function**, 1/1+ e to the - t."

**Formula:**
```
σ(z) = 1 / (1 + e^(-z))
```

Where:
```
z = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ
```

**Components:**
- **z**: Linear combination (also called **logit**)
- **σ(z)**: Sigmoid function output (probability)
- **β₀**: Bias/intercept
- **β₁, β₂, ..., βₚ**: Weights/coefficients
- **x₁, x₂, ..., xₚ**: Features

From lecture:
> "By the way, this **z is called logit** and this is a related decision boundary."

### 🎚️ The 0.5 Threshold

From lecture:
> "When it's set to zero, that means this is our threshold value and the probability here is going to be this one, and this **g is zero**, then it's **one-half**. It's going to meet the **0.5**."

**At z = 0:**
```
σ(0) = 1 / (1 + e^0) = 1 / (1 + 1) = 0.5
```

**Decision rule:**

From lecture:
> "With the **0.5 threshold** we can say this is going to be **malignant**, and **below 0.5 probability**, we can say it's going to be **benign**."

```
If σ(z) ≥ 0.5  →  Predict Class 1 (Malignant)
If σ(z) < 0.5  →  Predict Class 0 (Benign)
```

---

## 8. Why Not Use Linear Regression?

### 🤔 The Question

From lecture:
> "Well, some people might ask, **why don't we use linear regression** instead and maybe we can fit it here. We can fit this and then maybe find some threshold and it can also fit the probability of 0.5. We can try to do that."

**Idea:** Fit a line through binary data and use threshold at 0.5

### ❌ Problem 1: Finding the Threshold is Hard

From lecture:
> "It's **not easy**. First of all, we will have to **find out where this threshold is** and maybe we can just fit a line first and then just figure out which value will give 0.5 threshold."

**Issue:** No principled way to determine threshold

### ❌ Problem 2: Different Threshold Than Logistic Regression

From lecture:
> "But if we do that, it gives a **different threshold value** to the logistic regression."

**Issue:** Linear regression threshold ≠ logistic regression decision boundary

### ❌ Problem 3: Poor Interpretability

From lecture:
> "The one problem with the linear regression model, if it fit it, and then find the threshold where the probability value becomes 0.5 is that it's **not very interpretable**."

**Linear regression issues:**
1. Can predict values < 0 or > 1 (invalid probabilities!)
2. No natural probabilistic interpretation
3. Sensitive to outliers
4. No guarantee threshold is meaningful

### ✅ Why Logistic Regression Is Better

From lecture:
> "Whereas the **logistic regression with a sigmoid function**, it is a **well-defined the probability function**. It's **very interpretable** that we can find where probability becomes 0.5 and this gives the **right threshold** for us."

**Advantages:**
1. ✓ Outputs are valid probabilities [0, 1]
2. ✓ Natural probabilistic interpretation
3. ✓ Well-defined decision boundary at P = 0.5
4. ✓ Theoretically sound
5. ✓ Directly models class probabilities

---

## 9. Decision Boundaries

### 📏 Univariate Case (1 Feature)

From lecture:
> "In **univariate case**, where we have only **one feature**. The decision boundary is a **point** where it meets the **probability equals 0.5**."

**Decision boundary:**
```
z = β₀ + β₁x₁ = 0
```

**Solving for x₁:**
```
x₁ = -β₀ / β₁
```

From lecture:
> "The equation looks like this and you can get the value out of it."

**This gives a single point (threshold) on the x-axis**

### 📊 Bivariate Case (2 Features)

From lecture:
> "If we have a **two features**, the data will lie in the **two-dimensional space** and then the decision boundary becomes a **line**, so we can find the line equation here, which will draw this line."

**Decision boundary:**
```
z = β₀ + β₁x₁ + β₂x₂ = 0
```

**Solving for x₂:**
```
x₂ = (-β₀ - β₁x₁) / β₂
```

**This is the equation of a line in 2D space**

**Visualization:**
```
x₂ axis
    |
    |        Class 0
    |          •  •
    |        •  •
    |      •  •
    |----•----------•---- Decision Boundary (Line)
    |  •  •      •
    | •  •    •
    |•  •  •
    |________________ x₁ axis
       Class 1
```

### 🌐 Multivariate Case (3+ Features)

From lecture:
> "If it's a **multivariate** have a **multidimensional**, more than three, the decision boundary will be our **hyperplane**."

**Decision boundary:**
```
z = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ = 0
```

**This defines a hyperplane in p-dimensional space**

**Summary:**
- **1 feature** → Decision boundary is a **point**
- **2 features** → Decision boundary is a **line**
- **3 features** → Decision boundary is a **plane**
- **p features** → Decision boundary is a **hyperplane** (p-1 dimensional)

---

## 10. Multiclass Classification

### 🔍 The Problem

From lecture:
> "Let's talk about what if we have a **multiple categories**? Instead of having yes-or-no problem, maybe we can have **multiple categories**."

**Example:**

From lecture:
> "Such as, maybe we would like to predict whether this animal is **cat or dog, or maybe cow**."

**Now we have 3 classes instead of 2!**

### 📐 Mathematical Notation Change

**For binary (logistic regression):**

From lecture:
> "For the logistic regression, the logistic, which is the decision boundary, takes this form."

```
z = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ
```

No index for class (only 2 classes)

**For multiclass (softmax):**

From lecture:
> "Then for **softmax**, which is **multinomial**. This has another name, **multinomial**. Multinomial logistic regression has this form. They are very similar except that there is **no index for k category**. This is **index for category**."

```
z_k = β₀^(k) + β₁^(k)x₁ + β₂^(k)x₂ + ... + βₚ^(k)xₚ
```

**Key difference:** Superscript (k) indicates different weights for each class

From lecture:
> "For example, for **category number 1**, we can construct this model so there will be **different weights assigned to each category** and for **each feature**."

---

## 11. Softmax Function (Multinomial Logistic Regression)

### 🔍 From Sigmoid to Softmax

**Binary logistic (sigmoid):**

From lecture:
> "For logistic regression, we use the **sigmoid function** as a probability, and we show this form, but it can be **rewritten** as this form as well."

**Alternative form:**
```
P(y=1|x) = e^z / (1 + e^z)
```

This is equivalent to: `1 / (1 + e^(-z))`

### 🎯 Softmax Generalization

From lecture:
> "This is very similar to softmax. The **softmax function takes the same form** as this one, except that it **now has our index for the category**."

**Softmax formula:**
```
P(y=k|x) = e^(z_k) / Σⱼ e^(z_j)
```

From lecture:
> "Then instead of this, now it has **all the summation over of all the possible or exponent of this corresponding categories**."

**Where:**
- k: specific class we're computing probability for
- j: sum over all classes (1 to K)
- z_k: linear combination for class k

**Example for 3 classes:**
```
P(y=cat|x) = e^(z_cat) / (e^(z_cat) + e^(z_dog) + e^(z_cow))
P(y=dog|x) = e^(z_dog) / (e^(z_cat) + e^(z_dog) + e^(z_cow))
P(y=cow|x) = e^(z_cow) / (e^(z_cat) + e^(z_dog) + e^(z_cow))
```

### 📊 Alternative Name

From lecture:
> "**Softmax is called multinomial logistic regression**."

---

## 12. One-vs-Rest (OvR) Approach

### 🔍 Alternative Multiclass Strategy

From lecture:
> "However, there is another **similar way** that we can use the **original logistic regression** for multi-categories."

**Idea:** Convert multiclass into multiple binary problems

### 📋 How OvR Works

From lecture:
> "Maybe category A, B, C. We can construct such that it is a **binary classification for A versus naught A**, which we will have to **combine these two cases**."

**Example with 3 classes: A, B, C**

**Model 1: A vs Not-A**

From lecture:
> "This is maybe **logistic regression model 1**."

- Class A → Label 1
- Classes B and C combined → Label 0

**Model 2: B vs Not-B**

From lecture:
> "This is a **logistic regression model 2**. We're going to do **B versus naught B**."

- Class B → Label 1
- Classes A and C combined → Label 0

**Model 3: C vs Not-C**

From lecture:
> "Then we're going to construct certain model that says **C versus naught C**."

- Class C → Label 1
- Classes A and B combined → Label 0

### 🎯 Making Predictions with OvR

**Process:**
1. Run all K binary classifiers
2. Get probability from each: P(A), P(B), P(C)
3. Predict class with highest probability

**Example:**
```
Sample X:
- Model 1 (A vs not-A): P(A) = 0.2
- Model 2 (B vs not-B): P(B) = 0.7
- Model 3 (C vs not-C): P(C) = 0.3

Prediction: Class B (highest probability)
```

### 📝 Formal Name

From lecture:
> "This approach is called **one versus the rest**, an **OVR** problem."

---

## 13. Comparing Multiclass Approaches

### 🔄 Two Main Strategies

From lecture:
> "There are **different ways** to get the multi-category classification done. One is, like we mentioned, we use a **multinomial approach**, which is a **softmax**. Another way to do is using **OVR**."

### 📊 Comparison

| Aspect | Softmax (Multinomial) | One-vs-Rest (OvR) |
|--------|----------------------|-------------------|
| **Models trained** | 1 unified model | K separate models |
| **Parameters** | K sets of weights | K sets of weights |
| **Optimization** | Joint optimization | Independent models |
| **Probabilities sum to 1** | ✓ Yes, guaranteed | ✓ Yes (normalized) |
| **Common in practice** | ✓ More common | Less common |
| **Interpretation** | Direct multiclass | Multiple binary problems |

### 🔧 Implementation

From lecture:
> "You can find **SKLearn library** that utilize these two. But I think **softmax or multinomial is more common**."

**In scikit-learn:**
```python
# Softmax (multinomial)
clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')

# One-vs-Rest
clf = LogisticRegression(multi_class='ovr')
```

### 🎯 Model Preferences

From lecture:
> "You will see later other classification models such as **SVM and decision trees**. They have a **preferred way** of being multinomial verses logistic, or maybe some model is more convenient to use one versus the other. We'll talk about that later."

---

## 14. Probability Constraint

### 📊 Sum-to-One Property

From lecture:
> "By the way, both **OVR and softmax**, their **probabilities for categories they sum to 1**."

**Mathematical constraint:**
```
Σₖ P(y=k|x) = 1
```

**Example:**

From lecture:
> "For example, **probability for A** plus **probability for B** plus **probability for being C category** for the sample number 1, they **sum to 1**."

```
P(A) + P(B) + P(C) = 1

Example: 0.2 + 0.7 + 0.1 = 1.0
```

From lecture:
> "That's the same for **logistic and the softmax** regression."

**Why this matters:** Ensures valid probability distribution over classes

---

## 15. Multi-Label vs Multi-Class

### 🔍 Important Distinction

**Multi-class (what we've been discussing):**
- Must choose **exactly one** class
- Mutually exclusive categories
- Probabilities sum to 1

**Multi-label (different problem):**
- Can choose **zero, one, or multiple** classes
- Not mutually exclusive
- Probabilities don't necessarily sum to 1

### 📋 The Multi-Label Problem

From lecture:
> "However, there could be some problem where maybe there are A, B, C category and we **don't necessarily need to pick one of them**, but maybe the category **doesn't exist at all**. So **neither cat, nor dog nor cow, but something else**, then this should be **000**."

**Example: Animal Classification**

**Multi-class (must pick one):**
```
Sample 1: Cat    → [1, 0, 0]
Sample 2: Dog    → [0, 1, 0]
Sample 3: Cow    → [0, 0, 1]
```

**Multi-label (can pick any combination):**
```
Sample 1: Cat only       → [1, 0, 0]
Sample 2: Dog only       → [0, 1, 0]
Sample 3: Cat and Dog    → [1, 1, 0]
Sample 4: None of these  → [0, 0, 0]
Sample 5: All three      → [1, 1, 1]
```

### 📝 Terminology

From lecture:
> "In that case, it's called **multi-label problem**. I know it **sounds strange** because labeling categories, what's the difference? But this type of problem where we **don't necessarily have to pick one of them** in the categories are called **multi-label problem**."

From lecture:
> "Versus if we **have to pick one** of the categories, then it's a **multi-class problem**."

### 🎯 Model Applicability

From lecture:
> "In both the **logistic and softmax models** they are for **multi-class classification**."

**Standard logistic/softmax:** Multi-class only

From lecture:
> "Then there can be **some other ways** to treat the multi-label problem, but we can still **use the same models**, but we will have to **construct the labels differently** and **construct the training process a little differently**."

**For multi-label:** Need modifications to standard approach

### 📊 Prevalence

From lecture:
> "That's a little bit of difference, but you will see **more often the multi-class classification problems then multi-level problems**, but just keep in mind that they exist."

**Key takeaway:** Multi-class is more common, but multi-label problems exist

---

## 16. Visual Decision Boundaries

### 🎨 Softmax Visualization

From lecture:
> "But anyway, **softmax regression** can give this visualization. Let's say we had only **two features** in the data-set, and the data will lay in the **2D plane**, and this is going to be the **decision boundary** the softmax will give us."

**For 2 features and 3 classes:**

```
Feature 2
    |
    |    Region A    /|\ Region C
    |              /  |  \
    |            /    |    \
    |          /      |      \
    |    ----/--------+--------\----
    |      /          |          \
    |    /     Region B            \
    |  /                            \
    |/___________________________________ Feature 1
```

**Multiple decision boundaries:**
- Boundary between A and B
- Boundary between B and C
- Boundary between C and A

From lecture:
> "You can see more examples here."

**Each region corresponds to one class prediction**

---

## 17. Summary

### 🎯 Key Concepts

**1. Logistic Regression Purpose**
- Despite name, used for **classification** not regression
- Best for **binary classification**
- Can be extended to multiclass

**2. Sigmoid Function**
```
σ(z) = 1 / (1 + e^(-z))
```
- Maps real numbers to [0, 1]
- Interpreted as probability
- Has sharp transition at z = 0 (P = 0.5)

**3. Decision Boundary**
- **1 feature:** Point (threshold)
- **2 features:** Line
- **p features:** Hyperplane

**4. Why Not Linear Regression?**
- Can predict outside [0, 1]
- No natural probabilistic interpretation
- Logistic regression is theoretically sound for classification

**5. Multiclass Strategies**
- **Softmax (Multinomial):** Single unified model (more common)
- **One-vs-Rest (OvR):** Multiple binary models

**6. Multi-Class vs Multi-Label**
- **Multi-class:** Pick exactly one category
- **Multi-label:** Can pick zero, one, or multiple

### 📋 Binary Classification Applications

1. Credit card default
2. Insurance fraud detection
3. Spam filtering
4. Medical diagnosis
5. Survival prediction
6. Customer churn
7. Image recognition
8. Sentiment analysis

### 🧮 Mathematical Components

**Logit (linear combination):**
```
z = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ
```

**Sigmoid transformation:**
```
P(y=1|x) = 1 / (1 + e^(-z))
```

**Decision rule:**
```
If P(y=1|x) ≥ 0.5  →  Predict Class 1
If P(y=1|x) < 0.5  →  Predict Class 0
```

### 🔄 Next Steps

From lecture:
> "This ends our video. Then in the **next video**, we're going to talk about **how optimization works** in logistic regression and **how the coefficients are determined**."

---

**End of Lecture Notes - Module 03, Document 2**
