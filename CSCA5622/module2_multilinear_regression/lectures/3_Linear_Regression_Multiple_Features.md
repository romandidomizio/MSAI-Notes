# Linear Regression with Multiple Features - Detailed Lecture Notes
**CSCA5622 - Module 02**

---

## 📚 Overview

This document explores **multi-linear regression with multiple variables**, focusing on practical model building with many features. Topics include:

- Multi-linear regression model formulation
- Interpreting coefficients with multiple features
- Types of variables (real-valued, ordinal, non-ordinal categorical)
- Handling categorical variables (binary and N-1 encoding)
- Feature correlation analysis and pair plots
- Testing overall model significance (F-test)
- Testing individual coefficient significance (t-tests, p-values)
- Identifying and addressing multicollinearity
- Practical feature selection considerations

All concepts explained with examples from the lecture transcript.

---

## 1. Multi-Linear Regression with Multiple Variables

### 🔍 Transition from Previous Topics

**Previous:** Multi-linear regression with high-order terms of **single variable** (polynomial regression)

**Now:** Multi-linear regression with **multiple different variables**

### 🧮 Model Formulation

From lecture:
> "The multi linear regression model can be formulated by this, so all the variables are **linearly combined** to represent the model to predict the target variable, Y."

**General form:**
```
Y = β₀ + β₁X₁ + β₂X₂ + β₃X₃ + ... + βₚXₚ + ε
```

**Example - House Price:**
```
Price = β₀ + β₁×sqft_living + β₂×bedrooms + β₃×bathrooms + β₄×grade + ...
```

### 🎯 Learning Objectives

From lecture:
> "We're going to see **how to interpret these coefficients** and then **how to inspect whether these coefficients are significant**. And we're going to talk about **how to select the features** and what to consider when we select the features. And we'll talk about **high correlated features and multi collinearity**."

---

## 2. Interpreting Coefficients

### 🔍 Coefficient Meaning

From lecture:
> "Each coefficient is an **average effect of that variable to Y**, target variable **when we consider all other variables are independent and fixed**."

**Interpretation:** βᵢ = change in Y for one-unit increase in Xᵢ, **holding all other variables constant**

**Example:**
```
Price = 50000 + 150×sqft_living + 10000×bedrooms
```

- β₁ = 150: Each additional sq ft adds $150, holding bedrooms constant
- β₂ = 10000: Each additional bedroom adds $10,000, holding sqft constant

### ⚠️ Key Assumptions

From lecture:
> "This assumption may not be true in general. So variables or the **predictors might be correlated** in real world scenario."

And:
> "We also assume these **coefficients are constant**. However, that might not be true in general. So if there is an **interaction between two variables or more**, this constant or coefficient may not be true constant, but it is a **function of some other variables**."

**Assumptions often violated:**
1. Features are correlated (not independent)
2. Coefficients may vary (interactions exist)

---

## 3. Types of Variables

### 📊 Three Main Categories

From lecture:
> "There can be **real value number** and there could be **categorical variable**. And categorical variable can have **ordinal** and **non-ordinal** categorical variable."

#### Type 1: Real-Valued (Continuous)

**Examples:** sqft_living, price, latitude, longitude, yr_built

**Usage:** Use directly in model

#### Type 2: Ordinal Categorical

From lecture:
> "What is the ordinal categorical variable? These are **categories that have meaning in their order**. So something like **age group** or **grade a, b, c** or **one, two, three, four**."

**Examples:** grade (A, B, C), condition (Poor, Fair, Good, Excellent)

**Usage:** Can encode as numbers (1, 2, 3) since order matters

#### Type 3: Non-Ordinal Categorical

From lecture:
> "The examples of **non ordinal categorical variables** are **male, female, race, ethnicity** and so on. Some classes, they don't have any meaning in their order. So we can **permute the orders of categories** and they don't have any effect."

**Examples:** gender, race, city, color

**Challenge:**

From lecture:
> "So **non ordinal categories are difficult to use** in linear regression model. Because in linear regression model, a variable value times the coefficient present how much of the value in target variable is contributed by that variable. Therefore, if the variable can be **permuted arbitrarily**, it's not easy to use in the linear regression."

---

## 4. Encoding Non-Ordinal Categorical Variables

### ✅ Binary Variables (2 Categories)

From lecture:
> "For example, we can code the **male female into 0 or 1** or **1 or 0** or sometimes **-1 to 1**. So if you choose one of these, **it will work**."

**Gender encoding:**
- Male = 0, Female = 1
- Or Male = 1, Female = 0
- Or Male = -1, Female = +1

All work! Choose one consistently.

### ✅ N-1 Encoding for Multiple Categories

From lecture:
> "And then how about race? Let's say race had only **three categories, Asian Black and Caucasian**."

**Initial idea - create 3 binary variables:**

From lecture:
> "So we can convert this into **individual three binary categorical variable**. So is the person **Asian or not**? Is the person **black or not**? Is the person **Caucasian or not**?"

**But redundancy exists:**

From lecture:
> "However, we **don't need all three of them** because if the two are known, the other one can be known as well. So **they are dependent**. So, if we just **get rid of one of them** and use only **two into the model**, then it works better."

**General rule:**

From lecture:
> "So in general, if the **non ordinal categorical variable had N categories**, we could convert them into **N-1 binary categorical variables**."

**Example:** Race with 3 categories → Use 2 dummy variables

| Person    | is_Asian | is_Black | (Caucasian = reference) |
|-----------|----------|----------|-------------------------|
| Asian     | 1        | 0        |                         |
| Black     | 0        | 1        |                         |
| Caucasian | 0        | 0        | baseline                |

**Interpretation:**
- β₁ (is_Asian): Difference for Asian vs Caucasian
- β₂ (is_Black): Difference for Black vs Caucasian

### ⚠️ Warning: Large N

From lecture:
> "But you have to also **consider whether you want to include the N-1 new features** into your model. So what if you had a **really large N**, do you want to add **large N-1 features** into your model? **Probably not**."

**Example from lecture:**

From lecture:
> "That is an example of this **zip code**. So zip code had **70 something categories** and I didn't want to add **70 something new binary variables** into my model because my model then becomes **too big**. So hopefully, the **other variables** such as **latitude and longitude** can capture some information about the location of the house. So I'm going to just **get rid of zip code** and then use other variables."

**Decision:** For large N, consider alternatives or drop the variable

---

## 5. Qualitative Feature Inspection

### 🔍 Pre-Modeling Analysis

From lecture:
> "Before we build the model, let's have a **qualitative inspection**. So this case, we're going to see the **correlation between the price and all other variables** and see which variables might be useful to predict the price."

### 📊 Correlation with Target

From lecture:
> "So, **sqft_living** could be useful, **grade** could be useful. And some other variables such as **sqft_above** or **sqft_living15** could be useful."

**High correlation with price → potentially useful features**

### 🔗 Identifying Redundant Information

From lecture:
> "By the way, this **sqft_living15** is a square foot living of **similar 15 houses**. So therefore, this must have some **redundant information** as sqft_living."

**Linear dependency:**

From lecture:
> "And also the **sqft_living** is a **square foot above plus the square foot basement**. So they are **linearly dependent**. So they must have **redundant information**."

```
sqft_living = sqft_above + sqft_basement
```

### 📈 High Feature Correlations

From lecture:
> "And this is redundant information shows the **high correlation between the features**. So these two variables have **really high correlation** because they are **linearly dependent**. And this sqft_living and sqft_living15 because they are **similar in the definition**, they are also **highly correlated**."

### 📊 Pair Plots

From lecture:
> "This can be also visually inspected in the **para plot**. So para plot is distribution plot between two features. So in the **diagonal element**, it shows the **distribution of itself**, the feature itself and the **off diagonal elements** shows that the **distribution between one feature to the another**."

**Visual clue for collinearity:**

From lecture:
> "For example, **sqft_above and sqft_living** had a really high correlation. And you can see **very skinny distribution** of the data. Actually, this is a **very good indication of a collinearity**, which we will talk about later. But essentially that happens because these two features are **dependent to each other** or **nearly dependent to each other**."

**Look for:** Skinny, elongated scatter plots = high correlation

### 🧠 Summary

From lecture:
> "Okay, so we've **qualitatively inspected** variables and their correlations and we found that **some features may have some redundant information** and they may cause some problems."

---

## 6. Building and Evaluating the Model

### 🔍 Initial Full Model

From lecture:
> "So with that in mind, let's see what happens if we **throw all the features into the model** and fit to the data."

### 📊 Model Fit: R-Squared

From lecture:
> "So here are the **result summary table** and we can see that model fit with the **R squared is almost a 0.7**, which is good value."

**R² ≈ 0.70:** Model explains 70% of variance in price (good!)

### 📈 F-Statistic: Overall Model Significance

#### Purpose

From lecture:
> "And this actually shows whether there is **at least one significant variable** in the model."

#### Hypothesis Test

From lecture:
> "So the **null hypothesis** for F-test would be **all the coefficient values are zero**."

**Hypotheses:**
- H₀: β₁ = β₂ = ... = βₚ = 0 (all zero, model useless)
- H₁: At least one βᵢ ≠ 0 (model useful)

#### Formula

From lecture:
> "So the F value is defined by this formula. So **TSS - RSS divided by a number of features** and divided by **RSS times n - p - 1**."

```
F = [(TSS - RSS) / p] / [RSS / (n - p - 1)]
```

#### Interpretation

From lecture:
> "And then the **bigger this number** we are **more sure about there is at least one significant variable** in the model."

#### Our Result

From lecture:
> "So in our case, the **statistic value is big** and the **P-value for that is almost a zero**, that means it's **smaller than a certain threshold of error rate**. Therefore, we can conclude that our model has **at least one significant variable**."

**Conclusion:** Model is useful overall!

---

## 7. Individual Coefficient Significance

### 🔍 Examining P-Values

From lecture:
> "So let's have a look at the **P-values for individual variables** and we can see immediately that **some variables have insignificant coefficient values**."

### 🚨 Identifying Insignificant Features

From lecture:
> "So, **sqft_lot**, **floors** have a really high P-value as well as **sales_month** have high P-value. So we can **reject these features** because their coefficient values are **essentially zero**."

**Action:** Remove features with high p-values (typically > 0.05)

### ✅ Refined Model

From lecture:
> "All right., so after **removing the features that has a high P-values**, we get this result. So our, [R²] value is similar, F-statics value is still large and let's **inspect the t score and the P-values of each individual coefficients**. And **all of them looks statistically significant** and that's good."

**Good news:** All remaining features significant!

### ⚠️ Remaining Problem

From lecture:
> "However, this is **not the complete story** because we still see **some features such as these three are linearly dependent each other are still exist in the model** and they still have a **high correlation value**."

**Issue:** Multicollinearity still present (sqft_living, sqft_above, sqft_basement)

### 🔄 Next Steps

From lecture:
> "So we're going to talk about **some better ways to automatically add or remove the features** in the next video."

---

## 8. Summary

### 🎯 Key Concepts

**1. Multi-Linear Regression**
- Model: Y = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ
- Coefficients = partial effects holding others constant

**2. Variable Types**
- Real-valued: use directly
- Ordinal categorical: can encode as numbers
- Non-ordinal categorical: need N-1 binary encoding

**3. Categorical Encoding**
- Binary: 0/1 encoding
- N categories: N-1 dummy variables
- Large N: consider alternatives

**4. Qualitative Inspection**
- Check correlations with target
- Identify feature correlations
- Look for linear dependencies
- Use pair plots (skinny = collinearity)

**5. Statistical Testing**
- F-test: tests overall model (H₀: all β = 0)
- t-tests/p-values: test individual coefficients
- Remove high p-value features (> 0.05)

**6. Multicollinearity**
- High correlations between features
- Linear dependencies problematic
- Need better selection methods

### 📋 Workflow

```
1. Identify variable types
2. Encode categorical variables
3. Calculate correlations
4. Create pair plots
5. Build full model
6. Check F-statistic (overall)
7. Check p-values (individual)
8. Remove insignificant features
9. Address multicollinearity
```

---

**End of Lecture Notes - Module 02, Document 3**
