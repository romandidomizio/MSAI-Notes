# Feature Selection, Correlation, and Interaction - Detailed Lecture Notes
**CSCA5622 - Module 02**

---

## 📚 Overview

This document explores **automated feature selection methods** and the challenges of correlated features and interactions. Topics include:

- Motivation for automatic feature selection
- Forward selection (maximize R²)
- Backward selection (remove high p-values)
- Mixed selection (combined approach)
- Comparing selection method results
- Why features are correlated (redundancy, confounding, causality)
- Problems with multicollinearity
- Variance Inflation Factor (VIF)
- Model complexity vs performance trade-offs
- Interaction terms between features
- Hierarchical principle for interactions

All concepts explained with examples from the lecture transcript.

---

## 1. Motivation for Automatic Feature Selection

### 🔍 The Problem with Manual Selection

From lecture:
> "Last time we tried to fit the model that has all the features inside and found that some of the coefficients were not significant. After removing those, we tried again and found that the coefficients are significant, but still we had some linearly dependent features in it. **It's a lot of manual process** to figure out which features to select."

**Issues with manual approach:**
- Time-consuming
- Subjective decisions
- May miss important patterns
- Still leaves correlated features

### ✅ The Solution

From lecture:
> "Instead of doing that, we're going to introduce some **methods that automatically select the features**."

**Three main methods:**
1. Forward selection
2. Backward selection
3. Mixed selection

---

## 2. Forward Selection

### 🔍 The Algorithm

From lecture:
> "The first method is called the **forward selection**, which **add the feature one-by-one** by looking at the one that **maximize the R² value**. The add feature that maximize the R² value, and that's the forward selection."

### 📋 Step-by-Step Process

**Algorithm:**
```
1. Start with no features (only intercept)
2. Try adding each remaining feature one at a time
3. Calculate R² for each candidate model
4. Add the feature that gives maximum R²
5. Repeat steps 2-4 until stopping criterion met
```

**Stopping criteria options:**
- All features added
- R² improvement below threshold
- Predetermined number of features

### 📊 Results from Lecture Example

From lecture:
> "So **forward selection** gives this result that they **add squarefoot_living first** and then **latitude**, then **view**, and **grade**, and so on. It doesn't have a stock criteria so we're going to just fit all of them to the last feature."

**Order of feature addition (example):**
1. sqft_living (highest individual R²)
2. latitude
3. view
4. grade
5. ... (continues)

### 🎯 What Forward Selection Optimizes

**Focus:** Maximizing R² (model fit)

**Characteristic:** Greedy algorithm - chooses best feature at each step

---

## 3. Backward Selection

### 🔍 The Algorithm

From lecture:
> "Another method is called the **backward selection**, which **starts from the full model**, by full model I mean there are all the features inside the model already, and **remove the one feature that has maximum p-value**."

### 📋 Step-by-Step Process

From lecture:
> "Remove x_j that has **maximum p-value**. We **repeat this process** until we reach the **tolerance of the p-value** or sub-criteria."

**Algorithm:**
```
1. Start with all features in the model
2. Fit the model
3. Find feature with highest p-value
4. Remove that feature
5. Repeat steps 2-4 until all p-values < threshold
```

**Stopping criterion:** All remaining features have p-value < tolerance (e.g., 0.05)

### 📊 Results from Lecture Example

From lecture:
> "This is resulted from the **backward selection**. We start from the full model and then it removes one by one. First, it **removes the floors** and then **squarefoot_lot** second, and **sales_month** removed, and so on, all the way to the top."

**Order of feature removal (example):**
1. floors (highest p-value)
2. sqft_lot
3. sales_month
4. ... (continues removing)

### ⚖️ Forward vs Backward: Different Priorities

From lecture:
> "But as you can see, the **feature importance are the orders are very different** from the forward selection because the **forward selection cares about the R² value**, whereas a **backward selection cares about the p-value**."

**Key difference:**
- **Forward:** Optimizes model fit (R²)
- **Backward:** Optimizes statistical significance (p-values)

---

## 4. Mixed Selection (Best of Both Worlds)

### 🔍 The Algorithm

From lecture:
> "Another good method is called the **mixed selection**, which **combines the forward selection and backward selection** to mean that we add some feature that maximize the R² first, and then fit the new model and inspect the results and see if there are coefficients that are insignificant. If there are insignificant coefficients, just remove the features."

### 📋 Step-by-Step Process

From lecture:
> "Then we **add another feature again** and then **inspect the result** and **remove all the features that have large p-values** and so on."

**Algorithm:**
```
1. Start with no features
2. Add feature that maximizes R² (forward step)
3. Fit model with new feature
4. Check p-values of all features
5. Remove any features with p-value > threshold (backward step)
6. Repeat steps 2-5 until no more improvements
```

### 📊 Results from Lecture Example

From lecture:
> "As you can imagine, the **mixed selection resembles the forward selection result**, however, it **stops at some point**. Because it also look at the maximum R², the orders are pretty similar to forward selection. But then at some point, adding another feature will lead always have a p-value that's larger than certain criteria, so **it stops there**."

**Characteristics:**
- Similar order to forward selection (R² priority)
- But stops earlier (p-value constraint)
- More conservative than forward alone

### ✅ Practical Recommendation

From lecture:
> "Actually, **practically, mixed section is a good way to use** as a feature selection."

**Why mixed selection is preferred:**
- Balances model fit and statistical significance
- Avoids overfitting better than forward alone
- More efficient than backward (doesn't need full model initially)
- Automatic stopping criterion

### 📈 Impact on Multicollinearity

From lecture:
> "Here is a result of the **correlation matrix after we select the features from mixed selection**. **Good news** is that now we **don't have the linearly dependent feature**, such as a square foot above. **These are gone**, but still we see **large correlation values between the features**."

**Progress made:**
- ✅ Linearly dependent features removed
- ⚠️ High correlations still present

---

## 5. Why Features Are Correlated

### 🔍 Understanding Correlation Sources

From lecture:
> "Let's talk about correlated features. Why do they occur? **High correlation among features may occur from different regions**."

### 📊 Three Main Causes

#### Cause 1: Redundant Information

From lecture:
> "One of them would be **redundant information**. When the features are **linearly dependent on each other**, the information is redundant and they may have a high correlation."

**Example:** sqft_living = sqft_above + sqft_basement

**Characteristic:** Perfect or near-perfect mathematical relationship

#### Cause 2: Confounding

From lecture:
> "When there is **underlying effect** such as **confounding** or causality, the features may be highly correlated. For example, **ice cream sales and the sharks attack**, they have nothing to do with each other, however, they can be **caused by hot weather**."

**Explanation:**

From lecture:
> "**Hot weather** here is called the **confounding**. Then because of this or confounding ice cream sales and then sharks attack, they will have a **high correlation in the data**."

**Structure:**
```
        Hot Weather (Confounding)
          /        \
         /          \
    Ice Cream    Shark
     Sales      Attacks
         \          /
          \        /
        High Correlation
     (but no direct causation!)
```

**Key insight:** Two variables correlated due to common cause

#### Cause 3: Causality

From lecture:
> "An example of **causality** is heart disease can lead to heart attack and diabetes. Doesn't cause heart attack directly, but it can cause a heart disease, some type of heart disease, and then cause a heart attack."

**Explanation:**

From lecture:
> "When we look at the diabetes and heart attack, they may look **highly correlated**."

**Structure:**
```
Diabetes → Heart Disease → Heart Attack
                            ↑
                    Highly Correlated
```

**Key insight:** Indirect causal pathway creates correlation

#### Cause 4: Natural Correlation

From lecture:
> "In some cases, just the **variables are correlated in nature**. for example, **number of bedrooms and the size of house**, they don't cause each other, they don't have confounding. However, they are **correlated**, so they may have high correlation."

**Characteristic:** Variables naturally tend to vary together without direct causation or confounding

---

## 6. Problems with Highly Correlated Features

### 🚨 Why High Correlation Is Problematic

From lecture:
> "We mentioned that it is **problematic when there are highly correlated features** and why is that? When the predictors are highly correlated, the **coefficient estimate becomes very inaccurate**, and also the **interpretation of the coefficient** as a variable contribution to the response **becomes inaccurate**."

### ⚠️ Two Main Problems

**1. Inaccurate coefficient estimates**
- Coefficients become unstable
- Small data changes cause large coefficient changes
- Numerical instability

**2. Inaccurate interpretation**
- Can't isolate individual feature effects
- "Holding others constant" assumption violated
- Contribution gets "shared" among correlated features

### 📏 Rule of Thumb

From lecture:
> "When there is a **high correlation between features more than 0.7**, we consider it's **problematic**."

**Guideline:** Correlation > 0.7 → investigate and consider action

---

## 7. Collinearity vs Multicollinearity

### 🔍 Collinearity (Two Features)

From lecture:
> "This is specially called the **collinearity** when **two features are very similar to each other**, like we saw previously in the pair plot that the distribution of the data was **pretty skinny** between Feature 1 and Feature 2, for example, then they are very **collinear**."

**Definition:** High correlation between **two** features

**Visual indicator:** Skinny, elongated scatter plot in pair plot

### 🔍 Multicollinearity (Multiple Features)

From lecture:
> "Well, it's **not always possible to detect the collinearity using correlation matrix**. Because if there are **multiple variables that are involved in the collinearity**, they may look okay in the correlation matrix. However, they could be still co-linear. This special case is called the **multicollinearity**."

**Definition:** Linear relationship among **three or more** features

**Problem:** Pairwise correlations might look fine, but collective dependency exists

**Example:**
```
X₁ + X₂ - 2×X₃ ≈ 0
```

Each pairwise correlation might be moderate, but together they're dependent!

---

## 8. Variance Inflation Factor (VIF)

### 🔍 Better Detection Method

From lecture:
> "Beside the correlation matrix, **variance inflation factor** is a **better way to detect this multicollinearity**."

### 🧮 VIF Formula

From lecture:
> "VIF is defined by this formula. **VIF of our coefficient Beta i is 1 over 1 minus R squared value of this fitting**. And this fitting is actually **not hitting the target variable y**, but **fitting that variable x_i using all other variables**."

**Formula:**
```
VIFᵢ = 1 / (1 - R²ᵢ)
```

Where R²ᵢ is from the model:
```
Xᵢ = β₀ + β₁X₁ + ... + βᵢ₋₁Xᵢ₋₁ + βᵢ₊₁Xᵢ₊₁ + ... + βₚXₚ
```

### 📊 How to Calculate VIF

From lecture:
> "Using **other variables that are not this one** and we are fitting the model and we're going to get the **R-squared value** and we can get the **VIF value**."

**Process for VIFᵢ:**
1. Use Xᵢ as the target variable
2. Use all OTHER features as predictors
3. Fit a regression model
4. Get R²ᵢ from this model
5. Calculate VIFᵢ = 1/(1 - R²ᵢ)

### 📏 Interpretation

From lecture:
> "If the VIF value is **larger than 5** in general, or sometimes **10**, it means that there is **strong multicollinearity**."

**Guidelines:**
- **VIF < 5:** No multicollinearity concern
- **VIF = 5-10:** Moderate multicollinearity
- **VIF > 10:** Strong multicollinearity (problematic)

**Intuition:**
- R²ᵢ near 0 → VIF near 1 (feature independent of others) ✓
- R²ᵢ near 1 → VIF very large (feature predictable from others) ✗

### 📊 Example from Lecture

From lecture:
> "Let's have a look which variables had multicollinearity in the **original model** that we had all the features in it. Clearly **sqft_living, sqft_above, sqft_basement**, they are **linearly dependent each other**. They show **strong multicollinearity**."

**After mixed selection:**

From lecture:
> "Then after we have mixed a selection, some of the redundant features are gone, for example, this one is gone, and let's inspect, **sqft_living has still high VIF value**, but it's **much better than previous one** because of one of the dependent feature was gone."

### ✂️ Removing High Correlation Features

From lecture:
> "Let's **remove these highly correlated features** after the mixed selection. When we remove those variables with a **very high correlation to the square foot living**, we end up with a **much lower VIF value** for square for living because the **collinearity is gone**."

**Result:** VIF improves dramatically when correlated features removed

---

## 9. Feature Selection Considerations

### 🎯 Summary of Criteria

From lecture:
> "Here are some **things to consider when you select features**."

### 📋 Four Main Considerations

#### 1. Model Fitness (R²)

From lecture:
> "We talked about **model fitness** for selection gives a **maximum model fitness** by adding one features at a time."

**Method:** Forward selection
**Goal:** Maximize predictive power

#### 2. Statistical Significance

From lecture:
> "Also, we talked about **removing variables with the insignificant coefficients**, so **backward selection** was good at this."

**Method:** Backward selection
**Goal:** Keep only significant features

#### 3. Combined Approach

From lecture:
> "If we **combine this**, we can have **mixed selection**."

**Method:** Mixed selection
**Goal:** Balance fit and significance

#### 4. Multicollinearity

From lecture:
> "We also talked about some **problems that may occur when you have a multicollinearity**."

**Tool:** VIF
**Goal:** Remove redundant, correlated features

---

## 10. Model Complexity vs Performance

### 📊 Performance Analysis

From lecture:
> "Here is a graph that shows the **performance of each model**. This is including intercepts. This is just the intercept and this model is interceptors one feature and so on, all the way to **14th feature plus the intercept**."

### ⭐ Impact of Removing Correlated Features

From lecture:
> "This **star** represent the **model performance after we removing the variables with a high correlation**. When we remove the highly correlated features, they may have a **better estimation of the coefficient value** and then **better interpretation**. However, it can have some **less performance**."

**Trade-off:**
- ✅ Better coefficient estimates
- ✅ Better interpretability
- ⚠️ Slightly lower R² / performance

### 🎯 Optimal Model Complexity

From lecture:
> "Another thing that we can think about is that, **do we need all these 14 features**? It seems that the **model complexity six or seven gives efficient result**. This is still a less performance than having 14 features. However, **it's good enough**. We can consider that as well."

**Key insight:** Diminishing returns beyond 6-7 features

From lecture:
> "By looking at the **VIF of these six-feature model**, it has a **pretty good VIF as well**."

**Conclusion:** Simpler models (6-7 features) can be preferred:
- Good performance
- Better interpretability
- Lower VIF
- Less overfitting risk

---

## 11. Comparing All Models

### 📊 Coefficient Comparison

From lecture:
> "Here is all the result from the **models that we considered so far**. Again, this number of interesting include the intercept. It's actually **19-feature model** plus one intercept. **Mixed selection** gives **14 number of features** selected, and so on."

### 🔍 Focus on sqft_living Coefficient

From lecture:
> "Then if we look at just the **feature coefficient for square foot living**, the coefficient values are **all different** and **all of them are statistically significant**. However, if you can see the coefficient values, they are **very different**."

### 💡 Most Accurate Model

From lecture:
> "In particular, **this model removed all the features that are highly correlated to the square feet living**. Therefore, the **coefficient value for square foot living is more accurate and more interpretable**."

**Interpretation:**

From lecture:
> "This means that when we **increase the square foot living by one**, then the **house price goes up by 313 dollars**."

### ⚠️ Why Other Models Differ

From lecture:
> "On the other hand, the **coefficient value for square foot living is lower** in other models. That's because the other models still had **other variables that are highly correlated to the square foot living**. Therefore, the **coefficient values are not accurate**, and all these are **correlated features**. They **share the contribution to the house price**."

**Problem:** Correlated features "split" the effect
- Each gets partial credit
- Coefficients artificially lowered
- Interpretation becomes unclear

---

## 12. Interaction Terms

### 🔍 What Are Interactions?

From lecture:
> "Lastly, let's talk about **what to do when there are interactions**. What is the interactions? **Interactions can happen when this coefficient is not constant**, but is a **function of some other variable**, say x_3."

**Definition:** Effect of X₁ on Y depends on the value of X₃

### 🧮 Adding Interaction Terms

From lecture:
> "In that case, what we want to do is that we're going to have **interaction term, so x_1 * x_3**, and then **assign another coefficient**. Let's say Beta_13, and then **add to the model**."

**Model with interaction:**
```
Y = β₀ + β₁X₁ + β₂X₂ + β₃X₃ + β₁₃(X₁×X₃) + ...
```

### 🔢 Multiple Interactions

From lecture:
> "Not only this, we can also do **all the combinations** such as adding **x_1 * x_2**, adding **x_2 * x_3** and all combinations, and we can also have **higher-order terms**, something like that."

**Possible interactions:**
- Two-way: X₁×X₂, X₁×X₃, X₂×X₃, ...
- Three-way: X₁×X₂×X₃, ...
- And more...

### ⚠️ The Infinite Menu Problem

From lecture:
> "In that case, we have **infinite menu of features** and we don't want to do that, but however, we can just **choose the order**. Maybe the **maximum order is just one feature times another**, and then we can add them up."

**Solution:** Limit interaction order (typically just 2-way interactions)

### 🔍 Feature Selection for Interactions

From lecture:
> "Then we have a **problem of how to select all these many combination features**. Again, we're going to **apply the same method that we talked about before**. **Mixed selection method is a good way to do that**."

**Approach:** Use mixed selection on all candidate features (main + interaction terms)

### 📏 Hierarchical Principle

From lecture:
> "One thing that is a **little different from the previous case** is that when we have this **interaction term**, then we **must include also the Beta_1 x_1 + Beta_3 x_3**."

**Rule:** If you include X₁×X₃, you MUST include both X₁ and X₃

From lecture:
> "Sometimes you might see these **coefficient values may not be significant**. However, we should **still include these terms** in order to have this term."

**Important:** Keep main effects even if p-values are high, as long as interaction is significant

### 🔄 Same Process, Different Context

From lecture:
> "With that difference, **having interaction terms in the model is the same as having multiple features in the model**."

**Strategy:** Treat interactions as additional features, apply same selection methods

---

## 13. Summary

### 🎯 Key Concepts

**1. Automatic Feature Selection**
- **Forward:** Add features maximizing R²
- **Backward:** Remove features with high p-values
- **Mixed:** Combine both (recommended)

**2. Why Features Correlate**
- Redundant information (linear dependence)
- Confounding (common cause)
- Causality (indirect pathways)
- Natural correlation

**3. Problems with Correlation**
- Inaccurate coefficient estimates
- Poor interpretation
- Threshold: correlation > 0.7

**4. VIF for Multicollinearity**
- VIF = 1/(1 - R²ᵢ)
- Fit Xᵢ using other features
- VIF > 5-10 indicates problems

**5. Model Selection Trade-offs**
- Performance vs interpretability
- Complexity vs overfitting
- 6-7 features often sufficient

**6. Interaction Terms**
- Effect of X₁ depends on X₃
- Include X₁×X₃ term
- Must keep main effects
- Use mixed selection

### 📋 Practical Workflow

```
1. Start with all candidate features
2. Apply mixed selection
3. Check VIF for remaining features
4. Remove high-correlation features if VIF > 5-10
5. Consider model complexity (6-7 features?)
6. Add interaction terms if needed
7. Apply mixed selection again
8. Keep main effects with interactions
9. Final model: balance performance & interpretability
```

---

**End of Lecture Notes - Module 02, Document 4**
