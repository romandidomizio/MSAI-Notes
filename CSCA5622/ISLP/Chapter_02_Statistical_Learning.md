# Chapter 2 - Statistical Learning

## ISLP (Introduction to Statistical Learning with Python)

---

## Section 2.1 - What Is Statistical Learning?

Statistical learning refers to a vast set of tools for understanding data. We assume there exists a relationship between inputs (predictors) and outputs (response):

**Y = f(X) + ε**

Where:
- **f(X)** = true but unknown relationship
- **ε** = noise/randomness we cannot capture (irreducible error)
- **X** = predictors/features/independent variables
- **Y** = response/outcome/dependent variable

We build an estimate f̂(X) to approximate the true function f(X).

### 2.1.1 - Why Estimate f?

There are two main reasons we might want to estimate f:

#### Prediction
- We want to predict Y for new values of X
- We treat f̂ as a black box - we don't care about its exact form
- Accuracy of Ŷ = f̂(X) depends on two quantities:
  - **Reducible error**: can be improved by better estimation of f
  - **Irreducible error**: cannot be reduced no matter how well we estimate f

**Error Decomposition:**
E(Y - Ŷ)² = E[f(X) + ε - f̂(X)]² = [f(X) - f̂(X)]² + Var(ε)

#### Inference
- We want to understand the relationship between X and Y
- We want to know:
  - Which predictors are associated with the response?
  - What is the relationship between the response and each predictor?
  - Is the relationship linear or more complex?

### 2.1.2 - How Do We Estimate f?

Methods for estimating f can be characterized as either **parametric** or **non-parametric**.

#### Parametric Methods
1. **Step 1**: Make an assumption about the functional form of f
   - Example: assume f is linear in X
   - f(X) = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ

2. **Step 2**: Use training data to fit/train the model
   - Estimate parameters β₀, β₁, ..., βₚ
   - Most common: least squares

**Advantages:**
- Simplifies the problem of estimating f
- Reduces to estimating a set of parameters
- Generally need fewer observations

**Disadvantages:**
- The model we choose usually won't match the true unknown form of f
- Can lead to poor estimates if chosen model is too far from true f
- Can try more flexible models, but this can lead to overfitting

#### Non-parametric Methods
- No explicit assumptions about the functional form of f
- Seek an estimate of f that gets as close to the data points as possible
- Avoid the assumption of a particular functional form

**Advantages:**
- Can accurately fit a wider range of possible shapes for f
- No danger of a poor model choice

**Disadvantages:**
- Large number of observations required
- Prone to overfitting

### 2.1.3 - The Trade-Off Between Prediction Accuracy and Model Interpretability

**Interpretability vs. Flexibility:**
- **Restrictive methods** (low flexibility): more interpretable
- **Flexible methods** (high flexibility): less interpretable

**When do we prefer more restrictive methods?**
- When inference is the goal
- When we need to understand relationships
- When we have limited data

**When do we prefer more flexible methods?**
- When prediction accuracy is the primary concern
- When we have lots of data
- When interpretability is not important

### 2.1.4 - Supervised Versus Unsupervised Learning

#### Supervised Learning
- For each observation of the predictor measurements, there is an associated response measurement
- Goal: fit a model that relates the response to the predictors
- Goal: accurately predict the response for future observations
- **Examples**: regression, classification

#### Unsupervised Learning
- For every observation, we observe a vector of measurements, but no associated response
- Goal: understand the relationships between variables or observations
- More challenging because there's no response variable to guide analysis
- **Examples**: clustering, dimensionality reduction

#### Semi-supervised Learning
- Some observations have response measurements, others don't
- Combines supervised and unsupervised techniques

### 2.1.5 - Regression Versus Classification Problems

#### Regression
- **Quantitative response**: numerical values
- **Examples**: person's age, height, income
- **Methods**: linear regression, polynomial regression

#### Classification
- **Qualitative response**: categorical values
- **Examples**: brand of product purchased, whether person defaults on debt
- **Methods**: logistic regression, decision trees, support vector machines

**Note**: The distinction is not always clear-cut. Methods used for one can often be adapted for the other.

---

## Section 2.2 - Assessing Model Accuracy

No single method dominates all others over all possible data sets.

### 2.2.1 - Measuring the Quality of Fit

#### Mean Squared Error (MSE)
For regression:
MSE = (1/n) Σᵢ₌₁ⁿ (yᵢ - f̂(xᵢ))²

**Training MSE vs. Test MSE:**
- **Training MSE**: computed using training data
- **Test MSE**: computed using fresh test data
- **Key insight**: We care about test MSE, not training MSE

#### The Training vs. Test MSE Relationship
- As model flexibility increases:
  - Training MSE decreases monotonically
  - Test MSE initially decreases, then increases (U-shaped curve)
- **Overfitting**: when a method yields small training MSE but large test MSE

#### Cross-Validation
When test data is not available, we can estimate test MSE using cross-validation techniques.

### 2.2.2 - The Bias-Variance Trade-Off

For any given x₀, the expected test MSE can be decomposed into three components:

**E[y₀ - f̂(x₀)]² = Var(f̂(x₀)) + [Bias(f̂(x₀))]² + Var(ε)**

#### Variance
- **Var(f̂(x₀))**: amount by which f̂ would change if estimated using different training sets
- **High variance**: small changes in training data result in large changes in f̂
- **Generally**: more flexible methods have higher variance

#### Bias
- **Bias(f̂(x₀))**: error introduced by approximating real-life problem with simpler model
- **High bias**: method consistently misses relevant relations between features and response
- **Generally**: more flexible methods have lower bias

#### The Trade-off
- **Flexible methods**: low bias, high variance
- **Inflexible methods**: high bias, low variance
- **Goal**: find method with both low bias and low variance
- **Reality**: there's usually a trade-off

### 2.2.3 - The Classification Setting

#### Error Rate
For classification, the analog to MSE is the error rate:
Error Rate = (1/n) Σᵢ₌₁ⁿ I(yᵢ ≠ ŷᵢ)

Where I(yᵢ ≠ ŷᵢ) is an indicator variable (1 if incorrect, 0 if correct).

#### The Bayes Classifier
- **Bayes classifier**: assigns each observation to the most likely class
- **Bayes decision boundary**: where P(Y = j | X = x₀) = 0.5 for binary classification
- **Bayes error rate**: lowest possible error rate (analogous to irreducible error)

#### K-Nearest Neighbors (KNN)
Real data doesn't give us conditional probabilities, so we estimate them:

1. Given K and prediction point x₀
2. Identify K points in training data nearest to x₀ (call this set N₀)
3. Estimate conditional probability: P(Y = j | X = x₀) = (1/K) Σᵢ∈N₀ I(yᵢ = j)
4. Classify x₀ to class with highest estimated probability

**KNN Performance:**
- **Choice of K**: crucial for performance
- **Low K**: low bias, high variance (overfitting)
- **High K**: high bias, low variance (underfitting)

---

## Section 2.3 - Lab: Introduction to Python

### 2.3.1 - Getting Started
- Setting up Python environment
- Installing necessary packages: numpy, pandas, matplotlib, scikit-learn
- Jupyter notebooks for interactive analysis

### 2.3.2 - Basic Commands
- Basic Python syntax and operations
- Variable assignment and data types
- Mathematical operations

### 2.3.3 - Introduction to Numerical Python
- NumPy arrays and operations
- Creating and manipulating arrays
- Mathematical functions and broadcasting

### 2.3.4 - Graphics
- Matplotlib for data visualization
- Creating plots: line plots, scatter plots, histograms
- Customizing plots: labels, titles, colors

### 2.3.5 - Sequences and Slice Notation
- Python sequences: lists, tuples, strings
- Indexing and slicing operations
- List comprehensions

### 2.3.6 - Indexing Data
- Pandas DataFrames and Series
- Boolean indexing and conditional selection
- Handling missing data

### 2.3.7 - Loading Data
- Reading data from various sources: CSV, Excel, databases
- Data inspection and basic exploration
- Handling different data formats

### 2.3.8 - For Loops
- Iteration in Python
- Loop patterns and control structures
- Nested loops and loop optimization

### 2.3.9 - Additional Graphical and Numerical Summaries
- Summary statistics: mean, median, standard deviation
- Correlation and covariance
- Advanced plotting techniques

---

## Section 2.4 - Exercises

### Conceptual Questions
1. Explain the differences between parametric and non-parametric methods
2. Describe the bias-variance trade-off
3. When would you prefer a flexible approach over an inflexible one?

### Applied Questions
1. Implement KNN classifier from scratch
2. Compare training vs. test error for different levels of flexibility
3. Analyze bias-variance trade-off with simulation study

---

## Key Takeaways

1. **Statistical learning framework**: Y = f(X) + ε
2. **Two main goals**: prediction and inference
3. **Method types**: parametric vs. non-parametric
4. **Learning types**: supervised vs. unsupervised
5. **Problem types**: regression vs. classification
6. **Fundamental trade-offs**: 
   - Prediction accuracy vs. interpretability
   - Bias vs. variance
7. **Model assessment**: focus on test error, not training error
8. **No universal best method**: method choice depends on the specific problem
