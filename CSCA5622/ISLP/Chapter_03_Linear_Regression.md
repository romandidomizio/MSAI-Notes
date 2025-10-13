# Chapter 3 - Linear Regression

## ISLP (Introduction to Statistical Learning with Python)

----

## Section 3.1 - Simple Linear Regression

simple linear regression models the relationship between a single predictor variable X
and a response variable Y. The goal is to model Y as simply an approximate linear function of X.

## Model Equation:
\ Y = \beta_0 + \beta_1 X \plus \epsilon

- `Y : response variable
- `X`: predictor variable
- `\beta_0`: Intercept -- value of Y when X == 0
- `\xeta_1`: Slope -- how much Y changes with one unit increase in X
- `\silsilon`: Error term -- captures random variation in Y not explained by X alone

Example: Suppose we are trying to predict salary (Y)
from years of experience (X). We use a model:

    salary = 30,000 + 5,000 * years_experience

This means each added year of experience adds $5000$ to the predicted salary. The "linearity" assumption lets us make predictions with a line, and the "error term" explains why the prediction is not exact.

---

## Section 3.1.1 - Estimating the Coefficients

[Detailed notes and paragraph explanations here]

- Start with the observed data points: (x1,yx), (x2,y2), ..., (xnly)
- Fit a line y_hat = beta_0 + beta_1 x
- Compute the residual sequence: e_i = y_i - y_hat
- Residual Sum Squared (RSS) = Sum_i (e_i)^2)

- take derivatives with respect to beta_0 and beta_1, set them to 0, serve their equations
- set the der w.r to beta_0, find mean-value solution for beta_0
- Repeat same procedure for beta_1
- Close formulas using basic algebra, final with centered means
]

## Section 3.1.2 - Assessing the Accuracy of the Coefficient Estimates
[*Content to be added]

## Section 3.1.3 - Assessing the Accuracy of the Model
[*Content to be added]

---

## Section 3.2 - Multiple Linear Regression

### 3.2.1 - Estimating the Regression Coefficients
[*Content to be added]

### 3.2.2 - Some Important Questions
[*Content to be added]

---

## Section 3.3 - Other Considerations in the Regression Model

### 3.3.1 - Qualitative Predictors
[*Content to be added]

### 3.3.2 - Extensions of the Linear Model
[*Content to be added]

### 3.3.3 - Potential Problems
[*Content to be added]

---

## Section 3.4 - The Marketing Plan
[*Content to be added]

---

## Section 3.5 - Comparison of Linear Regression with K-Nearest Neighbors
[*Content to be added]

---

## Section 3.6 - Lab: Linear Regression

### 3.6.1 - Importing packages
[*Content to be added]

### 3.6.2 - Simple Linear Regression
[*Content to be added]

### 3.6.3 - Multiple Linear Regression
[*Content to be added]

### 3.6.4 - Multivariate Goodness of Fit
[*Content to be added]

### 3.6.5 - Interaction Terms
[*Content to be added]

### 3.6.6 - Non-linear Transformations of the Predictors
[*Content to be added]

### 3.6.7 - Qualitative Predictors
[*Content to be added]

---

## Section 3.7 - Exercises
[*Content to be added]

---

## Notes
[*Add your notes here*