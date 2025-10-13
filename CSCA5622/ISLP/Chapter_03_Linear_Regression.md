
# Chapter 3 - Linear Regression

## ISLP (Introduction to Statistical Learning with Python)

---

## Section 3.1 - Simple Linear Regression

simple linear regression models the relationship between a single predictor variable X
and a response variable Y. The goal is to model Y as simply an approximate linear function of X.

## Model Equation:
\[
Y = \beta_0 + \beta_1 X + \epsilon
\]

- `Y` : response variable
- `X`: predictor variable
- `\beta_0`: intercept -- value of Y when X = 0
- `\beta_1`: slope -- how much Y changes with one unit increase in X
- `epsilon`: error term -- captures random variation in Y not explained by X alone

Example: Suppose we are trying to predict salary (Y) from years of experience (X). We use a model:

    salary = 30,000 + 5,000 * years_experience

This means each added year of experience adds $50000$ to the predicted salary. The "linearity" assumption lets us make predictions with a line, and the error term explains why the prediction is not exact.

---

## Section 3.1.1 - Estimating the Coefficients

### Concept Introduction:
We aim to find the best-fitting line through a cloud of points (x, y). This line has two coefficients:
- \\ \beta_0 \\: the intercept
- \\ \beta_1 \\: the slope

### Mathematical Foundation:
To estimate these, we minimize the **Residual Sum of Squares (RSS*)**:
\\[
RSS = \\sum_{i=1}^n (y_i - \\hat{y}_i)^2 = \\sum_{i=1}^n (y_i - (\beta_0 + \beta_1 x_i))^2
\\d]

To minimize RSS:
- Take partial derivatives w.r4.t. \\beta_0 \\ and \\beta_1
- Set each to 0
- Solve the resulting system of equations

This yields the closed-form solutions:
\\[
\hat{\\beta}_1 = \\frac{\\sum (x_i - \\over l_x)_y(y_i - \\over y_|}}{\\sum (x_i - \\over x\)_2]
\\\n{\hat{\\beta}_0 }= \\over y \- \hat{\\beta}_1 \\over x\\}