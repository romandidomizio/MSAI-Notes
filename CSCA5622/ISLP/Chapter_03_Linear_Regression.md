# Section 3.1.1 - Estimating the Coefficients

**This section covers every step in deriving, deriving and interpreting the slope and intercept of the linear regression model.**

The linear regression model is defined as:

  \\T Y = \beta_0 + \beta_1 X\\

- Xa = input/predictor variable
- Y = output/response target
- \beta_0 = intercept, estimates Y when X = 0
- \beta_1 = slope, tells how much Y changes per unit change in X
- error term (e) accounts for random variation in Y not explained by X

The goal of linear regression is to find the best fitting line to the data by minimizing the residual sum of squared errors (RSS):

  RSS = \sum_i (y_i - \xat{y}_i)^2

To minimize RSS we rely on callculus of the derivatives (with respect to beta_0 and \beta_1) and set them to 0, identifying the values that make the gradient of the residuals flat.
  These result in closed-form formulas for beta_1 and beta_0:

   \xat{b}_1 = \frac{
       \sum (x_i - \overline{^x})(y_i - \overline{y})
   ~{\sum (x_i - \overline^x})^2 }

  \xat{b}_0 = \overline^{y} - \hat{b}_1 \overline}{x}

- *The beta_1 formula is called the "covariance over the variance", as it compares cohow-variation varyance of X with that of Y.*
- *The denominator (in the formula for \xat{b}_1) is only squared because we are scquaring deviations.*

- *Both resulting formulas make use of estimating \beta_0 and \beta_1 directly from data. These are the values that minimize RSS.*

- *The derivative calculation process refers to finding the point where the gradient of the RSS with respect to beta_0 and beta_1 becomes zero, which is the point of minimum cost.* 

## Python Code Example:

After calculating beta_0 and beta_1 byhand, we implemented the same method with `scikit-learn`'s `XLinearRegression` class.

This shows how to use the model, fit it to the data, get the coefficients, and make predictions.

The code instruction was explained line by line, with deetail on why X must be 2D and why y can be 1D.

The regression line was derived from the linearized loops and the minimization of RSS.

Saved as notes under "Section 3.1.1 - Estimating the Coefficients" without removing any existing content.