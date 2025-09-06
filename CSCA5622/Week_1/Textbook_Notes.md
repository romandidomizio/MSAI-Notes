## CSCA5622/ Week_1 / Textbook_Notes.md

### Chapter 3.1 – Simple Linear Regression (ISLP, Ch. 3, p. 61)

### – Key Concepts

Simple Linear Regression (SLR) is a foundational technique in supervised learning where we predict a continuous outcome (response variable) based on a single input (predictor variable).

### 射智版标

Intuitive:

- Imagine trying to draw a straight line through a scatterplot of data points.
- This line should summarize the linear relationship between an input variable X and an output variable Y.
- Our goal is to predict the values of X using new observations of X from the line.

### 🌉 Model Definition

The model has the form:  \n  \
Y \approx \beta_0 + \beta_1 X\n  \nWhere:\n  - \$Y\: response variable\n  - \X\: predictor variable\n  - \beta_0\: intercept (value of Y when X=0)\n  - \beta_1:\ slope (change in Y for each unit change in X)

### © Assumptions

- **Linearity**: The relationship between X and Y is approximately linear.
- **Additive Errors**: The randomness or noise is additive (i.e., \ V = beta_0 + beta_1 X + \epsilon \).
- **Independent Observations**: Each observation is not influenced by the others.

### 🐘 Purpose

- SLR allows both:  \n  - ** Inference**: understanding relationships between variables
  - **Prediction**: forecasting future outcomes

### 🌉 My Summary

- Simple linear regression is about fitting a line to predict one numerical variable from another.
- It introduces two unknown parameters: **intercept** and **slope**, which quantify the trend.
- These parameters are estimated using data.
- The model is simple but powerful – it forms the foundation for many more complex methods.
