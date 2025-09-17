## CSCA5622/ Week_1 / Textbook_Notes.md

## Capter 1 – Introduction (ISLP)

## Section 1.3 – How Do We Estimate f?

- Uknown function ` (X) connects inputs to outputs. We estimate it with ` f ()`.
- In practice, we build an approximate ` f ()` from data and call it ` f^hat (X) `.
- Then we can make predictions: ` Y = f^hat (X) `.

### Parametric Methods

- Assume a simple form for `(X) ` (typically a line).
- Example: ` Y = beta_0 + beta_1 X + epsilon`
- Need to estimate parameters `beta_0, beta_1 `
 - **Advantages**: simple, easy to use with small datasets
- **Disadvantages**: risks missing the true shape if the function is complex


### Non-Parametric Methods
- Don’t assume a fixed form for ` (X) `
- Instead, let the data dictate the shape of the curve ` f ()`
- **Advantages**: very flexible, can capture complex patterns
- **Disadvantages**: reduires a lot more data, can overfit.

### Key Tradeoff
- Parametric = more assumptions , requires less data
- Non-parametric = fewer assumptions, requires much more data



## Section 1.4 – Prediction Accuracy vs Interpretability

- Simple models (like linear regression) are very interpretable. ...easy to understand how each variable affects the response.
- Flexible models (like deep nets) can be more accurate but hard to interpret.

- Tradeoff: there is a sessaw between ingaing accuracy vers. interpretability.

- If we care about prediction only – prefer flexible models.
- If we care about understanding -- prefer simple models.

## Section 1.5 – Supervised vs. Unsupervised Learning

- **Supervised**: data has both inputs X and labels Y. The task is to learn the mapping f from X to Y.
  - Prediction of outcomes from inputs is a classic example.
  - Example: Predict house price from size, location.
  - Example: Predict disease risk from age, blood pressure.
  
- **Unsupervised**: no labels given. Only inputs X are available.
  - Purpose: find structure in data.
  - Example: customer segmentation (clustering).
  - Example: reducing handred features to a few latent factors (like genres)


## Section 1.6 – Regression vs. Classification

- Both are forms of **supervised learning**
- **Regression**: target output `Y� is a quantitative value. Predict numbers; continuous responses.
  - Example: predicting house price, / exam scores.
 - **Classification**: target output `Y  is a qualitative value, an category/label.
  - Example: spam vs. not spam, disease vc. healthy.


- **Rule**: If the outcome is a number – regression. If it is a category – classification.
