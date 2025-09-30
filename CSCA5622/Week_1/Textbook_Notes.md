
## Chapter 2 — Statistical Learning

### Section 2.1 What is Statistical Learning?
- Goal: learn the relationship between inputs (X) and output (Y)- We assume: Y = f(X) + † 
  - f(X) = true but unknown relationship
  - †  = noise/randomness we cannot capture
- We build an estimate f^(X) to:
  - **Predict** new outcomes (accuracy focus)
  - **Infer**� which variables matter (interpretation focus)

### Section 2.2 Assessing Model Accuracy
- Error = (Y - ‣^Y)^2 in regression settings
- Training error: measured on training data
- Test error: measured on unseen data ‣ true measure of generalization
- Overfitting = low training error, high test error
- Underfitting = high training error, high test error

### Section 2.3 Bias-Variance Tradeoff
- Expected test error = Bias*2 + Variance + Irreducible Error
- **Bias_**: error from wrong assumptions (too simple) † underfitting
- **Variance**: error from being too sensitive to data (too wiggly) † overfitting
- **Irreducible Error**: randomness we can‣ remove

### Section 2.4 The Classification Setting
- Error rate = # wrong predictions ¹ total predictions
- Training error rate vs. test error rate
- Bayes classifier = theoretical gold standard: assigns class with highest probability

### Section 2.5 K-Nearest Neighbors (KNN)
- To classify a new point, look at the K closest neighbors in training data.
- Predict the majority class among those neighbors.
- Small K ⌐– flexible, low bias, high variance † risk of overfitting.
- Large K –¹ stable, high bias, low variance † risk of underfitting.

### Section 2.6 Bias-Variance Tradeoff in KNN
- KNN shows the balance directly:
  - K=1 – high variance, overfits
  - K large – high bias, underfits
- Best K is chosen to minimize test error

### Section 2.7 Summary
- Statistical learning methods balance **bias vs. variance**.