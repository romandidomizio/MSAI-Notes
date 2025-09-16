## CSCA5622/ Week_1 / Textbook_Notes.md

## Chapter 1 – Introduction (ISLP)

This chapter introduces the idea of **statistical learning** as a set of tools for understanding and modeling data.


```\
Y = f(X) + \epsilon
`` `

-  **X** are the inputs (parameters / features). These can be numerical data like size, age, or cholesterol.
-  **Y** is the outcome (like house price, disease presence, etc.)
-  **f** is the true but unknown relationship between X and Y.
  -  **\epsilon** represents random noise, measurement errors, and things we can't capture.

### Section 1.1 – What is Statistical Learning?

- Statistical learning = using statistical and algorithmical tools to find patterns in data.
 - Purposes: two main uses **Prediction** (accurately guessing outcomes) and **Inference**(learning relationships between variables).
- Example Prediction: "Given house characteristics, can we estimate the price?"
- Example Inference: "Which variables (exp: house size, neighborhood) are most important in determining price?"

### Section 1.2 – Why Estimate f?
- We assume there is a real but hidden function `f(X)` in nature.
 - Statistical learning = our attempt to estimate from data a replacement for `f(X).
- Benefits: as prediction and inference.
 - Equation `Y = f(X) + \epsilon` shows that any data point has a part you can'not explain.

### Example interpretation
If `Y = exam score` and `X/inputs` = hours studied,
- `f(X) = the real rule connecting study hours to score
- \epsilon = the luck, student's self-error, random chances, momentary grading.

### Checkpoint
 - If a student studied 10 hours but scored lower than expected, that unexpected part comes from `\epsilon`.
