# Chapter 10 - Boosting and Additive Trees

## ESLII (The Elements of Statistical Learning)

---

## Overview

Boosting is one of the most powerful learning ideas introduced in the last twenty years. It was originally designed for classification problems, but can be extended to regression as well. The motivation for boosting was a procedure that combines the outputs of many "weak" classifiers to produce a powerful "committee."

---

## Section 10.1 - Boosting Methods

### Introduction to AdaBoost

**AdaBoost.M1** (Freund and Schapire, 1997) is the most popular boosting algorithm for two-class problems where Y ∈ {-1, 1}.

**Key Concepts:**

* **Weak Classifier**: A classifier whose error rate is only slightly better than random guessing
* **Purpose of Boosting**: Sequentially apply the weak classification algorithm to repeatedly modified versions of the data
* **Output**: Sequence of weak classifiers G_m(x), m = 1, 2, ..., M

**Final Classifier:**

The predictions are combined through a weighted majority vote:

[
G(x) = \text{sign}\left(\sum_{m=1}^M \alpha_m G_m(x)\right)
]

Where:
* α_1, α_2, ..., α_M are weights computed by the boosting algorithm
* Higher α values give more influence to more accurate classifiers

### How AdaBoost Works

**Data Modifications:**

* Apply weights w_1, w_2, ..., w_N to each training observation (x_i, y_i)
* Initially: w_i = 1/N (equal weights)
* At each iteration m:
  * Observations **misclassified** by G_{m-1}(x) → weights **increased**
  * Observations **correctly classified** → weights **decreased**
* Difficult observations receive ever-increasing influence

### Algorithm 10.1: AdaBoost.M1

1. **Initialize**: Set observation weights w_i = 1/N, i = 1, 2, ..., N

2. **For m = 1 to M**:
   
   a. Fit classifier G_m(x) to training data using weights w_i
   
   b. Compute weighted error rate:
      [
      \text{err}_m = \frac{\sum_{i=1}^N w_i I(y_i \neq G_m(x_i))}{\sum_{i=1}^N w_i}
      ]
   
   c. Compute classifier weight:
      [
      \alpha_m = \log\left(\frac{1 - \text{err}_m}{\text{err}_m}\right)
      ]
   
   d. Update observation weights:
      [
      w_i \leftarrow w_i \cdot \exp[\alpha_m \cdot I(y_i \neq G_m(x_i))]
      ]

3. **Output**: 
   [
   G(x) = \text{sign}\left(\sum_{m=1}^M \alpha_m G_m(x)\right)
   ]

### Discrete vs. Real AdaBoost

* **Discrete AdaBoost**: Base classifier G_m(x) returns discrete class label
* **Real AdaBoost**: Base classifier returns real-valued prediction (e.g., probability mapped to [-1, 1])

### Performance Example (Figure 10.2)

**Simulated Data**:
* Features X_1, ..., X_10 are standard independent Gaussian
* Target: Y = 1 if Σ X_j² > χ²_10(0.5), else Y = -1
* 2000 training cases, 10,000 test cases

**Results with Stumps** (two terminal-node classification tree):
* Single stump alone: 45.8% test error
* After 400 boosting iterations: 5.8% test error
* Single large 244-node tree: 24.7% test error

**Breiman's Quote**: AdaBoost with trees is the "best off-the-shelf classifier in the world"

### 10.1.1 - Outline of This Chapter

Key developments covered:

* AdaBoost fits an **additive model** optimizing exponential loss function (similar to binomial log-likelihood)
* Population minimizer of exponential loss = **log-odds of class probabilities**
* Loss functions for regression/classification more **robust** than squared error or exponential loss
* **Decision trees** are ideal base learner for data mining applications
* **Gradient Boosted Models (GBMs)** for boosting trees with any loss function
* Importance of **"slow learning"** via shrinkage and randomization
* Tools for **interpretation** of fitted models

---

## Section 10.2 - Boosting Fits an Additive Model

### Basis Function Expansion

Boosting fits an additive expansion in elementary basis functions:

[
f(x) = \sum_{m=1}^M \beta_m b(x; \gamma_m)
]

Where:
* β_m: expansion coefficients
* b(x; γ): basis functions characterized by parameters γ
* For boosting: basis functions are individual classifiers G_m(x) ∈ {-1, 1}

### Examples of Additive Expansions

1. **Neural Networks** (Chapter 11):
   * b(x; γ) = σ(γ_0 + γ_1^T x)
   * σ(t) = 1/(1 + e^{-t}) is sigmoid function
   * γ parameterizes linear combination of inputs

2. **Wavelets** (Section 5.9.1):
   * γ parameterizes location and scale shifts of "mother" wavelet

3. **MARS** (Section 9.4):
   * Truncated-power spline basis functions
   * γ parameterizes variables and knot values

4. **Trees**:
   * γ parameterizes split variables, split points, and terminal node predictions

### Fitting the Model

Typically minimize loss function over training data:

[
\min_{\{\beta_m, \gamma_m\}_1^M} \frac{1}{N} \sum_{i=1}^N L\left(y_i, \sum_{m=1}^M \beta_m b(x_i; \gamma_m)\right)
]

**Alternative approach** when fitting single basis function is feasible:

[
\min_{\beta, \gamma} \sum_{i=1}^N L(y_i, \beta b(x_i; \gamma))
]

---

## Section 10.3 - Forward Stagewise Additive Modeling

### Algorithm 10.2: Forward Stagewise Additive Modeling

**Approach**: Sequentially add new basis functions **without adjusting** previously added terms.

1. **Initialize**: f_0(x) = 0

2. **For m = 1 to M**:
   
   a. Compute optimal basis function and coefficient:
      [
      (\beta_m, \gamma_m) = \arg\min_{\beta, \gamma} \sum_{i=1}^N L(y_i, f_{m-1}(x_i) + \beta b(x_i; \gamma))
      ]
   
   b. Update:
      [
      f_m(x) = f_{m-1}(x) + \beta_m b(x; \gamma_m)
      ]

**Key Property**: Previously added terms are **not modified**.

### Squared-Error Loss

For L(y, f(x)) = (y - f(x))²:

[
L(y_i, f_{m-1}(x_i) + \beta b(x_i; \gamma)) = (r_{im} - \beta b(x_i; \gamma))²
]

Where r_{im} = y_i - f_{m-1}(x_i) is the **residual** of current model.

**Interpretation**: At each step, add the term β_m b(x; γ_m) that best fits the current residuals.

This is the basis for **least squares regression boosting** (Section 10.10.2).

**Note**: Squared-error loss is generally **not a good choice for classification**.

---

## Section 10.4 - Exponential Loss and AdaBoost

### Main Result

**AdaBoost.M1 is equivalent to forward stagewise additive modeling using exponential loss**:

[
L(y, f(x)) = \exp(-y f(x))
]

### Derivation

For basis functions G_m(x) ∈ {-1, 1}, must solve:

[
(\beta_m, G_m) = \arg\min_{\beta, G} \sum_{i=1}^N \exp[-y_i(f_{m-1}(x_i) + \beta G(x_i))]
]

This can be rewritten as:

[
(\beta_m, G_m) = \arg\min_{\beta, G} \sum_{i=1}^N w_i^{(m)} \exp(-\beta y_i G(x_i))
]

Where w_i^{(m)} = exp(-y_i f_{m-1}(x_i)) are weights that change with each iteration.

### Two-Step Solution

**Step 1**: For any β > 0, the optimal G_m(x) is:

[
G_m = \arg\min_G \sum_{i=1}^N w_i^{(m)} I(y_i \neq G(x_i))
]

This is the classifier that **minimizes the weighted error rate**.

**Step 2**: Plugging G_m back in and solving for β:

[
\beta_m = \frac{1}{2} \log\left(\frac{1 - \text{err}_m}{\text{err}_m}\right)
]

Where the minimized weighted error rate is:

[
\text{err}_m = \frac{\sum_{i=1}^N w_i^{(m)} I(y_i \neq G_m(x_i))}{\sum_{i=1}^N w_i^{(m)}}
]

### Weight Update

The approximation is updated:
[
f_m(x) = f_{m-1}(x) + \beta_m G_m(x)
]

Causing weights for next iteration to be:
[
w_i^{(m+1)} = w_i^{(m)} \cdot e^{-\beta_m y_i G_m(x_i)}
]

Using -y_i G_m(x_i) = 2·I(y_i ≠ G_m(x_i)) - 1:

[
w_i^{(m+1)} = w_i^{(m)} \cdot e^{\alpha_m I(y_i \neq G_m(x_i))} \cdot e^{-\beta_m}
]

Where α_m = 2β_m (from Algorithm 10.1, line 2c).

The factor e^{-β_m} multiplies all weights equally, so has no effect.

### Training vs. Exponential Loss (Figure 10.3)

**Simulated Data Observations**:
* Training misclassification error → 0 at ~250 iterations (stays there)
* Exponential loss keeps decreasing
* Test misclassification error continues improving after iteration 250

**Conclusion**: AdaBoost is **not optimizing training misclassification error**; exponential loss is more sensitive to changes in estimated class probabilities.

---

## Section 10.5 - Why Exponential Loss?
*[Content to be added]*

---

## Section 10.6 - Loss Functions for Regression and Classification
*[Content to be added]*

---

## Section 10.7 - "Off-the-Shelf" Procedures for Data Mining
*[Content to be added]*

---

## Section 10.8 - Example: Spam Data
*[Content to be added]*

---

## Section 10.9 - Boosting Trees
*[Content to be added]*

---

## Section 10.10 - Numerical Optimization via Gradient Boosting

### Overview

Fast approximate algorithms for solving optimization problems with any differentiable loss criterion can be derived by analogy to numerical optimization.

**Loss on Training Data:**

[
L(f) = \sum_{i=1}^N L(y_i, f(x_i))
]

**Goal**: Minimize L(f) with respect to f, where f(x) is constrained to be a sum of trees.

### Numerical Optimization View

Ignoring the constraint, minimizing can be viewed as:

[
\hat{f} = \arg\min_f L(f)
]

Where the "parameters" f ∈ ℝ^N are the values of approximating function f(x_i) at each of the N data points:

[
f = \{f(x_1), f(x_2), \ldots, f(x_N)\}^T
]

Numerical optimization procedures solve this as a sum of component vectors:

[
f_M = \sum_{m=0}^M h_m, \quad h_m \in \mathbb{R}^N
]

Where:
* f_0 = h_0 is an initial guess
* Each successive f_m is induced based on f_{m-1} (sum of previously induced updates)

### 10.10.1 - Steepest Descent

**Method**: Choose h_m = -ρ_m g_m where:
* ρ_m is a scalar (step length)
* g_m ∈ ℝ^N is the gradient of L(f) evaluated at f = f_{m-1}

**Gradient Components:**

[
g_{im} = \left[\frac{\partial L(y_i, f(x_i))}{\partial f(x_i)}\right]_{f(x_i) = f_{m-1}(x_i)}
]

**Step Length** (line search):

[
\rho_m = \arg\min_\rho L(f_{m-1} - \rho g_m)
]

**Update:**

[
f_m = f_{m-1} - \rho_m g_m
]

**Interpretation**: Steepest descent is a **very greedy strategy** - the direction -g_m is the local direction in ℝ^N for which L(f) is most rapidly decreasing at f = f_{m-1}.

### 10.10.2 - Gradient Boosting

**Forward Stagewise Boosting** is also a very greedy strategy:
* At each step, choose the tree that maximally reduces loss given current model f_{m-1}
* Tree predictions T(x_i; Θ_m) are **analogous to** components of negative gradient

**Key Differences from Steepest Descent:**

1. **Constraint on components**:
   * Tree components t_m = {T(x_1; Θ_m), ..., T(x_N; Θ_m)}^T are **NOT independent**
   * Constrained to be predictions of a J_m-terminal node decision tree
   * Negative gradient is the **unconstrained** maximal descent direction

2. **Line search**:
   * Stagewise approach performs **separate line search** for components corresponding to each terminal region
   * Steepest descent performs single global line search

**The Dilemma:**

* If minimizing training loss were the only goal, steepest descent would be preferred
* Gradient is trivial to calculate for any differentiable loss function
* But gradient is defined **only at training data points** x_i
* Ultimate goal: generalize f_M(x) to **new data** not in training set

**Resolution - Inducing Trees from Gradients:**

Induce a tree T(x; Θ_m) at iteration m whose predictions are **as close as possible** to the negative gradient.

Using squared error to measure closeness:

[
\tilde{\Theta}_m = \arg\min_\Theta \sum_{i=1}^N (-g_{im} - T(x_i; \Theta))^2
]

**Interpretation**: Fit the tree T to the negative gradient values by **least squares**.

* Solution regions R̃_{jm} may not be identical to exact solution regions R_{jm}
* Generally similar enough to serve the same purpose
* After constructing tree, constants in each region given by line search in each region

### Gradients for Common Loss Functions

**Table 10.2 Summary:**

| **Setting** | **Loss Function** | **-∂L(y_i, f(x_i))/∂f(x_i)** |
|-------------|-------------------|------------------------------|
| **Regression** | (1/2)[y_i - f(x_i)]² | y_i - f(x_i) |
| **Regression** | \|y_i - f(x_i)\| | sign[y_i - f(x_i)] |
| **Regression (Huber)** | Huber | y_i - f(x_i) for \|y_i - f(x_i)\| ≤ δ_m<br>δ_m sign[y_i - f(x_i)] for \|y_i - f(x_i)\| > δ_m<br>where δ_m = αth-quantile{\|y_i - f(x_i)\|} |
| **Classification** | Deviance | kth component: I(y_i = G_k) - p_k(x_i) |

**Interpretations:**

* **Squared error loss**: Negative gradient = ordinary residual
  * -g_{im} = y_i - f_{m-1}(x_i)
  * Fitting tree to gradient ≡ standard least-squares boosting

* **Absolute error loss**: Negative gradient = sign of residual
  * At each iteration, fit tree to **sign of current residuals** by least squares

* **Huber M-regression**: Negative gradient is compromise between above two

* **Classification (multinomial deviance)**: 
  * K least squares trees constructed at each iteration
  * Each tree T_{km} fit to its respective negative gradient:
    [
    -g_{ikm} = I(y_i = G_k) - p_k(x_i)
    ]
  * For binary classification (K=2), only one tree needed

### 10.10.3 - Implementations of Gradient Boosting

**Algorithm 10.3: Gradient Tree Boosting Algorithm**

1. **Initialize** f_0(x) = arg min_γ Σ L(y_i, γ)

2. **For m = 1 to M**:

   a. For i = 1, 2, ..., N compute (pseudo-residuals):
      [
      r_{im} = -\left[\frac{\partial L(y_i, f(x_i))}{\partial f(x_i)}\right]_{f = f_{m-1}}
      ]
   
   b. Fit a regression tree to targets r_{im} giving terminal regions R_{jm}, j = 1, 2, ..., J_m
   
   c. For j = 1, 2, ..., J_m compute:
      [
      \gamma_{jm} = \arg\min_\gamma \sum_{x_i \in R_{jm}} L(y_i, f_{m-1}(x_i) + \gamma)
      ]
   
   d. Update:
      [
      f_m(x) = f_{m-1}(x) + \sum_{j=1}^{J_m} \gamma_{jm} I(x \in R_{jm})
      ]

3. **Output**: f̂(x) = f_M(x)

**Key Points:**

* Line 1: Initialize to optimal constant model (single terminal node tree)
* Line 2(a): Components of negative gradient = **generalized/pseudo residuals** r
* Gradients for common loss functions in Table 10.2

**Classification Algorithm:**

* Similar structure
* Lines 2(a)–(d) repeated **K times** at each iteration m (once per class)
* Result: K different (coupled) tree expansions f_{kM}(x), k = 1, 2, ..., K
* Produce probabilities via softmax or classification via argmax

**Tuning Parameters:**

* **M**: Number of iterations
* **J_m**: Sizes of each constituent tree (m = 1, 2, ..., M)

**Software Implementations:**

* **MART**: "Multiple Additive Regression Trees" (original implementation)
* **R gbm package** (Ridgeway, 1999): Freely available
  * Used in Section 10.14.2 and extensively in Chapters 15-16
* **R mboost package** (Hothorn and Bühlmann, 2006)
* **TreeNet**: Commercial implementation (Salford Systems, Inc.)

---

## Section 10.11 - Right-Sized Trees for Boosting

### Historical Perspective

**Traditional View:**
* Boosting was considered a technique for **combining models** (trees)
* Tree building algorithm regarded as a "primitive" producing models to be combined
* Optimal size of each tree estimated **separately** when built (Section 9.2):
  1. Induce a very large (oversized) tree
  2. Use bottom-up procedure to **prune** to estimated optimal number of terminal nodes

**Problem with This Approach:**

* Assumes implicitly that **each tree is the last one** in the expansion
* Except perhaps for very last tree, this is a **very poor assumption**
* Result:
  * Trees tend to be **much too large**, especially during early iterations
  * **Substantially degrades performance**
  * **Increases computation**

### The Simpler Strategy: Fixed Tree Size

**Approach**: Restrict all trees to be **the same size**, J_m = J ∀m

* At each iteration: induce a J-terminal node regression tree
* J becomes a **meta-parameter** of entire boosting procedure
* Adjusted to **maximize estimated performance** for the data at hand

### Determining Useful Values for J

**Consider Properties of Target Function:**

[
\eta = \arg\min_f E_{XY} L(Y, f(X))
]

* Target function η(x) is the one with **minimum prediction risk** on future data
* This is what we're trying to approximate
* Expected value is over population joint distribution of (X, Y)

### ANOVA Expansion and Interaction Effects

**Key Property**: Degree to which coordinate variables X^T = (X_1, X_2, ..., X_p) **interact** with one another.

**ANOVA (Analysis of Variance) Expansion:**

[
\eta(X) = \sum_j \eta_j(X_j) + \sum_{jk} \eta_{jk}(X_j, X_k) + \sum_{jkl} \eta_{jkl}(X_j, X_k, X_l) + \cdots
]

**Terms Explained:**

1. **First sum** (Main Effects):
   * Functions of only a **single predictor** variable X_j
   * η_j(X_j) are those that jointly best approximate η(X) under loss criterion
   * Called "**main effect**" of X_j

2. **Second sum** (Two-way Interactions):
   * Two-variable functions η_{jk}(X_j, X_k)
   * When added to main effects, best fit η(X)
   * Called "**second-order interactions**" of variable pair (X_j, X_k)

3. **Third sum** (Three-way Interactions):
   * Third-order interactions η_{jkl}(X_j, X_k, X_l)

4. **Higher-order terms**: And so on...

**Practical Observation:**

* For many real-world problems, **low-order interaction effects tend to dominate**
* When this is the case, models producing strong higher-order interaction effects (e.g., **large decision trees**) suffer in accuracy

### Tree Size Limits Interaction Level

**Key Relationship**: Interaction level of tree-based approximations is **limited by tree size J**

* No interaction effects of level **greater than J-1** are possible
* Since boosted models are **additive in the trees**, this limit extends to them as well

**Specific Cases:**

* **J = 2** (single split "decision stump"):
  * Produces boosted models with **only main effects**
  * **No interactions** permitted

* **J = 3**:
  * **Two-variable interaction effects** also allowed

* **J = 4**:
  * Up to **three-way interactions**

* And so on...

**Implication**: Value chosen for J should **reflect the level of dominant interactions** of η(x)

### Practical Guidelines for Choosing J

**General Considerations:**

* Dominant interaction level is **generally unknown**
* In most situations, it will tend to be **low**

**Recommended Range:**

* **J = 2** may be insufficient in many applications
* **J > 10** unlikely to be required
* **Experience indicates**: **4 ≤ J ≤ 8 works well** in context of boosting
* Results fairly **insensitive** to particular choices in this range

**Fine-Tuning:**

* Can try several different values of J
* Choose one that produces **lowest risk on validation sample**
* However, this **seldom provides significant improvement** over using **J ≈ 6**

### Example: Simulated Additive Data (Figure 10.9)

**Setup** (using example 10.2 from Figure 10.2):
* Generative function is **additive** (sum of quadratic monomials)
* Compared boosting with different sized trees

**Results:**

* **Stumps (J=2)** perform the **best**
  * Makes sense since true model is additive (no interactions)
  
* **10-node trees**: Higher test error than stumps
  * Incurs unnecessary variance

* **100-node trees**: Even worse performance
  * Much higher test error

* **Boosting models with J > 2**: Incur unnecessary variance → higher test error

**Comparison**: Boosting algorithm used binomial deviance loss (Algorithm 10.3) vs. AdaBoost Algorithm 10.1

### Coordinate Functions (Figure 10.10)

**Visualization**: Coordinate functions estimated by **boosting stumps** for simulated example

* Shows η_j(X_j) for j = 1, 2, ..., 10
* **True quadratic functions** shown for comparison
* Demonstrates that boosted stumps can accurately recover additive structure

**Key Insight**: Even simple stumps, when boosted, can capture complex univariate relationships while avoiding spurious interactions

---

## Section 10.12 - Regularization

### 10.12.1 - Shrinkage
*[Content to be added]*

### 10.12.2 - Subsampling
*[Content to be added]*

---

## Section 10.13 - Interpretation

### 10.13.1 - Relative Importance of Predictor Variables
*[Content to be added]*

### 10.13.2 - Partial Dependence Plots
*[Content to be added]*

---

## Section 10.14 - Illustrations

### 10.14.1 - California Housing
*[Content to be added]*

### 10.14.2 - New Zealand Fish
*[Content to be added]*

### 10.14.3 - Demographics Data
*[Content to be added]*

---

## Notes
*[Add your notes here]*
