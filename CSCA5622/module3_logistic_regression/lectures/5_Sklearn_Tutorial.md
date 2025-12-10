# sklearn LogisticRegression Usage - Detailed Lecture Notes
**CSCA5622 - Module 03**

---

## 📚 Overview

This document covers practical usage of **sklearn's LogisticRegression** module, including parameters, methods, evaluation metrics, visualization, and statistical testing. All concepts from lecture transcript.

---

## 1. LogisticRegression Module Overview

### 📦 Import and Location

From lecture:
> "LogisticRegression module is inside of **sklearn.linear_model**."

```python
from sklearn.linear_model import LogisticRegression
```

### 🔧 Key Parameters

From lecture:
> "It has a bunch of options here. And interestingly, it **already have regularization term**, and they actually depend on the **type of solver**."

**Default settings:**

From lecture:
> "So by default, **solver is lbfgs**. And then in that case, by default, it uses **l2 regularization**."

### 🎯 Important Options

**fit_intercept:**

From lecture:
> "And the **fit_intercept goes through**, which is **better to have** in linear model."

**class_weight:**

From lecture:
> "You might want to change **class_weight = balanced**. If you have an **imbalanced labels**, then it will **automatically weigh** your class labels so you have a **slightly better performance**."

**multi_class:**

From lecture:
> "In case you have a **more than binary class**, it's going to **automatically apply** some multi_class. And usually, most of the time it will apply the **soft max**, which is **multi nomial**."

**solver:**

From lecture:
> "There are several **solver types**. Usually you **don't have to worry** about it. But if you want to **try out different solvers**, you can try. All of them uses some **sophisticated secondary or similar method**."

**n_jobs:**

From lecture:
> "For **end jobs**, if you have a **multiple core CPU**, then you can **utilize** it. So you can have **less computation time** with the **paralyzation**. If you do the **n_jobs equals -1**, it's going to use **all the CPU cores** in your computer."

---

## 2. Basic Usage

### 📋 Standard Workflow

From lecture:
> "Basic usage is like this. So you can just **call the module** and you can throw in your **preferred options**, and then you can do the **dot fit**."

```python
from sklearn.linear_model import LogisticRegression

# Create model
model = LogisticRegression(class_weight='balanced')

# Fit model
model.fit(X_train, y_train)
```

From lecture:
> "Inside of this fit function, you're going to throw your **data**. So it's **features for the training**, and this **y is the labels** for the training."

### 🔍 Model Attributes After Fitting

**Coefficients:**

From lecture:
> "From this model object, after this fitting has been done, it has a **number of useful stuff** inside. So for example, **model.coef_** will give us the **coefficient values** for all the features in this feature matrix."

```python
coefficients = model.coef_
```

**Intercept:**

From lecture:
> "And **intercept is a separate**. So you will have to do **model.intercept_**, then it's going to give the value for the intercept."

```python
intercept = model.intercept_
```

### 📊 Making Predictions

**Binary predictions:**

From lecture:
> "The **model.predict**, parenthesis, and throw your data such as **test data or train data** or any data that you want to get the prediction out. Then it's going to give the **binaries prediction**. So **why prediction**?"

```python
y_pred = model.predict(X_test)
```

**Probability predictions:**

From lecture:
> "Another good comment is **predict_proba**, and you can throw in your data features, then it's going to produce the **probability**. So **roll output from sigmoid** you can produce."

```python
y_proba = model.predict_proba(X_test)
```

From lecture:
> "So you can use that and **plot this kind of graph**, or you can **inspect the probability**."

---

## 3. Train-Test Split

### 📊 Splitting Data

From lecture:
> "In this example, I'm going to **split my original data x and y** into **train chunk** and then **test chunk**. And that's done by this very popular function called the **train_test_split**. It's inside of **sklearn_model_selection**."

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### ⚠️ Important Note

From lecture:
> "Be careful of these **detailed names**, because sometimes they may **upgrade** and they may **change the names** as they change the directory of their sub-libraries and things like that. But I think **for now it's solid**."

---

## 4. Complete Example

### 📋 Full Workflow

From lecture:
> "So we'll call the LogisticRegression module, and you can **name it differently** if it's too long. So I named it as a **LR**."

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and configure model
LR = LogisticRegression(class_weight='balanced')

# Fit model
clf = LR.fit(X_train, y_train)
```

### 🎯 Getting Accuracy

From lecture:
> "If you want to get **accuracy**, so this **score uses accuracy by default**, you can do the **fitted model**, that **score**, and throw your **test data** and it's going to give some result."

```python
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy}")
```

From lecture:
> "So for this example, the result was **accuracy of 0.96**, which is **pretty good**."

---

## 5. Evaluation Metrics

### 📦 Metrics Module

From lecture:
> "We can use **other kinds of metric**, and they are all in this **sklearn.metrics module**."

```python
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix
)
```

### 📊 Using Metrics

From lecture:
> "I can predict **yp**, and then most of them requires a **y truth** and then **y prediction**. So I'm going to throw in there."

```python
# Get predictions
y_pred = clf.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score:  {f1:.4f}")
```

### 📋 Confusion Matrix

From lecture:
> "I can do the **confusion metrics** using **confusion metrics function**. And again, it needs **y true value** and **y prediction value**. And it also requires the **labels**."

```python
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print("Confusion Matrix:")
print(cm)
```

From lecture:
> "As we mentioned in the previous lecture, this is going to be **y prediction** and this is going to be the **label**."

---

## 6. Precision-Recall Curve

### 📊 What Is It?

From lecture:
> "We can also draw **precision_recall_curve**. So previously we talked about **ROC curves**, and **precision_recall curve is similar**."

From lecture:
> "**ROC curve** had **true positive rate versus pulse positive rate**. **Precision recall curve** is **precision versus recall**, it has this shape."

### 🎯 Interpretation

From lecture:
> "So **ROC curve**, it was better when it's close to the **left top corner**. And **precision_recall_curve** is better when it's close to the **right top corner**, because we want to **high precision and high recall** as well."

### 📋 Usage

From lecture:
> "Using **precision_recall_curve function**, it also requires **true value** and **prediction probability**, actually, rather than **binarized label**. So I'm using this **predict_proba**."

```python
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Get probability predictions
y_proba = clf.predict_proba(X_test)

# Extract probability of positive class
y_proba_pos = y_proba[:, 1]

# Calculate curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba_pos)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='PR Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()
```

From lecture:
> "And the **column one** actually gives the **probability of label being one**. So I'm going to use that."

---

## 7. ROC Curve and AUC

### 📊 ROC Curve

From lecture:
> "So **ROC curve** also works as a similar. So I use **ROC curve** and then it's going to output **fpr, tpr, and the threshold** that was used to calculate this kind of spots."

```python
from sklearn.metrics import roc_curve, auc

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba_pos)

# Calculate AUC
roc_auc = auc(fpr, tpr)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()
```

### 🎯 AUC Score

From lecture:
> "The **AUC score** can be also **automatically calculated** using this function **out of AUC score**. Again, it needs a **true label** and then the **prediction probability**. So it's very **handy**."

```python
from sklearn.metrics import roc_auc_score

auc_score = roc_auc_score(y_test, y_proba_pos)
print(f"AUC Score: {auc_score:.4f}")
```

---

## 8. Getting Statistics (Method 1: statsmodels)

### 🔍 The Problem

From lecture:
> "So we talked about some **various metrics** and how to get the **coefficient values** from vit model. But **how about statistics**? Unfortunately, the **LogisticRegression module in sklearn doesn't give statistics right away**."

### ✅ Solution 1: statsmodels

From lecture:
> "So we have **two choices**. One is using the **statsmodel library** as we did before as in linear regression."

```python
import statsmodels.api as sm

# Prepare data (add constant for intercept)
X_with_const = sm.add_constant(X)

# Fit logistic regression
logit_model = sm.Logit(y, X_with_const)
result = logit_model.fit()

# Get summary
print(result.summary())
```

### ⚠️ Parameter Order

From lecture:
> "So inside of linear regression, we can use that **logit module** and then through our data. So **be careful** that their **order of feature and label is different** here. So they take the **label first** and then the **features**."

**sklearn:** `model.fit(X, y)`
**statsmodels:** `sm.Logit(y, X)`

### 📊 Output Differences

From lecture:
> "**Different from linear regression**, that gave a lot of other metric like r squared or just r squared and many other metrics such as f statistics. But here, they **don't have that**, perhaps because we **don't need that** in the nonlinear case."

From lecture:
> "However, it does give the **coefficient value** and then the **standard error** for that, and then **T test** instead of T test, but they are kind of the same. So here we can see that this **p value is very small**. So this **coefficient value is a significant**."

---

## 9. Getting Statistics (Method 2: Bootstrapping)

### 🔍 What Is Bootstrapping?

From lecture:
> "The other way to do it using a scale library is a **bootstrapping**. So **bootstrapping** is like this, as a reminder."

From lecture:
> "This is the **original sample**, and then we can **resample it multiple times** like this. We can **resample with the replacement**. So you might see some **duplicate data** in samples, and then we can **fit the model**."

**Process:**
1. Take original sample
2. Resample with replacement (same size)
3. Fit model on bootstrap sample
4. Repeat many times
5. Analyze distribution of coefficients

### 📦 BaggingClassifier

From lecture:
> "Conveniently, there is a module you just called the **BaggingClassifier**, which is essentially a **wrapper**. So this is class inside of **sklearn.ensemble module**."

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
```

### 🔧 Parameters

From lecture:
> "Takes the **base_estimator**. So it can be **any estimator**. So any kind of model, not only the LogisticRegression, but you can do a linear regression, or you can do tree models and others you can throw in here."

**Key parameters:**

From lecture:
> "And then **number of estimators** means that **how many times we will bootstrap** and then fit the model."

From lecture:
> "**Bootstrap is true**, so it's going to use bootstrapping."

From lecture:
> "**Oob_score** means the **outer bag**. So you can set aside some of the bootstrap samples and then you can use it as a **validation purposes**."

From lecture:
> "And **jobs** we can do also a **-1**, then it will **utilize all the computing resources** that we have."

### 📋 Complete Example

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression

# Create base estimator
base_lr = LogisticRegression(class_weight='balanced')

# Create bagging classifier
clf = BaggingClassifier(
    base_estimator=base_lr,
    n_estimators=1000,
    bootstrap=True,
    oob_score=False,
    n_jobs=-1,
    random_state=42
)

# Fit
clf.fit(X_train, y_train)
```

### 🔍 Extracting Bootstrap Results

From lecture:
> "I can pull some useful things from it. So for example, **.estimators_** will give all the the **fitted model objects** inside of a **list**. And since I asked for **number of estimators equals 1,000**, it's going to have **1,000 models** inside."

```python
# Get all fitted models
all_models = clf.estimators_

# Check number of models
print(f"Number of models: {len(all_models)}")

# Get coefficients from first model
first_model_coef = all_models[0].coef_
print(f"First model coefficients: {first_model_coef}")
```

### 📊 Collect All Coefficients

```python
import numpy as np

# Collect coefficients from all bootstrap samples
n_features = X_train.shape[1]
coef_bootstrap = np.zeros((len(all_models), n_features))
intercept_bootstrap = np.zeros(len(all_models))

for i, model in enumerate(all_models):
    coef_bootstrap[i, :] = model.coef_[0]
    intercept_bootstrap[i] = model.intercept_[0]

print(f"Coefficient distributions shape: {coef_bootstrap.shape}")
```

### 📈 Visualize Distributions

From lecture:
> "So I draw **histogram** first to see how they look like. And because and is written only big, they look like **normal distributions** skewed sometimes, but roughly they have some **mean and some bit**."

```python
import matplotlib.pyplot as plt

# Plot histograms for each coefficient
fig, axes = plt.subplots(1, n_features, figsize=(15, 4))

for i in range(n_features):
    axes[i].hist(coef_bootstrap[:, i], bins=50, edgecolor='black')
    axes[i].set_title(f'Feature {i}')
    axes[i].set_xlabel('Coefficient Value')
    axes[i].set_ylabel('Frequency')
    axes[i].axvline(0, color='red', linestyle='--', label='Zero')
    axes[i].legend()

plt.tight_layout()
plt.show()
```

---

## 10. Statistical Testing with Bootstrap

### 🔍 t-Test for Coefficients

From lecture:
> "So what do I do with this **all 1,000 values for each coefficient**? I can do the **t test**. So there is a convenient Python package here, **scipy.stats.ttest_1sample**. We are doing **t test** for the **p values**."

```python
from scipy.stats import ttest_1samp

# Test if coefficient is significantly different from 0
# Null hypothesis: coefficient = 0
# Alternative: coefficient ≠ 0

results = []
for i in range(n_features):
    t_stat, p_value = ttest_1samp(coef_bootstrap[:, i], popmean=0)
    results.append({
        'feature': i,
        't_statistic': t_stat,
        'p_value': p_value,
        'mean_coef': coef_bootstrap[:, i].mean(),
        'std_coef': coef_bootstrap[:, i].std()
    })
```

### 📋 Interpretation

From lecture:
> "The usage is like this. So I put the **list of coefficients**. So I'm going to put one kind of coefficient at a time, and then it can be in the by the way. And then this **value is the mean** that it wants to **compare with**. So for the **hypothesis testing**, the **null hypothesis** says that my **coefficient value is 0**. Therefore, I can put **0 here**."

From lecture:
> "The **alternative** says that my **coefficient is not 0**. And to test that, we're going to pull out the **p value**."

### 🎯 Decision Rule

From lecture:
> "And if **p value is smaller than certain threshold**, I'm going to choose **5% error**. That means, if p value is **smaller than 0.25**, because **t test or t test**, they have **two wings**. So a p value is smaller than this value, that means my **coefficient value is significant**, right?"

```python
import pandas as pd

# Create results dataframe
results_df = pd.DataFrame(results)

# Determine significance
results_df['significant'] = results_df['p_value'] < 0.05

print("Bootstrap t-test Results:")
print(results_df)
```

### 📊 Example Results

From lecture:
> "So as you can see, **all coefficients are very significant**, few **100 t values** away from the 0, and **p values are all 0s**. So **all of them are significant**."

---

## 11. Complete Code Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score,
    precision_recall_curve
)
from scipy.stats import ttest_1samp

# Generate data
X, y = make_classification(n_samples=1000, n_features=5, 
                          n_informative=3, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 1. Basic Model
clf = LogisticRegression(class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# 2. Predictions
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# 3. Metrics
print("="*50)
print("PERFORMANCE METRICS")
print("="*50)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"AUC:       {roc_auc_score(y_test, y_proba):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 4. Bootstrap for statistics
bagging_clf = BaggingClassifier(
    base_estimator=LogisticRegression(class_weight='balanced'),
    n_estimators=1000,
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)
bagging_clf.fit(X_train, y_train)

# 5. Extract coefficients
n_features = X_train.shape[1]
coef_bootstrap = np.array([model.coef_[0] for model in bagging_clf.estimators_])

# 6. t-tests
print("\n" + "="*50)
print("COEFFICIENT SIGNIFICANCE (Bootstrap t-tests)")
print("="*50)
for i in range(n_features):
    t_stat, p_value = ttest_1samp(coef_bootstrap[:, i], popmean=0)
    mean_coef = coef_bootstrap[:, i].mean()
    print(f"Feature {i}: Coef={mean_coef:8.4f}, t={t_stat:8.2f}, p={p_value:.4e}")

# 7. Visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
axes[0, 0].plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
axes[0, 0].plot([0, 1], [0, 1], 'k--')
axes[0, 0].set_xlabel('False Positive Rate')
axes[0, 0].set_ylabel('True Positive Rate')
axes[0, 0].set_title('ROC Curve')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
axes[0, 1].plot(recall, precision)
axes[0, 1].set_xlabel('Recall')
axes[0, 1].set_ylabel('Precision')
axes[0, 1].set_title('Precision-Recall Curve')
axes[0, 1].grid(True)

# Coefficient distributions (first two features)
axes[1, 0].hist(coef_bootstrap[:, 0], bins=50, edgecolor='black')
axes[1, 0].axvline(0, color='red', linestyle='--')
axes[1, 0].set_title('Feature 0 Coefficient Distribution')
axes[1, 0].set_xlabel('Coefficient Value')

axes[1, 1].hist(coef_bootstrap[:, 1], bins=50, edgecolor='black')
axes[1, 1].axvline(0, color='red', linestyle='--')
axes[1, 1].set_title('Feature 1 Coefficient Distribution')
axes[1, 1].set_xlabel('Coefficient Value')

plt.tight_layout()
plt.show()
```

---

## 12. Summary

### 🎯 Key Points

**sklearn LogisticRegression:**
- Default: lbfgs solver, L2 regularization
- Use `class_weight='balanced'` for imbalanced data
- Automatic multiclass support (softmax)

**Key Methods:**
- `.fit(X, y)` - Train model
- `.predict(X)` - Binary predictions
- `.predict_proba(X)` - Probability predictions
- `.score(X, y)` - Accuracy
- `.coef_` - Coefficients
- `.intercept_` - Intercept

**Metrics (sklearn.metrics):**
- accuracy_score, recall_score, precision_score, f1_score
- confusion_matrix
- roc_curve, auc, roc_auc_score
- precision_recall_curve

**Statistical Testing:**
- **Method 1:** statsmodels.Logit (parameter order: y, X)
- **Method 2:** Bootstrapping with BaggingClassifier
- Use scipy.stats.ttest_1samp for significance

---

**End of Lecture Notes - Module 03, Document 5**
