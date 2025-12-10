# Confusion Matrix - Comprehensive Reference Notes
**CSCA5622 - Module 03: Classification**

---

## 📚 Overview

A **confusion matrix** (also called an **error matrix**) is a fundamental tool in machine learning for evaluating the performance of classification algorithms. It provides a detailed breakdown of correct and incorrect predictions, allowing for deeper analysis than simple accuracy metrics.

This document covers:
- Definition and structure of confusion matrices
- Binary classification components (TP, TN, FP, FN)
- Detailed worked example
- Performance metrics derived from confusion matrix
- Multi-class confusion matrices
- Limitations and considerations
- Python implementation examples

**Source:** Based on Wikipedia article and standard machine learning literature.

---

## 1. What Is a Confusion Matrix?

### 🔍 Definition

From Wikipedia:
> "A confusion matrix is a specific table layout that allows visualization of the performance of an algorithm, typically a supervised learning one."

### 📊 Key Characteristics

**Structure:**
- Each **row** represents instances in an **actual class**
- Each **column** represents instances in a **predicted class**
- (Or vice versa - both conventions exist in literature)

**The diagonal:**
- Diagonal elements = **correct predictions**
- Off-diagonal elements = **errors/confusion**

**Why "confusion"?**
- Makes it easy to see if the system is **confusing two classes**
- Shows which classes are commonly mislabeled as others

### 🎯 Purpose

**Primary uses:**
1. **Visualize** classifier performance
2. **Identify patterns** in misclassification
3. **Calculate metrics** beyond simple accuracy
4. **Diagnose problems** with specific classes

---

## 2. Binary Classification Structure

### 📐 The 2×2 Confusion Matrix

For binary classification (two classes: positive and negative):

```
                    Predicted Class
                  Positive    Negative
Actual   Positive    TP          FN
Class    Negative    FP          TN
```

### 🔢 Four Fundamental Outcomes

#### True Positive (TP)
- **Actual:** Positive
- **Predicted:** Positive
- **Meaning:** Correctly identified positive case
- **Example:** Patient has cancer → Predicted cancer ✓

#### False Negative (FN)
- **Actual:** Positive
- **Predicted:** Negative
- **Meaning:** Missed a positive case (Type II error)
- **Example:** Patient has cancer → Predicted no cancer ✗
- **Also called:** Miss, Type II error

#### False Positive (FP)
- **Actual:** Negative
- **Predicted:** Positive
- **Meaning:** Incorrectly flagged negative as positive (Type I error)
- **Example:** Patient has no cancer → Predicted cancer ✗
- **Also called:** False alarm, Type I error

#### True Negative (TN)
- **Actual:** Negative
- **Predicted:** Negative
- **Meaning:** Correctly identified negative case
- **Example:** Patient has no cancer → Predicted no cancer ✓

### 📊 Visual Summary

```
┌─────────────────────────────────────────────────┐
│              PREDICTED                           │
│         Positive      Negative                   │
│  P  ┌──────────────┬──────────────┐             │
│  o  │   TRUE       │    FALSE     │             │
│  s  │  POSITIVE    │   NEGATIVE   │             │
│  i  │   (TP)       │    (FN)      │             │
│  t  │   ✓ Hit      │   ✗ Miss     │             │
│  i  ├──────────────┼──────────────┤   ACTUAL    │
│  v  │   FALSE      │    TRUE      │             │
│  e  │  POSITIVE    │   NEGATIVE   │             │
│     │   (FP)       │    (TN)      │             │
│  N  │ ✗ False Alarm│   ✓ Correct  │             │
│  e  │              │   Rejection  │             │
│  g  └──────────────┴──────────────┘             │
│  a                                               │
│  t                                               │
│  i                                               │
│  v                                               │
│  e                                               │
└─────────────────────────────────────────────────┘
```

---

## 3. Detailed Example: Cancer Detection

### 📋 Scenario Setup

**Dataset:** 12 patients
- **8 patients** have cancer (Actual Positive)
- **4 patients** are cancer-free (Actual Negative)

**Classification:**
- Class 1 (Positive) = Has cancer
- Class 0 (Negative) = Cancer-free

### 🔬 Individual Predictions

| Patient | Actual Status | Predicted Status | Outcome |
|---------|---------------|------------------|---------|
| 1       | Cancer (1)    | No Cancer (0)    | **FN** ✗ |
| 2       | Cancer (1)    | No Cancer (0)    | **FN** ✗ |
| 3       | Cancer (1)    | Cancer (1)       | **TP** ✓ |
| 4       | Cancer (1)    | Cancer (1)       | **TP** ✓ |
| 5       | Cancer (1)    | Cancer (1)       | **TP** ✓ |
| 6       | Cancer (1)    | Cancer (1)       | **TP** ✓ |
| 7       | Cancer (1)    | Cancer (1)       | **TP** ✓ |
| 8       | Cancer (1)    | Cancer (1)       | **TP** ✓ |
| 9       | No Cancer (0) | Cancer (1)       | **FP** ✗ |
| 10      | No Cancer (0) | No Cancer (0)    | **TN** ✓ |
| 11      | No Cancer (0) | No Cancer (0)    | **TN** ✓ |
| 12      | No Cancer (0) | No Cancer (0)    | **TN** ✓ |

### 📊 Summary Counts

**Counting each outcome:**
- **TP (True Positives):** 6 (patients 3-8)
- **FN (False Negatives):** 2 (patients 1-2)
- **FP (False Positives):** 1 (patient 9)
- **TN (True Negatives):** 3 (patients 10-12)

**Total:** 12 patients

### 📈 Resulting Confusion Matrix

```
                  Predicted
               Cancer  No Cancer
Actual  Cancer    6        2       (8 total)
        No Cancer 1        3       (4 total)
```

Or with labels:

```
                  Predicted
               Positive  Negative  │ Total
Actual Positive    6         2     │  8
       Negative    1         3     │  4
       ─────────────────────────────────
       Total       7         5     │ 12
```

### 🔍 Interpretation

**What the classifier got right:**
- 6 cancer patients correctly identified
- 3 cancer-free patients correctly identified
- **Total correct:** 9 out of 12 (75% accuracy)

**What the classifier got wrong:**
- 2 cancer patients missed (dangerous!)
- 1 healthy patient incorrectly flagged (unnecessary worry)
- **Total wrong:** 3 out of 12

**Key insight:** Diagonal elements (6 + 3 = 9) show correct predictions.

---

## 4. Performance Metrics from Confusion Matrix

### 📊 Primary Metrics

All metrics derived from TP, TN, FP, FN:

#### 1. Accuracy
**Definition:** Overall proportion of correct predictions

**Formula:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Example:**
```
Accuracy = (6 + 3) / (6 + 3 + 1 + 2) = 9/12 = 0.75 = 75%
```

**Interpretation:** 75% of all predictions were correct

**⚠️ Limitation:** Misleading with imbalanced datasets

#### 2. Precision (Positive Predictive Value)
**Definition:** Of all predicted positives, how many were actually positive?

**Formula:**
```
Precision = TP / (TP + FP)
```

**Example:**
```
Precision = 6 / (6 + 1) = 6/7 ≈ 0.857 = 85.7%
```

**Interpretation:** When model predicts cancer, it's correct 85.7% of the time

**Use case:** Important when false positives are costly (e.g., spam detection)

#### 3. Recall (Sensitivity, True Positive Rate)
**Definition:** Of all actual positives, how many did we catch?

**Formula:**
```
Recall = TP / (TP + FN) = TP / P
```

Where P = actual positives

**Example:**
```
Recall = 6 / (6 + 2) = 6/8 = 0.75 = 75%
```

**Interpretation:** Model catches 75% of all cancer cases

**Use case:** Critical when false negatives are dangerous (e.g., cancer detection)

#### 4. Specificity (True Negative Rate)
**Definition:** Of all actual negatives, how many were correctly identified?

**Formula:**
```
Specificity = TN / (TN + FP) = TN / N
```

Where N = actual negatives

**Example:**
```
Specificity = 3 / (3 + 1) = 3/4 = 0.75 = 75%
```

**Interpretation:** Model correctly identifies 75% of healthy patients

#### 5. F1 Score
**Definition:** Harmonic mean of precision and recall

**Formula:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Example:**
```
F1 = 2 × (0.857 × 0.75) / (0.857 + 0.75)
   = 2 × 0.643 / 1.607
   = 0.800 = 80%
```

**Interpretation:** Balanced measure when you care about both precision and recall

**⚠️ Note:** Can be misleading with imbalanced data

#### 6. False Positive Rate (Fall-out)
**Definition:** Of all actual negatives, what proportion were incorrectly flagged?

**Formula:**
```
FPR = FP / (FP + TN) = FP / N = 1 - Specificity
```

**Example:**
```
FPR = 1 / (1 + 3) = 1/4 = 0.25 = 25%
```

#### 7. False Negative Rate (Miss Rate)
**Definition:** Of all actual positives, what proportion were missed?

**Formula:**
```
FNR = FN / (FN + TP) = FN / P = 1 - Recall
```

**Example:**
```
FNR = 2 / (2 + 6) = 2/8 = 0.25 = 25%
```

### 📋 Metrics Summary Table

| Metric | Formula | Value (Example) | What It Measures |
|--------|---------|-----------------|------------------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | 75% | Overall correctness |
| **Precision** | TP/(TP+FP) | 85.7% | Positive prediction accuracy |
| **Recall** | TP/(TP+FN) | 75% | Positive detection rate |
| **Specificity** | TN/(TN+FP) | 75% | Negative detection rate |
| **F1 Score** | 2×(P×R)/(P+R) | 80% | Balance of P and R |
| **FPR** | FP/(FP+TN) | 25% | False alarm rate |
| **FNR** | FN/(FN+TP) | 25% | Miss rate |

---

## 5. Advanced Metrics

### Matthews Correlation Coefficient (MCC)

**Definition:** Most informative single metric for confusion matrix evaluation (according to research)

**Formula:**
```
MCC = (TP×TN - FP×FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]
```

**Range:** -1 to +1
- +1: Perfect prediction
- 0: Random prediction
- -1: Total disagreement

**Example:**
```
MCC = (6×3 - 1×2) / √[(6+1)(6+2)(3+1)(3+2)]
    = (18 - 2) / √[7×8×4×5]
    = 16 / √1120
    = 16 / 33.47
    ≈ 0.478
```

**Advantages:**
- Accounts for class imbalance
- Single value summary
- More reliable than F1 for imbalanced data

### Informedness (Youden's J)

**Formula:**
```
Informedness = Sensitivity + Specificity - 1
             = Recall + TNR - 1
```

**Example:**
```
Informedness = 0.75 + 0.75 - 1 = 0.50
```

**Interpretation:** 
- 0 = no better than random guessing
- 1 = perfect prediction

### Diagnostic Odds Ratio (DOR)

**Formula:**
```
DOR = (TP/FN) / (FP/TN) = (TP×TN) / (FP×FN)
```

**Example:**
```
DOR = (6×3) / (1×2) = 18/2 = 9
```

**Interpretation:** Odds of positive prediction in diseased vs non-diseased

---

## 6. The Imbalanced Data Problem

### ⚠️ Why Accuracy Can Be Misleading

**Scenario:** Cancer screening with highly imbalanced data
- 95 cancer samples
- 5 non-cancer samples
- Total: 100 samples

**Naive classifier:** Predict "cancer" for everyone

**Results:**
```
                Predicted
              Cancer  No Cancer
Actual Cancer   95       0       (95)
       Healthy   5       0       (5)
```

**Metrics:**
- **Accuracy:** 95/100 = 95% (Looks great!)
- **Precision:** 95/100 = 95% (Looks great!)
- **Recall (Sensitivity):** 95/95 = 100% (Perfect!)
- **Specificity:** 0/5 = 0% (Terrible!)
- **F1 Score:** 2×(0.95×1)/(0.95+1) = 97.4% (Misleading!)

**The problem:** Classifier has **zero ability** to identify healthy patients, yet metrics look good!

### ✅ Better Metrics for Imbalanced Data

**Recommended:**
1. **Matthews Correlation Coefficient (MCC)** - accounts for all four values
2. **Informedness** - removes bias from guessing
3. **Balanced Accuracy** = (Sensitivity + Specificity) / 2
4. **ROC-AUC** - considers all possible thresholds

**For imbalanced case above:**
- MCC ≈ 0 (correctly indicates no better than random)
- Informedness = 0 (correctly identifies pure guessing)

---

## 7. Multi-Class Confusion Matrices

### 📊 Structure for N Classes

For N classes, confusion matrix is **N × N**:

```
                    Predicted Class
                  C1    C2    C3    ...  CN
Actual  C1      a11   a12   a13   ...  a1N
Class   C2      a21   a22   a23   ...  a2N
        C3      a31   a32   a33   ...  a3N
        ...     ...   ...   ...   ...  ...
        CN      aN1   aN2   aN3   ...  aNN
```

**Diagonal (aii):** Correct predictions for class i
**Off-diagonal (aij, i≠j):** Class i instances predicted as class j

### 💡 Example: Image Classification (3 Classes)

**Classes:** Cat, Dog, Bird

**Results:**

```
                  Predicted
               Cat  Dog  Bird  │ Total
Actual  Cat     80   15    5   │  100
        Dog     10   85    5   │  100
        Bird     5   10   85   │  100
        ──────────────────────────────
        Total   95  110   95   │  300
```

**Interpretation:**
- **Cat:** 80% correctly classified, 15% confused with Dog, 5% with Bird
- **Dog:** 85% correct, main confusion with Cat (10%)
- **Bird:** 85% correct, equal confusion with Cat and Dog

**Overall Accuracy:** (80 + 85 + 85) / 300 = 250/300 = 83.3%

### 📏 Per-Class Metrics

For each class i in multi-class:

**Precision for class i:**
```
Precision_i = aii / Σ(aji)  [sum of column i]
```

**Recall for class i:**
```
Recall_i = aii / Σ(aij)  [sum of row i]
```

**Example for Cat:**
```
Precision_Cat = 80 / 95 ≈ 84.2%
Recall_Cat = 80 / 100 = 80%
```

---

## 8. Limitations of Confusion Matrices

### ⚠️ What Confusion Matrices DON'T Show

From Wikipedia:
> "Some researchers have argued that the confusion matrix, and the metrics derived from it, do not truly reflect a model's knowledge."

#### 1. Epistemic Luck
**Problem:** Can't distinguish between:
- Correct prediction from sound reasoning
- Correct prediction by chance/luck

**Example:** Model predicts "cancer" randomly but happens to be right

#### 2. Defeasibility
**Problem:** Doesn't capture when:
- Facts used for prediction later change
- Initial information was wrong

**Example:** Patient diagnosed with cancer, but later revealed to be misdiagnosis

#### 3. Confidence/Probability Information Lost
**Problem:** Binary predictions lose probability information

**Example:**
- Prediction A: 51% confidence → Predicted Positive
- Prediction B: 99% confidence → Predicted Positive

Both treated identically in confusion matrix!

#### 4. Doesn't Show Class Difficulty
**Problem:** Can't see which examples are inherently hard to classify

#### 5. Threshold Dependency
**Problem:** Binary confusion matrix depends on classification threshold

**Solution:** Use ROC curve to see performance across all thresholds

### 🔍 Complementary Tools

To address limitations, also use:
- **ROC Curves** - threshold-independent performance
- **Precision-Recall Curves** - for imbalanced data
- **Calibration Plots** - check probability estimates
- **Error Analysis** - examine individual misclassifications
- **Feature Importance** - understand what model learned

---

## 9. Python Implementation

### 📝 Basic Implementation

```python
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Example predictions and true labels
y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])  # Actual
y_pred = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])  # Predicted

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
print()

# Output:
# [[3 1]
#  [2 6]]
# 
# TN=3, FP=1 (top row)
# FN=2, TP=6 (bottom row)
```

### 📊 Visualization

```python
# Create heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
```

### 📈 Calculate All Metrics

```python
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, 
                             matthews_corrcoef, roc_auc_score)

# Extract values from confusion matrix
TN, FP, FN, TP = cm.ravel()

print(f"True Positives (TP): {TP}")
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")
print()

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)

# Manual calculations
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
fnr = FN / (FN + TP) if (FN + TP) > 0 else 0

print(f"Accuracy:    {accuracy:.3f}")
print(f"Precision:   {precision:.3f}")
print(f"Recall:      {recall:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"F1 Score:    {f1:.3f}")
print(f"MCC:         {mcc:.3f}")
print(f"FPR:         {fpr:.3f}")
print(f"FNR:         {fnr:.3f}")
```

### 📋 Classification Report

```python
# Comprehensive report
from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred, 
                           target_names=['Negative', 'Positive']))

# Output:
#               precision    recall  f1-score   support
#
#     Negative       0.60      0.75      0.67         4
#     Positive       0.86      0.75      0.80         8
#
#     accuracy                           0.75        12
#    macro avg       0.73      0.75      0.73        12
# weighted avg       0.77      0.75      0.76        12
```

### 🎯 Multi-Class Example

```python
# Multi-class confusion matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# Train classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Multi-class confusion matrix
cm_multi = confusion_matrix(y_test, y_pred)
print("Multi-class Confusion Matrix:")
print(cm_multi)

# Visualize
plt.figure(figsize=(8, 6))
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='viridis',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.ylabel('Actual Species')
plt.xlabel('Predicted Species')
plt.title('Iris Classification Confusion Matrix')
plt.show()

# Per-class report
print("\nPer-Class Metrics:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

---

## 10. Best Practices

### ✅ When to Use Which Metric

| Scenario | Recommended Metrics | Why |
|----------|---------------------|-----|
| **Balanced classes** | Accuracy, F1 Score | Simple and interpretable |
| **Imbalanced classes** | MCC, ROC-AUC, PR-AUC | Account for class imbalance |
| **False positives costly** | Precision, FPR | Minimize false alarms |
| **False negatives costly** | Recall, FNR | Maximize detection |
| **Medical diagnosis** | Recall (Sensitivity), MCC | Can't miss positive cases |
| **Spam detection** | Precision, F1 | Don't want false positives |
| **Multi-class** | Per-class metrics, Macro/Micro avg | Understand each class |

### 🎯 Analysis Workflow

```
1. Create confusion matrix
2. Visualize with heatmap
3. Calculate appropriate metrics for your domain
4. Identify patterns of confusion
5. Analyze misclassified examples
6. Iterate on model or features
```

### ⚠️ Common Pitfalls

1. **Relying only on accuracy** with imbalanced data
2. **Ignoring class-specific performance** in multi-class
3. **Not considering cost of errors** (FN vs FP)
4. **Forgetting to examine actual misclassifications**
5. **Using F1 as only metric** for highly imbalanced data

---

## 11. Summary

### 🎯 Key Takeaways

1. **Confusion matrix** shows detailed breakdown of classifier predictions
2. **Four outcomes** in binary classification: TP, TN, FP, FN
3. **Many metrics** derived from confusion matrix, each with different purpose
4. **Accuracy misleading** with imbalanced data
5. **MCC and Informedness** better for imbalanced scenarios
6. **Multi-class** confusion matrices extend naturally to N classes
7. **Limitations exist** - don't capture confidence, reasoning, or defeasibility
8. **Visual analysis** crucial for understanding model behavior

### 📋 Quick Reference

**Binary Confusion Matrix:**
```
              Predicted
           Pos      Neg
Act  Pos   TP       FN
     Neg   FP       TN
```

**Essential Formulas:**
- Accuracy = (TP + TN) / Total
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 × (Precision × Recall) / (Precision + Recall)
- MCC = (TP×TN - FP×FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]

---

**End of Confusion Matrix Notes - Module 03**
