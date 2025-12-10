# Performance Metrics in Classification - Detailed Lecture Notes
**CSCA5622 - Module 03**

---

## 📚 Overview

This document covers **performance metrics for classification models** - how to properly evaluate classification performance beyond simple accuracy. Topics include confusion matrix, various metrics (accuracy, precision, recall, F1, ROC-AUC), when to use each metric, and why cross-entropy is important.

All concepts explained from the lecture transcript.

---

## 1. Basic Terminology with Example

### 🏥 Binary Classification Example

From lecture:
> "Here is the example for **binary class classification**, where the **label is one** when the tumor is **malignant** and the **label is zero** when the tumor is **not malignant**. We created a **lossy regression model** based on this feature."

**Setup:**
- **Positive class (1):** Malignant tumor
- **Negative class (0):** Not malignant (benign)
- **Model:** Logistic regression classifier

### 📊 Four Possible Outcomes

### ✅ True Positive (TP)

From lecture:
> "This region where both the **labels and the predictions are positive** is called a **true positive**."

**Definition:** Model predicts positive AND actual label is positive

**Example:** Model predicts malignant, tumor is actually malignant ✓

### ✅ True Negative (TN)

From lecture:
> "This region where the **labels and the predictions are both negative**, we call **true negative**."

**Definition:** Model predicts negative AND actual label is negative

**Example:** Model predicts benign, tumor is actually benign ✓

### ❌ False Positive (FP)

From lecture:
> "This region is called the **false positive** because the **prediction says it's positive**, but actually the **labels as it's negative**."

**Definition:** Model predicts positive BUT actual label is negative

**Example:** Model predicts malignant, but tumor is actually benign ✗

### ❌ False Negative (FN)

From lecture:
> "On the other hand, this region is called the **false negative** because the **prediction says negative** whereas the **labels says it's positive**."

**Definition:** Model predicts negative BUT actual label is positive

**Example:** Model predicts benign, but tumor is actually malignant ✗

### 🎯 Optimization Goal

From lecture:
> "We would like to build a model that **maximize** the number of **true positives and true negatives** whereas we want to **minimize** the **false positive and false negatives**."

---

## 2. Type I and Type II Errors

### 📝 Alternative Names

From lecture:
> "This false negative and false positive have **different names** as well. **False positive** is called **Type I error** whereas the **false negative** is called the **Type II error**."

**Summary:**
- **Type I Error** = False Positive (FP)
- **Type II Error** = False Negative (FN)

### 😄 Memorable Analogy

From lecture:
> "If those terminology confuses you, then you can remember this **funny picture**. **Type I error** is like **telling a man that he's pregnant** whereas **Type II error** is like **telling a pregnant woman that she is not pregnant**."

**Type I (False Positive):** Saying yes when answer is no
- Man is NOT pregnant (truth = negative)
- But told he IS pregnant (prediction = positive)

**Type II (False Negative):** Saying no when answer is yes
- Woman IS pregnant (truth = positive)
- But told she is NOT pregnant (prediction = negative)

### ⚖️ Trade-offs

From lecture:
> "Both are **bad** and sometimes they are in **trade-off**. Depending on the **situation** and what are **important to us** in the problem, we will have to consider **one more seriously than the other**."

**Key insight:** Often can't minimize both simultaneously - must prioritize based on problem context

---

## 3. Confusion Matrix

### 🔍 Definition

From lecture:
> "We talked about true positive, true negative, and then false positive and false negative cases. A **former way to express that in a table** is called **confusion matrix**."

### 📊 Structure

From lecture:
> "The confusion matrix is like this. There is a **prediction label**, and there is a **target label**. They are in the table, this is **true positive** and this is **true negative**, this is **false positive**, this is **false negative**."

**Standard layout:**

```
                    Predicted
                Positive    Negative
Actual  Positive    TP          FN
        Negative    FP          TN
```

**Rows:** Actual labels (ground truth)
**Columns:** Predicted labels (model output)

### ⚠️ Notation Variation

From lecture:
> "Sometimes depending on the notation, the **row and the column may be exchanged** and this can be calculated by the scalar matrix confusion matrix module."

### 🔧 In scikit-learn

From lecture:
> "When you use a **computer matrix module**, it will give the **labels as row** and then **prediction is column**. But they **don't display this**, so sometimes it's confusing, but you can **figure out by looking at the data**."

**sklearn convention:**
```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
# Returns:
# [[TN  FP]
#  [FN  TP]]
```

---

## 4. Performance Metrics Formulas

From lecture:
> "Let's say we have all the **numbers that we collected** from this confusion matrix and now let's **calculate some performance metrics**."

### 📊 Accuracy

From lecture:
> "The **most popular one** in classification is **accuracy**. Accuracy is a **number of correct answers** divided by **all the data points**. It's a measure of **how many are accurate** out of all the data points."

**Formula:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Interpretation:** Proportion of all predictions that were correct

### 📊 True Positive Rate (TPR) / Recall / Sensitivity

From lecture:
> "**True positive rate**, in other words, **recall** or **sensitivity**, is a measure of **how many are truly positive** out over **all the positive cases in the data**."

**Formula:**
```
TPR = Recall = Sensitivity = TP / (TP + FN)
```

From lecture:
> "This is **all the positives**, and this is **true positive** in the data."

**Interpretation:** Of all actual positives, what fraction did we correctly identify?

### 📊 True Negative Rate (TNR) / Specificity

From lecture:
> "Another metric, **true negative rate**, which is **similar to true positive rate** except that they are **flipped**. Another name for it is **specificity** or cell activity. Measures **how many are true negative** out of **all the negative cases in the data**."

**Formula:**
```
TNR = Specificity = TN / (TN + FP)
```

From lecture:
> "By data I mean the **labels**."

**Interpretation:** Of all actual negatives, what fraction did we correctly identify?

### 📊 Positive Predictive Value (PPV) / Precision

From lecture:
> "Another good measure that we often use is a **positive predictive value** or in another words, a **precision** measures. **How many are correctly classified** this true positive over **prediction from the prediction**?"

**Formula:**
```
PPV = Precision = TP / (TP + FP)
```

**Interpretation:** Of all positive predictions, what fraction were actually correct?

### 📊 False Positive Rate (FPR) / Fall-out

From lecture:
> "**False-positive rate**, in other words, **fall-out rate** tells us **how many are false-positive** out of **all the negative cases**. How many were **falsely classified as positive** when it was **actually negative**? It's **all the negatives from the data**. How many of them are **falsely classified as positive**?"

**Formula:**
```
FPR = Fall-out = FP / (FP + TN)
```

From lecture:
> "Actually, as you can see, it is **1-TNR**."

**Relationship:**
```
FPR = 1 - Specificity = 1 - TNR
```

**Interpretation:** Of all actual negatives, what fraction were incorrectly flagged as positive?

### 📊 False Negative Rate (FNR) / Miss Rate

From lecture:
> "Similarly, **false-negative rate**, on other words, **miss rate** is also. How many of **positive in the data** are **falsely classified as negative**?"

**Formula:**
```
FNR = Miss Rate = FN / (FN + TP)
```

From lecture:
> "This is actually **1-TPR**, so they are **related to each other**."

**Relationship:**
```
FNR = 1 - Recall = 1 - TPR
```

**Interpretation:** Of all actual positives, what fraction did we miss?

### 📊 F1 Score

From lecture:
> "**F1 score** is a good metric because oftentimes, there is a **trade-off between recall and precision**. In some cases, recall is a good metric, but we want to see **both of them together**. In the case we want to use **F1 score** because it has **both of the precision and recall inside**."

**Formula:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Or equivalently:
```
F1 = 2TP / (2TP + FP + FN)
```

From lecture:
> "F1 score is usually **robust metric**, so it's **good to use**."

**Interpretation:** Harmonic mean of precision and recall - balances both

---

## 5. Summary Table of Metrics

| Metric | Formula | Interpretation | Alternative Names |
|--------|---------|----------------|-------------------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Overall correctness | - |
| **Recall** | TP/(TP+FN) | Positive detection rate | TPR, Sensitivity |
| **Specificity** | TN/(TN+FP) | Negative detection rate | TNR |
| **Precision** | TP/(TP+FP) | Positive prediction accuracy | PPV |
| **FPR** | FP/(FP+TN) | False alarm rate | Fall-out, 1-TNR |
| **FNR** | FN/(FN+TP) | Miss rate | 1-TPR, 1-Recall |
| **F1** | 2×(P×R)/(P+R) | Balance of P and R | - |

---

## 6. ROC Curve

### 🔍 What Is ROC?

From lecture:
> "Not only the F1 square, here are also **good metrics that are robust** in different kinds of situations. One of them is called the **ROC curve**, which is a **Receiver-Operating Characteristic curve**."

### 📊 Structure

From lecture:
> "It shows like this. In the **x-axis**, it has a **false-positive rate** and its **y-axis**, it has a **true positive rate**."

**Axes:**
- **X-axis:** False Positive Rate (FPR)
- **Y-axis:** True Positive Rate (TPR / Recall)

### 📈 Interpretation

From lecture:
> "This **red dotted line** represent the **random guess**. If the curve goes this way closer to this **left top corner**, this means **it's good**. We have **small false-positive rate** and **large true-positive rate**, so that's good."

**Good model:** Curve bends toward upper-left corner
```
TPR
 1 |    /--- ← Good (high TPR, low FPR)
   |   /
   |  /
0.5|_/_____ ← Random guess
   |/
 0 |_________  FPR
   0        1
```

From lecture:
> "However, if the curve is **below this random guess**, then it's **bad**. That means we have a **high false-positive rate** and **small true-positive rate**. This side is **good** and this side is **bad**."

**Bad model:** Curve below diagonal (worse than random)

### 🎯 Visual Summary

From lecture:
> "That's **ROC curve**. This is a **graphic way** to tell."

**Key zones:**
- **Above diagonal:** Better than random
- **On diagonal:** Random guessing
- **Below diagonal:** Worse than random (inverted predictions)
- **Top-left corner:** Perfect classifier

---

## 7. Area Under the Curve (AUC)

### 🔍 Numerical Summary of ROC

From lecture:
> "However, if you want to see a **number**, then you can use **area under the curve, AUC**. We measure the **area under the curve**, for example, this curve, the example, the **area under the curve** will be this value, usually, **between zero and one**."

**Range:** [0, 1]

**Interpretation:**
- **AUC = 1.0:** Perfect classifier
- **AUC = 0.5:** Random guessing
- **AUC < 0.5:** Worse than random (predictions inverted)
- **AUC > 0.5:** Better than random

From lecture:
> "The **bigger the value**, it's **better**."

### 📝 Names

From lecture:
> "That was **ROC and AUC**."

**Full name:** ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

---

## 8. When to Use Which Metric

### 🎯 General Recommendations

From lecture:
> "When to use these **different kinds of metrics**, we have a number of choices, accuracy, sensitivity, specificity, precision, fall-out rate, miss rate, F1 score, AUC, and confusion matrix."

From lecture:
> "In **general rule of thumb**, you should **always consider F1 score, AUC** because they are **robust** and **correct most of time**. **Confusion matrix** is also good way to **investigate** how many are wrong or correct, like in roll numbers."

**Always consider:**
1. **F1 Score** - Robust, balances precision and recall
2. **AUC** - Robust, threshold-independent
3. **Confusion Matrix** - Shows detailed breakdown

From lecture:
> "Depending on the **situations**, we might want **one versus the other** in these metrics."

---

## 9. Accuracy: When to Use and Avoid

### ✅ Generally Good

From lecture:
> "**Accuracy** is mostly **good and intuitive**."

**Advantages:**
- Easy to understand
- Single number summary
- Intuitive interpretation

### ❌ The Imbalanced Data Problem

From lecture:
> "However, when the accuracy **miserably fails** is that when the **label is very imbalanced**, accuracy might be **really bad**."

**Example from lecture:**

From lecture:
> "For example, if your data is **99.9% negative**, maybe **0.1% positive**, and maybe your model says **100% negative**, then it's going to give a **fantastic accuracy, 99.9% correct**. If it says **everything is negative**, so that's **not good**."

**The problem:**
```
Data: 999 negative, 1 positive
Naive model: Predict everything as negative

Accuracy = 999/1000 = 99.9% (looks great!)
But: Completely failed to detect the positive class!
```

From lecture:
> "Accuracy may have some **pitfall** there."

### ✅ When to Use Accuracy

From lecture:
> "Usually it's a **good idea to use accuracy** when you have a **balanced data**."

**Use accuracy when:**
- Classes are roughly balanced (e.g., 40-60% split)
- All errors have equal cost
- Simple, interpretable metric needed

---

## 10. Recall: When to Use

### 🔍 Definition Recap

From lecture:
> "**Recall**, which is **two positive divide by all the positive cases in the data**."

**Formula:** Recall = TP / (TP + FN)

### 🎯 Use Case: High Cost of Missing Positives

From lecture:
> "They are used mostly when we want to **capture as many positive cases as possible**, even though we **sacrifice false positives**."

### 🏥 Example: Cancer Detection

From lecture:
> "When it's good for. For example, **cancer detection** by **missing someone having cancer**. If you **miss the cancer** of a patient and the patient is **at risk**. There is a **high cost** associated with the **missing**."

**Scenario:**
- Missing a cancer diagnosis (FN) = potentially fatal
- False alarm (FP) = unnecessary worry, further testing (less severe)

**Priority:** Catch all cancer cases, even if means more false alarms

From lecture:
> "In that case, we want to use **recall** because we want to **capture as much positive cases as possible**. In that case also we want to look at **false negative rate**. If you have **too much false negatives** in the data, then we're in **trouble**. We want to look at **both of them the same time**."

### 📋 When to Prioritize Recall

From lecture:
> "If you have a **high cost of missing something**."

**Use recall when:**
- False negatives are very costly
- Must catch all (or most) positive cases
- False positives are acceptable
- Examples: disease detection, fraud detection, security screening

**Watch:** False Negative Rate (FNR) - minimize this!

---

## 11. False Positive Rate & Specificity: When to Use

### 🔍 False Positive Rate

From lecture:
> "On the other hand, the **false positive rate** or **false alarm rate** can be used when the **cost of false alarm is high**."

### 📧 Example: Spam Filtering

From lecture:
> "For example, **spam mail**. Having **spam mail is fine**, it's just **annoying**. However, if you have a **false alarms**, that means it's going to **erase the important mail**, then it's **problematic**. We want to **avoid these false alarms** than we want to look at the **false positive rate**."

**Scenario:**
- Spam gets through (FN) = minor annoyance
- Important email marked as spam (FP) = miss critical information!

**Priority:** Minimize false positives (don't block important emails)

From lecture:
> "Which is also similar to **specificity or sensitivity**. We want to look at it as well."

**Relationship:**
- Low FPR = High Specificity
- Want to maximize Specificity (correctly identify negatives)

### 📋 When to Prioritize Low FPR / High Specificity

**Use FPR/Specificity when:**
- False positives are very costly
- Must avoid false alarms
- Missing some positives is acceptable
- Examples: spam filtering, recommending surgery, criminal conviction

---

## 12. Precision: When to Use

### 🔍 Definition Recap

**Formula:** Precision = TP / (TP + FP)

**Meaning:** Of positive predictions, what fraction are correct?

### 🎯 Use Case: High Confidence in Action

From lecture:
> "**Precision** is used when we want to be **very sure about the action**."

### 💰 Example: Account Suspension

From lecture:
> "For example, when we identify **scammers in PayPal or Venmo** and we do like to **inactivate their account**, then we want to be **very sure** because otherwise we can just **delete innocent user's account** and it'll make the **customer unhappy**."

**Scenario:**
- Suspend real scammer (TP) = good outcome
- Suspend innocent user (FP) = customer loss, reputation damage

**Priority:** Only act when very confident (high precision)

### 📋 When to Prioritize Precision

**Use precision when:**
- False positives cause significant harm
- Actions based on predictions are irreversible
- Need high confidence before acting
- Examples: account suspension, medical interventions, product recalls

---

## 13. Summary of Metric Usage

### 📊 Decision Guide

| Metric | Use When | Example Scenarios |
|--------|----------|-------------------|
| **Accuracy** | Balanced classes, equal error costs | General classification with balanced data |
| **Recall** | Missing positives is costly | Cancer detection, fraud detection |
| **Specificity/FPR** | False alarms are costly | Spam filtering, criminal justice |
| **Precision** | Need confidence in positive predictions | Account suspension, recommendations |
| **F1 Score** | Want balance of precision and recall | Most classification tasks |
| **AUC** | Threshold-independent evaluation | Model comparison, imbalanced data |

### 🎯 Key Takeaway

From lecture:
> "In summary, some **performance metrics can be considered more important than other** depending on the **situation** and **what's important in your problem**. However, these **performance metrics are robust** and can be used in **almost any cases**."

**Robust metrics (use by default):**
- F1 Score
- AUC
- Confusion Matrix

**Problem-specific metrics:** Choose based on cost of different error types

---

## 14. Cross-Entropy as Performance Metric

### 🤔 Why Not Just Use Accuracy?

From lecture:
> "Let's talk about **cross-entropy as a performance metric**. Why do we want to use **cross-entropy and not accuracy**?"

From lecture:
> "Because **accuracy**, although it doesn't work very well in imbalanced data, if the data is balanced, it's a **pretty good** and it's **intuitive and interpretable**. But **why do we want to use cross-entropy**?"

### 💡 The Key Advantage

From lecture:
> "In a nutshell, **cross-entropy can use more granular information** about **how the prediction is more confident** whereas **accuracy only says that whether it's correct or not**."

**Accuracy:** Binary (correct/incorrect)
**Cross-entropy:** Considers confidence (probability values)

### 📊 Comparative Example

From lecture:
> "Let's have a look. For example, this is **Model A**."

**Model A:**

From lecture:
> "Model A's have **accuracy of two-third** because it's **correct two times and incorrect onetime** for these three samples. Although it's **correct here**, the **confidence is not that great**, whereas the **incorrect ones**, the **confidence is too confident** for the incorrect answer. We can say this **Model A is not very good model**."

```
Sample 1: True=1, Predicted=1, Confidence=0.55  ✓ (barely)
Sample 2: True=0, Predicted=1, Confidence=0.95  ✗ (very wrong!)
Sample 3: True=1, Predicted=1, Confidence=0.60  ✓ (barely)

Accuracy = 2/3 = 66.7%
```

**Model B:**

From lecture:
> "Maybe we can compare to **Model B**, which has the **same accuracy**. The third, however, when it's **correct**, it's **pretty confident** for the correct answers and when it's **not correct**, maybe it's **not sure**. Maybe it makes sense."

```
Sample 1: True=1, Predicted=1, Confidence=0.95  ✓ (very confident)
Sample 2: True=0, Predicted=1, Confidence=0.52  ✗ (barely wrong)
Sample 3: True=1, Predicted=1, Confidence=0.90  ✓ (very confident)

Accuracy = 2/3 = 66.7%
```

### ✅ Cross-Entropy Distinguishes Them

From lecture:
> "In this case, if we use the **cross-entropy**, you can **discern these two different cases**. It will give a **better score**, which is a **lower cross entropy value** for the **better model** and **higher cross entropy value** with the **less working model**."

**Cross-entropy calculation:**

Model A:
```
CE_A = -[log(0.55) + log(1-0.95) + log(0.60)]
     = -[log(0.55) + log(0.05) + log(0.60)]
     = High value (worse)
```

Model B:
```
CE_B = -[log(0.95) + log(1-0.52) + log(0.90)]
     = -[log(0.95) + log(0.48) + log(0.90)]
     = Lower value (better)
```

### 🎯 The Intuition

From lecture:
> "That's some **intuition behind** why you might want **cross-entropy and not accuracy** although **accuracy might be more intuitive**."

**Advantages of cross-entropy:**
1. ✓ Penalizes confident wrong predictions heavily
2. ✓ Rewards confident correct predictions
3. ✓ Uses full probability information
4. ✓ Better for model training (smooth gradient)

**When to use:**
- Training neural networks (standard loss)
- Evaluating probabilistic predictions
- Want to assess model confidence
- Imbalanced data

---

## 15. Complete Summary

### 🎯 Key Concepts

**1. Four Outcomes:**
- True Positive (TP) - Correct positive prediction
- True Negative (TN) - Correct negative prediction
- False Positive (FP) - Type I Error
- False Negative (FN) - Type II Error

**2. Confusion Matrix:**
```
                Predicted
            Positive  Negative
Actual Pos     TP        FN
       Neg     FP        TN
```

**3. Essential Metrics:**
- **Accuracy:** Overall correctness = (TP+TN)/Total
- **Recall:** Positive detection = TP/(TP+FN)
- **Precision:** Positive accuracy = TP/(TP+FP)
- **F1 Score:** Harmonic mean of precision and recall

**4. ROC-AUC:**
- ROC curve: TPR vs FPR at different thresholds
- AUC: Single number summary [0, 1]
- Higher is better

**5. Metric Selection:**
- **Balanced data, equal costs:** Accuracy
- **Missing positives costly:** Recall
- **False alarms costly:** Precision, Specificity
- **Need balance:** F1 Score
- **Model comparison:** AUC

**6. Cross-Entropy:**
- Considers prediction confidence
- Better than accuracy for training
- Penalizes confident mistakes

### 📋 Best Practices

**Always report:**
1. Confusion matrix (see all error types)
2. F1 score (balanced metric)
3. AUC (threshold-independent)

**Context-specific:**
- Add recall if missing positives is costly
- Add precision if false positives are costly
- Add accuracy only if data is balanced

---

**End of Lecture Notes - Module 03, Document 4**
