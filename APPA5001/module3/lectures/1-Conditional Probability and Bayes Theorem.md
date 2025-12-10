# Module 3: Conditional Probability and Bayes Theorem

## Introduction

In Module 3, we continue to extend and expand our foundations in probability. The module covers:
1. **Conditional Probability** - definition and concept
2. **Bayes Theorem** - using conditional probability to define Bayes theorem
3. **Independence and Mutually Exclusive Events** (covered in future videos)
4. **Relationship between conditional and independent events** in statistical experiments

---

## Conditional Probability

### Definition

Suppose we have two events, **A** and **B**, from the same sample space **S**.

We want to calculate the **probability of event A knowing that event B has occurred**.

**Notation:** $P(A \mid B)$ 

This is read as "the probability of A **given** B"

- **B** is called the **conditioning event**
- This represents the probability of A given knowledge that B has occurred

### Formal Definition

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

where $P(B) > 0$

> [!IMPORTANT]
> This definition only makes sense when $P(B) > 0$. If $P(B) = 0$, conditional probability is undefined.

---

### Example: Rolling Two Dice

**Experiment:** Roll a six-sided die twice.

**Sample Space:** $S = \{(i,j) \mid i,j \in \{1,2,3,4,5,6\}\}$ with $|S| = 36$ equally likely elements.

**Event A:** At least one of the two dice shows a 3
- A = {(3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (1,3), (2,3), (4,3), (5,3), (6,3)}
- $|A| = 11$ events
- $P(A) = \frac{11}{36}$

**Event B:** The sum of the two dice is 9
- B = {(3,6), (4,5), (5,4), (6,3)}
- $|B| = 4$ events
- $P(B) = \frac{4}{36}$

**Question:** Suppose you know that B has occurred (you got a sum of 9). What is the chance that event A happened (at least one die shows a 3)?

#### Intuitive Understanding with Venn Diagram

Without additional information:
- $P(A) = \frac{11}{36}$

But now we're given additional information: **we know B occurred** (the sum is 9).

How does this change the probability of event A?

Looking at the events:
- **Events in both A and B:** {(3,6), (6,3)} - these satisfy both conditions
- **Events in B but not A:** {(4,5), (5,4)} - sum is 9, but no 3's
- **Events in A but not B:** {(3,1), (1,3), (3,2), (2,3), (3,4), (4,3), (3,5), (5,3), (3,3)} - have a 3, but sum isn't 9

#### Calculation

Since we know **B has occurred**, B becomes the new relevant sample space.

For A to occur, we need $A \cap B$ to occur.

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

Where:
- $P(A \cap B) = \frac{2}{36}$ (the events {(3,6), (6,3)})
- $P(B) = \frac{4}{36}$

$$P(A \mid B) = \frac{\frac{2}{36}}{\frac{4}{36}} = \frac{2}{36} \times \frac{36}{4} = \frac{2}{4} = \frac{1}{2}$$

**Interpretation:** We know B has occurred. There are 4 equally likely events in B, and 2 of those events are in set A. So the probability is $\frac{2}{4} = \frac{1}{2}$.

This makes sense - given the new knowledge of event B occurring, our probability for event A has changed from $\frac{11}{36} \approx 0.306$ to $\frac{1}{2} = 0.5$.

---

## The Multiplication Rule

From the definition of conditional probability:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

We can multiply both sides by $P(B)$:

$$P(B) \times P(A \mid B) = P(A \cap B)$$

This is completely symmetric. We could also write:

$$P(B \mid A) = \frac{P(A \cap B)}{P(A)}$$

Which gives us:

$$P(A) \times P(B \mid A) = P(A \cap B)$$

---

## Bayes Theorem

Since both expressions equal $P(A \cap B)$, we can combine them:

$$P(B) \times P(A \mid B) = P(A) \times P(B \mid A)$$

Dividing both sides by $P(B)$:

$$\boxed{P(A \mid B) = \frac{P(A) \times P(B \mid A)}{P(B)}}$$

**This is Bayes Theorem.**

> [!NOTE]
> Sometimes it's easier to calculate $P(B \mid A)$, and we can use this information to calculate $P(A \mid B)$ using Bayes Theorem.

---

## Law of Total Probability

### Two Events Case

Suppose we have two events **A** and **B** from the same sample space **S**.

Event B can be partitioned into two disjoint parts:
1. The part of B that intersects with A: $B \cap A$
2. The part of B that intersects with A complement: $B \cap A^c$

We can write:
$$B = (B \cap A) \cup (B \cap A^c)$$

Since these two regions are disjoint (mutually exclusive), we can write:

$$P(B) = P(B \cap A) + P(B \cap A^c)$$

Using the definition of conditional probability (multiplication rule):
- $P(B \cap A) = P(B \mid A) \times P(A)$
- $P(B \cap A^c) = P(B \mid A^c) \times P(A^c)$

Therefore:

$$\boxed{P(B) = P(B \mid A) \times P(A) + P(B \mid A^c) \times P(A^c)}$$

**This is the Law of Total Probability.**

> [!NOTE]
> This looks like a confusing way to calculate $P(B)$. However, there are times when it's easier to calculate the conditional probabilities $P(B \mid A)$ or $P(B \mid A^c)$, and then use this formula to get $P(B)$.

### Extended Version with n Events

We can extend the law of total probability to **n** sets.

Let $A_1, A_2, \ldots, A_n$ be events that satisfy two conditions:

1. **Mutually Exclusive:** $A_i \cap A_j = \emptyset$ for all $i \neq j$
   - All the A's are mutually exclusive (no overlaps)

2. **Exhaustive:** $A_1 \cup A_2 \cup \cdots \cup A_n = S$
   - The union of all A's equals the entire sample space

Then we can write:

$$\boxed{P(B) = \sum_{i=1}^{n} P(B \mid A_i) \times P(A_i)}$$

Or written out:
$$P(B) = P(B \mid A_1) \times P(A_1) + P(B \mid A_2) \times P(A_2) + \cdots + P(B \mid A_n) \times P(A_n)$$

**Example with 4 sets:**

Suppose we partition S into 4 mutually exclusive events: $A_1, A_2, A_3, A_4$

Then:
$$P(B) = P(B \cap A_1) + P(B \cap A_2) + P(B \cap A_3) + P(B \cap A_4)$$

Using the multiplication rule:
$$P(B) = P(B \mid A_1)P(A_1) + P(B \mid A_2)P(A_2) + P(B \mid A_3)P(A_3) + P(B \mid A_4)P(A_4)$$

> [!NOTE]
> If $B \cap A_4 = \emptyset$ (B doesn't intersect with $A_4$), then $P(B \cap A_4) = 0$ and that term drops out.

---

## Comprehensive Example: Disease Testing

### Problem Setup

Your company has developed a new test for a disease.

**Given Information:**
- **Event A:** A randomly selected individual has the disease
  - $P(A) = 0.001$ (1 in 1000 people has the disease)
  
- **Event B:** A positive test result is received for that randomly selected individual

**Test Performance Data** (collected by your company):
1. $P(B \mid A) = 0.99$ 
   - Probability of a positive test result **given** the person has the disease
   - This is the **sensitivity** or **true positive rate**

2. $P(B^c \mid A) = 0.01$
   - Probability of a negative test result **given** the person has the disease
   - This is the **false negative rate**

3. $P(B \mid A^c) = 0.02$
   - Probability of a positive test result **given** the person doesn't have the disease
   - This is the **false positive rate**

4. $P(B^c \mid A^c) = 0.98$
   - Probability of a negative test result **given** the person doesn't have the disease
   - This is the **specificity** or **true negative rate**

### Question

**Calculate:** $P(A \mid B)$

That is, the probability that a person has the disease **given** a positive test result.

> [!IMPORTANT]
> **Critical Distinction:**
> - $P(B \mid A) = 0.99$: "If the person has the disease, what's the probability of a positive test?"
> - $P(A \mid B) = ?$: "If the test is positive, what's the probability the person has the disease?"
> 
> These are **different** questions! Knowing one thing or another changes the probabilities.

### Solution

**Step 1:** Start with Bayes Theorem

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

**Step 2:** Apply Bayes Theorem using the multiplication rule

$$P(A \mid B) = \frac{P(B \mid A) \times P(A)}{P(B)}$$

**Step 3:** Calculate $P(B)$ using the Law of Total Probability

Since we don't know $P(B)$ directly, we use the law of total probability:

$$P(B) = P(B \mid A) \times P(A) + P(B \mid A^c) \times P(A^c)$$

Where:
- $P(B \mid A) = 0.99$
- $P(A) = 0.001$
- $P(B \mid A^c) = 0.02$
- $P(A^c) = 1 - 0.001 = 0.999$

$$P(B) = (0.99)(0.001) + (0.02)(0.999)$$
$$P(B) = 0.00099 + 0.01998$$
$$P(B) = 0.02097$$

**Step 4:** Calculate $P(A \mid B)$

$$P(A \mid B) = \frac{P(B \mid A) \times P(A)}{P(B)}$$

$$P(A \mid B) = \frac{(0.99)(0.001)}{0.02097}$$

$$P(A \mid B) = \frac{0.00099}{0.02097}$$

$$\boxed{P(A \mid B) = 0.0472}$$

### Interpretation: Prior vs Posterior Probability

**Prior Probability:** $P(A) = 0.001$
- This is our probability **without any information**
- The probability that a randomly selected person (pulled off the street) has the disease is 0.001 or 0.1%

**Posterior Probability:** $P(A \mid B) = 0.0472$
- This is the probability of A **after** we have learned that event B has occurred
- After administering the test and getting a positive result, the probability of having the disease is 0.0472 or 4.72%

**Key Insight:**
- The probability has increased tremendously from 0.1% to 4.72% (a 47.2x increase!)
- However, it's still less than 5%
- In general, if someone goes to the doctor, they'll have **other symptoms** which would raise this probability even higher
- If we had additional symptoms as part of the conditioning, the probability of having the disease would go up further

---

## Tree Diagram Representation

A tree diagram provides another visual way to understand conditional probability and the calculations.

### Structure of the Tree

**First Branch (Prior):** Does the person have the disease?
- $P(A) = 0.001$ (has disease)
- $P(A^c) = 0.999$ (doesn't have disease)

**Second Branch (Conditional on First):**

If person **has** disease (on the A branch):
- $P(B \mid A) = 0.99$ (positive test | has disease)
- $P(B^c \mid A) = 0.01$ (negative test | has disease)

If person **doesn't have** disease (on the $A^c$ branch):
- $P(B \mid A^c) = 0.02$ (positive test | no disease)
- $P(B^c \mid A^c) = 0.98$ (negative test | no disease)

### Tree Diagram Visualization

```
Random Person
    |
    ├─── A (0.001) ──┬─── B|A (0.99) → P(A ∩ B) = 0.001 × 0.99 = 0.00099
    |                 └─── B^c|A (0.01) → P(A ∩ B^c) = 0.001 × 0.01 = 0.00001
    |
    └─── A^c (0.999) ─┬─── B|A^c (0.02) → P(A^c ∩ B) = 0.999 × 0.02 = 0.01998
                       └─── B^c|A^c (0.98) → P(A^c ∩ B^c) = 0.999 × 0.98 = 0.97902
```

### Key Properties of Tree Diagrams

1. **Branches sum to 1:** At any branching point, the probabilities sum to 1
   - $P(A) + P(A^c) = 0.001 + 0.999 = 1$ ✓
   - $P(B \mid A) + P(B^c \mid A) = 0.99 + 0.01 = 1$ ✓
   - $P(B \mid A^c) + P(B^c \mid A^c) = 0.02 + 0.98 = 1$ ✓

2. **Multiply along branches:** To get intersection probabilities
   - Top branch: $P(A) \times P(B \mid A) = P(A \cap B) = 0.00099$
   - Second branch: $P(A) \times P(B^c \mid A) = P(A \cap B^c) = 0.00001$
   - Third branch: $P(A^c) \times P(B \mid A^c) = P(A^c \cap B) = 0.01998$
   - Bottom branch: $P(A^c) \times P(B^c \mid A^c) = P(A^c \cap B^c) = 0.97902$

3. **All outcomes sum to 1:**
   - $0.00099 + 0.00001 + 0.01998 + 0.97902 = 1.00000$ ✓

4. **More complex experiments:** In more complicated experiments, you could have 2, 3, 4, or more branches at each node

---

## Sample Space Interpretation

Another way to think about this problem is to explicitly write out the sample space.

**Sample Space S:**

Let:
- **D** = has disease
- **N** = no disease  
- **+** = positive test result
- **−** = negative test result

There are only **4 possible events** in this experiment:

$$S = \{(D, +), (D, -), (N, +), (N, -)\}$$

**Matching to our calculations:**

| Event | Notation | Probability | Calculation |
|-------|----------|-------------|-------------|
| $(D, +)$ | $A \cap B$ | 0.00099 | Person has disease AND positive test |
| $(D, -)$ | $A \cap B^c$ | 0.00001 | Person has disease AND negative test |
| $(N, +)$ | $A^c \cap B$ | 0.01998 | No disease AND positive test |
| $(N, -)$ | $A^c \cap B^c$ | 0.97902 | No disease AND negative test |

> [!IMPORTANT]
> These probabilities are for a **randomly selected person** just pulled off the street.
> 
> When you add in other knowledge or additional symptoms, these probabilities all change!

---

## Summary

### Key Concepts

1. **Conditional Probability:**
   $$P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

2. **Multiplication Rule:**
   $$P(A \cap B) = P(B) \times P(A \mid B) = P(A) \times P(B \mid A)$$

3. **Bayes Theorem:**
   $$P(A \mid B) = \frac{P(A) \times P(B \mid A)}{P(B)}$$

4. **Law of Total Probability (2 events):**
   $$P(B) = P(B \mid A) \times P(A) + P(B \mid A^c) \times P(A^c)$$

5. **Law of Total Probability (n events):**
   $$P(B) = \sum_{i=1}^{n} P(B \mid A_i) \times P(A_i)$$
   where $A_1, \ldots, A_n$ are mutually exclusive and exhaustive

### Important Terms

- **Conditioning Event:** The event we are given (e.g., B in $P(A \mid B)$)
- **Prior Probability:** The probability without additional information (e.g., $P(A)$)
- **Posterior Probability:** The probability after learning new information (e.g., $P(A \mid B)$)
- **Mutually Exclusive:** Events $A_i$ and $A_j$ where $A_i \cap A_j = \emptyset$ for $i \neq j$
- **Exhaustive:** Events whose union equals the entire sample space: $\bigcup_{i=1}^{n} A_i = S$

### When to Use These Tools

- Use **conditional probability** when you have knowledge that one event has occurred and want to find the probability of another
- Use **Bayes Theorem** when it's easier to calculate $P(B \mid A)$ but you need $P(A \mid B)$
- Use **Law of Total Probability** when you can partition the sample space and know conditional probabilities
- Use **tree diagrams** to visualize complex probability problems with sequential events

---

## Next Topics

In the next video, we'll work on **independent events** and their relationship to conditional probability.
