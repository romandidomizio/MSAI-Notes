# Module 3: Independent Events

## Introduction

In this lecture, we continue learning about probability by exploring the concept of **independence**. Independence is a key concept in statistics and data science that will be used throughout future modules.

---

## What is Independence?

### Intuitive Definition

**Two events are independent if knowing the outcome of one event does not change the probability of the other.**

### Everyday Examples

**Example 1: Flipping a Coin Twice**
- Suppose you flip a coin twice
- If you know you got a head on the first flip, does that change the probability of getting a head or tail on the second flip?
- **Answer:** No. If it's a fair coin, the probability is still $\frac{1}{2}$ for a head and $\frac{1}{2}$ for a tail.

**Example 2: Rolling Dice**
- If you roll a die and get a 1 on the first roll, does that change the probabilities for the outcome of the second roll?
- **Answer:** No.

**Example 3: Polling (Not Independent)**
- Suppose you ask two randomly selected people about their political affiliation
- You might think if the people are truly chosen at random, the answer from the first person will not affect the answer from the second person
- **But what if** the two people are from the same family? Or what if they're friends?
- Then knowing the outcome of the first person **might affect** the outcome for the second
- **Conclusion:** That situation would **not be independent**

---

## Formal Definition of Independence

### Two Events

Two events, **A** and **B**, are **independent** if:

$$\boxed{P(A \mid B) = P(A)}$$

**Interpretation:**
- $P(A)$ is the **prior** probability (without additional information)
- $P(A \mid B)$ is the **posterior** probability (with knowledge that B occurred)
- Finding new information from event B doesn't change the probability for event A if the two events are independent

**By symmetry**, this is equivalent to:

$$P(B \mid A) = P(B)$$

So knowing A, the probability of B is still just the probability of B if A and B are independent.

### Multiplication Rule for Independent Events

Recall from the definition of conditional probability:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

If the events are **independent**, then $P(A \mid B) = P(A)$, so:

$$P(A) = \frac{P(A \cap B)}{P(B)}$$

Multiplying both sides by $P(B)$:

$$\boxed{P(A \cap B) = P(A) \times P(B)}$$

**This is the multiplication rule for independent events.**

> [!IMPORTANT]
> For independent events: $P(A \cap B) = P(A) \times P(B)$
>
> This is THE defining characteristic we use to check independence.

---

## Mutually Independent Events (Multiple Events)

We can extend the definition to multiple events.

Events $A_1, A_2, \ldots, A_n$ are **mutually independent** if for **every subset** of events (every possible grouping):

$$P(A_{i_1} \cap A_{i_2} \cap \cdots \cap A_{i_k}) = P(A_{i_1}) \times P(A_{i_2}) \times \cdots \times P(A_{i_k})$$

**This means:**
- Every **pairwise** grouping must satisfy independence (when $k=2$)
- Every **triple** must satisfy independence (when $k=3$)
- And so on for all possible subsets

> [!NOTE]
> **Pairwise independence does NOT necessarily imply mutual independence!** We will see an example of this shortly.

---

## Using the Definition of Independence

We can use the definition in **two ways**:

### Way 1: Testing for Independence

If we have two events A and B, we can test whether they're independent by:
1. Calculate $P(A)$
2. Calculate $P(B)$
3. Calculate $P(A \cap B)$
4. Check if $P(A \cap B) = P(A) \times P(B)$

**If they're equal:** Events A and B are **independent**
**If they're not equal:** Events A and B are **dependent**

### Way 2: Finding Intersection Probabilities

If we **know** two events are independent (either from an assumption or from other sources), then we can find the probability of their intersection:

$$P(A \cap B) = P(A) \times P(B)$$

---

## Example 1: Rolling Two Dice - Pairwise vs Mutual Independence

### Problem Setup

Roll a six-sided die twice.
- Sample space: $S = \{(i,j) \mid i,j \in \{1,2,3,4,5,6\}\}$
- $|S| = 36$ equally likely events

**Define events:**
- **E:** The sum is 7
- **F:** The first roll is a 4
- **G:** The second roll is a 3

**Question:** What can you say about the independence of E, F, and G?

### Solution

**Step 1: Calculate individual probabilities**

**Probability of E (sum is 7):**
- E = {(1,6), (2,5), (3,4), (4,3), (5,2), (6,1)}
- $|E| = 6$ events
- $P(E) = \frac{6}{36} = \frac{1}{6}$

**Probability of F (first roll is 4):**
- F = {(4,1), (4,2), (4,3), (4,4), (4,5), (4,6)}
- $P(F) = \frac{6}{36} = \frac{1}{6}$

**Probability of G (second roll is 3):**
- G = {(1,3), (2,3), (3,3), (4,3), (5,3), (6,3)}
- $P(G) = \frac{6}{36} = \frac{1}{6}$

**Step 2: Calculate pairwise intersections**

**$P(E \cap F)$ - sum is 7 AND first roll is 4:**
- If the first roll is 4 and the sum is 7, the second roll must be 3
- $E \cap F = \{(4,3)\}$
- $P(E \cap F) = \frac{1}{36}$
- Check: $P(E) \times P(F) = \frac{1}{6} \times \frac{1}{6} = \frac{1}{36}$ ✓

**$P(E \cap G)$ - sum is 7 AND second roll is 3:**
- If the second roll is 3 and the sum is 7, the first roll must be 4
- $E \cap G = \{(4,3)\}$
- $P(E \cap G) = \frac{1}{36}$
- Check: $P(E) \times P(G) = \frac{1}{6} \times \frac{1}{6} = \frac{1}{36}$ ✓

**$P(F \cap G)$ - first roll is 4 AND second roll is 3:**
- $F \cap G = \{(4,3)\}$
- $P(F \cap G) = \frac{1}{36}$
- Check: $P(F) \times P(G) = \frac{1}{6} \times \frac{1}{6} = \frac{1}{36}$ ✓

**Conclusion from Step 2:** Any pair of {E, F, G} is **pairwise independent**.

**Step 3: Check mutual independence**

For mutual independence, we need:
$$P(E \cap F \cap G) = P(E) \times P(F) \times P(G)$$

**Left side:**
- $E \cap F \cap G$ = sum is 7 AND first roll is 4 AND second roll is 3
- This is still just the single outcome {(4,3)}
- $P(E \cap F \cap G) = \frac{1}{36}$

**Right side:**
- $P(E) \times P(F) \times P(G) = \frac{1}{6} \times \frac{1}{6} \times \frac{1}{6} = \frac{1}{216}$

**Comparison:**
$$\frac{1}{36} \neq \frac{1}{216}$$

**Conclusion:** E, F, and G are **NOT mutually independent**.

### Key Insight

> [!IMPORTANT]
> Events E, F, and G are **pairwise independent** (any pair is independent), but they are **NOT mutually independent** (all three together are not independent).
>
> **Intuition:** If you know two of the events have occurred (say E and F), then the third event (G) **must** have occurred. So knowing the first two have occurred tells you something about the probability that the third event will occur.

---

## Example 2: School Statistics - Testing Independence

### Problem Setup

Suppose we have a school with:
- **Total students:** 1200
- **Juniors:** 250
- **Students taking stats:** 150
- **Juniors taking stats:** 40

**Define events:**
- **J:** A randomly selected student is a junior
- **S:** A randomly selected student is taking statistics

### Part A: Conditional Probability

**Question:** If the randomly selected student is a junior, what is the probability that they are also taking stats?

**Solution:**

From the problem, we know:
- $P(S) = \frac{150}{1200}$ (prior probability of taking stats)
- $P(J) = \frac{250}{1200}$ (probability of being a junior)
- $P(S \cap J) = \frac{40}{1200}$ (probability of being a junior AND taking stats)

We want to calculate:
$$P(S \mid J) = \frac{P(S \cap J)}{P(J)}$$

$$P(S \mid J) = \frac{\frac{40}{1200}}{\frac{250}{1200}} = \frac{40}{1200} \times \frac{1200}{250} = \frac{40}{250} = \frac{4}{25} = 0.16$$

**Interpretation:**
- This is the **posterior probability** that a student is taking statistics given the additional information that they're a junior
- The **prior probability** (without additional information) is $P(S) = \frac{150}{1200} = 0.125$
- The posterior probability (0.16) is different from the prior (0.125)

### Part B: Testing Independence

**Question:** Are J and S independent?

**Method 1: Compare conditional to marginal probability**

We calculated: $P(S \mid J) = 0.16$

We know: $P(S) = \frac{150}{1200} = 0.125$

Since $P(S \mid J) \neq P(S)$, the events are **NOT independent**.

**Method 2: Use multiplication rule**

For independence, we need: $P(S \cap J) = P(S) \times P(J)$

**Left side:**
$$P(S \cap J) = \frac{40}{1200}$$

**Right side:**
$$P(S) \times P(J) = \frac{150}{1200} \times \frac{250}{1200} = \frac{37500}{1440000}$$

When you calculate this:
- $\frac{40}{1200} = \frac{1}{30} \approx 0.0333$
- $\frac{37500}{1440000} \approx 0.0260$

Since these are not equal, J and S are **NOT independent**.

---

## Example 3: System Reliability with Independent Components

### Problem Setup

We have a system of 5 components arranged as follows:

```
Start → [1] → [2] → Finish
         ↓     ↑
        [3] ←--┘
         
        [4] → [5] → (connects to Finish)
```

**Assumptions:**
1. Each component works independently of every other component
2. Let $A_i$ be the event that the $i$-th component works
3. $P(A_i) = 0.9$ for all components ($i = 1, 2, 3, 4, 5$)

**System Operation:**
- For the entire system to work, you need a **path** of working components from start to finish
- Think of it as water flowing through the system, or electricity, where these are individual gates
- This is a system with **redundancy** built in to maximize the probability of the system working

**Possible paths:**
1. Components 1 → 2 (path through 1 and 2)
2. Components 1 → 3 (path through 1 and 3)
3. Components 4 → 5 (path through 4 and 5)

### Sample Space

The sample space consists of all possible states of the 5 components:

$$S = \{(x_1, x_2, x_3, x_4, x_5) \mid x_i \in \{0, 1\}\}$$

Where:
- $x_i = 1$ if the $i$-th component works
- $x_i = 0$ if the $i$-th component doesn't work

**Cardinality:** $|S| = 2^5 = 32$ possible events

> [!IMPORTANT]
> Each element in S is **NOT equally likely** because each component works with probability 0.9 (not 0.5).

**Examples:**
- $P(00000)$ = (none work) = $(0.1)^5 = 0.00001$
- $P(10101)$ = (components 1, 3, 5 work; 2, 4 don't) = $(0.9)^3 \times (0.1)^2 = 0.00729$

### Calculating the Probability the System Works

The system works if **at least one path is functional**:

$$P(\text{System works}) = P((A_1 \cap A_2) \cup (A_1 \cap A_3) \cup (A_4 \cap A_5))$$

**Understanding the events:**
- $A_1 \cap A_2$: Elements have form $(1, 1, ?, ?, ?)$ - first two components work, others can be anything
- $A_1 \cap A_3$: Elements have form $(1, ?, 1, ?, ?)$ - first and third work
- $A_4 \cap A_5$: Elements have form $(?, ?, ?, 1, 1)$ - fourth and fifth work

### Using the Inclusion-Exclusion Principle

Recall for the union of three events:

$$P(A \cup B \cup C) = P(A) + P(B) + P(C) - P(A \cap B) - P(A \cap C) - P(B \cap C) + P(A \cap B \cap C)$$

Let:
- Event A = $A_1 \cap A_2$
- Event B = $A_1 \cap A_3$
- Event C = $A_4 \cap A_5$

**Step 1: Individual probabilities (using independence)**

$$P(A_1 \cap A_2) = P(A_1) \times P(A_2) = (0.9)(0.9) = (0.9)^2$$

Similarly:
- $P(A_1 \cap A_3) = (0.9)^2$
- $P(A_4 \cap A_5) = (0.9)^2$

Sum: $3 \times (0.9)^2$

**Step 2: Pairwise intersections**

First intersection: $(A_1 \cap A_2) \cap (A_1 \cap A_3) = A_1 \cap A_2 \cap A_3$
- $P(A_1 \cap A_2 \cap A_3) = (0.9)^3$

Second intersection: $(A_1 \cap A_2) \cap (A_4 \cap A_5) = A_1 \cap A_2 \cap A_4 \cap A_5$
- $P(A_1 \cap A_2 \cap A_4 \cap A_5) = (0.9)^4$

Third intersection: $(A_1 \cap A_3) \cap (A_4 \cap A_5) = A_1 \cap A_3 \cap A_4 \cap A_5$
- $P(A_1 \cap A_3 \cap A_4 \cap A_5) = (0.9)^4$

Sum: $-(0.9)^3 - 2 \times (0.9)^4$

**Step 3: Triple intersection**

$$(A_1 \cap A_2) \cap (A_1 \cap A_3) \cap (A_4 \cap A_5) = A_1 \cap A_2 \cap A_3 \cap A_4 \cap A_5$$

$$P(A_1 \cap A_2 \cap A_3 \cap A_4 \cap A_5) = (0.9)^5$$

**Final Calculation:**

$$P(\text{System works}) = 3(0.9)^2 - (0.9)^3 - 2(0.9)^4 + (0.9)^5$$

$$= 3(0.81) - 0.729 - 2(0.6561) + 0.59049$$

$$= 2.43 - 0.729 - 1.3122 + 0.59049$$

$$= \boxed{0.97929}$$

**This is the overall probability that the system works.**

### Interpretation: The Power of Redundancy

**Comparison:**
- $P(A_1 \cap A_2) = (0.9)^2 = 0.81$
  - Just looking at components 1 and 2, the system would work with probability 0.81

- $P(\text{System works}) = 0.97929$
  - With three possible pathways (redundancy), the system works with probability 0.97929

**Key Insight:**

> [!IMPORTANT]
> **The key to increasing the probability that the system works is REDUNDANCY.**
>
> By having three possible paths instead of one, we increased the probability from 0.81 to 0.97929!

**Other ways to improve system reliability:**
1. **Improve component reliability:** Increase the probability that each component works above 0.9
2. **Add redundancy:** Add additional pathways through the system (as we did here)

### Extension

We could make this problem harder by having **different probabilities for each component**. Some components might be more delicate than others. The calculation method would be exactly the same, but we'd have different values instead of all 0.9s.

---

## Mutually Exclusive vs Independent

### Question

Suppose you have two events, A and B, and they are **mutually exclusive**. That is:

$$A \cap B = \emptyset$$

**Are A and B independent?**

### Initial Thought

You might initially think **yes**, since A and B are mutually exclusive, then knowing the probability of one maybe wouldn't influence the probability of the other.

### Analysis

Let's check using the definition of conditional probability:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

Since A and B are mutually exclusive:
$$A \cap B = \emptyset \implies P(A \cap B) = 0$$

Therefore:
$$P(A \mid B) = \frac{0}{P(B)} = 0$$

**But** if A and B were independent, we would need:
$$P(A \mid B) = P(A)$$

This means $P(A) = 0$, which would only be true for the trivial case.

### Conclusion

> [!IMPORTANT]
> **If A and B are mutually exclusive (and both have non-zero probability), then knowing that B has occurred means A cannot occur.**
>
> Therefore: **A and B are DEPENDENT, not independent.**

**Intuition:** Knowing B has occurred gives you information about A (specifically, that A didn't occur). This violates the definition of independence.

---

## Summary

### Key Concepts

1. **Independence (Two Events):**
   - $P(A \mid B) = P(A)$ or equivalently $P(B \mid A) = P(B)$
   - Multiplication rule: $P(A \cap B) = P(A) \times P(B)$

2. **Mutually Independent (Multiple Events):**
   - For ALL possible subsets: $P(A_{i_1} \cap \cdots \cap A_{i_k}) = P(A_{i_1}) \times \cdots \times P(A_{i_k})$

3. **Pairwise vs Mutual Independence:**
   - Pairwise independent does NOT imply mutually independent
   - Example 1 demonstrated this with the dice problem

4. **Testing Independence:**
   - Method 1: Check if $P(A \mid B) = P(A)$
   - Method 2: Check if $P(A \cap B) = P(A) \times P(B)$

5. **Using Independence:**
   - If events are known to be independent, use multiplication rule to find intersection probabilities
   - Essential for system reliability calculations

6. **Mutually Exclusive ≠ Independent:**
   - If events are mutually exclusive (and have non-zero probability), they are DEPENDENT

### Important Applications

- **Repeated experiments:** Coin flips, dice rolls (trials are independent)
- **System reliability:** Use independence and redundancy to calculate overall system probability
- **Statistical inference:** Independence assumptions are crucial in many statistical models

---

## Practice Problems

### Problem 1: Quick Check

If $P(A) = 0.5$, $P(B) = 0.3$, and A and B are independent, find $P(A \cap B)$.

**Solution:**
$$P(A \cap B) = P(A) \times P(B) = (0.5)(0.3) = 0.15$$

### Problem 2: Testing Independence

Given $P(A) = 0.4$, $P(B) = 0.6$, and $P(A \cap B) = 0.3$, are A and B independent?

**Solution:**

Check if $P(A \cap B) = P(A) \times P(B)$:
- Left side: $P(A \cap B) = 0.3$
- Right side: $P(A) \times P(B) = (0.4)(0.6) = 0.24$

Since $0.3 \neq 0.24$, events A and B are **NOT independent** (they are dependent).

### Problem 3: System with 3 Independent Components in Series

Three components are arranged in series (all must work for the system to work). Each works independently with probability 0.95. What is the probability the system works?

**Solution:**

Let $A_i$ = component $i$ works, $i = 1, 2, 3$

System works if all three work:
$$P(\text{System works}) = P(A_1 \cap A_2 \cap A_3)$$

Since components are independent:
$$P(A_1 \cap A_2 \cap A_3) = P(A_1) \times P(A_2) \times P(A_3) = (0.95)^3 = 0.857375$$

The system works with probability approximately **0.857** or **85.7%**.
