# Module 2: Descriptive Statistics and the Axioms of Probability

## Axioms of Probability

The **axioms of probability** are the foundation of probability theory. They are:

1. **Non-negativity**: $P(A) \geq 0$ for any event $A$.
2. **Normalization**: $P(S) = 1$.
3. **Finite additivity**: For any finite collection of pairwise disjoint events $A_1, \dots, A_n$, $P\!\left(\cup_{i=1}^n A_i\right) = \sum_{i=1}^n P(A_i)$.
4. **Countable (σ-)additivity**: For any countably infinite collection of pairwise disjoint events $A_1, A_2, \ldots$, $P\!\left(\cup_{i=1}^\infty A_i\right) = \sum_{i=1}^\infty P(A_i)$.

### Example 1

Experiment: Flip a coin until the first tail appears. 0 represents heads, 1 represents tails. The sample space is:
$$S = \{1, 01, 001, 0001, ...\}$$

Let $A_n$ be the event that the first tail appears on the $n^{th}$ flip. $A_n = \{00...01\}$ Find $P(A_1)$, $P(A_2)$, $P(A_5)$ and $P(A_n)$, where $n$ is a positive integer.

$P(A_1) = 1/2$ <br>
$P(A_2) = P({01}) = 1/4$ <br>
$P(A_5) = P({00001}) = 1/2^5 = 1/32$ <br>
$P(A_n) = P({00...01}) = 1/2^n$

**Note:** $$ P(S) = P(\cup_{k=1}^{\infty} A_k) = \sum_{k=1}^{\infty} P(A_k) = \sum_{k=1}^{\infty} \frac{1}{2^k} = 1 $$

If $B$ is the event that it takes at least 3 flips to obtain a tail, find $P(B)$.

$B^c$ is the event that it takes at most 2 flips to obtain a tail.

$$ P(B^c) = P(\{1, 01\}) = P(A_1) + P(A_2) = \frac{1}{2} + \frac{1}{4} = \frac{3}{4} $$

We also note:
$$ P(S) = P(B \cup B^c) = P(B) + P(B^c) = 1 $$
So,
$$ P(B) = 1 - P(B^c) = 1 - \frac{3}{4} = \frac{1}{4} $$

### Consequences of the Axioms

If $A$ and $B$ are two events contained in the same sample space $S$,
- $A \cap A^c = \emptyset$ and $A \cup A^c = S$ so, <br>
1 = $P(S) = P(A \cup A^c) = P(A) + P(A^c)$ which implies $P(A^c) = 1 - P(A)$
- If $A \cap B = \emptyset$, then $P(A \cap B) = 0 = P(\emptyset)$
- $P(A \cup B) = P(A) + P(B) - P(A \cap B)$

These three consequences will help us calculate many probabilities more efficiently.

![Hand-drawn Venn diagrams](../img/venn2.png)

### Example 2

Return to our car example: a randomly selected car is inspected for three defects. The sample space is
$$S = \{000, 100, 010, 001, 110, 101, 011, 111\}.$$
Consider the three events:

- $A$ is the event defect 1 is present, $A = \{100, 110, 101, 111\}$.
- $B$ is the event defect 2 is present, $B = \{010, 110, 011, 111\}$.
- $C$ is the event defect 3 is present, $C = \{001, 011, 101, 111\}$.

Suppose over many days of data collection we find:

- 20% of cars have defect 1, so $P(A) = 0.20$
- 25% have defect 2, so $P(B) = 0.25$
- 30% have defect 3, so $P(C) = 0.30$
- 5% have defects 1 and 2, so $P(A \cap B) = 0.05$
- 7.5% have defects 2 and 3, so $P(B \cap C) = 0.075$
- 6% have defects 1 and 3, so $P(A \cap C) = 0.06$
- 1.5% have all three defects, so $P(A \cap B \cap C) = 0.015$

Calculate the probability of each of the following events for the randomly selected car:

- Defect 1 did not occur:
$$ P(A^c) = 1 - P(A) = 1 - 0.20 = 0.80 $$
- At least one defect occurs:
$$ P(A \cup B \cup C) = P(A) + P(B) + P(C) - P(A \cap B) - P(B \cap C) - P(A \cap C) + P(A \cap B \cap C) $$
$$ = 0.20 + 0.25 + 0.30 - 0.05 - 0.075 - 0.06 + 0.015 = 0.58 $$
- No defect occurs:
$$ P(A^c \cap B^c \cap C^c) = P((A \cup B \cup C)^c) = 1 - P(A \cup B \cup C) = 1 - 0.58 = 0.42 $$
- Defect 1 and 3 occur but 2 does not:
$$ P(A \cap C \cap B^c) = P(A \cap C) - P(A \cap B \cap C) = 0.06 - 0.015 = 0.045 $$

![Hand-drawn Venn diagrams](../img/venn3.png)