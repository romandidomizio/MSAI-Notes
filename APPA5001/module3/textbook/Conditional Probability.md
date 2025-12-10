# Chapter 2: Conditional Probability

## 1. Motivation
"Thinking conditionally" — reasoning with information we already know — is one of the most powerful concepts in probability. This chapter introduces:
- Conditional probability
- Law of Total Probability (LOTP)
- Bayes' Rule
- Independence
- Classic problems: Birthday Problem, Monty Hall, Gambler's Ruin
- Introduction to Random Variables (focusing on Binomial)

---

## 2. Conditional Probability

### Definition
**Conditional Probability** is the probability of event $A$ occurring given that event $B$ has already occurred.

**Notation**: $P(A|B)$ read as "probability of A given B"

**Formula**:
$$ P(A|B) = \frac{P(A \cap B)}{P(B)} $$

**Intuition**: When we condition on $B$, we're shrinking the sample space to only include outcomes where $B$ occurred. Within this smaller space, we find the proportion where $A$ also occurred.

> **Example**: Professor's class enrollment depends on ratings.
> - Let $A$ = over 300 students enroll
> - Let $B$ = poor ratings
> - $P(A|B)$ is lower than $P(A)$ because poor ratings discourage enrollment.

---

## 3. Law of Total Probability (LOTP)

### Definition
LOTP allows us to find $P(A)$ by partitioning the sample space into cases.

**Formula** (two cases):
$$ P(A) = P(A|B)P(B) + P(A|B^c)P(B^c) $$

**General Form** (partitioning into $n$ cases):
$$ P(A) = \sum_{i=1}^{n} P(A|B_i)P(B_i) $$
where $B_1, B_2, \ldots, B_n$ partition the sample space.

> **Think of it as**: A weighted average! Each case is weighted by its probability.

### Example: Anne's Commute
- $W$ = Anne goes to work
- $S$ = Sunny day
- $P(W|S) = 0.95$ (goes to work if sunny)
- $P(W|S^c) = 0.3$ (goes to work if rainy)
- $P(S) = 0.6$

**Find $P(W)$**:
$$ P(W) = P(W|S)P(S) + P(W|S^c)P(S^c) $$
$$ = (0.95)(0.6) + (0.3)(0.4) = 0.57 + 0.12 = 0.69 $$

```r
# Simulation in R
set.seed(110)
sims = 10000
work = rep(0, sims)

for(i in 1:sims){
  weather = runif(1)  # Random draw for weather
  go = runif(1)       # Random draw for work decision
  
  if(weather <= 0.6){  # Sunny
    if(go <= 0.95) work[i] = 1
  } else {             # Rainy
    if(go <= 0.3) work[i] = 1
  }
}

mean(work)  # Should be ~0.69
```

---

## 4. Bayes' Rule

### Definition
Bayes' Rule allows us to "flip" conditional probabilities.

**Formula**:
$$ P(A|B) = \frac{P(B|A)P(A)}{P(B)} $$

**With LOTP** (more useful):
$$ P(A|B) = \frac{P(B|A)P(A)}{P(B|A)P(A) + P(B|A^c)P(A^c)} $$

### Example: Frodo's Journey
- $F$ = Frodo gets jewelry to store
- $S$ = Sam goes with Frodo
- $P(F|S) = 0.9$ (success rate with Sam)
- $P(F|S^c) = 0.1$ (success rate without Sam)
- $P(S) = 0.8$
- **Find**: $P(S|F)$ (prob Sam went, given Frodo succeeded)

**Solution**:
$$ P(S|F) = \frac{P(F|S)P(S)}{P(F|S)P(S) + P(F|S^c)P(S^c)} $$
$$ = \frac{(0.9)(0.8)}{(0.9)(0.8) + (0.1)(0.2)} = \frac{0.72}{0.72 + 0.02} = \frac{0.72}{0.74} \approx 0.97 $$

**Interpretation**: If Frodo succeeded, there's a 97% chance Sam was with him!

---

## 5. Inclusion-Exclusion Principle

### Two Events
$$ P(A \cup B) = P(A) + P(B) - P(A \cap B) $$

**Why subtract?** We double-count the intersection when we add $P(A)$ and $P(B)$.

### General Form
$$ P(\text{Union}) = P(\text{Singles}) - P(\text{Doubles}) + P(\text{Triples}) - \ldots $$

### Example: Hospital Baby Mix-Up
$n$ couples each have 1 baby. Babies are randomly redistributed. What's the probability at least one couple gets their own baby back?

Let $A_i$ = couple $i$ gets their baby back.  
We want $P(A_1 \cup A_2 \cup \ldots \cup A_n)$.

**Solution** (using symmetry and inclusion-exclusion):
- $P(A_i) = \frac{(n-1)!}{n!} = \frac{1}{n}$
- $P(A_i \cap A_j) = \frac{(n-2)!}{n!}$
- General: $P(A_{i_1} \cap \ldots \cap A_{i_k}) = \frac{(n-k)!}{n!}$

By inclusion-exclusion:
$$ P(\text{at least 1 match}) = \sum_{k=1}^{n} \frac{(-1)^{k+1}}{k!} $$

**As $n \to \infty$**:
$$ P(\text{at least 1 match}) \to 1 - \frac{1}{e} \approx 0.632 $$

---

## 6. Independence

### Definition
Events $A$ and $B$ are **independent** if knowing one occurred doesn't change the probability of the other.

**Formally**:
$$ P(A|B) = P(A) \quad \text{and} \quad P(B|A) = P(B) $$

**Equivalent definition** (often easier to check):
$$ P(A \cap B) = P(A) \cdot P(B) $$

### Important Distinctions
- **Independent** $\neq$ **Disjoint**
- Disjoint events are *dependent*! (If $A$ occurs, $B$ cannot occur)

### Conditional Independence
$A$ and $B$ are **conditionally independent given $C$** if:
$$ P(A \cap B | C) = P(A|C) \cdot P(B|C) $$

> **Warning**: Conditional independence does NOT imply marginal independence, and vice versa!

**Example**: Rolling two dice
- Die results are marginally independent
- But conditionally dependent given their sum equals 7

```r
# Simulation showing independence
set.seed(110)
sims = 10000
H = rep(0, sims)  # Coin: Heads
E = rep(0, sims)  # Die: Even
O = rep(0, sims)  # Die: Odd

for(i in 1:sims){
  flip = runif(1)
  roll = sample(1:6, 1)
  
  if(flip <= 0.5) H[i] = 1
  if(roll %% 2 == 0) E[i] = 1
  if(roll %% 2 == 1) O[i] = 1
}

# H and E are independent
mean(H)           # ~0.5
mean(H[E == 1])   # Still ~0.5

# E and O are dependent
mean(E[O == 1])   # 0 (mutually exclusive)
```

---

## 7. The Birthday Problem

### Problem Statement
In a room of $n$ people, what's the probability that at least 2 people share a birthday?

**Assumptions**:
- 365 days in a year (no leap year)
- Birthdays are uniformly distributed
- Independence

### Solution Strategy
Use the **complement**! It's easier to find "no matches" than "at least one match".

Let $A$ = at least one match.  
Then $A^c$ = no matches (all unique birthdays).

**Probability of no match**:
- Person 1: $\frac{365}{365}$ (any day is unique)
- Person 2: $\frac{364}{365}$ (must avoid person 1's birthday)
- Person 3: $\frac{363}{365}$ (must avoid first 2)
- ...
- Person $n$: $\frac{365-n+1}{365}$

$$ P(A^c) = \frac{365 \cdot 364 \cdot 363 \cdots (365-n+1)}{365^n} $$

$$ P(A) = 1 - P(A^c) $$

**Surprising Result**: With just **23 people**, $P(\text{match}) > 0.5$!

**Why?** There are $\binom{23}{2} = 253$ pairs of people, which is a significant fraction of 365 days.

```r
# Birthday probability function
birthday_prob = function(n){
  if(n > 365) return(1)
  numerator = prod(365:(365-n+1))
  1 - numerator / 365^n
}

# Plot
n_values = 1:100
probs = sapply(n_values, birthday_prob)

plot(n_values, probs, type="l", lwd=3, col="darkred",
     main="Birthday Problem",
     xlab="Number of People", 
     ylab="P(at least one match)")
abline(h=0.5, lty=2)
abline(v=23, lty=2, col="blue")
```

---

## 8. Monty Hall Problem

### Setup
Game show with 3 doors:
- 1 door has a car (prize)
- 2 doors have goats
- You pick Door 1
- Monty (who knows where the car is) opens Door 2 to reveal a goat
- **Should you switch to Door 3?**

### Answer: YES! Switch!

**Probability of winning**:
- Stay: $\frac{1}{3}$
- Switch: $\frac{2}{3}$

### Proof via Bayes' Rule
Let:
- $C$ = car behind Door 1
- $G$ = Monty opens Door 2

We want $P(C|G)$.

$$ P(C|G) = \frac{P(G|C)P(C)}{P(G)} $$

- $P(C) = \frac{1}{3}$ (car equally likely behind any door)
- $P(G|C) = \frac{1}{2}$ (if car behind Door 1, Monty picks Door 2 or 3 randomly)
- $P(G) = \frac{1}{2}$ (by symmetry, Monty opens Door 2 or 3 equally)

$$ P(C|G) = \frac{\frac{1}{2} \cdot \frac{1}{3}}{\frac{1}{2}} = \frac{1}{3} $$

So $P(\text{Car behind Door 3}|G) = 1 - \frac{1}{3} = \frac{2}{3}$.

**Intuition**: Your initial pick had $\frac{1}{3}$ chance. The other two doors combined had $\frac{2}{3}$. Monty revealing one goat doesn't change your door's probability, but concentrates the $\frac{2}{3}$ onto the remaining door.

---

## 9. Gambler's Ruin

### Setup
- Player $A$ starts with $i$ dollars
- Player $B$ starts with $N - i$ dollars
- Each round: $A$ wins with probability $p$, loses with probability $q = 1-p$
- Winner takes $1 from loser
- Game ends when someone reaches $0

**Question**: What's $P(A \text{ wins})$?

### Solution

**Case 1**: $p = \frac{1}{2}$ (fair game)
$$ P(A \text{ wins}) = \frac{i}{N} $$

**Case 2**: $p \neq \frac{1}{2}$
$$ P(A \text{ wins}) = \frac{1 - (q/p)^i}{1 - (q/p)^N} $$

**Key Insight**: With a fair game, probability of winning equals your fraction of total money!

---

## 10. Random Variables: Introduction

### What is a Random Variable?
A **random variable** is a function that maps outcomes of a random experiment to numbers. Think of it as a "machine that spits out random numbers according to some pattern."

**Notation**: Usually denoted with capital letters like $X, Y, Z$.

**Two Types**:
1. **Discrete**: Takes on countable values (1, 2, 3, ...)
2. **Continuous**: Takes on any value in an interval (e.g., all real numbers between 0 and 1)

### Key Properties

#### 1. Distribution
The "recipe" or "pattern" the random variable follows.  
**Notation**: $X \sim \text{Distribution}(\text{parameters})$

#### 2. Expectation (Expected Value)
The **mean** or **average** value.  
**Notation**: $E(X)$ or $\mu$

**Formula** (discrete):
$$ E(X) = \sum_{i} x_i \cdot P(X = x_i) $$

Think: *Weighted average* of all possible values.

#### 3. Variance
Measures **spread** or **variability**.  
**Notation**: $Var(X)$ or $\sigma^2$

**Formula**:
$$ Var(X) = E[(X - E(X))^2] = \sum_i (x_i - E(X))^2 \cdot P(X = x_i) $$

**Standard Deviation**: $SD(X) = \sqrt{Var(X)}$

#### 4. PMF (Probability Mass Function)
For **discrete** random variables.  
Gives $P(X = x)$ for each value $x$.

#### 5. CDF (Cumulative Distribution Function)
For both discrete and continuous.  
Gives $P(X \leq x)$.  
**Notation**: $F(x) = P(X \leq x)$

**Properties**:
- Non-decreasing
- Right-continuous
- $\lim_{x \to -\infty} F(x) = 0$
- $\lim_{x \to \infty} F(x) = 1$

#### 6. Support
The set of values the random variable can take on.

### Example: Lottery Game
Roll a fair 6-sided die:
- Win \$10 if roll is 5 or 6
- Win \$0 if roll is 2, 3, or 4
- Lose \$5 if roll is 1

**Expectation**:
$$ E(X) = (10)\left(\frac{2}{6}\right) + (0)\left(\frac{3}{6}\right) + (-5)\left(\frac{1}{6}\right) = \frac{20 - 5}{6} = 2.5 $$

**Variance**:
$$ Var(X) = (10-2.5)^2\left(\frac{2}{6}\right) + (0-2.5)^2\left(\frac{3}{6}\right) + (-5-2.5)^2\left(\frac{1}{6}\right) $$
$$ = (56.25)(1/3) + (6.25)(1/2) + (56.25)(1/6) = 31.25 $$

---

## 11. Binomial Distribution

### Story
Perform $n$ **independent** trials, each with:
- Two outcomes: success or failure
- Constant probability of success $p$
- Count the number of successes

**Examples**:
- Flip a coin $n$ times, count heads
- Survey $n$ people, count "yes" responses
- Take $n$ free throws, count makes

### Notation
$$ X \sim \text{Bin}(n, p) $$
where:
- $n$ = number of trials
- $p$ = probability of success on each trial

### Properties

**Support**: $X \in \{0, 1, 2, \ldots, n\}$

**PMF**:
$$ P(X = k) = \binom{n}{k} p^k (1-p)^{n-k} $$

**Expectation**:
$$ E(X) = np $$

**Variance**:
$$ Var(X) = np(1-p) $$

### Understanding the PMF
$$ P(X = k) = \underbrace{\binom{n}{k}}_{\text{# of ways}} \times \underbrace{p^k(1-p)^{n-k}}_{\text{prob of one way}} $$

- $\binom{n}{k}$: Number of ways to choose which $k$ trials are successes
- $p^k$: Probability of $k$ successes
- $(1-p)^{n-k}$: Probability of $n-k$ failures

### Example: Coin Flips
$X$ = number of heads in 5 flips of a fair coin.  
$X \sim \text{Bin}(5, 0.5)$

**Find $P(X = 3)$**:
$$ P(X = 3) = \binom{5}{3} (0.5)^3 (0.5)^2 = 10 \times 0.03125 = 0.3125 $$

```r
# R functions for Binomial
# PMF: P(X = k)
dbinom(3, size=5, prob=0.5)  # 0.3125

# CDF: P(X <= k)
pbinom(3, size=5, prob=0.5)  # 0.5

# Quantile: find x such that P(X <= x) = q
qbinom(0.9, size=10, prob=0.5)  # 7

# Random draws
rbinom(10, size=5, prob=0.5)  # Generates 10 random values
```

---

## 12. Practice Problems

### Conditional Probability
**2.1**: Disease testing with false positives/negatives (Bayes' Rule application)

**2.2**: Weather forecasting with conditional probabilities

**2.3**: Card games with conditioning

### Independence
**2.4**: Prove or disprove independence in various scenarios

**2.5**: Conditional independence examples

### Classic Problems
**2.6**: Birthday problem variations (e.g., what if 3 people must share?)

**2.7**: Monty Hall with 4 doors

**2.8**: Gambler's Ruin simulations

### Binomial
**2.9**: Free throw shooter makes 70% of shots. Takes 10 shots. What's probability of making exactly 7?

**2.10**: Quality control: 5% of items are defective. Sample 20 items. Expected number of defects?

**2.11**: Multiple choice test: 10 questions, 4 choices each. Guessing randomly, what's probability of getting at least 7 correct?

---

## Summary

**Key Formulas**:

| Concept | Formula |
|:--------|:--------|
| Conditional Probability | $P(A\|B) = \frac{P(A \cap B)}{P(B)}$ |
| Law of Total Probability | $P(A) = \sum_i P(A\|B_i)P(B_i)$ |
| Bayes' Rule | $P(A\|B) = \frac{P(B\|A)P(A)}{P(B)}$ |
| Independence | $P(A \cap B) = P(A)P(B)$ |
| Inclusion-Exclusion | $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ |
| Binomial PMF | $P(X=k) = \binom{n}{k}p^k(1-p)^{n-k}$ |
| Binomial Expectation | $E(X) = np$ |
| Binomial Variance | $Var(X) = np(1-p)$ |

**Key Concepts**:
✓ Conditional probability narrows the sample space  
✓ LOTP breaks problems into cases (weighted average)  
✓ Bayes' Rule "flips" conditional probabilities  
✓ Independence means no information gain  
✓ Birthday problem shows power of pairs  
✓ Monty Hall illustrates conditional thinking  
✓ Random variables map outcomes to numbers  
✓ Binomial counts successes in fixed trials
