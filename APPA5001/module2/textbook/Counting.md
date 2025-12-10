# Chapter 1: Counting

## 1. Motivation
**"Learning to count is not as easy as it sounds."**
Counting is the foundation of probability. Many probability problems boil down to:
$$ P(Event) = \frac{\text{Number of Favorable Outcomes}}{\text{Total Number of Outcomes}} $$
To solve these, we need robust methods to count complex outcomes without listing them all one by one.

---

## 2. Set Theory Basics
Probability uses the language of sets. Here are the essential definitions:

*   **Set**: A collection of distinct objects (elements). E.g., $A = \{1, 2, 3\}$.
*   **Subset**: Set $A$ is a subset of $B$ ($A \subseteq B$) if every element in $A$ is also in $B$.
*   **Empty Set ($\emptyset$)**: The set containing no elements.
*   **Union ($A \cup B$)**: "A **OR** B". The set of elements in $A$, in $B$, or in both.
*   **Intersection ($A \cap B$)**: "A **AND** B". The set of elements common to both $A$ and $B$.
*   **Complement ($A^c$ or $\bar{A}$)**: "Not A". The set of all elements *not* in $A$.
*   **Disjoint Sets**: Sets with no overlap ($A \cap B = \emptyset$).
*   **Partition**: A collection of disjoint subsets ($A_1, A_2, ...$) that together cover the entire set.

> **Example 1.1: Harvard Statistics**
> - $A$ = Statistics Majors
> - $B$ = Winthrop House Students
> - **Union ($A \cup B$)**: Stats majors OR Winthrop students (or both).
> - **Intersection ($A \cap B$)**: Stats majors who live in Winthrop.
> - **Disjoint Example**: Set of US Presidents and Set of people named "Matt" (assuming no President was named Matt).

---

## 3. Naive Definition of Probability
If all outcomes are **equally likely**, the probability of an event $A$ is:
$$ P(A) = \frac{\text{Number of favorable outcomes}}{\text{Total number of outcomes}} $$

> **Critical Context**: This definition *only* works if the outcomes are equally weighted. It fails for things like "Will I win the lottery?" (2 outcomes: Win/Lose, but they are NOT equally likely).

---

## 4. Counting Tools

### A. The Multiplication Rule
Used for multi-step processes.
**Rule**: If step 1 has $n_1$ choices, step 2 has $n_2$ choices, ..., step $r$ has $n_r$ choices, then the total number of outcomes is:
$$ n_1 \times n_2 \times \dots \times n_r $$

**Example: Pizza**
- Sizes: 3 (S, M, L)
- Toppings: 4 (Pep, Saus, Meat, Cheese)
- Order: 2 (Delivery, Pickup)
- **Total Pizzas**: $3 \times 4 \times 2 = 24$.

```r
# R Code: Generating Pizza Combinations
size = c("S", "M", "L")
topping = c("pepperoni", "sausage", "meatball", "extra cheese")
order = c("deliver", "pick-up")

# Create all combinations (Cartesian product)
# We can use expand.grid() in R for this, but here is the loop logic:
pizzas = expand.grid(size, topping, order)
nrow(pizzas) # Returns 24
```

### B. Factorials (Permutations)
Used for ordering distinct objects.
**Definition**: $n! = n \times (n-1) \times \dots \times 1$.
**Concept**: How many ways can you arrange $n$ items?
- Slot 1: $n$ choices
- Slot 2: $n-1$ choices
- ...
- Slot $n$: 1 choice

**Example**: Ordering letters A, B, C.
Permutations: ABC, ACB, BAC, BCA, CAB, CBA ($3! = 6$).

```r
# R Code: Permutations
# install.packages("combinat")
perms = combinat::permn(c("A", "B", "C"))
length(perms) # 6
```

### C. Binomial Coefficient (Combinations)
Used for choosing groups where **order does not matter**.
**Notation**: $\binom{n}{k}$ read as "$n$ choose $k$".
**Formula**:
$$ \binom{n}{k} = \frac{n!}{k!(n-k)!} $$

**Intuition**:
1.  Start with $n!$ (total permutations of all items).
2.  Divide by $(n-k)!$ (remove ordering of the items we *didn't* pick).
3.  Divide by $k!$ (remove ordering of the items we *did* pick, because order doesn't matter).

**Example**: Choosing a committee of 3 from 5 people.
$$ \binom{5}{3} = \frac{5!}{3!2!} = \frac{120}{6 \times 2} = 10 $$

```r
# R Code: Combinations
choose(5, 3) # Returns 10

# Generating the actual combinations
# install.packages("gtools")
gtools::combinations(n = 5, r = 3)
```

---

## 5. The Sampling Table
This table summarizes how to count based on two constraints: **Order** and **Replacement**.

| | **Order Matters** | **Order Doesn't Matter** |
| :--- | :---: | :---: |
| **With Replacement** | $n^k$ | $\binom{n+k-1}{k}$ |
| **Without Replacement** | $\frac{n!}{(n-k)!}$ | $\binom{n}{k}$ |

### Explaining the Quadrants
1.  **Top-Left ($n^k$)**: Password with $k$ digits from $n$ options. repeats allowed, order matters (112 $\neq$ 211).
2.  **Bottom-Left ($\frac{n!}{(n-k)!}$)**: Password with unique digits. No repeats, order matters.
3.  **Bottom-Right ($\binom{n}{k}$)**: Lottery ticket / Committee. No repeats, order doesn't matter ({1,2} is same as {2,1}).
4.  **Top-Right ($\binom{n+k-1}{k}$)**: **Bose-Einstein**.
    *   **Scenario**: Putting $k$ indistinguishable balls into $n$ distinguishable boxes.
    *   **Analogy**: Buying $k$ donuts from a shop with $n$ flavors. You can pick the same flavor multiple times (replacement), and the order you put them in the box doesn't matter (order doesn't matter).
    *   **Stars and Bars**: Imagine $k$ balls (stars) and $n-1$ dividers (bars). Any arrangement of these represents a selection. Total positions: $n+k-1$. We choose $k$ positions for the balls.

---

## 6. Story Proofs
A **Story Proof** proves a mathematical identity by interpreting both sides of the equation as counting the same thing in two different ways.

**Example 1.3: Symmetry**
$$ \binom{n}{k} = \binom{n}{n-k} $$
*   **Story**: Choosing a team of $k$ people from $n$ is exactly the same as choosing the $n-k$ people who are *not* on the team.

**Example 1.4: The President**
$$ k \binom{n}{k} = n \binom{n-1}{k-1} $$
*   **Story**: Choosing a committee of $k$ people and then picking a president from them.
    *   **LHS**: Choose committee ($\binom{n}{k}$), then choose president from committee ($k$).
    *   **RHS**: Choose president from population ($n$), then choose rest of committee from remaining people ($\binom{n-1}{k-1}$).

---

## 7. Symmetry in Probability
Using symmetry can simplify complex problems.

**Example 1.5: Spades**
*   **Problem**: You are dealt 4 cards. On average, how many are Spades?
*   **Hard Way**: Calculate prob of 0, 1, 2, 3, 4 spades, multiply by count, sum up.
*   **Symmetry Way**:
    *   Imagine the 52 cards are split into 13 groups of 4.
    *   There are 13 Spades total in the deck.
    *   By symmetry, each group of 4 should have the same expected number of spades.
    *   $13 \text{ Spades} / 13 \text{ Groups} = 1 \text{ Spade per group}$.
    *   **Answer**: 1.

---

## 8. Practice Problems

**1.1 Adjacent Aces**
Prob that 4 Aces are adjacent in a shuffled deck.

**1.2 Ivy League Rankings**
Ways to rank 8 schools? What if "Big 3" (Harvard, Yale, Princeton) are identical?

**1.3 Kickball Teams**
Ways to split 20 kids into teams of 9 and 11? Teams of 10 and 10?

**1.4 Poker Hands**
Prob of Royal Flush? Prob of 3 of a kind?

**1.5 Scheduling**
Tony has 5 meetings M-F. How many schedules if not all on one day?

**1.6 Counties & Towns**
5 counties, 6 towns each. Visit each county once. How many paths?

**1.7 Tic-Tac-Toe**
Prob X wins on 3rd move?

**1.8 Tic-Tac-Max**
Play until board full. How many game sequences? How many final boards?

**1.9 Pairing People**
Ways to pair $2n$ people?

**1.10 Digits**
Ways to make a 3-digit and 7-digit number from 0-9?

**1.11 Pentagon**
Prob that 5 random lines between 5 points form a pentagon?

**1.12 Shoes**
$n$ people, $2n$ shoes (L/R). Ways to assign shoes correctly?

**1.13 Nick's Mistake**
Explain why $\binom{n}{k}\binom{k}{n} \neq 1$.

**1.14 Coin Flips**
Compare P(5 Heads) vs P(10 Heads). Compare sequence HTHTHTHTHT vs HHHHHHHHHH.

**1.15 No Repeats**
5-letter words with no repeating letters.

**1.16 Restaurant 7367**
Dice game for free meal.

**1.17 Board Game**
Prob Dan sits next to Alec at round table.

**1.18 Knights**
Jon (10) and Robb (30) picking knights. Does going first matter?

**1.19 Cards Roulette**
Drawing Aces. Red Ace loses. Go first or second?

**1.20 Good Cop / Bad Cop**
Story proof for $\binom{n}{x} 2^x$.

**1.21 Skip-Counting**
Ways to count from 1 to $n$.

**1.22 NFL Pool**
Strategy to guarantee survival in betting pool.

### BH Problems (Blitzstein & Hwang)
*   **1.8**: Splitting people into teams.
*   **1.9**: Lattice paths.
*   **1.16**: Pascal's Identity ($\binom{n}{k} + \binom{n}{k-1} = \binom{n+1}{k}$).
*   **1.18**: Hockey Stick Identity.
*   **1.22**: Family birth order.
*   **1.23**: Robberies in districts.
*   **1.26**: Course conflicts.
*   **1.27**: Dice sums and Palindromes.
*   **1.29**: Capture-recapture.
*   **1.31**: Drawing balls from jar.
*   **1.32**: Poker probabilities.
*   **1.40**: No-repeat words.
*   **1.48**: Void in suit.
*   **1.52**: Class scheduling.
*   **1.59**: The Drunken Passenger (Airplane seating problem).
