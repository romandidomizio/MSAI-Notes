# Module 2: Descriptive Statistics and the Axioms of Probability

## Permutations and Combinations

The goal of probability is to assign some number, $P(A)$, called the probability of event $A$, which will give us a measure of the likelihood of $A$ occurring.

If a sample space $S$ has $N$ single events, and if each of these events is equally likely to occur, then we need only count the number of events to find the probability.

For example, if $S = \{E_1, E_2, \ldots, E_N\}$, then $P(E_i) = \frac{1}{N}$ for all $i = 1, \ldots, N$, and if $A$ is an event in $S$, then $P(A) = \frac{\text{number of simple events in } A}{N}$

### Example 1

Experiment: Roll a six-sided die twice.

Sample space: $S = \{(i,j) | i,j \in \{1,2,3,4,5,6\}\}, |S| = 6^2 = 36$ and each of the events is equally likely to occur.

- Let $A$ be the event of rolling a 1 on the first roll. <br>
$P(A) = P(\{11, 12, 13, 14, 15, 16\}) = \frac{6}{36} = \frac{1}{6}$
- Let $B$ be the event that the sum of the two rolls is 8. <br>
$P(B) = P(\{26, 35, 44, 53, 62\}) = \frac{5}{36}$
- Let $C$ be the event that the value of the second roll is exactly two more than the value of the first roll. <br>
$P(C) = P(\{13, 24, 35, 46\}) = \frac{4}{36} = \frac{1}{9}$

### Permutations

Any **ordered sequence** of $k$ objects from a set of $n$ distinct objects is called a **permutation** of size $k$ from $n$ objects.

Notation: $P(k,n)$

**Example:**

Suppose an organization has 60 members. One person is selected at random to be president, another person is selected at random to be vice president, and a third person is selected at random to be treasurer. How many different ways can this be done? (This would be the cardinality of the sample space.)

*Order matters*.

$$P(3, 60) = 60 \times 59 \times 58 = \frac{60!}{(60-3)!} = \frac{60!}{57!} = 205,320$$

**Definition:**

$n! = n(n-1)(n-2)\cdots 3 \times 2 \times 1$ for any positive integer $n$. By definition, $0! = 1$.

### Combinations

Given $n$ distinct objects, any **unordered** subset of size $k$ of the objects is called a **combination** of size $k$ from $n$ objects.

Notation: $C(k,n)$

**Example:**

Suppose we have 60 people and want to choose a 3 person team. How many different combinations are possible? (This would be the cardinality of the sample space.)

*Order does not matter*.

$$C(3, 60) = \frac{P(3, 60)}{3!} = \frac{60!}{57!3!} = \frac{60 \times 59 \times 58}{3 \times 2 \times 1} = 34,220$$

**General Formula:**
$$C(k, n) = \binom{n}{k} = \frac{n!}{k!(n-k)!}$$


This represents the number of combinations of $k$ chosen from $n$ distinct objects.

$\binom{n}{k}$ is read as "n choose k"

60 choose 3 is the same as 60 choose 57 because 60 choose 3 = 60 choose (60-3) = 60 choose 57 and 60 choose 57 = 60 choose (60-57) = 60 choose 3. This is because the order does not matter.

**Example - continued:**

Suppose we have the same 60 people, 35 are female and 25 are male. We need to select a committee of 11 people.

- How many ways can such a committee be formed?
$$\text{\# of committees of 11} = \binom{60}{11} = \frac{60!}{11!(60-11)!}$$
$$|S| = \binom{60}{11} = \frac{60!}{11!(60-11)!} = \frac{60!}{11!49!} = \frac{60 \times 59 \times 58 \times 57 \times 56 \times 55 \times 54 \times 53 \times 52 \times 51 \times 50}{11 \times 10 \times 9 \times 8 \times 7 \times 6 \times 5 \times 4 \times 3 \times 2 \times 1} = 1,679,600$$

- What is the probability that a randomly selected committee will contain at least 5 men and at least 5 women? (Assume each committee is equally likely to be selected.)
$$P(\text{at least 5 men and at least 5 women})$$
$$=P(5m + 6w) + P(6m + 5w)$$
$$=\frac{\binom{25}{5}\binom{35}{6}}{\binom{60}{11}} + \frac{\binom{25}{6}\binom{35}{5}}{\binom{60}{11}}$$

**Example 2:**

A city has bought 20 buses. Shortly after being put into service, some of them develop cracks in the frame. The buses are inspected and 8 have visible cracks.

- How many ways can the city selects a sample of 5 for thorough inspection? (Assume each bus is equally likely to be selected.)

$$|S| = \binom{20}{5} = \frac{20!}{5!(20-5)!} = \frac{20!}{5!15!} = \frac{20 \times 19 \times 18 \times 17 \times 16}{5 \times 4 \times 3 \times 2 \times 1} = 15,504$$

- If 5 buses are chosen at random, find the probability that exactly 4 have cracks.

$$P(\text{exactly 4 have cracks}) = \frac{\binom{8}{4}\binom{12}{1}}{\binom{20}{5}}$$

- If 5 buses are chosen at random, find the probability that at least 4 have cracks.

$$P(\text{at least 4 have cracks})$$
$$=P(\text{exactly 4 have cracks}) + P(\text{exactly 5 have cracks})$$
$$=\frac{\binom{8}{4}\binom{12}{1}}{\binom{20}{5}} + \frac{\binom{8}{5}\binom{12}{0}}{\binom{20}{5}}$$

*Must account for all possible ways to choose 5 buses out of 20 (cracked vs non-cracked). Order does not matter.*