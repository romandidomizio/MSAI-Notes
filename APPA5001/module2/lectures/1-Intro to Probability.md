# Module 2: Descriptive Statistics and the Axioms of Probability

## Intro to Probability

**Statistics is the science of using data effectively to gain new knowledge**

**Population**: Those individuals or objects from which we want to acquire information or draw a conclusion. Population is usually very large, so we can only collect data on a subset of it. We call this our **sample**

In **probability**, we assume we know the characteristics of the entire population. Then we can pose and answer questions about the nature of a sample. **Statistics**, on the other hand, is the opposite. If we have a sample with particular characteristics, we want to be able to say with some degree of confidence whether the whole population has that characteristic or not.

So we start by learning probability and figuring out what we can say about individual samples. And then we go to the samples in statistics, and what can we say about the whole population.

**Probability** studies randomness and uncertainty by giving these concepts a mathematical foundation.

* An **experiment** is any action or process that generates observations
* The **sample space** of an experiment, denotes $S$, is the set of all possible outcomes of an experiment
* An **event** is any possible outcome, or combination of outcomes, of an experiment
* The **cardinality** of a sample space or an event, is the number of outcomes it contains. $|S|$ represents the cardinality of the sample space.

### Examples

- Experiment 1: Flip a coin once
    * Sample space: $S = \{H, T\} = \{0, 1\}$
    * Cardinality: $|S| = 2$
- Experiment 2: Flip a coin twice
    * Sample space: $S = \{HH, HT, TH, TT\} = \{00, 01, 10, 11\}$
    * Cardinality: $|S| = 4$
- Experiment 3: Flip a coin until you get a tail
    * Sample space: $S  = \{T, HT, HHT, HHHT, ...\} = \{1, 01, 001, 0001, ...\}$
    * Cardinality: $|S| = \infty$
- Experiment 4: Select a car coming off an assembly line and inspect it for 3 different defects (engine, seat belt, paint)
    * Sample space: $S = \{000, 001, 010, 011, 100, 101, 110, 111\}$
    * Cardinality: $|S| = 8$

### Set Notation

For events $A$ and $B$:

- $A \cup B$, the **union** of $A$ and $B$, is the set of outcomes in $A$ or $B$.
- $A \cap B$, the **intersection** of $A$ and $B$, is the set of outcomes contained in both $A$ and $B$.
- $A^c$, the **complement** of $A$, is the set of outcomes in $S$ that are not in $A$.
- $A$ and $B$ are **mutually exclusive** (disjoint) if they have no outcomes in common; we write $A \cap B = \emptyset$.

### Examples Continued

Let $S = \{000, 100, 010, 001, 110, 101, 011, 111\}$ describe the quality-control experiment, where each digit indicates whether a defect is present (engine, seat belt, paint).

- $A$: there is an engine problem (defect 1). $A = \{100, 110, 101, 111\}$.
- $B$: there is exactly one defect. $B = \{100, 010, 001\}$.
- $C$: there are exactly two defects. $C = \{110, 101, 011\}$.

Using these sets:

- $A \cap B = \{100\}$ (engine-only failures).
- $A^c = \{000, 010, 001, 011\}$ (all cars without an engine defect).
- $A^c \cup B = \{000, 010, 001, 011, 100\}$.
- $B \cap C = \emptyset$ because a car cannot simultaneously have exactly one and exactly two defects.

### Visualizing with Venn Diagrams

Venn diagrams help reinforce how unions, intersections, and complements partition the sample space $S$.

![Hand-drawn Venn diagrams](../img/venn1.png)

The left panel highlights $A$ and its complement $A^c$ for the engine-defect example. The right panel illustrates how $A$, $B$, and $C$ overlap: the region where $A$ and $C$ intersect contains $110$ and $101$, while $B$ remains disjoint from $C$.