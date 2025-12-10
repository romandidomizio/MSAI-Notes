# Getting Started in R

## Introduction
R is a powerful language used for statistical analysis, data visualization, and more. While it has a learning curve, it is highly readable and logical once you understand the basics.

**Tools:**
- **R**: The programming language itself.
- **RStudio**: An Integrated Development Environment (IDE) that makes using R much easier. It combines the console, script editor, environment view, and plots into one interface.

---

## 1. Basic Arithmetic & The Console
At its simplest, R is a glorified calculator. You can type commands directly into the **console** and press `Enter` to run them.

### Basic Operations
```r
2 + 2
## [1] 4
```
*Note: The `[1]` indicates that the output is a vector and this is the first element.*

You can chain commands with semicolons `;` (though usually, we put them on separate lines for readability).
```r
2*3; 2/2; 2^2
## [1] 6
## [1] 1
## [1] 4
```

---

## 2. Variables
To store values for later use, we assign them to variables.
*   **Assignment Operator**: The text uses `=`, but `<-` is also the standard assignment operator in R (e.g., `x <- 2`). Both work, but `<-` is preferred by many R style guides.

```r
x = 2
x * 3   # 6
x / x   # 1
x^x     # 2^2 = 4
```

---

## 3. Functions
Functions are tools that take inputs (arguments), perform an action, and return an output.
Syntax: `function_name(argument1, argument2, ...)`

### Built-in Functions
R has many built-in functions.
```r
sqrt(1)   # Square root of 1 -> 1
exp(2)    # Exponential function (e^2) -> ~7.389
```

> **Tip**: To get help on any function, type `?` before the function name in the console (e.g., `?sqrt`) to open the documentation.

---

## 4. Vectors
A **vector** is the fundamental data structure in R. It is a list of elements of the same type (e.g., all numbers).

### Creating Vectors
Use the `c()` function (short for **concatenate** or **combine**).
```r
x = c(3, 1, 9)
x
## [1] 3 1 9
```

### Vectorized Operations
R is "vectorized," meaning operations often apply to *every element* of a vector at once without needing a loop.
```r
# Math on vectors
mean(x)  # Average of 3, 1, 9 -> 4.33
sum(x)   # Sum -> 13

# Adding vectors (element-wise addition)
y = c(2, 5, 6)
x + y
## [1] 5 6 15  (3+2, 1+5, 9+6)
```

### Indexing
Access specific elements using square brackets `[]`. **Note: R is 1-indexed** (the first element is at index 1, not 0).
```r
x[2]  # Returns the 2nd element (1)
```

---

## 5. Writing Custom Functions
You can define your own functions to automate tasks.

**Syntax:**
```r
function_name = function(input_variable) {
  # Body of the function
  result = ...
  return(result)
}
```

**Example 1: Square a number**
```r
my.first.function = function(x){
  return(x^2)
}
my.first.function(2) # 4
```

**Example 2: Pythagorean Theorem**
$$a^2 + b^2 = c^2 \implies c = \sqrt{a^2 + b^2}$$
```r
pyth = function(a, b){
  c = sqrt(a^2 + b^2)
  return(c)
}
pyth(3, 4) # 5
```

**Example 3: Logic & Modulo**
The modulo operator `%%` returns the remainder of division.
- `3 %% 2` is `1` (odd)
- `2 %% 2` is `0` (even)

```r
is.even = function(x){
  if(x %% 2 == 0){
    return(TRUE)
  }
  if(x %% 2 != 0){
    return(FALSE)
  }
}
```

---

## 6. Loops
Loops allow you to repeat code multiple times.

### For Loops
Runs code a specific number of times.
```r
# Create an empty vector of 10 NAs (Not Available) to hold results
x = rep(NA, 10)

# Loop from i = 1 to i = 10
for(i in 1:10){
  x[i] = i  # Set the i-th element of x to i
}
print(x)
## [1] 1 2 3 4 5 6 7 8 9 10
```

**Advanced Example: Stochastic Process**
Simulating a random walk: $S_i = S_{i-1} + X_i$ where $X_i \sim N(0,1)$.
```r
set.seed(110)        # Make random numbers reproducible
S = rep(NA, 100)     # Initialize vector
S[1] = rnorm(1, 0, 1) # First value

for(i in 2:100){
  S[i] = S[i-1] + rnorm(1, 0, 1) # Add random noise to previous value
}
plot(S, type="l")    # Plot as a line
```

### While Loops
Runs code *while* a condition is true.
```r
i = 0
while(i < 10){
  i = i + 1
}
print(i) # 10
```

**Break Statement**: Exits a loop immediately.
```r
while(TRUE){
  # Infinite loop until 'break' is called
  if(condition_met){
    break
  }
}
```

---

## 7. Graphics
R's base graphics are robust for quick visualization.

### `plot()` Function
The generic function for plotting.
```r
x = 1:10
y = x^2

plot(x, y,
     main = "Our First Plot!", # Title
     xlab = "x axis",          # X label
     ylab = "y axis",          # Y label
     xlim = c(0, 10),          # X limits
     ylim = c(0, 100),         # Y limits
     type = "l",               # 'l' for line, 'p' for points
     lwd = 5,                  # Line width
     col = "darkred")          # Color
```

### Other Plotting Tools
- **`par(mfrow = c(rows, cols))`**: Splits the plot window into a grid (e.g., 3x3).
- **`abline(h = y, v = x)`**: Adds horizontal (`h`) or vertical (`v`) straight lines to an existing plot.
- **`hist(x)`**: Creates a histogram.

---

## 8. Glossary & Key Commands

| Command | Description | Example |
| :--- | :--- | :--- |
| `?func` | Help for a function | `?mean` |
| `#` | Comment (ignored by R) | `# This is a comment` |
| `install.packages("pkg")` | Install a library | `install.packages("dplyr")` |
| `library(pkg)` | Load a library | `library(dplyr)` |
| `c(...)` | Concatenate (create vector) | `c(1, 2, 3)` |
| `rep(x, n)` | Replicate x, n times | `rep(NA, 10)` |
| `seq(from, to, by)` | Generate sequence | `seq(1, 10, by=0.5)` |
| `1:10` | Sequence operator | `1, 2, ..., 10` |
| `length(x)` | Number of elements | `length(c(1,2))` -> 2 |
| `unique(x)` | Unique elements | `unique(c(1,1,2))` -> 1, 2 |
| `paste0(...)` | Concatenate strings | `paste0("A", "B")` -> "AB" |
| `sample(x, size)` | Random sample | `sample(1:10, 3)` |
| `set.seed(n)` | Set RNG seed | `set.seed(123)` |
| `NA` | Missing value | |
| `TRUE` / `FALSE` | Boolean values | |

### Random Distributions
R has functions for distributions (prefix with `r` for random, `d` for density, `p` for probability, `q` for quantile).
- `rnorm(n, mean, sd)`: Normal distribution
- `runif(n, min, max)`: Uniform distribution
- `rbinom`, `rpois`, `rexp`, etc.

---

## 9. Practice Problems

### Problem 0.1: Unit Conversion
There are 1.60934 kilometers in every mile. Write a function in R that converts kilometers to miles.

### Problem 0.2: Binomial Coefficient Maximization
Show graphically that $\frac{n!}{k!(n-k)!}$ is maximized at $k = n/2$ (for even $n$) and $k \approx n/2$ (for odd $n$).
*Hint: The binomial coefficient is available in R as `choose(n, k)`.*

### Problem 0.3: Random Walk Simulation
Demren is wandering among the letters A to E. He starts at C. Every step, he moves up or down a letter with equal probability (50/50). He stops at A or E. Let $X$ be the number of steps. Estimate the mean and median of $X$ using simulation.
*Hint: Map A=1, B=2, C=3, D=4, E=5. Start at 3. Stop at 1 or 5.*

### Problem 0.4: Compound Random Variable
Roll a fair 6-sided die. Flip a fair coin the number of times shown on the die. Let $X$ be the number of heads. Estimate the mean and median of $X$.
