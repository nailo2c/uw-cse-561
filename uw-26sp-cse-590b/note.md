# Week 1 - 2026.03.31

+ SAT: Boolean Satisfiability Problem
    + Pure Boolean logic problem
    + e.g., (A or B) and (not A)
+ SMT: Satisfiability Modulo Theories
    + Boolean logic plus richer mathematical theories
    + e.g., (x > 3) and (x < 5)

A simple example I came out:

Let's say I have a file contianing `1111` and I want to check whether all of the numbers in this file are `1`.

1. Normal programming style:
```python
for x in data:
    if x != 1:
        print("Found one that is not 1")
```

2. SAT style:
```asm
(x1 = 1) AND (x2 = 1) AND (x3 = 1) AND (x4 = 1)
```

A simple example to explain implication truth table.

| P | Q | P->Q |
|---|---|------|
| T | T | T    |
| T | F | F    |
| F | T | T    |
| F | F | T    |

> If it rains, I will bring an umbrella.

Let:
+ P = it rains
+ Q = I bring an umbrella
