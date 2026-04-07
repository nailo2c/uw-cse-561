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

4. Biconditional (<->)

| P | Q | P<->Q |
|---|---|-------|
| T | T | T     |
| T | F | F     |
| F | T | F     |
| F | F | T     |

5. Defintion of satisfiable/valid/unsatisfiable

A formula is `satisfiable` if some interpretation makes it true.  
`valid` if every interpretation makes it true.  
`unsatisfiable` if no interpretation makes it true.

6. Normal Forms

CNF is subset of NNF, and NNF is subset of Propositional Logic.

DNF is also a subset of NNF, but not a subset of CNF.

If we want to use SAT solver, the input must be a CNF form.

DNF can check if it's satisfiability immediately.

7. Tseitin Transformation

Tseitin can transform any formula to CNF in linear trade-off.

8. BCP & DPLL

BCP (Boolean Constraint Propagation) is a way to simplify the formula by find a unit clause and set it to true. (LOOP)

DPLL (Davis-Putnam-Logemann-Loveland) is a SAT solving algorithm that combines BCP with branching and backtracking.  
It first calls BCP to deduce all forced assignments.  
If the formula is not yet resolved, it picks an unassigned literal, guesses it as true, and recurses itself.  
If that leads to a conflict, it backtracks and tries the opposite value.  

9. Recap

This class walk through the process below:

Any formula (Propositional Logic) -> NNF -> Tseitin -> CNF -> DPLL/SAT solver -> Find a model or prove it's UNSAT
