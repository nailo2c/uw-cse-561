"""Part 2 Option 1 — Zebra puzzle with roommates.

Six people share three houses (two per house, as "roommates"). Every
non-House category has six distinct values — one per person — while
House has three values, each held by two people. You are given some
clues about people's properties and about their roommates'. Decide who
has what.

Your encoding should be quantifier-free EUF: every assertion a ground
equality or disequality between terms over uninterpreted functions.

### Input / Output

    solve_zebra(categories, clues) -> list[dict[str, str]] | None

    categories : dict[str, list[str]] mapping category name to its
                 ordered list of values. The five categories are:

                     "House"       : 3 distinct values
                     "Nationality" : 6 distinct values
                     "Pet"         : 6 distinct values
                     "Drink"       : 6 distinct values
                     "Reading"     : 6 distinct values

    clues      : list[Clue] — see `zebra_types.py`. Each clue has:
                     subject          : Property identifying a person
                                        (or a pair, for House)
                     target           : Property about the subject or
                                        the subject's roommate
                     about_roommate   : False → target applies to the subject
                                        True → to the subject's roommate
                     subject_negated  : False → IS the subject property;
                                        True  → is NOT
                     target_negated   : False → HAS the target property;
                                        True  → does NOT

                 Note that about_roommate must be False when subject is "House".

    Return an array of dicts, each dict mapping categories to values for a 
    particular person, or None if the puzzle is unsatisfiable. You may return 
    the people in any order.

### Scaffolding we've provided

The five category EnumSorts (`Nationality`, `House`, `Pet`, `Drink`,
`Reading`) are built for you at the top of `solve_zebra`, along with a
`VALUE` dict that maps a category name and value name to the matching
Z3 constant. Use `VALUE[clue.subject.category][clue.subject.value]`
to turn a clue's `Property` into a Z3 term. Everything else — how you
represent people, what uninterpreted functions you declare, every
axiom, and the final extraction loop — is up to you.

### Z3 primitives you may use

You may only use the Z3 primitives imported below. Do not use anything
else from Z3.

    EnumSort(name, v1, v2, ...)
        Creates a type of finite domain whose values are exactly v1, v2, ... (strings).
        Returns (sort, tuple_of_constants).

    Function(name, s1, s2, ...)
        Uninterpreted function mapping the first n-1 sorts (types) to the last sort.
        Examples:
            Function("f", U, U)         f: U → U
            Function("g", U, U, U)      g: U × U → U
        Returns an object that you can call with () like a normal function. 

    Const(name, sort)
        A constant of the given sort.  Unlike EnumSort constants, the
        solver picks its value — useful for declaring a free "witness"
        person whose identity is determined by the clue constraints.

    Distinct([e1, e2, ...])
        Syntactic sugar for pairwise disequality (conjunction of != literals).

    Solver()
        Supports the following operations:
            s.add(formula): add all constraints like this
            s.check(): returns sat or unsat
            s.model(): if sat, gets a solution

    sat, unsat      result possibilities
    is_true(expr)   True iff `expr` is the Z3 constant True;
                    apply to m.evaluate(bool_var) to get a Python bool


### Hint

You'll probably want an uninterpreted function `Roommate` that outputs
the roommate of a given person. Think through what it means for two 
people to share a house, and sanity-check that your axioms rule out 
every "obviously wrong" model a solver might return. In particular:

- Being roommates is symmetric: if q is p's roommate, then p is
  q's roommate.
- Nobody is their own roommate.
- Roommates live in the same house.
- Every house has exactly two occupants. This is the subtlest one.
  Distinctness on the other categories can come from `Distinct([...])`,
  but `House` has only 3 values across 6 people, so `Distinct` won't 
  help here. How can you force each house to hold exactly two people?
"""
import itertools

from z3 import EnumSort, Function, Const, Distinct, Solver, sat, unsat, is_true

from zebra_types import Clue, Property


N = 6  # six people, three houses (fixed per the assignment spec).

_CALL_COUNTER = itertools.count()


def solve_zebra(categories, clues):
    # Fresh per-call tag so repeat calls don't collide in Z3's global
    # sort namespace (Z3 forbids redeclaring an EnumSort name).
    tag = next(_CALL_COUNTER)

    # ---- Category sorts (scaffolding) ---------------------------------
    # Each category is an EnumSort whose values are exactly the strings
    # from `categories`. Use `VALUE[cat][val]` to turn a clue's
    # Property into its Z3 constant.
    Nationality, nat_consts = EnumSort(
        f"Nationality_{tag}", categories["Nationality"]
    )
    House, house_consts = EnumSort(
        f"House_{tag}", categories["House"]
    )
    Pet, pet_consts = EnumSort(
        f"Pet_{tag}", categories["Pet"]
    )
    Drink, drink_consts = EnumSort(
        f"Drink_{tag}", categories["Drink"]
    )
    Reading, reading_consts = EnumSort(
        f"Reading_{tag}", categories["Reading"]
    )

    VALUE = {
        "Nationality": dict(zip(categories["Nationality"], nat_consts)),
        "House":       dict(zip(categories["House"],       house_consts)),
        "Pet":         dict(zip(categories["Pet"],         pet_consts)),
        "Drink":       dict(zip(categories["Drink"],       drink_consts)),
        "Reading":     dict(zip(categories["Reading"],     reading_consts)),
    }

    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    CATEGORIES = {
        "House":       ["Red", "Yellow", "Blue"],
        "Nationality": ["Englishman", "Spaniard", "Ukrainian",
                        "Japanese", "Norwegian", "Italian"],
        "Pet":         ["Dog", "Cat", "Rabbit",
                        "Guinea Pig", "Parrot", "Hamster"],
        "Drink":       ["Tea", "Coffee", "Milk",
                        "Juice", "Water", "Cocoa"],
        "Reading":     ["Novel", "Newspaper", "Magazine",
                        "Tabloid", "Encyclopedia", "Almanac"],
    }

    def clue(subj_cat, subj_val, tgt_cat, tgt_val,
             about_rm=False, subj_neg=False, tgt_neg=False):
        return Clue(Property(subj_cat, subj_val),
                    Property(tgt_cat, tgt_val),
                    about_rm, subj_neg, tgt_neg)

    def find(result, category, value):
        hits = [row for row in result if row[category] == value]
        assert len(hits) == 1
        return hits[0]

    # ---- Example 1 -----------------------------------------------------
    # Three nationalities are all pinned to the Red house. Only two
    # people share a house, so no assignment satisfies all three
    # clues — expect None.
    unsat_clues = [
        clue("Nationality", "Englishman", "House", "Red"),
        clue("Nationality", "Spaniard",   "House", "Red"),
        clue("Nationality", "Ukrainian",  "House", "Red"),
    ]
    assert solve_zebra(CATEGORIES, unsat_clues) is None

    # ---- Example 2 -----------------------------------------------------
    # All six pets are pinned to nationalities (so Spaniard uniquely
    # owns the Dog), and the Englishman's roommate owns the Dog. The
    # only consistent assignment places the Englishman and the Spaniard
    # in the same house.
    rm_clues = [
        clue("Nationality", "Englishman", "Pet", "Cat"),
        clue("Nationality", "Spaniard",   "Pet", "Dog"),
        clue("Nationality", "Ukrainian",  "Pet", "Rabbit"),
        clue("Nationality", "Japanese",   "Pet", "Guinea Pig"),
        clue("Nationality", "Norwegian",  "Pet", "Parrot"),
        clue("Nationality", "Italian",    "Pet", "Hamster"),
        clue("Nationality", "Englishman", "Pet", "Dog", about_rm=True),
    ]
    result = solve_zebra(CATEGORIES, rm_clues)
    assert result is not None
    assert find(result, "Nationality", "Englishman")["House"] == \
           find(result, "Nationality", "Spaniard")["House"]

    # ---- Example 3 -----------------------------------------------------
    # A larger Zebra-style puzzle covering every clue shape
    # solve_zebra must handle: positive and negated subject-direct,
    # positive and negated roommate, house-subject, and clues keyed off
    # Pet / Drink / Reading as well as Nationality.
    sat_clues = [
        # The Englishman lives in the Red house.
        clue("Nationality", "Englishman", "House", "Red"),
        # The Yellow house is home to the Ukrainian.
        clue("House", "Yellow", "Nationality", "Ukrainian"),
        # The Japanese owns a Guinea Pig.
        clue("Nationality", "Japanese", "Pet", "Guinea Pig"),
        # The Spaniard drinks Coffee.
        clue("Nationality", "Spaniard", "Drink", "Coffee"),
        # The Norwegian drinks Water.
        clue("Nationality", "Norwegian", "Drink", "Water"),
        # The Italian reads the Almanac.
        clue("Nationality", "Italian", "Reading", "Almanac"),
        # The Japanese reads the Tabloid.
        clue("Nationality", "Japanese", "Reading", "Tabloid"),
        # The Ukrainian reads the Magazine.
        clue("Nationality", "Ukrainian", "Reading", "Magazine"),
        # The Englishman owns the Dog.
        clue("Nationality", "Englishman", "Pet", "Dog"),
        # Whoever reads the Almanac drinks Cocoa.
        clue("Reading", "Almanac", "Drink", "Cocoa"),
        # Whoever owns the Parrot reads the Encyclopedia.
        clue("Pet", "Parrot", "Reading", "Encyclopedia"),
        # Whoever drinks Milk owns the Rabbit.
        clue("Drink", "Milk", "Pet", "Rabbit"),
        # Whoever owns the Cat reads the Newspaper.
        clue("Pet", "Cat", "Reading", "Newspaper"),
        # The Englishman's roommate drinks Coffee.
        clue("Nationality", "Englishman", "Drink", "Coffee", about_rm=True),
        # The Ukrainian's roommate drinks Juice.
        clue("Nationality", "Ukrainian", "Drink", "Juice", about_rm=True),
        # The Norwegian's roommate owns the Hamster.
        clue("Nationality", "Norwegian", "Pet", "Hamster", about_rm=True),
        # The Italian's roommate does not own the Dog.
        clue("Nationality", "Italian", "Pet", "Dog", about_rm=True, tgt_neg=True),
        # The Spaniard does not own the Dog.
        clue("Nationality", "Spaniard", "Pet", "Dog", tgt_neg=True),
    ]
    print("Full puzzle:")
    result = solve_zebra(CATEGORIES, sat_clues)
    assert result is not None
    for i, person in enumerate(result):
        print(f"Person {i}:")
        for property in person:
            print(f"  {property}: {person[property]}")
