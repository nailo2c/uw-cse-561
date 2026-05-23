from z3 import BitVec
from engine.se_engine import check_assertions
from partA_clamp_sub import clamp_sub, MAX_DEPTH, BITWIDTH
from partA_safe_mul import safe_mul, MAX_DEPTH, BITWIDTH
from partA_count_groups import count_groups, MAX_DEPTH, BITWIDTH

def to_signed(n, bitwidth):
    if n >= 2 ** (bitwidth - 1):
        return n - 2 ** bitwidth
    return n


# violations = check_assertions(
#     clamp_sub,
#     max_depth=MAX_DEPTH,
#     bitwidth=BITWIDTH,
#     timeout_ms=5000,
# )

# print("violations:", len(violations))
# for v in violations:
#     print(v["model"])


# violations = check_assertions(
#     safe_mul,
#     max_depth=MAX_DEPTH,
#     bitwidth=BITWIDTH,
#     timeout_ms=5000,
# )

violations = check_assertions(
    count_groups,
    max_depth=MAX_DEPTH,
    bitwidth=BITWIDTH,
    timeout_ms=5000,
)

print("violations:", len(violations))
for v in violations:
    model = v["model"]
    print("raw model:", model)

    # a = model.eval(BitVec("a", BITWIDTH), model_completion=True).as_long()
    # b = model.eval(BitVec("b", BITWIDTH), model_completion=True).as_long()

    # print("signed:")
    # print("a =", to_signed(a, BITWIDTH))
    # print("b =", to_signed(b, BITWIDTH))

    x_raw = model.eval(BitVec("x", BITWIDTH), model_completion=True).as_long()
    print("x unsigined =", x_raw)
    print("x signed =", to_signed(x_raw, BITWIDTH))
