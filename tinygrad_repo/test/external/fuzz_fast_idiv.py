import random
import z3
from tinygrad import dtypes
from tinygrad.uop.spec import z3_renderer, z3_cdiv
from tinygrad.uop.ops import UOp, graph_rewrite
from tinygrad.uop.decompositions import fast_idiv
random.seed(42)

powers_of_two = [2**i for i in range(64)]
if __name__ == "__main__":
  for i in range(10_000):
    if i % 1000 == 0:
      print(f"Progress: {i}")
    dt = random.choice(dtypes.ints)
    u = UOp.variable('x', random.randint(dt.min, 0), random.randint(1, dt.max), dtype=dt)
    d = random.randint(1, max(1, u.arg[2]))
    if d in powers_of_two: continue
    expr = fast_idiv(None, u, d)
    if expr is None: continue

    solver = z3.Solver()
    z3_sink = graph_rewrite(expr.sink(u), z3_renderer, ctx=(solver, {}))
    z3_expr, x = z3_sink.src[0].arg, z3_sink.src[1].arg

    if solver.check(z3_expr != z3_cdiv(x, d)) == z3.sat:
      assert False, f"Failed: {expr.render()} != x//{d} at x={solver.model()}\nx={u}\nd={d}\n{z3_expr=}\n{x/d=}"
