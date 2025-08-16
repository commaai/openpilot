import random
from z3 import Int, Solver, sat
from tinygrad import dtypes, Device
from tinygrad.uop.ops import UOp, Ops, UPat, graph_rewrite, PatternMatcher
from tinygrad.uop.transcendental import fast_idiv
random.seed(42)

z3_renderer = PatternMatcher([
  (UPat((Ops.DEFINE_VAR, Ops.SPECIAL), name="x"), lambda x: UOp(Ops.NOOP, arg=x.arg[0])),
  # Because fast_idiv only works for non-negative integers we can emulate machine arithmetic with modulo operations.
  (UPat(Ops.SHR, src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP, arg=f"(({x.src[0].arg}/(2**{x.src[1].arg}))%{dtypes.max(x.dtype)+1})")),
  (UPat(Ops.MUL, src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP, arg=f"(({x.src[0].arg}*{x.src[1].arg})%{dtypes.max(x.dtype)+1})")),
  (UPat((Ops.CONST, Ops.VCONST), name="x"), lambda x: UOp(Ops.NOOP, arg=str(x.arg))),
  (UPat(Ops.CAST, src=UPat(Ops.NOOP), name="x"), lambda x:  UOp(Ops.NOOP, arg=f"{x.src[0].arg}")),
])

def render(self) -> str:
  ret = graph_rewrite(self.simplify(), z3_renderer)
  return ret.arg if ret.op is Ops.NOOP else str(ret)

if __name__ == "__main__":
  x = Int('x')
  for _ in range(10_000):
    dt = random.choice(dtypes.ints)
    u = UOp(Ops.DEFINE_VAR, dt, arg=('x', 0, random.randint(1, dtypes.max(dt))), src=())
    d = random.randint(1, max(1, u.arg[2]))

    expr = fast_idiv(Device[Device.DEFAULT].renderer, u, d)
    if expr is None: continue
    solver = Solver()
    solver.add(x>=u.arg[1], x<=u.arg[2])
    if solver.check(eval(render(expr)) != x/d) == sat:
      assert False, f"Failed: {render(expr)} != x//{d} at x={solver.model()[x]}\nx={u}\nd={d}"
