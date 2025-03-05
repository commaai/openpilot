import itertools
import random
from tinygrad import Variable, dtypes
from tinygrad.ops import UOp
from tinygrad.helpers import DEBUG
random.seed(42)

def add_v(expr, rng=None):
  if rng is None: rng = random.randint(0,2)
  return expr + v[rng], rng

def div(expr, rng=None):
  if rng is None: rng = random.randint(1,9)
  return expr // rng, rng

def mul(expr, rng=None):
  if rng is None: rng = random.randint(-4,4)
  return expr * rng, rng

def mod(expr, rng=None):
  if rng is None: rng = random.randint(1,9)
  return expr % rng, rng

def add_num(expr, rng=None):
  if rng is None: rng = random.randint(-4,4)
  return expr + rng, rng

def lt(expr, rng=None):
  if rng is None: rng = random.randint(-4,4)
  return expr < rng, rng

def ge(expr, rng=None):
  if rng is None: rng = random.randint(-4,4)
  return expr >= rng, rng

def le(expr, rng=None):
  if rng is None: rng = random.randint(-4,4)
  return expr <= rng, rng

def gt(expr, rng=None):
  if rng is None: rng = random.randint(-4,4)
  return expr > rng, rng

# NOTE: you have to replace these for this test to pass
from tinygrad.ops import python_alu, Ops
python_alu[Ops.MOD] = lambda x,y: x%y
python_alu[Ops.IDIV] = lambda x,y: x//y

if __name__ == "__main__":
  ops = [add_v, div, mul, add_num, mod]
  for _ in range(1000):
    upper_bounds = [*list(range(1, 10)), 16, 32, 64, 128, 256]
    u1 = Variable("v1", 0, random.choice(upper_bounds))
    u2 = Variable("v2", 0, random.choice(upper_bounds))
    u3 = Variable("v3", 0, random.choice(upper_bounds))
    v = [u1,u2,u3]
    tape = [random.choice(ops) for _ in range(random.randint(2, 30))]
    # 10% of the time, add one of lt, le, gt, ge
    if random.random() < 0.1: tape.append(random.choice([lt, le, gt, ge]))
    expr = UOp.const(dtypes.int, 0)
    rngs = []
    for t in tape:
      expr, rng = t(expr)
      if DEBUG >= 1: print(t.__name__, rng)
      rngs.append(rng)
    if DEBUG >=1: print(expr)
    space = list(itertools.product(range(u1.vmin, u1.vmax+1), range(u2.vmin, u2.vmax+1), range(u3.vmin, u3.vmax+1)))
    volume = len(space)
    for (v1, v2, v3) in random.sample(space, min(100, volume)):
      v = [v1,v2,v3]
      rn = 0
      for t,r in zip(tape, rngs): rn, _ = t(rn, r)
      num = eval(expr.render())
      if num != rn:
        unsimplified_num = eval(expr.render(simplify=False))
        assert unsimplified_num == rn, "UNSIMPLIFIED MISMATCH!"
        assert num == rn, f"mismatched {expr.render()} at {v1=} {v2=} {v3=} = {num} != {rn}\n{expr.render(simplify=False)}"
      if DEBUG >= 1: print(f"matched {expr.render()} at {v1=} {v2=} {v3=} = {num} == {rn}")
