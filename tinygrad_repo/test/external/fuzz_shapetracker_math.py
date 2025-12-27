import random
from tinygrad.helpers import getenv, DEBUG, colored, trange
from tinygrad.shape.shapetracker import ShapeTracker
from test.external.fuzz_shapetracker import shapetracker_ops
from test.unit.test_shapetracker_math import st_equal, MultiShapeTracker

def fuzz_plus() -> tuple[ShapeTracker, ShapeTracker]:
  m = MultiShapeTracker([ShapeTracker.from_shape((random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)))])
  for _ in range(4): random.choice(shapetracker_ops)(m)
  backup = m.sts[0]
  m.sts.append(ShapeTracker.from_shape(m.sts[0].shape))
  for _ in range(4): random.choice(shapetracker_ops)(m)
  st_sum = backup + m.sts[1]
  return m.sts[0], st_sum

if __name__ == "__main__":
  if seed:=getenv("SEED"): random.seed(seed)
  total = getenv("CNT", 1000)
  for fuzz in [globals()[f'fuzz_{x}'] for x in getenv("FUZZ", "plus").split(",")]:
    same_but_neq = 0
    for _ in trange(total, desc=f"{fuzz}"):
      st1, st2 = fuzz()
      eq = st_equal(st1, st2)
      if getenv("CHECK_NEQ") and eq and st1.simplify() != st2.simplify():
        print(colored("same but unequal", "yellow"))
        print(st1.simplify())
        print(st2.simplify())
        same_but_neq += 1
      if DEBUG >= 1:
        print(f"EXP: {st1}")
        print(f"GOT: {st2}")
        print(colored("****", "green" if eq else "red"))
      if not eq: exit(0)
    if getenv("CHECK_NEQ"): print(f"same but unequal {same_but_neq}/{total} = {(same_but_neq/total)*100:.2f}%")
