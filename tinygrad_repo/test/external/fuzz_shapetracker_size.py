from tinygrad.shape.shapetracker import ShapeTracker
from test.external.fuzz_shapetracker import shapetracker_ops as st_ops
from test.unit.test_shapetracker_math import MultiShapeTracker
from tinygrad.helpers import getenv
import random

random.seed(getenv("SEED", 42))
for i in range(getenv("CNT", 2000)):
  if getenv("DEBUG", 0) >= 1: print()
  N = random.randint(1, 10000)
  mst = MultiShapeTracker([ShapeTracker.from_shape((N,))])  # st_ops don't mutate regular shapetrackers for some reason
  for j in range(20): random.choice(st_ops)(mst)
  assert mst.sts[0].real_size() <= N, f"{N=}, real_size={mst.sts[0].real_size()}, st={mst.sts[0]}"
