import pickle, sys
from tinygrad.helpers import getenv, is_image_shape, temp
from tinygrad.uop.ops import Ops
from tinygrad.viz.serve import VizData, _reconstruct

with open(sys.argv[1] if len(sys.argv) > 1 else temp("rewrites.pkl", append_user=True), "rb") as f: data = VizData(pickle.load(f))
steps = [s for key, ctx in zip(data.trace.keys, data.trace.rewrites) if key.display_name.startswith("JIT ")
         for s in ctx if s.name == "View captured linear"]
assert len(steps) == 1, f"expected one JIT compilation, found {len(steps)}"
asts = [call.without_after.src[0] for call in _reconstruct(data, steps[0].sink).src]
by_ast = {key.keys[1]: _reconstruct(data, s.sink) for key, ctx in zip(data.trace.keys, data.trace.rewrites)
          for s in ctx if s.name == "View Program" and key.keys[1] in asts}
programs = [by_ast[ast] if ast.op is Ops.SINK else ast for ast in asts if ast.op in {Ops.SINK, Ops.PROGRAM}]
read_image, gated_read_image = 0, 0
for prg in programs:
  uops = prg.src[1].src
  reads = {u for u in uops if u.op is Ops.LOAD and u.src[0].op is Ops.INDEX and is_image_shape(u.src[0].src[0]._shape)}
  gated = {u for u in reads if len(u.src) == 3}
  # include image channels masked after loading, count each load only once.
  for u in uops:
    if u.op is not Ops.WHERE: continue
    for value in u.src[1:]:
      while value.op in {Ops.INDEX, Ops.SHRINK, Ops.CAST, Ops.BITCAST}: value = value.src[0]
      if value in reads: gated.add(value)
  read_image += len(reads)
  gated_read_image += len(gated)

for name, count in zip(("KERNEL_COUNT", "READ_IMAGE", "GATED_READ_IMAGE"), (len(programs), read_image, gated_read_image)):
  print(f"{name}={count}")
  if (allowed:=getenv(f"ALLOWED_{name}", -1)) != -1: assert count == allowed, f"ALLOWED_{name}={allowed}, got {count}"
