import pickle, sys
from tinygrad.helpers import getenv, is_image_shape
from tinygrad.uop.ops import Ops

# Inspect the actual compiled calls, including repeated launches of the same program.
with open(sys.argv[1], "rb") as f: linear = pickle.load(f)['run'].captured._linear
by_key = {u.key: u for u in linear.toposort() if u.op is Ops.PROGRAM}
programs = []
for entry in linear.src:
  call = entry.without_after
  if call.arg.hcq is not None:  # count the kernels in the batch, not the host program that submits it
    programs.extend(by_key[key] for _, _, _, _, key, _, _ in call.arg.hcq.kernels if key is not None)
  elif call.body.op is Ops.PROGRAM: programs.append(call.body)
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
