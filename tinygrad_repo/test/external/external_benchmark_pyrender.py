# benchmark speed of pyrender for all created UOps saved with TRACK_MATCH_STATS=2
import functools, pickle
from tinygrad.uop.ops import UOp, Ops
from tinygrad.helpers import tqdm, temp, time_to_str, cpu_profile

BENCHMARK_OPS = {Ops.INDEX, Ops.BUFFERIZE}

@functools.cache
def create_uop(a:int) -> UOp:
  op, dtype, src, arg, *rest = trace.uop_fields[a]
  return UOp(op, dtype, tuple(create_uop(s) for s in src), arg, *rest)

if __name__ == "__main__":
  # load rewrite trace
  with open(temp("rewrites.pkl", append_user=True), "rb") as f:
    trace = pickle.load(f)

  # benchmark
  result:list[tuple[str, int]] = []
  try:
    for steps in tqdm(trace.rewrites):
      for r in steps:
        for _,yn,_,__ in r.matches:
          y = create_uop(yn)
          if y.op in BENCHMARK_OPS:
            with cpu_profile("pyrender") as e:
              try: ren = y.render()
              except Exception: ren = "PYRENDER_ERR"
            result.append((ren, float(e.en-e.st)/1e6))
  finally:
    N = 10
    print(f"Slowst {N} renders from {len(result)} samples:")
    for ren,tm in sorted(result, key=lambda x:x[1], reverse=True)[:N]:
      print(f"{time_to_str(tm).strip():<10s} {ren}")
