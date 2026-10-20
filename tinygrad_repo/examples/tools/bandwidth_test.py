#!/usr/bin/env python3
from tinygrad import Tensor, Device, GlobalCounters, Context, dtypes
from tinygrad.helpers import getenv, colored

SZ = 8_000_000_000
GPUS = getenv("GPUS", 4) # TODO: expose a way in tinygrad to access this

if __name__ == "__main__":
  # create tensors
  tens = [Tensor.ones(SZ, dtype=dtypes.uint8, device=f"{Device.DEFAULT}:{i}").contiguous() for i in range(GPUS)]
  Tensor.realize(*tens)

  bw = [[0.0]*GPUS for _ in range(GPUS)]
  for i in range(GPUS):
    for j in range(GPUS):
      GlobalCounters.reset()
      with Context(DEBUG=2):
        if i == j:
          # this copy would be optimized out, just add 1
          (tens[i]+1).realize()
        else:
          tens[i].to(f"{Device.DEFAULT}:{j}").realize()
      t = max(GlobalCounters.time_sum_s, 1e-9)
      bw[i][j] = SZ / t / 1e9  # GB/s

  def fmt(x):
    c = "green" if x > 50 else "yellow" if x > 20 else "red"
    return colored(f"{x:6.1f}", c)

  # header
  print(" " * 8 + " ".join(f"{'d'+str(j):>6}" for j in range(GPUS)))
  # rows
  for i in range(GPUS):
    print(f"{'s'+str(i):>6} -> " + " ".join(fmt(x) for x in bw[i]))
