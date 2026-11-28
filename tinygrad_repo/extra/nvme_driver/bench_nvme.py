#!/usr/bin/env python3
# GPUS=4 DRIVES=4 GB=2 python3 extra/nvme_driver/bench_nvme.py: each gpu reads GB GiB from its DRIVES drives at once. never writes
import time
from tinygrad import Tensor, Device, TinyJit, dtypes
from tinygrad.helpers import getenv

GPUS, DRIVES, N = getenv("GPUS", 4), getenv("DRIVES", 4), getenv("GB", 2) << 30
gpus = [f"AMD:{g}" for g in range(GPUS)]
srcs = [Tensor.empty(sz:=Device[f"NVME:{d}"].nvme.size, dtype=dtypes.uint8, device=f"NVME:{d}")[sz // 2:sz // 2 + N] for d in range(GPUS * DRIVES)]
dsts = [Tensor.empty(N, dtype=dtypes.uint8, device=gpus[d // DRIVES]).realize() for d in range(GPUS * DRIVES)]
@TinyJit
def read() -> list[Tensor]: return [dst.assign(src.to(dst.device)).realize() for dst, src in zip(dsts, srcs)]
def run(n:int) -> float:
  for g in gpus: Device[g].synchronize()
  st = time.perf_counter()
  for _ in range(n): read()
  for g in gpus: Device[g].synchronize()
  return time.perf_counter() - st
run(3)
for n in (1, getenv("ITERS", 8)):
  dt = run(n)
  print(f"{len(srcs)} drives -> {GPUS} gpus, {n} x {len(srcs) * N / 2**30:.0f} GiB in {dt * 1e3:.0f} ms: "
        f"{n * len(srcs) * N / dt / 1e9:.1f} GB/s, {n * DRIVES * N / dt / 1e9:.1f} GB/s a gpu")
