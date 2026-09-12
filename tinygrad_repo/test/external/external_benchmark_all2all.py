from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.helpers import Timing, Context

GPUS, DEPTH, SZ = 8, 4, 128 * 2**20
WARMUP, ITERS = 3, 5
devs = tuple(f"{Device.DEFAULT}:{i}" for i in range(GPUS))
bufs = tuple(Tensor.empty(SZ, dtype=dtypes.uint8, device=dev).contiguous().realize() for _ in range(DEPTH) for dev in devs)

@TinyJit
def all_to_all(*srcs:Tensor): return Tensor.realize(*(src.to(dst) for i,src in enumerate(srcs) for j,dst in enumerate(devs) if i % GPUS != j))

if __name__ == "__main__":
  with Context(ALL2ALL=1, JIT_BATCH_SIZE=0):
    for i in range(-WARMUP, ITERS):
      with Timing("ALL2ALL ", lambda ns: f" {SZ*GPUS*(GPUS-1)*DEPTH/ns:.2f} GB/s", enabled=i>=0):
        all_to_all(*bufs)
        for dev in devs: Device[dev].synchronize()
