from tinygrad import Tensor, Device, TinyJit, dtypes
from tinygrad.helpers import getenv

GPUS = getenv("GPUS", 4) # TODO: expose a way in tinygrad to access this
N = 6144

@TinyJit
def many_matmul(A, B):
  out = A
  for _ in range(8): out = out@B
  return out

if __name__ == "__main__":
  A = Tensor.ones(GPUS, N, N, dtype=dtypes.half).shard(devices=tuple([f"{Device.DEFAULT}:{i}" for i in range(GPUS)]), axis=0).contiguous()
  B = Tensor.ones(GPUS, N, N, dtype=dtypes.half).shard(devices=tuple([f"{Device.DEFAULT}:{i}" for i in range(GPUS)]), axis=0).contiguous()
  while 1: many_matmul(A, B)
