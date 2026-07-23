from tinygrad import Tensor, Device, TinyJit, dtypes

GPUS = Device[Device.DEFAULT].count()
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
