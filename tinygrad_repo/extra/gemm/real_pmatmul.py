import time
from tinygrad import Tensor, Device, TinyJit
from tinygrad.helpers import getenv

if __name__ == "__main__":
  DEVS = [f"NV:{i}" for i in range(getenv("GPUS", 2))]
  N = getenv("N", 8192)
  A = Tensor.rand(N, N).shard(DEVS, 0).realize()
  B = Tensor.rand(N, N).shard(DEVS, 1).realize()
  print("***** MUL *****")
  jmatmul = TinyJit(Tensor.dot)
  for i in range(10):
    Device["NV:0"].synchronize()
    Device["NV:1"].synchronize()
    st = time.perf_counter()
    jmatmul(A, B)
    Device["NV:0"].synchronize()
    Device["NV:1"].synchronize()
    et = time.perf_counter()
    print(f"{(N*N*N*2*1e-12)/(et-st):.2f} TFLOPS")
