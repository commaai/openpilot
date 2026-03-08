from tinygrad import Tensor, TinyJit, Device
import numpy as np

GPUS = 4
N = 128
ds = tuple([Device.canonicalize(f"{Device.DEFAULT}:{i}") for i in range(GPUS)])
t = Tensor.rand(N, N, N).shard(ds, 0)
n = t.numpy()

@TinyJit
def allreduce(t:Tensor) -> Tensor:
  return t.sum(0) #.realize()

for i in range(10):
  print(i)
  tn = allreduce(t).numpy()
  np.testing.assert_allclose(tn, n.sum(0), atol=1e-4, rtol=1e-4)
