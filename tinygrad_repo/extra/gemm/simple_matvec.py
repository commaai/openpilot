import numpy as np
from tinygrad.helpers import getenv
from tinygrad import dtypes, Tensor, Device
dtype_in = dtypes.half if getenv("HALF") else dtypes.bfloat16 if getenv("BFLOAT16") else dtypes.float
acc_dtype = dtypes.half if getenv("ACC_HALF") else dtypes.bfloat16 if getenv("ACC_BFLOAT16") else None

GPUS = getenv("GPUS", 0)
M = getenv("M", 16384)
N = getenv("N", 4096)
CNT = getenv("CNT", 10)
ATOL = getenv("ATOL", 1e-4)
RTOL = getenv("RTOL", 3e-2)

def _rand(device):
  a, b = Tensor.rand(M, N, dtype=dtype_in).realize(), Tensor.rand(N, dtype=dtype_in).realize()
  if isinstance(device, tuple):
    a.shard_(device, axis=1)
    b.shard_(device, axis=0)
  return a, b

if __name__ == "__main__":
  device = tuple(f"{Device.DEFAULT}:{i}" for i in range(GPUS)) if GPUS > 1 else Device.DEFAULT
  a, b = _rand(device)
  for i in range(CNT):
    if i > 0 and getenv("RAND", 0) != 0:
      a, b = _rand(device)
    c = a.matmul(b, acc_dtype=acc_dtype).realize()
  nc = c.numpy()
  comp = a.numpy().astype(np.float32) @ b.numpy().astype(np.float32)
  np.testing.assert_allclose(nc, comp, atol=ATOL, rtol=RTOL)
