import numpy as np
from tinygrad.helpers import getenv
from tinygrad import dtypes, Tensor
dtype_in = dtypes.half if getenv("HALF") else dtypes.bfloat16 if getenv("BFLOAT16") else dtypes.float
acc_dtype = dtypes.half if getenv("ACC_HALF") else dtypes.bfloat16 if getenv("ACC_BFLOAT16") else None
if getenv("INT"):
  dtype_in = dtypes.int8
  acc_dtype = dtypes.int32
N = getenv("N", 4096)
M = getenv("M", N)
K = getenv("K", N)
CNT = getenv("CNT", 10)
ATOL = getenv("ATOL", 1e-4)
RTOL = getenv("RTOL", 3e-2)

if __name__ == "__main__":
  a, b = Tensor.rand(M, K, dtype=dtype_in).realize(), Tensor.rand(K, N, dtype=dtype_in).realize()
  for i in range(CNT):
    if i > 0 and getenv("RAND", 0) != 0:
      a, b = Tensor.rand(M, K, dtype=dtype_in).realize(), Tensor.rand(K, N, dtype=dtype_in).realize()
    c = a.matmul(b, acc_dtype=acc_dtype).realize()
  comp = a.numpy().astype(np.float32) @ b.numpy().astype(np.float32)
  nc = c.numpy()
  try:
    np.testing.assert_allclose(nc, comp, atol=ATOL, rtol=RTOL)
  except AssertionError as e:
    if getenv("DEBUG_VALUES") > 0:
      indices = np.where(~np.isclose(nc, comp, rtol=RTOL, atol=ATOL))
      non_matching_elements_nc = nc[indices]
      non_matching_elements_comp = comp[indices]
      print(indices)
      print("result      :", non_matching_elements_nc)
      print("ground truth:", non_matching_elements_comp)
    raise e
