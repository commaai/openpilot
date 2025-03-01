import numpy as np
from tinygrad.helpers import getenv
from tinygrad import dtypes, Tensor

dtype_in = dtypes.half if getenv("HALF") else dtypes.bfloat16 if getenv("BFLOAT16") else dtypes.float
acc_dtype = dtypes.half if getenv("ACC_HALF") else dtypes.bfloat16 if getenv("ACC_BFLOAT16") else None
if getenv("INT"):  dtype_in, acc_dtype = dtypes.int8, dtypes.int32
if getenv("UINT"): dtype_in, acc_dtype = dtypes.uint8, dtypes.int32

N = getenv("N", 4096)
M = getenv("M", N)
K = getenv("K", N)
CNT = getenv("CNT", 10)
ATOL = getenv("ATOL", 1e-4)
RTOL = getenv("RTOL", 3e-2)

if __name__ == "__main__":
  def init_matrix(rows, cols):
    if dtype_in in dtypes.ints:
      return Tensor.randint((rows, cols), dtype=dtype_in).realize()
    return Tensor.rand(rows, cols, dtype=dtype_in).realize()

  a, b = init_matrix(M, K), init_matrix(K, N)
  for i in range(CNT):
    if i > 0 and getenv("RAND", 0) != 0:
      a, b = init_matrix(M, K), init_matrix(K, N)
    c = a.matmul(b, acc_dtype=acc_dtype).realize()

  ref = a.numpy().astype(np.float32) @ b.numpy().astype(np.float32)
  res = c.numpy()
  try:
    np.testing.assert_allclose(res, ref, rtol=RTOL, atol=ATOL)
  except AssertionError as e:
    if getenv("DEBUG_VALUES", 0) > 0:
      mismatch = np.where(~np.isclose(res, ref, rtol=RTOL, atol=ATOL))
      print("Mismatch indices:", mismatch)
      print("Result          :", res[mismatch])
      print("Ground truth    :", ref[mismatch])
    raise e
