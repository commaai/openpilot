import numpy as np
from tinygrad.helpers import getenv
from tinygrad import dtypes, Tensor
dtype_in = dtypes.half if getenv("HALF") else dtypes.bfloat16 if getenv("BFLOAT16") else dtypes.float
acc_dtype = dtypes.half if getenv("ACC_HALF") else dtypes.bfloat16 if getenv("ACC_BFLOAT16") else None
N_START = getenv("N_START", 1)
M_START = getenv("M_START", 1)
K_START = getenv("K_START", 1)
N_STOP = getenv("N_STOP", 32)
M_STOP = getenv("M_STOP", N_STOP)
K_STOP = getenv("K_STOP", N_STOP)
N_STEP = getenv("N_STEP", 1)
M_STEP = getenv("M_STEP", 1)
K_STEP = getenv("K_STEP", 1)
ATOL = getenv("ATOL", 1e-4)
RTOL = getenv("RTOL", 3e-2)

if __name__ == "__main__":
  failed = []
  for M in range(M_START, M_STOP+1, M_STEP):
    for N in range(N_START, N_STOP+1, N_STEP):
      for K in range(K_START, K_STOP+1, K_STEP):
        print(f"testing {M=} {N=} {K=}")
        a, b = Tensor.rand(M, K, dtype=dtype_in).realize(), Tensor.rand(K, N, dtype=dtype_in).realize()
        c = a.matmul(b, acc_dtype=acc_dtype).realize()
        comp = a.numpy().astype(np.float32) @ b.numpy().astype(np.float32)
        nc = c.numpy()
        try:
          np.testing.assert_allclose(nc, comp, atol=ATOL, rtol=RTOL)
        except AssertionError as e:
          failed.append((M,N,K,))
          if getenv("DEBUG_VALUES") > 0:
            indices = np.where(~np.isclose(nc, comp, rtol=RTOL, atol=ATOL))
            non_matching_elements_nc = nc[indices]
            non_matching_elements_comp = comp[indices]
            print(indices)
            print("result      :", non_matching_elements_nc)
            print("ground truth:", non_matching_elements_comp)
          print(e)
          pass
  print(f"failed sizes: {failed}")
  print(f"num failures: {len(failed)}")
  if len(failed) > 0:
    raise RuntimeError(f"failed on {len(failed)} kernels")