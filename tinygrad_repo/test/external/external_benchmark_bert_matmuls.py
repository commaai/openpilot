from tinygrad import Tensor, dtypes
dtypes.default_float = dtypes.float16
from tinygrad.dtype import to_dtype
from tinygrad.helpers import getenv

if __name__ == "__main__":
  # matmuls in bert layers
  BS = getenv("BS", 96//6)
  acc_dtype = to_dtype(getenv("ACC_DTYPE", "half"))
  tensors = [
    (Tensor.empty(BS, 512, 1024), Tensor.empty(1024, 1024).T),                                          # linear to get qkv
    (Tensor.empty(BS, 512, 16, 64).permute(0,2,1,3), Tensor.empty(BS, 512, 16, 64).permute(0,2,3,1)),   # q@k
    (Tensor.empty(BS, 16, 512, 512), Tensor.empty(BS, 512, 16, 64).permute(0,2,1,3)),                   # qk@v
  ]
  for t0, t1 in tensors:
    print(f"{t0.shape=}, {t0.lazydata.st.real_strides()=}, {t1.shape=}, {t1.lazydata.st.real_strides()=}")
    for _ in range(5):
      t0.dot(t1, dtype=acc_dtype).realize()
