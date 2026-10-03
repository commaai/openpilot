from tinygrad import Tensor, dtypes, GlobalCounters
from tinygrad.dtype import to_dtype
from tinygrad.helpers import getenv, Context
from test.backend.test_softmax_fusion import single_kernel_softmax

Context(DEFAULT_FLOAT=dtypes.float16).__enter__()

if __name__ == "__main__":
  # softmax in bert layers
  BS = getenv("BS", 96//6)
  acc_dtype = to_dtype(getenv("ACC_DTYPE", "half"))
  t = Tensor.empty(BS, 16, 512, 512)
  t.softmax(-1, dtype="half").realize()

  # test single kernel softmax
  GlobalCounters.reset()
  single_kernel_softmax(t, -1, acc_dtype).realize()

