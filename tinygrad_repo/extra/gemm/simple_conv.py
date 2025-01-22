from tinygrad.helpers import getenv
from tinygrad import dtypes, Tensor

dtype_in = dtypes.half if getenv("HALF") else dtypes.bfloat16 if getenv("BFLOAT16") else dtypes.float
acc_dtype = dtypes.half if getenv("ACC_HALF") else dtypes.bfloat16 if getenv("ACC_BFLOAT16") else None

CNT = getenv("CNT", 8)
BS = getenv("BS", 16)
CIN = getenv("CIN", 128)
COUT = getenv("COUT", 128)
HW = getenv("HW", 128)
K = getenv("K", 3)
PADDING = getenv("PADDING", 1)
COMP = getenv("COMP", 0)
ATOL = getenv("ATOL", 1e-4)
RTOL = getenv("RTOL", 3e-2)

FLOPS = BS*K*K*CIN*HW*HW*COUT*2
def rand_input(): return Tensor.rand(BS, CIN, HW, HW, dtype=dtype_in).realize(), Tensor.rand(COUT, CIN, K, K, dtype=dtype_in).realize()

if __name__ == "__main__":
  a, b = rand_input()
  for i in range(CNT):
    if i > 0 and getenv("RAND", 0) != 0:
      a, b = rand_input()
    c = a.conv2d(b, padding=PADDING, acc_dtype=acc_dtype).realize()

  if COMP:
    import numpy as np, time, torch
    torch_device = "cuda:0" if torch.cuda.is_available() else ("mps" if getenv("MPS", 0) else "cpu")
    ta, tb = torch.from_numpy(a.numpy()).to(torch_device), torch.from_numpy(b.numpy()).to(torch_device)
    tc = torch.nn.functional.conv2d(ta, tb, padding=PADDING)
    np.testing.assert_allclose(c.numpy(), tc.cpu(), atol=ATOL, rtol=RTOL)
