#!POPCORN leaderboard grayscale
#!POPCORN gpu A100
# not a stable API, but works

import torch, functools
from tinygrad import Tensor, TinyJit, Device
from tinygrad.engine.realize import CompiledRunner
from tinygrad.helpers import get_single_element, Context, OSX
from tinygrad.dtype import _from_torch_dtype

@TinyJit
def f(tg_out, tg_data): return tg_out.assign(tg_data[:, :, 0] * 0.2989 + tg_data[:, :, 1] * 0.5870 + tg_data[:, :, 2] * 0.1140).realize()

def custom_kernel(data: torch.Tensor, device="CUDA") -> torch.Tensor:
  assert data.dtype == torch.float32
  tg_data = Tensor.from_blob(data.data_ptr(), data.shape, dtype=_from_torch_dtype(data.dtype), device=device)

  out = torch.empty((data.shape[0], data.shape[1]), dtype=data.dtype, device=data.device)
  tg_out = Tensor.from_blob(out.data_ptr(), out.shape, dtype=_from_torch_dtype(out.dtype), device=device)

  # Need to sync torch to make sure the data is valid.
  if data.device.type == "mps": torch.mps.synchronize()
  else: torch.cuda.synchronize()

  with Context(BEAM=2): f(tg_out, tg_data)

  # Wait for computation to finish and the data is valid.
  Device[device].synchronize()

  return out

if __name__ == "__main__":
  for i in range(3):
    if OSX:
      out = custom_kernel(inp:=torch.rand(16, 16, 3, device=torch.device("mps")), device="METAL")
    else:
      out = custom_kernel(inp:=torch.rand(16, 16, 3, device=torch.device("cuda")), device="CUDA")
    assert torch.allclose(out, inp[:, :, 0] * 0.2989 + inp[:, :, 1] * 0.5870 + inp[:, :, 2] * 0.1140)
