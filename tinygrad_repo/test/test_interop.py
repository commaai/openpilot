#!/usr/bin/env python
import unittest
import torch
import numpy as np

from tinygrad.helpers import getenv, CI
from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.dtype import _from_torch_dtype, _to_torch_dtype

MOCKGPU = getenv("MOCKGPU")

@unittest.skipIf(Device.DEFAULT not in ["METAL", "CUDA"] or MOCKGPU, f"no support on {Device.DEFAULT}")
class TestInterop(unittest.TestCase):
  def setUp(self):
    if Device.DEFAULT == "CUDA": self.torch_device = "cuda"
    elif Device.DEFAULT == "METAL": self.torch_device = "mps"

  def test_torch_interop(self):
    inp = torch.rand(2, 2, 3, device=torch.device(self.torch_device))

    if self.torch_device == "mps": torch.mps.synchronize()
    else: torch.cuda.synchronize()

    tg_data = Tensor.from_blob(inp.data_ptr(), inp.shape, dtype=_from_torch_dtype(inp.dtype))

    tg_out = tg_data[:, :, 0] * 0.2989 + tg_data[:, :, 1] * 0.5870 + tg_data[:, :, 2] * 0.1140
    tg_res = tg_out.numpy()

    if self.torch_device == "mps" and CI:
      # MPS backend out of memory: https://discuss.pytorch.org/t/mps-back-end-out-of-memory-on-github-action/189773
      # Calculate expected value on cpu.
      inp = inp.cpu()
    torch_out = inp[:, :, 0] * 0.2989 + inp[:, :, 1] * 0.5870 + inp[:, :, 2] * 0.1140

    np.testing.assert_allclose(tg_res, torch_out.cpu().numpy(), atol=1e-5, rtol=1e-5)

  def test_torch_interop_write(self):
    tg_data = Tensor.randn((4, 4), device=Device.DEFAULT)

    out = torch.empty(4, 4, device=torch.device(self.torch_device), dtype=_to_torch_dtype(tg_data.dtype))
    tg_out = Tensor.from_blob(out.data_ptr(), out.shape, dtype=_from_torch_dtype(out.dtype))

    tg_out.assign(tg_data).realize()
    Device[Device.DEFAULT].synchronize()

    torch_out_np = out.cpu().numpy()

    np.testing.assert_allclose(tg_data.numpy(), torch_out_np, atol=1e-5, rtol=1e-5)

if __name__ == '__main__':
  unittest.main()
